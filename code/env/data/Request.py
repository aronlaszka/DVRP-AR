from common.logger import logger
from common.types import RequestActions


class Request:
    def __init__(
            self,
            request_id,
            offset,
            origin,
            destination,
            arrival_time,
            scheduled_pickup,
            capacity,
            **kwargs
    ):
        """
        :param request_id: request-id of the request
        :param offset: offset to the request-gpu_index
        :param origin: origin of the request as (latitude, longitude)
        :param destination: destination of the request as (latitude, longitude)
        :param arrival_time: arrival time of the requests in seconds
        :param scheduled_pickup: the earliest time at which the request can be picked up in seconds
        :param capacity: the passenger capacity of individual request in the form of dictionary
        :param kwargs: key-word arguments that used to define the pickup and dropoff windows

        create the request object
        """
        self.request_id = request_id
        self.__request_id = request_id  # this request_id is never updated
        self.origin = origin
        self.destination = destination
        self.pickup_idx = 2 * (request_id + offset)
        self.dropoff_idx = self.pickup_idx + 1
        self.relative_pickup_idx = None
        self.relative_dropoff_idx = None
        self.direct_dist = self.origin.distance(self.destination)
        self.direct_time = self.origin.duration(self.destination)
        self.arrival_time = arrival_time
        self.expected_status = None
        self.__scheduled_pickup = scheduled_pickup
        self.capacity = capacity  # dictionary of multiple capacity values !!!

        # placeholder values
        self.__earliest_pickup = scheduled_pickup
        self.__latest_pickup = scheduled_pickup
        self.__earliest_dropoff = scheduled_pickup + self.direct_time
        self.__latest_dropoff = scheduled_pickup + self.direct_time

        self.configure_time_windows(**kwargs)

        self.__already_in_the_manifest = False  # indicates whether the request is already there and must be served
        self.__fixed_to_route = False  # indicator whether the requests considered in solving
        self.__picked = False  # indicator for whether this request is picked up or not
        self.__served = False  # indicator for whether this request is served or not

        self.__assigned_route_id = -1
        self.__assigned_pickup_time = -1
        self.__assigned_dropoff_time = -1

        self.max_search_duration = 1

    def configure_time_windows(self, **kwargs):
        window_size = kwargs["window_size"]
        detour_ratio = kwargs["detour_ratio"]
        detour_minimum = kwargs["detour_minimum"]
        if detour_minimum > 0:
            detour = min(detour_ratio * self.direct_time, detour_minimum)
        else:
            detour = detour_ratio * self.direct_time

        # dwell time needs to be added for Google OR Tools
        self.__earliest_pickup = self.__scheduled_pickup + self.origin.dwell_time
        if window_size > 0:
            self.__latest_pickup = self.__scheduled_pickup + window_size + self.origin.dwell_time
        else:
            self.__latest_pickup = self.__scheduled_pickup + detour + self.origin.dwell_time
        self.__earliest_dropoff = self.__earliest_pickup + self.direct_time + self.destination.dwell_time
        self.__latest_dropoff = (self.__earliest_pickup + window_size + self.direct_time +
                                 detour + self.destination.dwell_time)

        # making into integer so that GOOGLE OR tools works fine !!!
        self.__earliest_pickup = int(self.__earliest_pickup)
        self.__latest_pickup = int(self.__latest_pickup)
        self.__earliest_dropoff = int(self.__earliest_dropoff)
        self.__latest_dropoff = int(self.__latest_dropoff)

        assert self.__latest_dropoff >= self.__earliest_dropoff >= self.__earliest_pickup

    def update(self, **kwargs):
        """
        :param kwargs:
        `pickup_entry`: actual pickup using previous solution
        `dropoff_entry`: actual dropoff using previous solution
        `fixed_timeahead`: before how many seconds the fixing of vehicle and pickup window should be determined
        `fixed_vehicle`: if enabled fix the vehicle for previous requests
        `fixed_pickup`: if enabled fix the pickup time for previous request
        `fixed_dropoff`: if enabled fix the dropoff time for previous request
        `current_time`: time of the system, to make sure the request is no-longer changed
        :return: the updated request
        """
        # once received the request will be already in manifest !!!
        self.__already_in_the_manifest = True

        pickup_entry = kwargs['pickup_entry'] if 'pickup_entry' in kwargs else None
        dropoff_entry = kwargs['dropoff_entry'] if 'dropoff_entry' in kwargs else None

        look_ahead_time = kwargs["current_time"] + kwargs["fixed_timeahead"]

        pickup_fixed_reached = False
        if pickup_entry is not None:
            if look_ahead_time >= pickup_entry.action_time:
                pickup_fixed_reached = True
            # assert self.__earliest_pickup <= pickup_entry.action_time <= self.__latest_pickup
            self.__assigned_pickup_time = pickup_entry.action_time

        dropoff_fixed_reached = False
        if dropoff_entry is not None:
            if look_ahead_time >= dropoff_entry.action_time:
                dropoff_fixed_reached = True
                # assert self.__earliest_dropoff <= dropoff_entry.action_time <= self.__latest_dropoff
                self.__assigned_dropoff_time = dropoff_entry.action_time

        if look_ahead_time >= self.__earliest_pickup:
            # until this point assignment not-need to be finalized and after-wards there will be no-change
            self.__fixed_to_route = True
            if pickup_entry is not None:
                if self.__assigned_route_id < 0:
                    self.__assigned_route_id = pickup_entry.real_route_id
                else:
                    if not self.__picked:
                        self.__assigned_route_id = pickup_entry.real_route_id
                    else:
                        assert self.__assigned_route_id == pickup_entry.real_route_id

        if pickup_entry is not None:
            if kwargs["current_time"] >= pickup_entry.action_time:
                # note, we will look the future requests earlier to make sure they are unaffected by other requests
                self.__picked = True  # marked as picked
                self.__enforce_pickup(pickup_entry.action_time)

            if pickup_fixed_reached:
                # this part is where the pickup is not officially happened but solution will not
                # update the pickup hear after
                self.__enforce_pickup(pickup_entry.action_time)

        if dropoff_entry is not None:
            if kwargs["current_time"] >= dropoff_entry.action_time:
                # note, we will look the future requests earlier to make sure they are unaffected by other requests
                self.__served = True  # marked as served
                self.__enforce_dropoff(dropoff_entry.action_time)

            if dropoff_fixed_reached:
                # this part is where the dropoff is not officially happened but solution will not
                # update the dropoff hear after
                self.__enforce_dropoff(dropoff_entry.action_time)

        return self

    def __enforce_pickup(self, action_time):
        """
        :param action_time: pickup action time
        this will fix the pickup time
        """
        self.__earliest_pickup = action_time
        self.__latest_pickup = action_time
        # earliest dropoff also needs to be updated
        self.__earliest_dropoff = action_time + self.direct_time + self.destination.dwell_time
        error_msg = f"Earliest Dropoff {self.__earliest_dropoff} is after Latest Dropoff {self.__latest_dropoff}"
        if self.__earliest_dropoff > self.__latest_dropoff:
            logger.critical(error_msg.upper())

    def __enforce_dropoff(self, action_time):
        """
        :param action_time: dropoff action time
        this will fix the dropoff time
        """
        self.__earliest_dropoff = action_time
        self.__latest_dropoff = action_time

    def update_id(self, request_id, offset=0):
        """
        this function will only update both original and relative pickup and dropoff index
        :param request_id: request identifier
        :param offset: offset for the pickup index
        :return: request object
        """
        self.request_id = request_id
        self.pickup_idx = 2 * (request_id + offset)
        self.dropoff_idx = self.pickup_idx + 1
        self.relative_pickup_idx = None
        self.relative_dropoff_idx = None
        return self

    def update_relative_id(self, request_id, offset=0):
        """
        this function will only update both original and relative pickup and dropoff index
        :param request_id: request identifier
        :param offset: offset for the pickup index
        :return: request object
        """
        self.request_id = request_id
        self.pickup_idx = 2 * (request_id + offset)
        self.dropoff_idx = self.pickup_idx + 1
        self.relative_pickup_idx = self.pickup_idx
        self.relative_dropoff_idx = self.relative_pickup_idx + 1
        return self

    def set_relative_index(self, idx=None, action_type=RequestActions.Pickup):
        if action_type == RequestActions.Pickup:
            self.relative_pickup_idx = idx
            self.relative_dropoff_idx = idx + 1
        else:
            self.relative_pickup_idx = None
            self.relative_dropoff_idx = idx
        return self

    def time_window_feasible(self, depot_loc):
        """
        :param depot_loc: location of the depot
        :return: whether the request can be considered to serve mathematically
        """
        return self.__latest_pickup - depot_loc.duration(self.origin) >= 0

    def get_earliest_pickup(self, current_time):
        """
        :param current_time: current-time
        :return: the earliest pickup time of the request
        """
        if self.__picked:
            assert self.__assigned_pickup_time <= current_time
            assert self.__assigned_pickup_time > -1
            return self.__assigned_pickup_time
        return int(self.__earliest_pickup)

    def get_earliest_dropoff(self, current_time):
        """
        :param current_time: current-time
        :return: the earliest dropoff time of the request
        """
        if self.__served:
            assert self.__assigned_dropoff_time <= current_time
            assert self.__assigned_dropoff_time > -1
            return self.__assigned_dropoff_time
        elif self.__picked:
            assert self.__latest_dropoff >= current_time
            return int(self.__earliest_dropoff)  # this should be updated in the update function
        return int(self.__earliest_dropoff)

    def get_latest_pickup(self, current_time):
        """
        :param current_time: current-time
        :return: the latest pickup time of the request
        """
        if self.__picked:
            assert self.__assigned_pickup_time <= current_time
            assert self.__assigned_pickup_time > -1
            return self.__assigned_pickup_time
        return int(self.__latest_pickup)

    def get_latest_dropoff(self, current_time):
        """
        :param current_time: current-time
        :return: the latest dropoff time of the request
        """
        if self.__served:
            assert self.__assigned_dropoff_time <= current_time
            assert self.__assigned_dropoff_time > -1
            return self.__assigned_dropoff_time
        elif self.__picked:
            assert self.__earliest_dropoff <= self.__latest_dropoff
            assert self.__latest_dropoff >= current_time
            return int(self.__latest_dropoff)  # this should be updated in the update function
        return int(self.__latest_dropoff)

    def fixed_to_route(self):
        """
        configure the request as fixed to a single specific route when feeding as an input to the solver
        """
        self.__fixed_to_route = True

    def is_fixed_to_route(self):
        """
        :return: indicates whether the request is fixed to a single specific route when feeding as an input to the solver
        (at this point assigned route is no-longer changed)
        """
        return self.__fixed_to_route

    def is_already_in_manifest(self):
        """
        :return: indicates whether the request is added to at-least one route
        """
        return self.__already_in_the_manifest

    def is_picked(self):
        """
        :return: indicates whether the request is picked up or not
        """
        return self.__picked

    def is_served(self):
        """
        :return: indicates whether the request is served or not
        """
        return self.__served

    def get_assigned_route_id(self):
        """
        :return: provides the assigned route id if the request assigned to a route
        """
        return self.__assigned_route_id

    def get_assigned_pickup_time(self):
        """
        :return: provides the assigned pickup time if the request assigned to a route
        """
        return self.__assigned_pickup_time

    def get_assigned_dropoff_time(self):
        """
        :return: provides the assigned dropoff time if the request assigned to a route
        """
        return self.__assigned_dropoff_time

    def assert_pickup_time(self, pickup_time, current_time):
        """
        :param pickup_time: pickup time output by the solver
        :param current_time: current system time
        assert whether the pickup time input by the solver is valid or not
        """
        assert (self.get_earliest_pickup(current_time) <= pickup_time <= self.get_latest_pickup(current_time))

    def assert_dropoff_time(self, dropoff_time, current_time):
        """
        :param dropoff_time: dropoff time output by the solver
        :param current_time: current system time
        assert whether the dropoff time input by the solver is valid or not
        """
        assert (self.get_earliest_dropoff(current_time) <= dropoff_time <= self.get_latest_dropoff(current_time))
