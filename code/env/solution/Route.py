import copy

from common.logger import logger
from common.types import RequestActions, ObjectiveTypes, LocationInterpolationTypes, ExecutionModes
from env.base.ImplementationError import ImplementationError
from env.data.Location import LocationTypes
from env.solution.Action import ComprehensiveAction
from env.solution.RouteBase import RouteBase
from learn.agent.AgentManager import AgentManager


class Route(RouteBase):
    def __init__(
            self,
            route_id,
            start_loc,
            end_loc,
            capacity,
            start_time,
            end_time,
            time_ahead,
            look_ahead_horizon,
            interpolation_type,
            search_mode,
            execution_mode
    ):
        """
        :param route_id: id of the vehicle
        :param start_loc: location of the depot
        :param end_loc: capacity of the vehicle in dictionary format
        :param capacity: maximum capacity for different passenger classes such as "am" and "wc"
        :param start_time: start time of the operation of the vehicle
        :param end_time: end time of the operation of the vehicle
        :param time_ahead: time ahead that needs to be fixed
        :param look_ahead_horizon: time horizon that needs to be considered when choosing an objective function
        :param interpolation_type: type of interpolation when adjusting the location
        :param search_mode: indicates whether the route is initialized in search mode or not

        create the vehicle object
        """
        self.route_id = route_id
        self.original_start_loc = copy.deepcopy(start_loc)
        self.original_end_loc = copy.deepcopy(end_loc)
        self.capacity = capacity
        self.original_start_time = start_time
        self.original_end_time = end_time
        self.time_ahead = time_ahead
        self.look_ahead_horizon = look_ahead_horizon
        self.interpolation_type = interpolation_type
        self.search_mode = search_mode
        self.execution_mode = execution_mode

        self.allow_negative_capacity = False

        self.nodes = []
        self.times = []
        self.un_served_request_ids = {}
        self.movable_idx = -1
        self.actual_start = 0
        self.actual_end = 0
        self.total_travel_time = 0
        self.total_drive_time = 0
        self.depot_travel_time = 0

        self.start_time = self.original_start_time
        self.end_time = self.original_end_time

        self.start_pos = self.original_start_loc  # this will update over the time
        self.end_pos = self.original_end_loc

        self.completed_pointer = None
        self.add_completed_cost = False  # whether to consider the completed part of the cost of not

    def reset(self):
        """
        :return: make the route as empty
        """
        self.nodes = []
        self.times = []
        self.un_served_request_ids = {}
        self.movable_idx = -1
        self.actual_start = 0
        self.actual_end = 0
        self.total_travel_time = 0
        self.total_drive_time = 0
        self.depot_travel_time = 0
        return self

    def get_interpolated_position(self, completed_nodes, completed_times, remainder_nodes, remainder_times, curr_time):
        """
        :param completed_nodes: list of nodes that are completed as of the time 'curr_time'
        :param completed_times:  list of times corresponding to completed nodes as of the time 'curr_time'
        :param remainder_nodes: list of nodes that needs to be visited as of the time 'curr_time'
        :param remainder_times: scheduled times for visiting the remainder nodes as of the time 'curr_time'
        :param curr_time: current time of the system
        :return: the current geographical position of the route plan
        """
        interpolated_pos = None
        if len(completed_nodes) > 0:
            if curr_time < completed_times[-1]:
                raise ImplementationError("NODE IS ALREADY COMPLETED ILLEGAL ACCESS")
            elif curr_time == completed_times[-1]:
                # curr time is exactly at last node of completed route
                last_loc = completed_nodes[-1].loc
                interpolated_pos = last_loc.as_location(last_loc.latitude, last_loc.longitude, 0)
                interpolated_pos.loc_type = LocationTypes.OTHER
            else:
                if len(remainder_nodes) > 0:
                    # minimum time between last node of completed route and first node of remainder route
                    time_to_connect = completed_nodes[-1].duration(remainder_nodes[0])
                    time_of_dwell = remainder_times[0] - self._get_dwell_time(remainder_nodes[0])

                    if curr_time <= completed_times[-1] + time_to_connect:
                        # if the complete route not completely moves toward the location of the
                        # first node of remainder node, we need to interpolate distance based on the partial journey
                        if time_to_connect > 0:
                            numerator = (curr_time - completed_times[-1])
                            interpolated_pos = completed_nodes[-1].interpolate(
                                remainder_nodes[0], numerator, time_to_connect
                            )
                        else:
                            # if both node, the last node of completed route and first node
                            # of remainder route are on same location then vehicle is already in the
                            interpolated_pos = remainder_nodes[0].loc

                    else:
                        # vehicle already reached the end position for completed route
                        # and start position for remainder route first node of remainder route
                        if curr_time >= time_of_dwell:
                            # passenger is in the process of pick_up or drop_off
                            interpolated_pos = copy.deepcopy(remainder_nodes[0].loc)
                            # make sure to make copy !!!
                            passed_time = curr_time - time_of_dwell
                            if interpolated_pos.loc_type == LocationTypes.PICKUP:
                                interpolated_pos.loc_type = LocationTypes.PICKUP_MIDDLE
                            elif interpolated_pos.loc_type == LocationTypes.DROPOFF:
                                interpolated_pos.loc_type = LocationTypes.DROPOFF_MIDDLE
                            else:
                                assert passed_time == 0
                                interpolated_pos.loc_type = LocationTypes.INTERPOLATED
                            interpolated_pos.captured_dwell_time = passed_time

                        else:
                            # vehicle already reached the end position for completed route
                            # and start position for remainder route first node of remainder route
                            # but not yet started pick_up or drop_off
                            interpolated_pos = copy.deepcopy(remainder_nodes[0].loc)

                else:
                    time_to_end = completed_nodes[-1].duration(self.original_end_loc)

                    # time at which vehicle needs start moving to depot, based on the current last location
                    estimated_end_time = self.original_end_time - time_to_end

                    if curr_time <= estimated_end_time:
                        # dropped off the passenger and waiting at the last drop_off until time to reach the depot
                        last_loc = completed_nodes[-1].loc
                        interpolated_pos = last_loc.as_location(last_loc.latitude, last_loc.longitude, 0)
                        interpolated_pos.loc_type = LocationTypes.OTHER

                    elif curr_time > estimated_end_time:
                        if curr_time < self.original_end_time:
                            if time_to_end > 0:
                                # vehicle is on the way to depot
                                numerator = curr_time - estimated_end_time
                                interpolated_pos = completed_nodes[-1].interpolate(
                                    self.original_end_loc, numerator, time_to_end
                                )
                            else:
                                # vehicle already reached the end position
                                # for completed route and also reached the depot
                                interpolated_pos = self.original_end_loc
                        else:
                            # vehicle already reached the end position for completed route and also reached the depot
                            interpolated_pos = self.original_end_loc

        elif len(completed_nodes) == 0 and len(remainder_nodes) > 0:
            time_to_start_wt = (self.original_start_loc.duration(remainder_nodes[0]) +
                                self._get_dwell_time(remainder_nodes[0]))
            time_to_start_wo = self.original_start_loc.duration(remainder_nodes[0])
            expected_start_time_wt = remainder_times[0] - time_to_start_wt
            expected_start_time_wo = remainder_times[0] - self._get_dwell_time(remainder_nodes[0])

            if expected_start_time_wt <= curr_time <= expected_start_time_wo:
                if time_to_start_wo > 0:
                    # vehicle is on the way to first node of the remainder node
                    numerator = curr_time - expected_start_time_wt
                    interpolated_pos = self.original_start_loc.interpolate(
                        remainder_nodes[0], numerator, time_to_start_wo
                    )
                else:
                    # vehicle is already at the first node of the remainder node
                    interpolated_pos = remainder_nodes[0]

            elif curr_time > expected_start_time_wo:
                interpolated_pos = remainder_nodes[0].loc
                passed_time = curr_time - expected_start_time_wo
                if interpolated_pos.loc_type == LocationTypes.PICKUP:
                    interpolated_pos.loc_type = LocationTypes.PICKUP_MIDDLE
                elif interpolated_pos.loc_type == LocationTypes.DROPOFF:
                    interpolated_pos.loc_type = LocationTypes.DROPOFF_MIDDLE
                else:
                    assert passed_time == 0
                    interpolated_pos.loc_type = LocationTypes.INTERPOLATED
                interpolated_pos.captured_dwell_time = passed_time

            elif curr_time < expected_start_time_wt:
                # just position the route in start location (i.e., depot, and it is undefined if the
                # current time is less than original start time of the depot
                if curr_time >= self.original_start_time:
                    interpolated_pos = self.original_start_loc
                else:
                    assert interpolated_pos is None

        from env.solution.RouteNode import RouteNode
        if isinstance(interpolated_pos, RouteNode):
            interpolated_pos = interpolated_pos.loc
        return copy.deepcopy(interpolated_pos)

    def interpolate_routes(self, completed_route, curr_time):
        completed_nodes = completed_route.nodes
        completed_times = completed_route.times
        remainder_nodes = []
        remainder_times = []

        for k, node in enumerate(self.nodes):
            # consider the requests that are in process of picking up or dropping off as also completed
            if self.times[k] <= curr_time:
                completed_nodes.append(node)
                completed_times.append(self.times[k])
            else:
                remainder_nodes.append(node)
                remainder_times.append(self.times[k])

        if self.interpolation_type == LocationInterpolationTypes.RouteBased.value:
            interpolated_pos = self.get_interpolated_position(
                completed_nodes, completed_times, remainder_nodes, remainder_times, curr_time
            )
            completed_route = self.__perform_route_based_interpolation(
                completed_route, completed_nodes, completed_times,
                remainder_nodes, remainder_times, curr_time, interpolated_pos
            )
            assert self.start_pos.duration(completed_route.end_pos) == 0, (
                self.start_pos.duration(completed_route.end_pos))

        return {
            "completed": {"route": completed_route, "nodes": completed_nodes},
            "current": {"route": self, "nodes": self.nodes},
        }

    def __perform_route_based_interpolation(
            self, completed_route, completed_nodes, completed_times,
            remainder_nodes, remainder_times, curr_time, interpolated_pos
    ):
        # due to practical issues, we incorporate errors beyond 1-seconds when using route based interpolation
        if len(completed_nodes) > 0:
            completed_route.end_time = curr_time
            if interpolated_pos:
                dwell_time = 0
                if interpolated_pos.loc_type in [LocationTypes.PICKUP_MIDDLE, LocationTypes.DROPOFF_MIDDLE]:
                    dwell_time = interpolated_pos.captured_dwell_time

                expected_travel_time = completed_nodes[-1].duration(interpolated_pos) + dwell_time
                adj_curr_end_time = completed_times[-1] + expected_travel_time
                error_correction = 0
                if adj_curr_end_time - curr_time > 0:
                    discrepancy = adj_curr_end_time - curr_time
                    error_correction = discrepancy
                completed_route.end_time = curr_time + error_correction
                completed_route.end_pos = copy.deepcopy(interpolated_pos)

            if completed_route.execution_mode == ExecutionModes.GatherExperience.value:
                # just preserve the last node and time
                completed_route.add([completed_nodes[-1]], [completed_times[-1]], verify=True)
            else:
                completed_route.add(completed_nodes, completed_times, verify=True)
                assert completed_route.start_time <= curr_time
        else:
            completed_route = completed_route.reset()
            completed_route.end_time = curr_time
            if interpolated_pos:
                if completed_route.original_start_time <= curr_time <= completed_route.original_end_time:
                    dwell_time = 0
                    if interpolated_pos.loc_type in [LocationTypes.PICKUP_MIDDLE, LocationTypes.DROPOFF_MIDDLE]:
                        dwell_time = interpolated_pos.captured_dwell_time

                    expected_travel_time = completed_route.start_pos.duration(interpolated_pos) + dwell_time
                    adj_curr_end_time = completed_route.start_time + expected_travel_time
                    error_correction = 0
                    if adj_curr_end_time - curr_time > 0:
                        discrepancy = adj_curr_end_time - curr_time
                        error_correction = discrepancy
                    completed_route.end_time = curr_time + error_correction
                completed_route.end_pos = copy.deepcopy(interpolated_pos)

        if len(remainder_nodes) > 0:
            self.verify()
            self.start_time = curr_time
            if interpolated_pos:
                dwell_time_miss = 0
                if interpolated_pos.loc_type in [LocationTypes.PICKUP_MIDDLE, LocationTypes.DROPOFF_MIDDLE]:
                    dwell_time_miss = -interpolated_pos.captured_dwell_time

                dwell_time_first = self._get_dwell_time(remainder_nodes[0])
                expected_travel_time = (interpolated_pos.duration(remainder_nodes[0]) +
                                        dwell_time_miss + dwell_time_first)

                adj_curr_start_time = remainder_times[0] - expected_travel_time
                error_correction = 0
                if curr_time - adj_curr_start_time > 0:
                    discrepancy = curr_time - adj_curr_start_time
                    error_correction = discrepancy
                self.start_time = curr_time - error_correction
                self.start_pos = copy.deepcopy(interpolated_pos)

            self.allow_negative_capacity = True  # this will allow to have negative capacity
            self.add(remainder_nodes, remainder_times, verify=True)
        else:
            self.reset()
            self.start_time = curr_time
            if interpolated_pos:
                if self.original_start_time <= curr_time <= self.original_end_time:
                    dwell_time_miss = 0
                    if interpolated_pos.loc_type in [LocationTypes.PICKUP_MIDDLE, LocationTypes.DROPOFF_MIDDLE]:
                        dwell_time_miss = -interpolated_pos.captured_dwell_time

                    expected_travel_time = interpolated_pos.duration(self.original_end_loc) + dwell_time_miss
                    adj_curr_start_time = self.original_end_time - expected_travel_time
                    error_correction = 0
                    if curr_time - adj_curr_start_time > 0:
                        discrepancy = curr_time - adj_curr_start_time
                        error_correction = discrepancy
                    self.start_time = curr_time - error_correction
                self.start_pos = copy.deepcopy(interpolated_pos)

        if len(completed_nodes) == 0 and len(remainder_nodes) == 0:
            self.start_time = curr_time
            completed_route.start_time = curr_time
            completed_route.end_time = curr_time

        return completed_route

    def add(self, nodes, times, verify=False):
        """
        :param nodes: ordering of the nodes at which each pick_up and drop-off the requests happen
        :param times: time at which each pick_up and drop-off the requests happen
        :param verify: verify the nodes and times

        this function computes the time taken and distance traveled
        """
        self.nodes = nodes
        self.times = [int(time_val) for time_val in times]

        if self.search_mode:
            self.un_served_request_ids = {}

            self.movable_idx = -1
            for i, node in enumerate(nodes):
                if times[i] > node.current_time + self.time_ahead:
                    self.movable_idx = i
                    break

            if self.movable_idx == -1:
                self.movable_idx = len(self.nodes)

            seen = {}
            for i, node in enumerate(nodes):
                if node.request_id not in seen:
                    if times[i] >= node.current_time + self.time_ahead:
                        # only consider the requests that are far away to pick_up current_time + time_ahead
                        seen[node.request_id] = i, -1
                else:
                    pick_up_idx, _ = seen[node.request_id]
                    seen[node.request_id] = pick_up_idx, i

            for request_id, (start_idx, ref_idx) in seen.items():
                if ref_idx != -1:
                    self.un_served_request_ids[request_id] = start_idx, ref_idx

        self.total_travel_time = 0
        self.total_drive_time = 0
        self.depot_travel_time = 0

        dwell_time_correction_start = self._get_dwell_time(nodes[0]) - self.start_pos.captured_dwell_time
        time_from_start = self.start_pos.duration(nodes[0]) + dwell_time_correction_start

        time_to_end = nodes[-1].duration(self.end_pos)
        if self.end_pos.loc_type in [LocationTypes.PICKUP_MIDDLE, LocationTypes.DROPOFF_MIDDLE]:
            time_to_end += self.end_pos.captured_dwell_time

        assert self.start_pos.captured_dwell_time * self.end_pos.captured_dwell_time == 0

        self.actual_start = int(self.times[0] - time_from_start)
        self.actual_end = int(self.times[-1] + time_to_end)

        self.total_travel_time = self.actual_end - self.actual_start  # this includes depot travel
        self.depot_travel_time = time_from_start + time_to_end  # this is place-holder thing
        self.total_drive_time = time_from_start + time_to_end

        if self.execution_mode != ExecutionModes.GatherExperience.value:
            # computing total-drive time
            for i, node in enumerate(nodes[:-1]):
                self.total_drive_time += node.duration(nodes[i + 1])

            if verify:
                self.verify()

        return self

    def remove(self, pick_up_idx, drop_off_idx, verify=False):
        """
        :param pick_up_idx: position of pick_up node in self.nodes (0-indexed)
        :param drop_off_idx: position of drop_off node in self.nodes (0-indexed)
        :param verify: whether the removal of request satisfy the standard constraints
        """
        current_other_nodes = (
                self.nodes[:pick_up_idx] +
                self.nodes[pick_up_idx + 1:drop_off_idx] +
                self.nodes[drop_off_idx + 1:]
        )
        current_other_times = (
                self.times[:pick_up_idx] +
                self.times[pick_up_idx + 1:drop_off_idx] +
                self.times[drop_off_idx + 1:]
        )
        if len(current_other_nodes) > 0:
            self.add(current_other_nodes, current_other_times, verify=verify)
        else:
            self.reset()
        return self

    def remove_and_place_new_request(
            self, instance, pick_up_idx, drop_off_idx, in_req_pick_up_node, in_req_drop_off_node
    ):
        """
        :param instance: complete problem instance
        :param pick_up_idx: position of pick_up node in self.nodes (0-indexed)
        :param drop_off_idx: position of drop_off node in self.nodes (0-indexed)
        :param in_req_pick_up_node: incoming pick_up node
        :param in_req_drop_off_node: incoming drop_off node
        :return: route after inserting the incoming request
        """
        # this will perform soft-removal (not actually remove self.nodes)
        nodes_without_removed_one = (
                self.nodes[:pick_up_idx] +
                self.nodes[pick_up_idx + 1:drop_off_idx] +
                self.nodes[drop_off_idx + 1:]
        )
        return self.place_new_request(
            instance, nodes_without_removed_one, in_req_pick_up_node, in_req_drop_off_node
        )

    def verify(self):
        """
        :return: True if the route is valid by respecting all real-world constraints otherwise return False
        """
        if len(self.nodes) == 0:
            return True

        dwell_time_correction_start = self._get_dwell_time(self.nodes[0]) - self.start_pos.captured_dwell_time
        time_from_start = self.start_pos.duration(self.nodes[0]) + dwell_time_correction_start

        time_to_end = self.nodes[-1].duration(self.end_pos)
        if self.end_pos.loc_type in [LocationTypes.PICKUP_MIDDLE, LocationTypes.DROPOFF_MIDDLE]:
            time_to_end += self.end_pos.captured_dwell_time

        self.actual_start = self.times[0] - time_from_start
        self.actual_end = self.times[-1] + time_to_end

        assert len([node.idx for node in self.nodes]) == len(set([node.idx for node in self.nodes]))


    def get_total_travel_distance(self):
        """
        :return: get total travel distance
        """
        if len(self.nodes) == 0:
            return 0

        total_travel_distance = self.start_pos.distance(self.nodes[0]) + self.nodes[-1].distance(self.end_pos)
        for i, node in enumerate(self.nodes[:-1]):
            total_travel_distance += node.distance(self.nodes[i + 1])
        return total_travel_distance

    def get_total_dead_head_distance(self):
        """
        :return: get total dead head distance
        """
        if len(self.nodes) == 0:
            return 0
        dead_head_dist = self.start_pos.distance(self.nodes[0]) + self.nodes[-1].distance(self.end_pos)
        capacity_value = self.nodes[0].capacity["am"] + self.nodes[0].capacity["wc"]
        for i, node in enumerate(self.nodes[:-1]):
            if capacity_value == 0:
                dead_head_dist += self.nodes[i].distance(self.nodes[i + 1])
            capacity_value += self.nodes[i + 1].capacity["am"]
            capacity_value += self.nodes[i + 1].capacity["wc"]
        return dead_head_dist

    def get_total_dead_head_time(self):
        """
        :return: get total dead head time
        """
        if len(self.nodes) == 0:
            return 0
        dead_head_time = self.start_pos.duration(self.nodes[0]) + self.nodes[-1].duration(self.end_pos)
        capacity_value = self.nodes[0].capacity["am"] + self.nodes[0].capacity["wc"]
        for i, node in enumerate(self.nodes[:-1]):
            if capacity_value == 0:
                dead_head_time += self.times[i + 1] - self.times[i]
            capacity_value += self.nodes[i + 1].capacity["am"]
            capacity_value += self.nodes[i + 1].capacity["wc"]
        return dead_head_time

    def get_total_shared_distance(self):
        """
        :return: get total shared distance
        """
        if len(self.nodes) == 0:
            return 0
        shared_dist = 0
        capacity_value = self.nodes[0].capacity["am"] + self.nodes[0].capacity["wc"]
        for i, node in enumerate(self.nodes[:-1]):
            if capacity_value > 1:
                shared_dist += self.nodes[i].distance(self.nodes[i + 1])
            capacity_value += self.nodes[i + 1].capacity["am"]
            capacity_value += self.nodes[i + 1].capacity["wc"]
        return shared_dist

    def get_total_shared_time(self):
        """
        :return: get total shared time
        """
        if len(self.nodes) == 0:
            return 0
        shared_time = 0
        capacity_value = self.nodes[0].capacity["am"] + self.nodes[0].capacity["wc"]
        for i, node in enumerate(self.nodes[:-1]):
            if capacity_value > 1:
                shared_time += self.times[i + 1] - self.times[i]
            capacity_value += self.nodes[i + 1].capacity["am"]
            capacity_value += self.nodes[i + 1].capacity["wc"]
        return shared_time

    def get_passenger_miles(self):
        """
        return get total passenger miles
        """
        if len(self.nodes) == 0:
            return 0

        visited_requests = {}
        passenger_miles = 0
        for i, node in enumerate(self.nodes):
            if node.request_id not in visited_requests:
                visited_requests[node.request_id] = node
            else:
                passenger_miles += visited_requests[node.request_id].distance(node)
        return passenger_miles

    def get_assignments(self, offset=0):
        entries = [
            {
                "request_id": node.request_id,
                "action": node.node_type.value,
                "expected_status": node.expected_status,
                "action_time": int(self.times[i]),
                "location_lat": node.loc.latitude,
                "location_lon": node.loc.longitude,
                "position": i + len(self.completed_pointer.nodes) if self.completed_pointer else i,
                "action_index": node.idx,
                "real_route_id": self.route_id % offset if offset > 0 else self.route_id,
                "route_id": self.route_id
            }
            for i, node in enumerate(self.nodes)
        ]
        return entries

    def check_feasible(self, instance, request, node_map):
        """
        :param instance: full instance object
        :param node_map: mapping of the all nodes in the system
        :param request: incoming request
        it will check all the possible choice and choose the one with better utility
        """
        from env.solution.RouteNode import generate_pick_up_node, generate_drop_off_node
        current_time = instance["current_time"]
        pick_up_node = generate_pick_up_node(request, current_time=current_time)
        drop_off_node = generate_drop_off_node(request, current_time=current_time)
        if len(self.nodes) == 0:
            temp_nodes = [pick_up_node, drop_off_node]
            stat = self.adjust(temp_nodes)
        else:
            adjusted_nodes = [node_map[node.idx] for node in self.nodes]
            stat = self.place_new_request(instance, adjusted_nodes, pick_up_node, drop_off_node)
        return stat

    def get_paired_placement_indices(self, current_nodes, pick_up_node, drop_off_node):
        """
        :param current_nodes: current nodes in the routes
        :param pick_up_node: pick_up node of the incoming request
        :param drop_off_node drop_off node of the incoming request
        return possible index pair combinations
        """
        p_indices = self.get_placement_indices(current_nodes, pick_up_node, 0)
        d_indices = []
        if len(p_indices) > 0:
            d_indices = self.get_placement_indices(current_nodes, drop_off_node, min(p_indices) - 1)
        idx_pairs = []
        if len(p_indices) > 0 and len(d_indices) > 0:
            for p_idx in p_indices:
                for d_idx in d_indices:
                    if p_idx <= d_idx:
                        idx_pairs.append((p_idx, d_idx))
        return idx_pairs

    def place_new_request(self, instance, current_nodes, pick_up_node, drop_off_node):
        """
        :param instance: full instance object
        :param current_nodes: current nodes in the routes
        :param pick_up_node: pick_up node of the incoming request
        :param drop_off_node drop_off node of the incoming request
            best placement if we need to consider all possible placement
        it will check all the possible choice and choose the one with better utility
        """
        if len(current_nodes) == 0:
            temp_nodes = [pick_up_node, drop_off_node]
            stat = self.adjust(temp_nodes)
        else:
            import math
            stat = None
            actions = []
            min_cost = math.inf
            idx_pairs = self.get_paired_placement_indices(current_nodes, pick_up_node, drop_off_node)
            for p_idx, d_idx in idx_pairs:
                temp_nodes = current_nodes[:p_idx] + [pick_up_node] + current_nodes[p_idx:d_idx] + \
                             [drop_off_node] + current_nodes[d_idx:]
                temp_stat = self.adjust(temp_nodes)
                if temp_stat:
                    if instance["objective"] == ObjectiveTypes.CustomObjectiveByRL.value:
                        actions.append(
                            ComprehensiveAction(
                                initial_routes=instance["routes"],
                                changed_routes={self.route_id: temp_stat}
                            )
                        )
                    else:
                        insertion_cost = self.get_cost(
                            objective=instance["objective"],
                            consider_threshold=True,
                            state=instance,
                            action=ComprehensiveAction(
                                initial_routes=instance["routes"],
                                changed_routes={self.route_id: temp_stat}
                            ),
                        )
                        if insertion_cost < min_cost:
                            min_cost = insertion_cost
                            stat = temp_stat

            if len(actions) > 0:
                best_action, value = AgentManager.instance().get_best_action(instance, actions)
                stat = best_action.changed_routes[self.route_id]
        return stat

    def get_feasible_actions(self, req, node_map, current_time):
        """
        :param node_map: mapping of the all nodes in the system
        :param req: incoming request
        :param current_time: current time of the system
        it will check all the possible choice and choose the one with better utility
        """
        from env.solution.RouteNode import generate_pick_up_node, generate_drop_off_node
        pick_up_node = generate_pick_up_node(req, current_time=current_time)
        drop_off_node = generate_drop_off_node(req, current_time=current_time)
        adjusted_nodes = [node_map[node.idx] for node in self.nodes]
        return self._get_feasible_actions(pick_up_node, drop_off_node, adjusted_nodes)

    def get_feasible_actions_after_removal(
            self, pick_up_idx, drop_off_idx, in_req_pick_up_node, in_req_drop_off_node
    ):
        """
        :param pick_up_idx: position of pick_up node in self.nodes (0-indexed)
        :param drop_off_idx: position of drop_off node in self.nodes (0-indexed)
        :param in_req_pick_up_node: incoming pick_up node
        :param in_req_drop_off_node: incoming drop_off node
        :return: route after inserting the incoming request
        """
        adjusted_nodes = (
                self.nodes[:pick_up_idx] +
                self.nodes[pick_up_idx + 1:drop_off_idx] +
                self.nodes[drop_off_idx + 1:]
        )
        return self._get_feasible_actions(in_req_pick_up_node, in_req_drop_off_node, adjusted_nodes)

    def _get_feasible_actions(self, pick_up_node, drop_off_node, adjusted_nodes):
        """
        :param pick_up_node: incoming pick_up node
        :param drop_off_node: incoming drop_off node
        :param adjusted_nodes: routes nodes after performing alteration (from original set of nodes)
        :return: route after inserting the incoming request
        """
        stats = []
        if len(adjusted_nodes) == 0:
            temp_nodes = [pick_up_node, drop_off_node]
            temp_stat = self.adjust(temp_nodes)
            if temp_stat:
                stats.append(temp_stat)
        else:
            import math
            idx_pairs = self.get_paired_placement_indices(adjusted_nodes, pick_up_node, drop_off_node)
            for p_idx, d_idx in idx_pairs:
                temp_nodes = adjusted_nodes[:p_idx] + [pick_up_node] + adjusted_nodes[p_idx:d_idx] + \
                             [drop_off_node] + adjusted_nodes[d_idx:]
                temp_stat = self.adjust(temp_nodes)
                if temp_stat:
                    stats.append(temp_stat)
        return stats

    def get_placement_indices(self, nodes, node, start_index=0):
        """
        this will provide the possible insertion location of the node
        :param nodes: list of previous nodes
        :param node: selected node
        :param start_index: start index
        :return: insertion indices
        """
        n_indices = []
        for i, node_i in enumerate(nodes):
            if i >= start_index:
                # add before all the requests
                if i == 0:
                    if node.reachable(node_i, self._get_dwell_time(node_i)):
                        n_indices.append(0)

                # add at the end of all the requests
                if i == len(nodes) - 1:
                    if node_i.reachable(node, self._get_dwell_time(node)):
                        n_indices.append(len(nodes))

                if i < len(nodes) - 1:
                    node_i_next = nodes[i + 1]
                    if (node_i.reachable(node, self._get_dwell_time(node)) and
                            node.reachable(node_i_next, self._get_dwell_time(node_i_next))):
                        n_indices.append(i + 1)

        return n_indices

    def adjust(self, nodes):
        success = True
        total_travel_time = 0
        total_drive_time = 0
        times = []
        node_times = []
        capacities = {"am": [], "wc": []}

        time_from_start = self.start_pos.duration(nodes[0])
        time_to_end = nodes[-1].duration(self.end_pos)

        earliest_time_to_reach_first_node = self.start_time + time_from_start + self._get_dwell_time(nodes[0])
        earliest_time_to_reach_end = nodes[-1].earliest_arrival + time_to_end

        if earliest_time_to_reach_first_node > nodes[0].latest_arrival:
            success = False

        if earliest_time_to_reach_end > self.end_time:
            success = False

        if success:
            first_node_time = max(earliest_time_to_reach_first_node, nodes[0].earliest_arrival)
            node_times = {nodes[0].idx: first_node_time}
            times = [first_node_time]
            capacities = {"am": [nodes[0].capacity["am"]], "wc": [nodes[0].capacity["wc"]]}

            for key in capacities.keys():
                if capacities[key][0] > self.capacity[key]:
                    success = False

        if success:
            # if the initial conditions are met perform a forward pass
            success, nodes, times, node_times, total_drive_time, capacities = self.__forward_optimize(
                nodes=nodes,
                times=times,
                node_times=node_times,
                capacities=capacities
            )

        if success:
            # if the forward pass is successful then perform a backward pass
            times, node_times = self.__backward_optimize(
                nodes=nodes,
                times=times,
                node_times=node_times
            )
            self.assert_times(nodes, node_times, times)

            start_time_cond = self.start_time <= times[0] - time_from_start - self._get_dwell_time(nodes[0])
            end_time_cond = times[-1] + time_to_end <= self.end_time
            if not (start_time_cond and end_time_cond):
                success = False
            else:
                total_travel_time = times[-1] - times[0] + self._get_dwell_time(nodes[0])

        from env.solution.InsertStat import InsertStat
        stat = None
        if success:
            stat = InsertStat(
                route_id=self.route_id,
                capacity=self.capacity,
                start_pos=self.start_pos,
                end_pos=self.end_pos,
                nodes=copy.deepcopy(nodes),
                times=times,
                capacities=capacities,
                total_travel_time=total_travel_time,
                total_drive_time=total_drive_time,
                start_time=self.start_time,
                original_start_time=self.original_start_time,
                original_end_time=self.original_end_time,
            )
        return stat

    def __forward_optimize(self, nodes, times, node_times, capacities):
        """
        Optimize the route plans by performing a forward pass
        :param nodes: nodes in the vehicle route
        :param times: initial time-values by ensuring time-window constraints
        for the ``nodes''
        :param node_times: times index based on node-index
        :param capacities:  list of capacity values
        :return: adjusted travel time values and dictionary of time values for the corresponding
        nodes
        """
        success = True
        total_drive_time = 0
        for i, node in enumerate(nodes[:-1]):
            # first check the TIME-WINDOW constraint

            # computing travel time for a node
            travel_time = node.duration(nodes[i + 1])

            total_drive_time += travel_time

            # total time take to move to the node
            total_time_taken = travel_time + self._get_dwell_time(nodes[i + 1])

            # next time point value
            time_value = max(times[i] + total_time_taken, nodes[i + 1].earliest_arrival)

            upper_limit = nodes[i + 1].latest_arrival
            if time_value > upper_limit:
                success = False
                break
            elif time_value >= 0:
                times.append(time_value * 1.0)
                node_times[nodes[i + 1].idx] = time_value
            else:
                raise ImplementationError(f"Invalid time value {time_value}")

            # then check CAPACITY constraint
            for key in capacities.keys():
                spec_capacity_value = capacities[key][i] + nodes[i + 1].capacity[key]
                if spec_capacity_value > self.capacity[key]:
                    success = False
                    break
                elif spec_capacity_value >= 0 or self.allow_negative_capacity:
                    capacities[key].append(spec_capacity_value)
                else:
                    raise ImplementationError(f"Invalid capacity value ({key}) {spec_capacity_value}")

            if not success:
                break
        return success, nodes, times, node_times, total_drive_time, capacities

    def __backward_optimize(self, nodes, times, node_times):
        """
        Optimize the route plans by performing a backward pass
        :param nodes: nodes in the vehicle route
        :param times: initial time-values by ensuring time-window constraints
        for the ``nodes''
        :param node_times: times index based on node-index
        :return: adjusted travel time values and dictionary of time values for the corresponding
        nodes
        """
        # adjusting travel times to minimize the waiting time
        # in the first step, I start with the first node, the earliest pick_up, and continue
        # adjusting travel times

        # but that is not actually required and can introduce wait times,
        # those wait times are fixed later once all nodes are assigned

        # this helps to reduce the total vehicle route duration and improves the overall solution
        node_rev = copy.deepcopy(nodes[::-1])
        len_v = len(times)
        for k, _r_node in enumerate(node_rev):
            k_rev = len_v - k - 1
            if k > 0:
                _prev_travel_time = _r_node.duration(node_rev[k - 1])
                _prev_time = node_times[node_rev[k - 1].idx]
                _cur_time = node_times[_r_node.idx]
                _prev_dwell_time = self._get_dwell_time(node_rev[k - 1])

                _min_prev_time = _cur_time + _prev_travel_time + _prev_dwell_time
                if _min_prev_time < _prev_time:
                    _cur_time = min(_prev_time - _prev_travel_time - _prev_dwell_time, _r_node.latest_arrival)
                    assert _r_node.earliest_arrival <= _cur_time <= _r_node.latest_arrival
                times[k_rev] = _cur_time
                node_times[_r_node.idx] = _cur_time
        return times, node_times

    @staticmethod
    def assert_times(nodes, node_times, time_values):
        for i, node in enumerate(nodes):
            if not (node.earliest_arrival <= node_times[node.idx] <= node.latest_arrival):
                time_window_violation_msg = f"The route reach the {node.node_type} node, with time window " + \
                                            f"({node.earliest_arrival}, {node.latest_arrival}) " + \
                                            f"at {node_times[node.idx]}"
                error_msg = f"Expected {node_times[node.idx]}, Actual {time_values[i]}"
                assert node_times[node.idx] == time_values[i], error_msg
                raise ImplementationError(
                    f"Time-window constraint validation failed, {time_window_violation_msg}"
                )

    @staticmethod
    def is_pick_up_and_drop_off_inorder(nodes):
        # check whether drop_off nodes only comes after pick_up node if it presents in the list of nodes
        seen_req_ids = set()
        feasible_shift = True
        for node in nodes:
            if node.request_id in seen_req_ids:
                if node.node_type == RequestActions.Pickup or node.node_type.value == RequestActions.Pickup.value:
                    feasible_shift = False
                    break
            else:
                seen_req_ids.add(node.request_id)
        return feasible_shift

    def get_fixed_cost(self, objective):
        """
        :param objective: type of the objective either 'travel-time' or 'drive-time' or 'idle-time'
        :return: the objective value of the completed route or fixed route for Google OR Tools only (*)
        """
        match objective:
            case ObjectiveTypes.IdleTime.value:
                return -1 * (self.original_end_time - self.start_time)
            case _:
                return 0

    def get_current_cost(self, **kwargs):
        """
        :param kwargs: keyword arguments such as objective
                        (i.e., type of the objective either 'travel-time' or 'drive-time' or 'idle-time'),
                         consider_threshold, wait_time_threshold, action (that represents the change in the manifests)
        :return: the objective value of the completed route
        """
        objective = kwargs['objective']
        current_time = kwargs['current_time']
        consider_threshold = kwargs['consider_threshold'] if 'consider_threshold' in kwargs else False
        wait_time_threshold = kwargs['wait_time_threshold'] if 'wait_time_threshold' in kwargs else 0

        action = kwargs['action'] if 'action' in kwargs else None

        if action:
            instance_obj = action
        else:
            instance_obj = self

        match objective:
            case ObjectiveTypes.TravelTime.value:
                # this is useless for infinite horizon where look-ahead horizon always returned !!!
                # because start-time always current time and end-time is always infinite
                start_time = max(self.start_time, current_time)
                end_time = min(self.end_time, current_time + self.look_ahead_horizon)
                return end_time - start_time
            case ObjectiveTypes.DriveTime.value:
                return (
                        self.look_ahead_horizon -
                        instance_obj.get_wait_time(
                            current_time=current_time,
                            horizon=self.look_ahead_horizon,
                            threshold=wait_time_threshold,
                            consider_threshold=consider_threshold
                        )
                )
            case ObjectiveTypes.IdleTime.value:
                return -1 * instance_obj.get_wait_time(
                    current_time=current_time,
                    horizon=self.look_ahead_horizon,
                    threshold=wait_time_threshold,
                    consider_threshold=consider_threshold
                )

    @staticmethod
    def get_cost(objective, **kwargs):
        """
        :param objective: objective (i.e., type of the objective either 'travel-time' or 'drive-time' or 'idle-time')
        :param kwargs: keyword arguments such as state (represent current state of the environment),
                       action (that represents the change in the manifests),
                       consider_threshold, wait_time_threshold
        :return: the cost
        """
        from learn.agent.AgentManager import AgentManager
        if objective == ObjectiveTypes.CustomObjectiveByRL.value:
            if 'state' not in kwargs:
                kwargs['state'] = None
            if 'action' not in kwargs:
                kwargs['action'] = None

        return AgentManager.instance().get_cost(objective=objective, **kwargs)
