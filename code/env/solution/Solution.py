import copy

from env.structures.Measure import Duration, Distance


class Solution:
    def __init__(self, routes=None, completed_routes=None, copy_completed=False):
        if routes is None:
            routes = []
        if completed_routes is None:
            completed_routes = []
        self.routes = routes
        if copy_completed:
            self.completed_routes = [copy.deepcopy(route) for route in routes]
            for complete_route in completed_routes:
                complete_route.end_pos = copy.deepcopy(complete_route.start_pos)
                complete_route.end_time = complete_route.start_time
        else:
            self.completed_routes = completed_routes

        for i, route in enumerate(self.routes):
            self.routes[i].completed_pointer = self.completed_routes[i]

        self.total_travel_time = Duration(0, unit="s")
        self.total_drive_time = Duration(0, unit="s")
        self.total_dwell_time = Duration(0, unit="s")
        self.total_depot_travel_time = Duration(0, unit="s")
        self.total_dead_head_time = Duration(0, unit="s")
        self.total_shared_time = Duration(0, unit="s")
        self.total_wait_time = Duration(0, unit="s")
        self.total_travel_distance = Distance(0, unit="mi")
        self.total_passenger_miles = Distance(0, unit="mi")
        self.total_dead_head_distance = Distance(0, unit="mi")
        self.total_shared_distance = Distance(0, unit="mi")

    def reset(self):
        self.routes = []
        self.total_travel_time = Duration(0, unit="s")
        self.total_drive_time = Duration(0, unit="s")
        self.total_depot_travel_time = Duration(0, unit="s")
        self.total_dead_head_time = Duration(0, unit="s")
        self.total_shared_time = Duration(0, unit="s")
        self.total_wait_time = Duration(0, unit="s")
        self.total_travel_distance = Distance(0, unit="mi")
        self.total_passenger_miles = Distance(0, unit="mi")
        self.total_dead_head_distance = Distance(0, unit="mi")
        self.total_shared_distance = Distance(0, unit="mi")
        return self

    def add(self, route, verify=True):
        if verify:
            route.verify()
        if route.route_id >= len(self.routes):
            adj_idx = route.route_id - len(self.routes)
            self.completed_routes[adj_idx] = route
            self.routes[adj_idx].completed_pointer = self.completed_routes[adj_idx]
        else:
            self.routes[route.route_id] = route
            self.routes[route.route_id].completed_pointer = self.completed_routes[route.route_id]

    def summarize(self):
        for route in self.completed_routes:
            # total travel duration from the start of the route at depot and end of the route again at depot
            self.total_travel_time += route.total_travel_time

            # only consider the driving time
            self.total_drive_time += route.total_drive_time

            # consider all the dwell times
            self.total_dwell_time += route.get_sum_of_dwell_times()

            # only considering the driving time with no passengers
            self.total_dead_head_time += route.get_total_dead_head_time()

            # duration for how long where more than 1 passenger in the vehicle
            self.total_shared_time += route.get_total_shared_time()

            # time consumed for back and forth traveling of the routes
            self.total_depot_travel_time += route.depot_travel_time

            # total time the vehicle waits without serving any requests
            self.total_wait_time += (route.total_travel_time - route.total_drive_time - route.get_sum_of_dwell_times())

            # total travel distance from the start of the route at depot and end of the route again at depot
            self.total_travel_distance += route.get_total_travel_distance()

            # individual passenger miles if the passenger takes direct trip from origin to destination
            self.total_passenger_miles += route.get_passenger_miles()

            # only considering the driving distance with no passengers
            self.total_dead_head_distance += route.get_total_dead_head_distance()

            # distance for how long where more than 1 passenger in the vehicle
            self.total_shared_distance += route.get_total_shared_distance()

    def get_summary(self):
        return {
            "number_of_routes": len(self),
            "total_travel_time": self.total_travel_time,
            "total_drive_time": self.total_drive_time,
            "total_dwell_time": self.total_dwell_time,
            "total_wait_time": self.total_wait_time,
            "total_dead_head_time": self.total_dead_head_time,
            "total_shared_time": self.total_shared_time,
            "total_travel_distance": self.total_travel_distance.miles,
            "total_passenger_miles": self.total_passenger_miles.miles,
            "VMT/PMT": self.total_travel_distance.miles / self.total_passenger_miles.miles
            if self.total_passenger_miles.miles > 0 else 0,
            "total_dead_head_distance": self.total_dead_head_distance.miles,
            "total_shared_distance": self.total_shared_distance.miles,
            "total_depot_drive_time": self.total_depot_travel_time,
            "total_travel_duration_wo_depot": self.total_travel_time - self.total_depot_travel_time,
            "total_drive_time_wo_depot": self.total_drive_time - self.total_depot_travel_time,
        }

    def get_readable_summary(self):
        return {
            "number_of_routes": len(self),
            "total_travel_time": self.total_travel_time.hhmmss(),
            "total_drive_time": self.total_drive_time.hhmmss(),
            "total_dwell_time": self.total_dwell_time.hhmmss(),
            "total_wait_time": self.total_wait_time.hhmmss(),
            "total_dead_head_time": self.total_dead_head_time.hhmmss(),
            "total_shared_time": self.total_shared_time.hhmmss(),
            "total_travel_distance": self.total_travel_distance.miles,
            "total_passenger_miles": self.total_passenger_miles.miles,
            "VMT/PMT": self.total_travel_distance.miles / self.total_passenger_miles.miles
            if self.total_passenger_miles.miles > 0 else 0,
            "total_dead_head_distance": self.total_dead_head_distance.miles,
            "total_shared_distance": self.total_shared_distance.miles,
            "total_depot_drive_time": self.total_depot_travel_time.hhmmss(),
            "total_travel_duration_wo_depot": (self.total_travel_time - self.total_depot_travel_time).hhmmss(),
            "total_drive_time_wo_depot": (self.total_drive_time - self.total_depot_travel_time).hhmmss(),
        }

    def __len__(self):
        return len(self.routes)
