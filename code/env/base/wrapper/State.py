class MinimalState:
    def __init__(self, current_time, routes):
        self.current_time = current_time
        self.routes = [route.minimal() for route in routes]

    def minimal(self):
        return self


class State:

    def __init__(self):
        self.current_time = 0
        self.new_request = None
        self.current_requests = []
        self.accepted_requests = 0
        self.current_idx = -1
        self.instance_info = None
        self.routes = None
        self.completed_routes = None
        self.value = None
        self.action_map = {}

    def minimal(self):
        if isinstance(self, MinimalState):
            return self
        return MinimalState(self.current_time, self.routes)

    def is_valid(self):
        for route in self.routes:
            if route.start_time + 1 < self.current_time:
                raise AssertionError(
                    "Start time should be greater than or equal to Current time,"
                    f"Start time {route.start_time}, and Current time {self.current_time}"
                )

    def __deepcopy__(self, mem_dict=None):
        """
        :return: provides a partial deepcopy of state object !!!
        """
        import copy

        state = type(self)()
        state.current_time = self.current_time
        state.new_request = copy.deepcopy(self.new_request)
        state.current_requests = copy.deepcopy(self.current_requests)
        state.accepted_requests = copy.deepcopy(self.accepted_requests)
        state.current_idx = self.current_idx
        state.instance_info = copy.deepcopy(self.instance_info)
        state.routes = copy.deepcopy(self.routes)
        state.completed_routes = copy.deepcopy(self.completed_routes)
        state.value = self.value
        state.action_map = copy.deepcopy(self.action_map)
        return state

    def reset(
            self,
            sample_idx,
            model_idx,
            input_files,
            depot_loc,
            capacity,
            time_ahead,
            dwell_time_pickup,
            dwell_time_dropoff,
            **time_window_kwargs
    ):
        """
        :param sample_idx: sample index to select one problem instance
        :param model_idx: model index (only applicable for RL based approach)
        :param input_files: file path containing input data such as requests, routes, dates and expectation
        :param depot_loc: location of the depot, initially defined start and end location
        :param capacity: dictionary of capacities {"am": 8, "wc": 2}
        :param time_ahead: time ahead to consider which scheduling
        :param dwell_time_pickup: dwell time at pickup
        :param dwell_time_dropoff: dwell time at dropoff

        reset the environment by loading the requests specific to the sample_idx
        where the requests, routes, dates and expectations are stored in file path specified input files
        """
        self._clear()
        import pandas as pd
        sample_dates = pd.read_csv(input_files["dates"])["sample_date"].tolist()
        sample_date = sample_dates[sample_idx % len(sample_dates)]
        self.instance_info = {
            "sample_idx": sample_idx,
            "model_idx": model_idx,
            "sample_date": sample_date,
            "input_files": input_files,
            "depot_loc": depot_loc,
            "capacity": capacity,
            "time_ahead": time_ahead,
            "dwell_time_pickup": dwell_time_pickup,
            "dwell_time_dropoff": dwell_time_dropoff,
            "time_window_kwargs": time_window_kwargs
        }

    def process_action(self, action, reset_new_action=True):
        self.action_map = {}
        for k, action_entry in action["solution_df"].iterrows():
            self.action_map[action_entry.action_index] = action_entry
        self.routes = action["computed_solution"].routes
        self.completed_routes = action["computed_solution"].completed_routes
        if reset_new_action:
            self.new_request = None

    def _load_requests_and_routes(self):
        """
        Load requests and Routes
        """
        from common.general import convert_sec_to_hh_mm_ss
        from common.logger import logger

        routes = self.__load_routes()
        requests = self.__load_requests(offset=len(routes))

        if len(requests) > 0:
            simulation_time = requests[-1].get_latest_dropoff(0) - requests[0].arrival_time
            logger.info(f"TOTAL SIMULATION TIME {convert_sec_to_hh_mm_ss(simulation_time)}")
        return requests, routes

    def __load_routes(self):
        """
        :return: list of routes
        """
        import copy
        from common.arg_parser import get_parsed_args
        from common.general import convert_time_to_sec
        from common.logger import logger
        from env.solution.Route import Route

        args = get_parsed_args()
        routes = [
            Route(
                route_id=route_id,
                start_loc=self.instance_info["depot_loc"],
                end_loc=self.instance_info["depot_loc"],
                capacity=copy.deepcopy(self.instance_info["capacity"]),
                start_time=convert_time_to_sec(args.route_start_time),
                end_time=convert_time_to_sec(args.route_end_time),
                time_ahead=args.time_ahead,
                look_ahead_horizon=args.look_ahead_horizon,
                interpolation_type=args.interpolation_type,
                search_mode=args.perform_search,
                execution_mode=args.execution_mode
            ) for route_id in range(args.number_of_routes)
        ]
        logger.info(f"Successfully loaded {len(routes)} routes".upper())
        return routes

    def __load_requests(self, offset=0):
        """
        :param offset: offset the request_idx
        :return: list of requests
        """
        import pandas as pd
        from common.logger import logger
        from common.arg_parser import get_parsed_args
        from env.data.Request import Request
        from env.data.Location import Location, LocationTypes

        args = get_parsed_args()
        df = pd.read_csv(self.instance_info["input_files"]["requests"], low_memory=False)
        df = df[df.sample_date == self.instance_info["sample_date"]]

        if args.number_of_requests != -1:
            # optional test channel
            # inorder to perform quick checks on change in implementation
            df = df.iloc[:args.number_of_requests]
            df = df.reset_index()

        if len(df) == 0:
            return []

        df = df.sort_values(by="arrival_time").reset_index()
        df["next_arrival_time"] = df["arrival_time"].to_list()[1:] + [df["scheduled_pickup"].iloc[-1]]
        df["maximum_search_time"] = df.apply(func=lambda x: max(1, x.next_arrival_time - x.arrival_time), axis=1)
        df.arrival_time = df.arrival_time.astype(int)
        df.next_arrival_time = df.next_arrival_time.astype(int)
        df.maximum_search_time = df.maximum_search_time.astype(int)

        requests = []
        for i, entry in df.iterrows():
            request = Request(
                request_id=len(requests),
                offset=offset,
                origin=Location(
                    latitude=entry.origin_lat,
                    longitude=entry.origin_lon,
                    dwell_time=self.instance_info["dwell_time_pickup"],
                    loc_type=LocationTypes.PICKUP
                ),
                destination=Location(
                    latitude=entry.destination_lat,
                    longitude=entry.destination_lon,
                    dwell_time=self.instance_info["dwell_time_dropoff"],
                    loc_type=LocationTypes.DROPOFF
                ),
                arrival_time=entry.arrival_time,
                scheduled_pickup=entry.scheduled_pickup,
                capacity={"am": entry.am_count, "wc": entry.wc_count},
                **self.instance_info["time_window_kwargs"]
            )
            request.max_search_duration = entry.maximum_search_time
            if hasattr(entry, "Request Status"):
                request.expected_status = entry["Request Status"]
            requests.append(request)
        logger.info(f"Successfully loaded {len(requests)} requests".upper())
        return requests

    def _clear(self):
        self.current_time = 0
        self.new_request = None
        self.current_requests = []
        self.accepted_requests = 0
        self.current_idx = -1
        self.instance_info = None
        self.routes = None
        self.completed_routes = None
        self.value = None
        self.action_map = {}

    def get_service_rate(self, percentage=False):
        service_rate = 0
        if self.current_idx > 0:
            service_rate = round(self.accepted_requests * (100 if percentage else 1) / self.current_idx, 2)
        return service_rate

    def log_summary(self):

        from common.logger import logger

        fail_count = self.current_idx - self.accepted_requests
        service_rate = self.get_service_rate(percentage=True)
        logger.info(
            f"SERVICE-RATE: {service_rate} %, INSERT: {self.accepted_requests}, FAILED: {fail_count}"
        )

    def service_summary(self):

        fail_count = self.current_idx - self.accepted_requests
        service_rate = self.get_service_rate(percentage=True)

        return {
            "total_requests": self.accepted_requests + fail_count,
            "assigned_requests": self.accepted_requests,
            "dropped_requests": fail_count,
            "service_rate": service_rate
        }

    def get_day_of_week(self):
        """
        :return: get day of the week of given problem instance
        """
        from datetime import datetime

        if "sample_date" in self.instance_info:
            return datetime.strptime(self.instance_info["sample_date"], "%Y-%m-%d").date().weekday()
        return 0
