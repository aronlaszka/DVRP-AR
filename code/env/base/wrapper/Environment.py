from common.general import measure_time
from common.types import InsertionHeuristics, MetaHeuristics


class Environment:

    def __init__(self, objective, output_dir):
        from env.base.wrapper.SearchSummary import SearchSummary
        self.objective = objective
        self.output_dir = output_dir
        self.solver = None
        self.state_class = None
        self.store_compute_times = False
        self.summaries = []
        self.readable_summaries = []
        self.compute_times = []
        self.search_summary = SearchSummary()

        # temporary storage
        self.requests = None  # represents all the requests for given problem instance (during simulation)
        self.total_requests = 0

        self.current_instance = None
        # current RoutingAPI instance (for ease of access when using Routing API based search
        self.current_routing_instance = None
        self.current_state = None  # copy of current state
        self.current_response = None  # current computation (just for easy of access)
        self.current_solution = None  # current Solution
        self.last_insertion_success = False  # indicator of whether last insertion is successful or not
        self.last_save_state = None  # precomputed stats to reuse in the event of last insertion is success

    def clear(self):
        """
            clear summaries for the given problem instance
        """
        from env.base.wrapper.SearchSummary import SearchSummary
        self.summaries = []
        self.readable_summaries = []
        self.compute_times = []
        self.search_summary.clear()
        del self.search_summary
        self.search_summary = SearchSummary()

        del self.requests
        del self.current_instance
        del self.current_routing_instance
        del self.current_response
        del self.current_solution
        del self.last_insertion_success
        del self.last_save_state

        self.requests = None
        self.total_requests = 0
        self.current_instance = None
        self.current_routing_instance = None
        self.current_response = None
        self.current_solution = None
        self.last_insertion_success = False
        self.last_save_state = None

    @measure_time
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
        :param capacity: dictionary of capacities {"am": 8, "wc": 2},
        :param time_ahead: time ahead to consider which scheduling
        :param dwell_time_pickup: dwell time at pickup
        :param dwell_time_dropoff: dwell time at dropoff

        reset the environment by loading the requests specific to the sample_idx
        where the requests, routes, dates and expectations are stored in file path specified input files
        """
        from common.arg_parser import get_parsed_args
        args = get_parsed_args()
        self.clear()

        self.current_state = self.state_class()

        requests, solution = self.current_state.reset(
            sample_idx=sample_idx,
            model_idx=model_idx,
            input_files=input_files,
            depot_loc=depot_loc,
            capacity=capacity,
            time_ahead=time_ahead,
            dwell_time_pickup=dwell_time_pickup,
            dwell_time_dropoff=dwell_time_dropoff,
            **time_window_kwargs
        )

        self.requests = requests
        self.total_requests = len(requests)
        self.current_solution = solution
        self.current_state.routes = self.current_solution.routes
        self.current_state.completed_routes = self.current_solution.completed_routes
        if self.has_more_requests():
            from env.data.TimeMatrix import MatrixManager
            MatrixManager.instance().load(self.get_route_locations(), init_matrix=True)
        self.log_main(sample_idx, time_ahead)

    @measure_time
    def process_action(self, action, value):
        from common.logger import logger
        from env.data.TimeMatrix import MatrixManager
        # always reset routing instance and add it if it actually presents for current state
        self.current_routing_instance = None
        self.current_state.value = value
        if len(action.changed_route_ids) > 0:
            resp = self.solver.apply_insertion(self.current_instance, action)
            if "instance" in resp.keys():
                self.current_routing_instance = resp["instance"]
            self.last_insertion_success = True
            self.current_solution = resp["computed_solution"]

            self.current_state.process_action(resp)
            if self.current_response["new_request_size"] == 1:
                if len(resp["dropped_request_ids"]) == 0:
                    self.current_state.current_requests = self.current_response["all_requests"]
                    self.current_state.accepted_requests += 1
            else:
                newly_added_requests = [
                    request for request in self.current_response["all_requests"]
                    if request.request_id in resp["assigned_request_ids"]
                ]
                self.current_state.current_requests = newly_added_requests
                self.current_state.accepted_requests += len(newly_added_requests)

            self.last_save_state = self.current_response
            MatrixManager.instance().update()
            logger.info("SUCCESSFULLY INSERTED THE NEW REQUEST")
        else:
            self.last_insertion_success = False
            self.last_save_state = None
            logger.error("UNABLE TO INSERT THE NEW REQUEST")
        self.current_state.log_summary()
        MatrixManager.instance().reset_temp()
        self.log_remain(self.current_state.current_time)

    def is_done(self):
        return not self.has_more_requests()

    def has_more_requests(self):
        if self.requests:
            return len(self.requests) > 0 and self.current_state.current_idx < len(self.requests)
        return False

    def has_more_than(self, percentage=0.1):
        if self.requests:
            return self.current_state.current_idx < (1 - percentage) * self.total_requests
        return False

    def get_route_locations(self):
        locations_at_time_t = []
        if self.current_solution is not None:
            for route in self.current_solution.routes:
                locations_at_time_t.extend([route.start_pos, route.end_pos])
        return locations_at_time_t

    def log_main(self, sample_idx, time_ahead):
        from common.general import convert_sec_to_hh_mm_ss
        from common.logger import logger
        logger.info(f"SAMPLE INDEX ({sample_idx})")
        logger.info(f"LOOKAHEAD TIME ({convert_sec_to_hh_mm_ss(time_ahead)})")
        logger.info(f"TOTAL NUMBER REQUESTS: {len(self.requests)}")
        logger.info(f"TOTAL ROUTES: {len(self.current_solution.routes)}")

    def get_max_time(self):
        if len(self.requests) > 0:
            return self.requests[-1].get_latest_dropoff(0)
        return -1

    def log_remain(self, curr_time):
        from common.general import convert_sec_to_hh_mm_ss
        from common.logger import logger
        if self.get_max_time() > 0 and self.get_max_time() >= curr_time:
            logger.info(f"REMAINING SIMULATION TIME: {convert_sec_to_hh_mm_ss(self.get_max_time() - curr_time)}")
