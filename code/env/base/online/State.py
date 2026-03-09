from env.base.wrapper.State import State


class OnlineState(State):

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
        super(OnlineState, self).reset(
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
        from env.solution.Solution import Solution
        requests, routes = self._load_requests_and_routes()
        solution = Solution(routes, copy_completed=True)
        self.current_idx = 0
        return requests, solution
