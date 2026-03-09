##################################################################
#              GOOGLE OR-TOOL ROUTING IMPLEMENTATION             #
##################################################################

import copy

from common.types import SolvingModes, TravelTimeSources, ExecutionModes
from env.base.wrapper.RoutingSolver import RoutingSolver
from env.structures.MappedList import MappedList


class OnlineRoutingSolver(RoutingSolver):
    """
        This is a wrapper class for Google OR-Tools PDPTW solver where the problem solves only online schedules
    """

    @staticmethod
    def _init_computed_solution(instance):
        """
        Initializes the ComputedSolution object based on the initial solution at the start
        """
        from env.solution.Solution import Solution
        return Solution(instance["routes"], instance["completed_routes"])

    def exhaustive_search_insert(self, instance):
        """
        :param instance: dictionary object describing the problem instance

        perform exhaustive search insertion

        :return: dictionary of status including whether the new request is assigned or not,
        if assigned the complete routing-instance object, and assignment dataframe
        """
        import numpy as np

        routes = instance["routes"]
        completed_routes = instance["completed_routes"]

        routing, manager = None, None
        if self.config["travel_time_source"] == TravelTimeSources.OSMnx.value:
            routing, manager = self._setup_problem(instance)

        assignments = []
        if self.config["execution_mode"] != ExecutionModes.GatherExperience.value:
            for i in range(len(completed_routes)):
                assignments.extend(completed_routes[i].get_assignments())

        instance["objective"] = self.config["objective"]

        acceptance_value_threshold = self.config["acceptance_value_threshold"]

        allow_rejection = False
        if acceptance_value_threshold > 0.0:
            allow_rejection = self.config["allow_rejection"]
        actions = self.get_feasible_actions(instance, allow_rejection=allow_rejection)
        if acceptance_value_threshold and acceptance_value_threshold > 0.0:
            assert acceptance_value_threshold >= 0
            values = [
                action.get_value(
                    objective=self.config["objective"],
                    current_time=instance["current_time"]
                )
                for action in actions[:-1]
            ]
            filtered_values = [value for value in values if value >= acceptance_value_threshold]
            if len(filtered_values) > 0:
                # to ensure the same placement
                idx = np.argmax(values)
                action = actions[idx]
            else:
                # reject action
                action = actions[-1]
        else:
            # default setting with no-threshold
            values = [
                action.get_value(
                    objective=self.config["objective"],
                    current_time=instance["current_time"]
                )
                for action in actions
            ]
            idx = np.argmax(values)
            action = actions[idx]

        computed_solution = self._init_computed_solution(instance)
        assigned = False
        for i in range(len(routes)):
            route = copy.deepcopy(routes[i])
            route.verify()
            initial_nodes = len(route.nodes)
            if action and route.route_id in action.changed_route_ids:
                assigned = True
                changed_route = action.routes_after[route.route_id]
                route = route.add(copy.deepcopy(changed_route.nodes), copy.deepcopy(changed_route.times), verify=True)
                if not (initial_nodes + 2 == len(route.nodes)):
                    raise AssertionError(
                        f"Expected the nodes {initial_nodes + 2} but actual nodes {len(route.nodes)}"
                    )
            else:
                if not (initial_nodes == len(route.nodes)):
                    raise AssertionError(
                        f"Expected the nodes {initial_nodes} but actual nodes {len(route.nodes)}"
                    )

            assignments.extend(route.get_assignments())
            computed_solution.add(route, verify=True)

        return self._exhaustive_search_insertion_finalize(
            instance, routing, manager, computed_solution, assigned, assignments
        )

    def get_feasible_actions(self, instance, allow_rejection=False):
        """
        :param instance: dictionary object describing the problem instance
        :param allow_rejection: allow rejection as one of the feasible actions

        obtain feasible actions

        :return: dictionary of status including whether the new request is assigned or not,
        if assigned the complete routing-instance object, and assignment dataframe
        """
        from env.solution.Action import ComprehensiveAction
        node_map = self._get_node_map(instance)

        routes = instance["state"].routes
        new_request = instance["state"].new_request

        actions = []
        for i in range(len(routes)):
            route = copy.deepcopy(routes[i])
            route.verify()
            temp_stats = route.get_feasible_actions(new_request, node_map, instance["state"].current_time)

            if len(temp_stats) > 0:
                for temp_stat in temp_stats:
                    actions.append(
                        ComprehensiveAction(initial_routes=routes, changed_routes={route.route_id: temp_stat})
                    )

        if len(actions) == 0 or allow_rejection:
            actions.append(ComprehensiveAction(initial_routes=routes, changed_routes={}))
        return actions

    def apply_insertion(self, instance, action):
        """
        :param instance: dictionary object describing the problem instance
        :param action: specific insertion action

        perform RL based insertion

        :return: dictionary of status including whether the new request is assigned or not,
        if assigned the complete routing-instance object, and assignment dataframe
        """
        routes = instance["routes"]
        completed_routes = instance["completed_routes"]

        assignments = []

        if self.config["execution_mode"] != ExecutionModes.GatherExperience.value:
            for i in range(len(completed_routes)):
                assignments.extend(completed_routes[i].get_assignments())

        computed_solution = self._init_computed_solution(instance)
        assigned = False
        instance["objective"] = self.config["objective"]
        for i in range(len(routes)):
            route = copy.deepcopy(routes[i])
            route.verify()
            initial_nodes = len(route.nodes)
            if route.route_id in action.changed_route_ids:
                assigned = True
                route = route.add(
                    copy.deepcopy(action.routes_after[i].nodes), copy.deepcopy(action.routes_after[i].times),
                    verify=True
                )
                if not (initial_nodes + 2 == len(route.nodes)):
                    raise AssertionError(
                        f"Expected the nodes {initial_nodes + 2} but actual nodes {len(route.nodes)}"
                    )
            else:
                if not (initial_nodes == len(route.nodes)):
                    raise AssertionError(
                        f"Expected the nodes {initial_nodes} but actual nodes {len(route.nodes)}"
                    )

            assignments.extend(route.get_assignments())
            computed_solution.add(route, verify=True)

        # for the moment, I leave them none
        routing = None
        manager = None
        return self._exhaustive_search_insertion_finalize(
            instance, routing, manager, computed_solution, assigned, assignments
        )

    def exhaustive_search_bulk_insert(self, instance):
        """
        :param instance: dictionary object describing the problem instance

        perform exhaustive search bulk insertion

        :return: dictionary of status including whether the new request is assigned or not,
        if assigned the complete routing-instance object, and assignment dataframe
        """
        import math
        from env.solution.Action import ComprehensiveAction
        from env.solution.RouteNode import generate_pick_up_node, generate_drop_off_node

        node_map = self._get_node_map(instance)

        routes = instance["routes"]
        completed_routes = instance["completed_routes"]
        requests = instance["requests"][instance["insertion_request_idx"]:]

        assignments = []
        if self.config["execution_mode"] != ExecutionModes.GatherExperience.value:
            for i in range(len(completed_routes)):
                assignments.extend(completed_routes[i].get_assignments())

        routing, manager = self._setup_problem(instance)

        assigned_requests = set()

        computed_solution = self._init_computed_solution(instance)

        def check_feasibility(_args):
            _route, _, _request, _node_map = _args
            return _route.check_feasible(instance, _request, _node_map)

        instance["objective"] = self.config["objective"]

        while True:
            min_cost = math.inf
            stat = None
            best_route_idx = -1
            best_request = None
            arguments = []
            for k in range(len(requests)):
                request = copy.deepcopy(requests[k])
                if request.request_id not in assigned_requests:
                    for i in range(len(routes)):
                        route = copy.deepcopy(routes[i])
                        arguments.append((route, i, request, node_map))

            from common.multi_proc import thread_pool_executor_wrapper

            temp_stat_list = thread_pool_executor_wrapper(
                check_feasibility,
                arguments,
                number_of_workers=self.config["threads"],
                wait_for_response=True,
                flatten=False
            )
            for p, temp_stat in enumerate(temp_stat_list):

                route, route_id, request, _ = arguments[p]
                if temp_stat:
                    # at insertion threshold is not considered
                    insertion_cost = route.get_cost(
                        objective=self.config["objective"],
                        state=instance,
                        action=ComprehensiveAction(
                            initial_routes=routes,
                            changed_routes={route.route_id: temp_stat}
                        )
                    )
                    if insertion_cost < min_cost:
                        min_cost = insertion_cost
                        stat = temp_stat
                        best_route_idx = route_id
                        best_request = request

            if best_route_idx >= 0:
                route = routes[best_route_idx]
                route.add(copy.deepcopy(stat.nodes), copy.deepcopy(stat.times), verify=True)
                pickup_node = generate_pick_up_node(best_request, current_time=instance["current_time"])
                dropoff_node = generate_drop_off_node(best_request, current_time=instance["current_time"])
                node_map[best_request.pickup_idx] = pickup_node
                node_map[best_request.dropoff_idx] = dropoff_node
                assigned_requests.add(best_request.request_id)

                if len(requests) == len(assigned_requests):
                    break
            else:
                break

        for i in range(len(routes)):
            route = copy.deepcopy(routes[i])
            assignments.extend(route.get_assignments())
            computed_solution.add(route, verify=True)

        assigned = True
        return self._exhaustive_search_insertion_finalize(
            instance, routing, manager, computed_solution, assigned, assignments
        )

    def simulated_annealing(self, instance):
        """
        :param instance: initial set of routes
        :return: do the simulated annealing search for the minimum solution,
        return the minimum solution, costs and summary file
        """
        from common.general import get_df_from_dict
        from env.solution.Action import ComprehensiveAction
        from learn.agent.AgentManager import AgentManager

        completed_routes = instance["completed_routes"]
        current_routes = instance["routes"]
        max_nodes = max([len(route.nodes) for route in current_routes])

        instance["current_cost"] = AgentManager.instance().get_cost(
            objective=self.config["search_objective"],
            state=instance,
            action=ComprehensiveAction(initial_routes=current_routes),
            use_target=instance["use_target"]
        )

        if max_nodes <= 2:
            assignments, computed_solution, updated_routes = self._wrap_solution(current_routes, completed_routes)
            return {
                "success": True,
                "info": "Not sufficient nodes to perform alterations !!!",
                "start_cost": instance["current_cost"],
                "end_cost": instance["current_cost"],
                "solution_df": get_df_from_dict(assignments),
                "computed_solution": computed_solution,
                "final_action": ComprehensiveAction(initial_routes=current_routes)
            }

        final_action, start_cost, end_cost = self._simulated_annealing(instance)

        assignments, computed_solution, updated_routes = self._wrap_solution(
            final_action.routes_after, completed_routes
        )

        search_summaries = None
        if self.config["write_individual_summaries"]:
            search_summaries = [
                {
                    "iteration": i + 1,
                    "cost": objective_value,
                    "compute_time": final_action.compute_times[i],
                    **final_action.operation_counts_list[i]
                }
                for i, objective_value in enumerate(final_action.costs)
            ]

        operations = copy.deepcopy(final_action.operation_counts_list)
        assignment_df = self._write_summaries(instance, assignments, search_summaries)

        # clear unwanted statistics
        final_action.clear()
        return {
            "success": True,
            "info": f"Performed simulated annealing for {self.config['search_duration']}",
            "start_cost": start_cost,
            "end_cost": end_cost,
            "operations": operations,
            "solution_df": assignment_df,
            "computed_solution": computed_solution,
            "final_action": final_action
        }

    @staticmethod
    def _wrap_solution(current_routes, completed_routes):
        """
        :param current_routes: current active parts of routes as list of route
        :param completed_routes: complete part of the routes as list of route
        """
        from env.solution.Solution import Solution
        computed_solution = Solution(current_routes, completed_routes)
        assignments = []
        for i, completed_route in enumerate(completed_routes):
            assignments.extend(completed_route.get_assignments())

        updated_routes = []
        for route in current_routes:
            if len(route.nodes) == 0:
                route.reset()
            updated_routes.append(route)
            assignments.extend(route.get_assignments())
            computed_solution.add(route, verify=True)
        return assignments, computed_solution, updated_routes

    def _setup_problem(self, instance):
        """
        :param instance: dictionary object describing the problem instance

        set up the problem instance using Google Routing API

        :return: routing, manager generated using Google Routing API
        for given problem instance
        """
        from common.logger import logger

        matrix = instance["matrix"]
        requests = instance["requests"]
        routes = instance["routes"]
        node_map = instance["node_map"]
        manager, routing = self._init_manager_and_model(instance)

        time_callback_index = self._add_time_dimension(routing, matrix)

        for i, request in enumerate(requests):
            self.__add_request_constraints(request, routing, manager, instance)

        self._add_capacity_dimension(routing, manager, node_map, routes)

        self._add_route_constraints(routing, instance)

        self._add_dropping_penalties(routing, manager, instance)

        self._add_costs(routing, instance, time_callback_index, add_fixed_cost=True)

        logger.info(f"SETUP ROUTING INSTANCE WITH {len(requests)} REQUESTS")
        return routing, manager

    def __add_request_constraints(self, request, routing, manager, instance):
        current_time = instance["current_time"]
        number_of_routes = len(instance["routes"])

        time_dimension = routing.GetDimensionOrDie(self.time_dimension_name)
        if not request.is_served():
            if request.relative_dropoff_idx is None:
                raise AssertionError("Dropoff Index should have a value, currently it is None")

            if request.relative_dropoff_idx < 2 * number_of_routes:
                raise AssertionError(
                    f"Dropoff Index should be greater than or equal to {2 * number_of_routes},"
                    f"currently it is {request.relative_dropoff_idx}"
                )
            delivery_index = manager.NodeToIndex(request.relative_dropoff_idx)
            time_dimension.CumulVar(delivery_index).SetRange(
                request.get_earliest_dropoff(current_time=current_time),
                request.get_latest_dropoff(current_time=current_time)
            )
            assigned_route_id = request.get_assigned_route_id()
            if request.is_picked():
                if assigned_route_id < 0:
                    raise AssertionError(f"Picked up request must be assigned to a vehicle")
                if assigned_route_id >= number_of_routes:
                    raise AssertionError("Assigned vehicle ID should be less than number of vehicles")
                # active requests (i.e., picked up but not dropped off)
                routing.VehicleVar(delivery_index).SetValues([assigned_route_id])
            else:
                # full fresh requests
                pickup_index = manager.NodeToIndex(request.relative_pickup_idx)
                if not request.is_served():
                    if request.relative_pickup_idx is None:
                        raise AssertionError("Pickup Index should have a value, currently it is None")

                    if request.relative_pickup_idx < 2 * number_of_routes:
                        raise AssertionError(
                            f"Pickup Index should be greater than or equal to {2 * number_of_routes},"
                            f"currently it is {request.relative_pickup_idx}"
                        )
                routing.AddPickupAndDelivery(pickup_index, delivery_index)
                routing.solver().Add(
                    time_dimension.CumulVar(pickup_index) <= time_dimension.CumulVar(delivery_index)
                )

                time_dimension.CumulVar(pickup_index).SetRange(
                    request.get_earliest_pickup(current_time=current_time),
                    request.get_latest_pickup(current_time=current_time)
                )

                fix_vehicle = assigned_route_id >= 0 and request.is_fixed_to_route()

                if fix_vehicle:
                    # vehicle-ids start with zero !!!
                    routing.VehicleVar(pickup_index).SetValues([assigned_route_id])
                    routing.VehicleVar(delivery_index).SetValues([assigned_route_id])

                # this will ensure the pick-up and drop-off happen using same vehicle.
                routing.solver().Add(
                    routing.VehicleVar(pickup_index) == routing.VehicleVar(delivery_index)
                )

    def _process_solution(self, instance, routing, manager, solution, routing_monitor):
        from common.logger import logger
        from env.solution.RouteNode import generate_pick_up_node, generate_drop_off_node

        routes = instance["routes"]
        requests = instance["requests"]

        number_of_vehicles = len(routes)
        expected_request_ids = set([request.request_id for request in requests])

        # extract the assignment from the output and compute the results
        # to match with our heuristics
        nodes = {}
        pick_up_request = {}
        drop_off_request = {}
        for request in requests:
            if not request.is_served():
                if not request.is_picked():
                    nodes[request.relative_pickup_idx] = generate_pick_up_node(
                        request, current_time=instance["current_time"]
                    )
                    pick_up_request[request.relative_pickup_idx] = request
                nodes[request.relative_dropoff_idx] = generate_drop_off_node(
                    request, current_time=instance["current_time"]
                )
                drop_off_request[request.relative_dropoff_idx] = request

        dropped_request_ids = set()
        for node in range(routing.Size()):
            if routing.IsStart(node) or routing.IsEnd(node):
                continue
            if solution.Value(routing.NextVar(node)) == node:
                dropped_request_ids.add(nodes[manager.IndexToNode(node)].request_id)

        time_dimension = routing.GetDimensionOrDie(self.time_dimension_name)
        computed_solution = self._init_computed_solution(instance)
        assigned_request_ids = set()

        assignments = []
        if self.config["execution_mode"] != ExecutionModes.GatherExperience.value:
            for i in range(len(instance["completed_routes"])):
                assignments.extend(instance["completed_routes"][i].get_assignments())

        for r_i in range(number_of_vehicles):
            route_id = routes[r_i].route_id
            time_values = []
            route_nodes = MappedList(discard_duplicate=True)
            index = routing.Start(route_id)
            time_values.append(solution.Value(time_dimension.CumulVar(index)))
            while not routing.IsEnd(index):
                index = solution.Value(routing.NextVar(index))
                time_value = solution.Value(time_dimension.CumulVar(index))
                node_idx = manager.IndexToNode(index)

                if node_idx in pick_up_request.keys():
                    req = pick_up_request[node_idx]
                    if not req.is_picked():
                        req.assert_pickup_time(time_value, instance["current_time"])
                        time_values.append(time_value)

                elif node_idx in drop_off_request.keys():
                    req = drop_off_request[node_idx]
                    assigned_request_ids.add(req.request_id)
                    if not req.is_served():
                        req.assert_dropoff_time(time_value, instance["current_time"])
                        time_values.append(time_value)

                else:
                    time_values.append(time_value)

                if node_idx in nodes.keys():
                    route_nodes.append(nodes[node_idx])

            route = routes[route_id]

            if len(route_nodes) > 0:
                route = route.add(route_nodes.get_list(), time_values[1:-1], verify=True)
                assignments.extend(route.get_assignments())
            else:
                route = route.reset()
            computed_solution.add(route, verify=True)

        if not (len(expected_request_ids) == len(assigned_request_ids) + len(dropped_request_ids)):
            raise AssertionError(
                f"Total Requests ({len(expected_request_ids)}) = "
                f"Accepted Requests ({len(assigned_request_ids)}) + Dropped Requests ({len(dropped_request_ids)})"
            )
        assignments_df = self._process_solution_finalize(instance, routing_monitor, assignments)
        logger.info(f"OBJECTIVE ({instance['mode'].value.upper()}): {solution.ObjectiveValue()}")
        return computed_solution, assignments_df, assigned_request_ids, dropped_request_ids

    def _update(self, state=None, new_request=None, change_environment=True):
        from common.general import convert_sec_to_hh_mm_ss
        from common.logger import logger
        if state is None:
            state = self.environment.current_state
        logger.info(f"UPDATE THE STATE AT {convert_sec_to_hh_mm_ss(state.current_time)}")
        response = self.environment.update(
            state.current_requests,
            state.routes,
            state.completed_routes,
            state.current_time,
            state.action_map,
            new_request=new_request
        )
        if change_environment:
            self.environment.current_solution.routes = response["routes"]
            self.environment.current_solution.completed_routes = response["completed_routes"]
        state.routes = response["routes"]
        state.completed_routes = response["completed_routes"]
        return response

    def _get_config(self, state, update_response, temp_matrix=True, mode=SolvingModes.INSERT):
        from env.data.TimeMatrix import MatrixManager
        if str(state.instance_info["model_idx"]) != "-1":
            output_dir = f"{self.environment.output_dir}_IND_SUMMARY/" + \
                         f"{state.instance_info['model_idx']}/" + \
                         f"{state.instance_info['sample_idx']}/" + \
                         f"{state.current_idx}"
        else:
            output_dir = f"{self.environment.output_dir}_IND_SUMMARY/" + \
                         f"{state.instance_info['sample_idx']}/" + \
                         f"{state.current_idx}"

        matrix_obj = MatrixManager.instance().get_matrix(temp=temp_matrix)
        return {
            "matrix": matrix_obj.matrix if matrix_obj else [[]],
            "locations": matrix_obj.locations if matrix_obj else [],
            "routes": update_response["routes"],
            "completed_routes": update_response["completed_routes"],
            "node_map": update_response["node_map"],
            "requests": update_response["all_requests"],
            "output_dir": output_dir,
            "current_time": state.current_time,
            "instance": self.environment.current_routing_instance,
            "mode": mode
        }

    def _get_insertion_config(self, state, new_request, change_environment=True):
        from env.data.TimeMatrix import MatrixManager
        from common.types import SolvingModes
        response = self._update(state=state, new_request=new_request, change_environment=change_environment)
        MatrixManager.instance().temporary_reconstruct(response["locations"], update_temp=True)
        return self._get_config(state, response, temp_matrix=True, mode=SolvingModes.INSERT), response

    def _get_search_config(self, state, change_environment=True):
        from env.data.TimeMatrix import MatrixManager
        from common.types import InsertionModes, SolvingModes
        if self.environment.last_insertion_success and change_environment:
            config = self._get_config(
                state, self.environment.last_save_state, temp_matrix=False, mode=SolvingModes.SEARCH
            )
        else:
            response = self._update(state=state, change_environment=change_environment)
            MatrixManager.instance().temporary_reconstruct(response["locations"], update_temp=True)
            config = self._get_config(state, response, temp_matrix=True, mode=SolvingModes.RESTORE)
            config["insert_mode"] = InsertionModes.BULK
        return config
