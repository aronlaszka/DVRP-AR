##################################################################
#              GOOGLE OR-TOOL ROUTING IMPLEMENTATION             #
##################################################################
import copy
import json
import os
import shutil
import sys
from datetime import datetime

from common.general import measure_time, directory_exists
from common.logger import logger
from common.types import InsertionHeuristics, MetaHeuristics, ExecutionModes


class RoutingSolver:
    """
        This is a wrapper class for Pickup and Delivery Problem with Time Windows Solver
        [half of the implementation based on Google OR Tools and rest based on my implementation that could support
        any custom objective functions]
    """

    def __init__(self, args, custom_config):
        from ortools.constraint_solver import routing_enums_pb2

        # default configurations
        self.config = {
            "threads": args.threads,
            "execution_mode": args.execution_mode,
            "travel_time_source": args.travel_time_source,
            "interpolation_type": args.interpolation_type,
            "objective": args.objective,
            "search_objective": args.search_objective,
            "allow_rejection": args.allow_rejection,
            "acceptance_value_threshold": args.acceptance_value_threshold,
            "data_dir": args.data_dir,
            "data_id": args.data_id,
            "rtv_rh": args.rtv_rh,
            "rtv_interval": args.rtv_interval,
            "rtv_time_limit": args.rtv_time_limit,
            "rtv_bin_name": args.rtv_bin_name,
            "allow_dropping": False,
            "dwell_time_pickup": args.dwell_time_pickup,
            "dwell_time_dropoff": args.dwell_time_dropoff,
            "first_solution_strategy": routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_COST_INSERTION,
            "meta_heuristic": routing_enums_pb2.LocalSearchMetaheuristic.AUTOMATIC,
            "use_full_propagation": True,
            "solution_constrained": False,  # if solution constrained consider the number of solutions,
            "search_duration": 1,
            "max_search_duration": 1,
            "solution_limit": 1,
            "max_solution_limit": 1,
            "fixed_timeahead": args.time_ahead,  # 30 minutes
            "start_prob": 0.99,
            "end_prob": 0.01,
            "write_individual_summaries": args.write_individual_summaries,
            "log_search": args.log_search,
            "log_search_detailed": args.log_search_detailed,
        }
        for key, value in custom_config.items():
            self.config[key] = value
        self.time_dimension_name = "TIME"
        self.am_capacity_dimension_name = "AM_CAPACITY"
        self.wc_capacity_dimension_name = "WC_CAPACITY"
        self.environment = None

    def _setup_problem(self, instance):
        raise NotImplementedError

    @staticmethod
    def _init_manager_and_model(instance):

        from ortools.constraint_solver import pywrapcp

        matrix = instance["matrix"]
        routes = instance["routes"]
        locations = instance["locations"]

        number_of_locations = len(matrix[0])  # pickup, dropoff for each request and depot
        number_of_vehicles = len(routes)

        start_indices = [locations.index(routes[r_i].start_pos) for r_i in range(number_of_vehicles)]
        end_indices = [locations.index(routes[r_i].end_pos) for r_i in range(number_of_vehicles)]

        # create the routing manager
        manager = pywrapcp.RoutingIndexManager(
            number_of_locations, number_of_vehicles, start_indices, end_indices
        )

        # Create Routing model.
        routing = pywrapcp.RoutingModel(manager)
        return manager, routing

    def _add_time_dimension(self, routing, matrix):
        time_callback_index = routing.RegisterTransitMatrix(matrix)

        # add time dimension
        routing.AddDimension(
            time_callback_index,
            self.config["day_max_time"],  # maximum wait time
            self.config["day_max_time"],  # maximum time a vehicle can reach
            False,  # Don't force start cumulative to zero.
            self.time_dimension_name
        )
        return time_callback_index

    def _add_capacity_dimension(self, routing, manager, node_map, routes):
        number_of_vehicles = len(routes)

        # Add Capacity constraint.
        def am_demand_callback(from_index):
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            if from_node in node_map.keys():
                return node_map[from_node].capacity["am"]
            return 0

        am_demand_callback_index = routing.RegisterUnaryTransitCallback(am_demand_callback)

        vehicle_am_capacities = [routes[r_i].capacity["am"] for r_i in range(number_of_vehicles)]
        routing.AddDimensionWithVehicleCapacity(
            am_demand_callback_index,
            0,  # null capacity slack
            vehicle_am_capacities,  # vehicle maximum capacities
            False,  # start cumulative to zero
            self.am_capacity_dimension_name
        )

        def wc_demand_callback(from_index):
            # Convert from routing variable Index to demands NodeIndex.
            from_node = manager.IndexToNode(from_index)
            if from_node in node_map.keys():
                return node_map[from_node].capacity["wc"]
            return 0

        wc_demand_callback_index = routing.RegisterUnaryTransitCallback(wc_demand_callback)

        vehicle_wc_capacities = [routes[r_i].capacity["wc"] for r_i in range(number_of_vehicles)]
        routing.AddDimensionWithVehicleCapacity(
            wc_demand_callback_index,
            0,  # null capacity slack
            vehicle_wc_capacities,  # vehicle maximum capacities
            False,  # start cumulative to zero
            self.wc_capacity_dimension_name
        )

    def _add_route_constraints(self, routing, instance):
        routes = instance["routes"]
        number_of_vehicles = len(routes)
        time_dimension = routing.GetDimensionOrDie(self.time_dimension_name)
        # Add time window constraints for each run start node.
        for r_i in range(number_of_vehicles):
            route = routes[r_i]
            s_idx = routing.Start(route.route_id)

            time_dimension.CumulVar(s_idx).SetRange(route.start_time, route.end_time)
            e_idx = routing.End(route.route_id)

            time_dimension.CumulVar(e_idx).SetRange(route.start_time, route.end_time)

    def _add_dropping_penalties(self, routing, manager, instance):
        from common.types import InsertionModes, SolvingModes
        routes = instance["routes"]
        requests = instance["requests"]
        insert_mode = InsertionModes.BULK
        if instance["mode"] == SolvingModes.INSERT:
            insert_mode = instance["insert_mode"]

        # request can be dropped only during insertion (it can be single insertion or bulk insertion)
        allow_dropping = (instance["mode"] in [SolvingModes.INSERT, SolvingModes.BULK_INSERT]
                          and self.config["allow_dropping"])

        if allow_dropping:
            penalty = 24 * 60 * 60 * len(routes) * 1000
            if insert_mode == InsertionModes.SINGLE:  # only allow dropping last requests
                pickup_index = manager.NodeToIndex(requests[-1].relative_pickup_idx)
                delivery_index = manager.NodeToIndex(requests[-1].relative_dropoff_idx)
                routing.AddDisjunction([pickup_index], penalty)
                routing.AddDisjunction([delivery_index], penalty)
            else:
                for request in requests:
                    if not request.is_already_in_manifest():
                        pickup_index = manager.NodeToIndex(request.relative_pickup_idx)
                        delivery_index = manager.NodeToIndex(request.relative_dropoff_idx)
                        routing.AddDisjunction([pickup_index], penalty)
                        routing.AddDisjunction([delivery_index], penalty)

    def _add_costs(self, routing, instance, time_callback_index, add_fixed_cost=False):
        from common.types import SolvingModes, ObjectiveTypes

        if instance["mode"] == SolvingModes.RESTORE and self.config["objective"] == ObjectiveTypes.IdleTime.value:
            # at restoring mode adding fixed cost option will not work for maximizing idle times
            add_fixed_cost = False

        routes = instance["routes"]
        number_of_vehicles = len(routes)
        time_dimension = routing.GetDimensionOrDie(self.time_dimension_name)
        # Add time window constraints for each run start node.
        for r_i in range(number_of_vehicles):
            route = routes[r_i]

            match self.config["objective"]:
                # total_travel_time = total_drive_time + total_idle_time
                # idle_time = - (total_drive_time - total_travel_time)
                case ObjectiveTypes.TravelTime.value:
                    # objective = MIN(total_travel_time)
                    factor = 1
                    time_dimension.SetSpanCostCoefficientForVehicle(factor, route.route_id)

                case ObjectiveTypes.DriveTime.value:
                    # objective = MIN(total_drive_time)
                    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

                case ObjectiveTypes.IdleTime.value:
                    # objective = MIN(total_drive_time - total_travel_time)
                    # objective = MAX(idle_time)
                    routing.SetArcCostEvaluatorOfAllVehicles(time_callback_index)

                    # restore is not properly supported !!!
                    if add_fixed_cost:
                        if self.config["objective"] == ObjectiveTypes.IdleTime.value:
                            routing.SetFixedCostOfVehicle(
                                route.get_fixed_cost(self.config["objective"]), route.route_id
                            )

            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.Start(r_i)))
            routing.AddVariableMinimizedByFinalizer(time_dimension.CumulVar(routing.End(r_i)))

    def solve(self, instance):
        from common.types import InsertionModes
        from env.base.wrapper.RoutingSupport import RoutingInstance

        func_name = f"routing_{instance['mode'].value}"

        routing, manager, solution, routing_monitor = self.__getattribute__(func_name)(instance)

        if routing is not None:
            error_msg = self._error_msg(routing.status())
        else:
            error_msg = "UNABLE TO GENERATE ROUTING MODEL"

        insert_objective = None
        search_objective = None
        if solution:
            if instance["instance"] and instance["instance"].solution:
                insert_objective = instance["instance"].solution.ObjectiveValue()

            computed_solution, assignments_df, assigned_request_ids, dropped_request_ids = (
                self._process_solution(
                    instance, routing, manager, solution, routing_monitor
                )
            )
            service_rate = -1
            if len(dropped_request_ids) > 0:
                service_rate = len(assigned_request_ids) / (len(assigned_request_ids) + len(dropped_request_ids))

            improvement = 0
            improvement_percentage = 0
            if insert_objective is not None:
                search_objective = solution.ObjectiveValue()
                # skip the restoring cases
                if insert_objective > search_objective and insert_objective != 0:
                    improvement = insert_objective - search_objective
                    improvement_percentage = improvement / insert_objective

            success = True
            if "insert_mode" in instance:
                if instance["insert_mode"] == InsertionModes.SINGLE:
                    if self.config["allow_dropping"]:
                        if len(dropped_request_ids) > 0:
                            success = False

            return {
                "success": success,  # otherwise assertion already fails
                "assigned": len(assigned_request_ids),
                "assigned_request_ids": assigned_request_ids,
                "dropped_request_ids": dropped_request_ids,
                "error_msg": error_msg,
                "start_cost": insert_objective,
                "end_cost": search_objective,
                "solution_df": assignments_df,
                "service_rate": service_rate,
                "improvement": improvement,
                "improvement_percentage": improvement_percentage,
                "computed_solution": computed_solution,
                "instance": RoutingInstance(routing, manager, solution)
            }
        return {
            "success": False,
            "start_cost": insert_objective,
            "end_cost": insert_objective,
            "error_msg": error_msg,
        }

    def _solve(self, routing, depth=1, solution=None):
        # meta heuristics
        from common.logger import logger
        from env.base.wrapper.RoutingSupport import RoutingMonitor
        from ortools.constraint_solver import pywrapcp

        routing_monitor = RoutingMonitor(routing)
        routing.AddAtSolutionCallback(routing_monitor)

        if solution:
            solution = routing.SolveFromAssignmentWithParameters(
                assignment=solution, search_parameters=self._generate_params()
            )
        else:
            solution = routing.SolveWithParameters(search_parameters=self._generate_params())

        # copied error codes from
        # https://developers.google.com/optimization/routing/routing_options#search-status
        # in-order to provide better description
        if routing.status() not in [pywrapcp.RoutingModel.ROUTING_SUCCESS, pywrapcp.RoutingModel.ROUTING_OPTIMAL]:
            if self.config['solution_constrained']:
                if routing.status() == pywrapcp.RoutingModel.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED:
                    if self.config['solution_limit'] * depth >= self.config["max_solution_limit"]:
                        logger.warning(
                            f"ALREADY REACHED THE MAXIMUM NUMBER OF SOLUTIONS "
                            f"({self.config['max_solution_limit']}) FOR THE PROBLEM INSTANCE."
                        )
                        return solution, routing_monitor
                    logger.warning(
                        f"RUNNING THE SEARCH WITH AN ADDITIONAL {self.config['solution_limit']} SOLUTIONS; "
                        f"SO FAR, SOLVER SEARCHED FOR {self.config['solution_limit'] * depth} SOLUTIONS."
                    )
                    return self._solve(routing, depth + 1, solution)
                else:
                    error_msg = self._error_msg(routing.status())
                    logger.error(error_msg)
            else:
                # if time-constrained
                if routing.status() in [
                    pywrapcp.RoutingModel.ROUTING_FAIL_TIMEOUT,
                    pywrapcp.RoutingModel.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED
                ]:
                    if self.config['search_duration'] * depth >= self.config["max_search_duration"]:
                        logger.warning(
                            f"ALREADY REACHED THE MAXIMUM SEARCH DURATION"
                            f"({self.config['max_search_duration']}) FOR THE PROBLEM INSTANCE."
                        )
                        return solution, routing_monitor
                    logger.warning(
                        f"RUNNING THE SEARCH WITH AN ADDITIONAL DURATION OF {self.config['search_duration']} SECONDS; "
                        f"SO FAR, SOLVER SEARCHED FOR {self.config['search_duration'] * depth} SECONDS."
                    )
                    return self._solve(routing, depth + 1, solution)
                else:
                    error_msg = self._error_msg(routing.status())
                    logger.error(error_msg)

        return solution, routing_monitor

    def _process_solution(self, instance, routing, manager, solution, routing_monitor):
        raise NotImplementedError

    def routing_insert(self, instance):
        """
        :param instance: dictionary object describing the problem instance

        perform quick insertion using Routing API

        :return: routing, manager, solution and routing-monitor generated using Google Routing API
        for given problem instance
        """
        routing, manager = self._setup_problem(instance)
        solution, routing_monitor = self._solve(routing)
        return routing, manager, solution, routing_monitor

    def routing_bulk_insert(self, instance):
        """
        :param instance: dictionary object describing the problem instance

        perform quick insertion using Routing API

        :return: routing, manager, solution and routing-monitor generated using Google Routing API
        for given problem instance
        """
        from common.types import SolvingModes, InsertionModes

        self.config["allow_dropping"] = True
        instance['mode'] = SolvingModes.BULK_INSERT
        instance['insert_mode'] = InsertionModes.BULK
        routing, manager = self._setup_problem(instance)
        solution, routing_monitor = self._solve(routing)
        self.config["allow_dropping"] = False
        return routing, manager, solution, routing_monitor

    def routing_restore(self, instance):
        """
        :param instance: dictionary object describing the problem instance

        perform restoration of assignment using Routing API

        :return: routing, manager, solution and routing-monitor generated using Google Routing API
        for given problem instance
        """
        from common.logger import logger
        from common.types import SolvingModes
        from env.base.wrapper.RoutingSupport import RoutingMonitor

        logger.info("RESTORING ROUTING SOLUTION TO MATCH WITH THE CURRENT STATE")
        instance['mode'] = SolvingModes.RESTORE
        routing, manager = self._setup_problem(instance)
        routing_monitor = RoutingMonitor(routing)
        solution = self._convert_to_routing_api_routes(manager, routing, instance, exit_on_error=True)
        return routing, manager, solution, routing_monitor

    def routing_search(self, instance):
        """
        :param instance: dictionary object describing the problem instance

        perform local search using Routing API

        :return: routing, manager, solution and routing-monitor generated using Google Routing API
        for given problem instance
        """
        # search mode
        if instance is not None and "instance" in instance and instance["instance"] is not None:
            routing = instance["instance"].model
            manager = instance["instance"].manager
            solution = instance["instance"].solution

            solution, routing_monitor = self._solve(routing=routing, solution=solution)
            return routing, manager, solution, routing_monitor
        return None, None, None, None

    def exhaustive_search_insert(self, instance):
        raise NotImplementedError

    def exhaustive_search_bulk_insert(self, instance):
        raise NotImplementedError

    def simulated_annealing(self, instance):
        raise NotImplementedError

    def _simulated_annealing(self, instance):
        import math
        import json
        import random
        from datetime import datetime
        from common.logger import logger
        from env.solution.Action import ComprehensiveAction
        from env.base.wrapper.NearestNeighbor import nearest_neighbour
        from learn.agent.AgentManager import AgentManager

        start = datetime.now()
        start_prob = float(self.config['start_prob'])
        end_prob = float(self.config['end_prob'])

        if self.config["log_search"]:
            logger.info(
                f"SIMULATED ANNEALING CONFIGURATIONS | RUNNING TIME: {self.config['search_duration']} | "
                f"START PROB: {start_prob} | END PROB: {end_prob}"
            )

        current_routes = copy.deepcopy(instance["routes"])
        initial_routes = copy.deepcopy(instance["routes"])
        best_routes = copy.deepcopy(instance["routes"])

        initial_cost = instance["current_cost"]
        current_cost = initial_cost
        min_cost = initial_cost

        temp_start = -1.0 / math.log(start_prob)
        temp_end = -1.0 / math.log(end_prob)
        search_duration_delta = self.config["search_duration"] - 1.0 if self.config["search_duration"] > 1 else 1
        rate_of_temp = (temp_end / temp_start) ** (1.0 / search_duration_delta)

        selected_temp = temp_start
        delta_e_avg = 0.0
        number_of_accepted = 1
        i = 0
        compute_times = []
        costs = []
        operation_counts_list = []
        current_time = datetime.now()
        changed_route_ids = set()
        while (current_time - start).total_seconds() < self.config["search_duration"]:
            nearest_neighbor_input = {
                "objective": self.config["search_objective"],
                "state": instance["state"],
                "routes": current_routes,
                "use_target": instance["use_target"],
                "current_time": instance["current_time"],
                "iteration": i
            }

            mutation_action = nearest_neighbour(nearest_neighbor_input)

            instance["routes"] = copy.deepcopy(current_routes)
            nn_cost = AgentManager.instance().get_cost(
                objective=self.config["search_objective"],
                state=instance,
                action=mutation_action,
                use_target=instance["use_target"]
            )

            delta_reward = abs(nn_cost - current_cost)
            if len(mutation_action.operation_records) == 0:
                # filter the scenarios where there is no-mutations
                accept = False
            elif nn_cost > current_cost:
                # if the reward is worse than current best reward, but still accept at the acceptance probability
                if i == 0:
                    delta_e_avg = delta_reward
                denominator = (delta_e_avg * selected_temp)
                accept_prob = round(
                    math.exp(-1 * math.inf) if denominator == 0 else math.exp(-delta_reward / denominator), 5
                )
                random_prob = round(random.random(), 5)
                accept = True if accept_prob > random_prob else False
                if accept:
                    if self.config["log_search_detailed"]:
                        logger.info(
                            f"[ACCEPT-COND] CYCLE: {i + 1}, TEMPERATURE: {round(selected_temp, 5)}, COST: {nn_cost}, "
                            f"ACCEPTANCE PROBABILITY {accept_prob} > RANDOM PROBABILITY {random_prob}, "
                            f"\nRECORDS: {json.dumps(mutation_action.operation_records, indent=4)}"
                        )
                    elif self.config["log_search"]:
                        logger.info(
                            f"[ACCEPT-COND] CYCLE: {i + 1}, TEMPERATURE: {round(selected_temp, 5)}, COST: {nn_cost}, "
                            f"ACCEPTANCE PROBABILITY {accept_prob} > RANDOM PROBABILITY {random_prob}"
                        )
            else:
                accept = True
                if self.config["log_search_detailed"]:
                    logger.info(
                        f"[ACCEPT] CYCLE: {i + 1}, TEMPERATURE: {round(selected_temp, 5)}, COST: {nn_cost},"
                        f"\nRECORDS: {json.dumps(mutation_action.operation_records, indent=4)}"
                    )
                elif self.config["log_search"]:
                    logger.info(
                        f"[ACCEPT] CYCLE: {i + 1}, TEMPERATURE: {round(selected_temp, 5)}, COST: {nn_cost}"
                    )

            if accept:
                if nn_cost <= min_cost:
                    best_routes = copy.deepcopy(mutation_action.routes_after)
                    min_cost = nn_cost

                    for route_id in mutation_action.changed_route_ids:
                        changed_route_ids.add(route_id)

                current_routes = copy.deepcopy(mutation_action.routes_after)
                current_cost = nn_cost
                delta_e_avg = delta_e_avg + (delta_reward - delta_e_avg) / number_of_accepted
                number_of_accepted += 1

            compute_times.append((datetime.now() - current_time).total_seconds())
            costs.append(nn_cost)
            operation_counts_list.append(mutation_action.operation_counts)

            current_time = datetime.now()

            selected_temp = temp_start * math.pow(rate_of_temp, (current_time - start).total_seconds())
            i += 1

        final_action = ComprehensiveAction(
            initial_routes=initial_routes,
            changed_routes={
                route.route_id: route for route in best_routes if route.route_id in changed_route_ids
            },
            costs=costs,
            compute_times=compute_times,
            operation_counts_list=operation_counts_list
        )
        return final_action, initial_cost, min_cost

    def rh_solve(self, instance):
        import math
        from common.general import create_dir, load_data_as_pandas_df
        from common.logger import logger
        from env.solution.RouteNode import generate_pick_up_node, generate_drop_off_node
        requests = instance["requests"]
        routes = instance["routes"]

        locations = [self.config["depot_location"].get_raw()]

        for request in requests:
            locations += [
                generate_pick_up_node(request, 0).loc.get_raw(),
                generate_drop_off_node(request, 0).loc.get_raw()
            ]

        network_data_dir = f"{self.config['data_dir']}/{self.config['data_id']}/networks"
        data_dir = f"{instance['output_dir']}"
        requests_dir = f"{data_dir}/requests"
        vehicles_dir = f"{data_dir}/vehicles"
        results_dir = f"{data_dir}/results"
        if directory_exists(data_dir):
            shutil.rmtree(data_dir, ignore_errors=True)
        create_dir(data_dir)
        for directory in [requests_dir, vehicles_dir, results_dir]:
            create_dir(directory)

        vehicles_info = self._generate_vehicle_contents(routes)
        requests_info, request_list, count = self._generate_request_contents(requests)

        with open(f"{data_dir}/requests/requests.csv", "w+") as requests_file:
            requests_file.write(requests_info)

        with open(f"{data_dir}/vehicles/vehicles.csv", "w+") as vehicles_file:
            vehicles_file.write(vehicles_info)

        from sys import platform
        mosek_home = os.environ['MSKHOME']
        mosek_library_path = ""
        if platform == "linux" or platform == "linux2":
            if 'LD_LIBRARY_PATH' not in os.environ or os.environ['LD_LIBRARY_PATH'] == "":
                os.environ['LD_LIBRARY_PATH'] = f'{mosek_home}/mosek/8/tools/platform/linux64x86/bin'
                logger.info(f"Environment variable LD_LIBRARY_PATH is set to {os.environ['LD_LIBRARY_PATH']}")
        elif platform == "darwin":
            # os.system() will directly take the environment variables so, we need to provide them
            if 'DYLD_LIBRARY_PATH' not in os.environ or os.environ['DYLD_LIBRARY_PATH'] == "":
                os.environ['DYLD_LIBRARY_PATH'] = f'{mosek_home}/mosek/10.0/tools/platform/osxaarch64/bin'
                logger.info(f"Environment variable DYLD_LIBRARY_PATH is set to {os.environ['DYLD_LIBRARY_PATH']}")

            if 'DYLD_FALLBACK_LIBRARY_PATH' not in os.environ or os.environ['DYLD_FALLBACK_LIBRARY_PATH'] == "":
                os.environ['DYLD_FALLBACK_LIBRARY_PATH'] = f'{mosek_home}/mosek/10.0/tools/platform/osxaarch64/bin'
                logger.info(
                    f"Environment variable DYLD_FALLBACK_LIBRARY_PATH is set to "
                    f"{os.environ['DYLD_FALLBACK_LIBRARY_PATH']}"
                )
            mosek_library_path = f"DYLD_LIBRARY_PATH={os.environ['DYLD_LIBRARY_PATH']}"
        else:
            logger.error(f"The OS platform: {platform} is not currently supported")
            sys.exit(-1)

        divider = len(routes)
        command = f"{mosek_library_path} " \
                  f"baselines/rh/{self.config['rtv_bin_name']} " \
                  f"{int(self.config['threads'])} " \
                  f"DATAROOT {data_dir} " \
                  f"NETWORKDATAROOT {network_data_dir} " \
                  f"RESULTS_DIRECTORY {results_dir} " \
                  f"RH {self.config['rtv_rh']} " \
                  f"VEHICLE_LIMIT {len(routes)} " \
                  f"CARSIZE {routes[0].capacity['am']} " \
                  f"INTERVAL {self.config['rtv_interval']} " \
                  f"MAX_WAITING 1800 " \
                  f"MAX_DETOUR 4500 " \
                  f"DWELL_PICKUP {self.config['dwell_time_pickup']} " \
                  f"DWELL_ALIGHT {self.config['dwell_time_dropoff']} " \
                  f"RTV_TIMELIMIT {math.floor(self.config['rtv_time_limit'] * 1000 / divider)}"

        logger.info(f"Executed command {command}")
        start_time = datetime.now()
        os.system(command)  # for some reason, subprocess.call() doesn't wait until all process finishes
        end_time = datetime.now()

        # deleting these files to save space !!!
        os.system(f"rm -rf {results_dir}/*.log")
        os.system(f"rm -rf {results_dir}/assign_*")
        os.system(f"rm -rf {results_dir}/ilp*")

        with open(f"{results_dir}/compute_time.csv", "w+") as compute_time:
            compute_time.write(str((end_time - start_time).total_seconds()))

        assignment_file_path = f"{results_dir}/assign.csv"
        df_assignment = load_data_as_pandas_df(assignment_file_path, exit_on_error=False)
        success = False
        if df_assignment is not None:
            if len(df_assignment) == 2 * count:
                with open(f"{results_dir}/served_requests.csv", "w+") as compute_time:
                    compute_time.write(str(len(request_list.keys())))
                success = True
        return {
            "success": success
        }

    def _generate_request_contents(self, requests):
        """
        :param requests: list of requests
        :return: provides requests in the form of string seperated by comma and newline
        """

        from env.data.TimeMatrix import MatrixManager
        from env.solution.RouteNode import generate_pick_up_node, generate_drop_off_node
        from env.structures.Measure import Duration
        _request_str = ""
        _request_list = {}
        _ptr = 0
        for _i, _request in enumerate(requests):
            _pick_up_node_obj = generate_pick_up_node(_request, 0)
            _drop_off_node_obj = generate_drop_off_node(_request, 0)
            _pick_up_node = MatrixManager.instance().get_node(_pick_up_node_obj.loc.get_raw())
            _drop_off_node = MatrixManager.instance().get_node(_drop_off_node_obj.loc.get_raw())
            _request_list[_request.request_id] = []
            for _k in range(_request.capacity["am"]):
                _request_list[_request.request_id].append(_ptr)
                _request_str += f"{_ptr},{_pick_up_node},{_pick_up_node_obj.loc.get_raw()[1]}," \
                                f"{_pick_up_node_obj.loc.get_raw()[0]},{_drop_off_node}," \
                                f"{_drop_off_node_obj.loc.get_raw()[1]},{_drop_off_node_obj.loc.get_raw()[0]}," \
                                f"{Duration(_pick_up_node_obj.earliest_arrival).hhmmss()}," \
                                f"{Duration(_pick_up_node_obj.latest_arrival).hhmmss()}," \
                                f"{Duration(_drop_off_node_obj.latest_arrival - self.config['dwell_time_dropoff']).hhmmss()}\n"
                _ptr += 1
        return _request_str, _request_list, _ptr

    @staticmethod
    def _generate_vehicle_contents(routes):
        """
        :return: provides vehicles in the form of string seperated by comma and newline
        """

        from env.data.TimeMatrix import MatrixManager
        from env.structures.Measure import Duration
        _vehicles_str = ""
        for _route in routes:
            _location = _route.start_pos.get_raw()
            _node_id = MatrixManager.instance().get_node(_location)
            _start_time = Duration(_route.original_start_time)
            _vehicles_str += f"{_route.route_id + 1},{_node_id},{_location[0]},{_location[1]}," + \
                             f"{_start_time.hhmmss()},{_route.capacity['am']}\n"
        return _vehicles_str

    @staticmethod
    def _get_node_map(instance):
        """
        :param instance: current instance object
        :return: mapping of node based on the original indices
        """
        from env.solution.RouteNode import generate_pick_up_node, generate_drop_off_node
        prev_requests = instance["requests"][:instance["insertion_request_idx"]]
        node_map = {}
        for prev_request in prev_requests:
            node_map[prev_request.pickup_idx] = generate_pick_up_node(
                prev_request, current_time=instance["current_time"]
            )
            node_map[prev_request.dropoff_idx] = generate_drop_off_node(
                prev_request, current_time=instance["current_time"]
            )
        return node_map

    def _convert_to_routing_api_routes(self, manager, routing, instance, exit_on_error=False):
        from env.base.ImplementationError import ImplementationError
        routes = instance["routes"]
        current_time = instance["current_time"]

        converted_routes = []
        descriptive_routes = []

        def get_possible_route_ids(idx):
            var_obj = routing.VehicleVar(manager.NodeToIndex(idx))
            id_values = var_obj.DebugString().replace(var_obj.Name(), "")
            id_values = id_values[1:-1]
            if 'inner' in id_values:
                possible_ids = [0]
            elif ".." in id_values:
                start_id, end_id = id_values.split("..")
                start_id, end_id = int(start_id), int(end_id)
                possible_ids = [i for i in range(start_id + 1, end_id + 1)]
            else:
                possible_ids = [int(id_values)]
            return possible_ids

        def respect_constraints(_route, _node, _i):
            time_window_constraint = (_node.earliest_arrival <= _route.times[_i] <= _node.latest_arrival)
            vehicle_assignment_constraint = int(_route.route_id) in get_possible_route_ids(_node.relative_idx)
            general_constraints = time_window_constraint and vehicle_assignment_constraint
            if _i == 0:
                enough_time_at_start = _route.start_time <= _route.times[_i] - _node.duration(_route.start_pos)
                return enough_time_at_start and general_constraints
            elif _i == len(_route.times) - 1:
                enough_time_at_end = _route.end_time >= _route.times[_i] + _route.end_pos.duration(_node)
                return enough_time_at_end and general_constraints
            return general_constraints

        time_dimension = routing.GetDimensionOrDie(self.time_dimension_name)

        request_ids = []
        for route in routes:
            request_ids.extend([node.request_id for node in route.nodes])
            converted_routes.append(tuple([manager.NodeToIndex(node.relative_idx) for node in route.nodes]))
            descriptive_routes.append(
                [
                    {
                        "route_id": route.route_id,
                        "expected_route_id": get_possible_route_ids(node.relative_idx),
                        "current_time": current_time,
                        "request_id": node.request_id,
                        "node_type": node.node_type.value,
                        "earliest_arrival": (
                            node.earliest_arrival,
                            time_dimension.CumulVar(manager.NodeToIndex(node.relative_idx)).Min()
                        ),
                        "latest_arrival": (
                            node.latest_arrival,
                            time_dimension.CumulVar(manager.NodeToIndex(node.relative_idx)).Max()
                        ),
                        "action_time": route.times[i]
                    }
                    for i, node in enumerate(route.nodes) if not respect_constraints(route, node, i)
                ]
            )

        solution = routing.ReadAssignmentFromRoutes(converted_routes, ignore_inactive_indices=True)
        if not solution:
            descriptive_route_json = json.dumps(descriptive_routes, indent=4)
            error_msg = f"FAILED TO GENERATE ASSIGNMENTS FROM {converted_routes}, "
            f"VIOLATED ROUTE NODES: {descriptive_route_json}"
            if exit_on_error:
                raise ImplementationError(error_msg)
            else:
                logger.error(error_msg)

        return solution

    def _exhaustive_search_insertion_finalize(
            self, instance, routing, manager, computed_solution, assigned, assignments
    ):
        """
        convert the solution to support RoutingAPI and write the summaries
        """
        from common.logger import logger
        from env.base.wrapper.RoutingSupport import RoutingInstance

        solution = None
        if assigned:
            if instance["verify"]:
                solution = self._convert_to_routing_api_routes(manager, routing, instance, exit_on_error=False)
        if solution:
            logger.info(f"OBJECTIVE (EXHAUSTIVE-SEARCH-INSERT): {solution.ObjectiveValue()}")
        assignments_df = self._write_summaries(instance, assignments)
        assigned_request_ids = set()
        dropped_request_ids = set()
        for route in instance["routes"]:
            for node in route.nodes:
                assigned_request_ids.add(node.request_id)

        for request in instance["requests"]:
            if request.request_id not in assigned_request_ids:
                dropped_request_ids.add(request.request_id)

        return {
            "success": assigned,
            "assigned": len(assigned_request_ids),
            "assigned_request_ids": assigned_request_ids,
            "dropped_request_ids": dropped_request_ids,
            "computed_solution": computed_solution,
            "instance": RoutingInstance(routing, manager, solution),
            "solution_df": assignments_df
        }

    def _process_solution_finalize(self, instance, routing_monitor, assignments):
        search_summaries = None
        if self.config["write_individual_summaries"]:
            search_summaries = [
                {
                    "iteration": i + 1,
                    "objective_value": objective_value,
                    "compute_time": routing_monitor.iteration_compute_times[i]
                }
                for i, objective_value in enumerate(routing_monitor.objective_values)
            ]
        return self._write_summaries(instance, assignments, search_summaries)

    def _write_summaries(self, instance, assignments, search_summaries=None):
        """
        :param instance: problem instance as dictionary object
        :param assignments: assignments of related to all the requests arrived so-far
        :param search_summaries: optional parameter that includes search summaries which includes
        cost/reward at each iteration and time consumed to perform each iteration
        """
        import math
        from common.general import get_df_from_dict, create_dir

        assignments_df = get_df_from_dict(assignments)

        if self.config["write_individual_summaries"]:
            output_dir = instance["output_dir"]
            create_dir(output_dir)

            operation_mode = instance["mode"].value
            assignments_df.to_csv(f"{output_dir}/assignments_{operation_mode}.csv", index=False)
            if search_summaries and len(search_summaries) > 0:
                search_summaries = get_df_from_dict(search_summaries)
                search_summaries["iteration"] = search_summaries.apply(
                    lambda row: math.ceil(row.iteration / 100) * 100, axis=1
                )
                search_summaries_sum = search_summaries.groupby(search_summaries.iteration).sum().reset_index()
                search_summaries_sum = search_summaries_sum.round(decimals=3)
                search_summaries_sum.to_csv(f"{output_dir}/search_summary_{operation_mode}.csv", index=False)
                search_summaries_mean = search_summaries.groupby(search_summaries.iteration).mean().reset_index()
                search_summaries_mean = search_summaries_mean.round(decimals=3)
                search_summaries_mean.to_csv(f"{output_dir}/search_summary_mean_{operation_mode}.csv", index=False)
        return assignments_df

    def _generate_params(self):
        """
        :return: RoutingSearchParameters object with custom values
        """
        from ortools.constraint_solver import pywrapcp

        search_parameters = pywrapcp.DefaultRoutingSearchParameters()
        search_parameters.first_solution_strategy = self.config["first_solution_strategy"]
        search_parameters.local_search_metaheuristic = self.config["meta_heuristic"]
        search_parameters.use_full_propagation = self.config["use_full_propagation"]
        if self.config['solution_constrained']:
            search_parameters.solution_limit = self.config["solution_limit"]
        else:
            time_limit = self.config['search_duration'] if 'search_duration' in self.config else 0
            search_parameters.time_limit.seconds = int(time_limit)
        search_parameters.log_search = self.config["log_search"]
        return search_parameters

    def _error_msg(self, status):
        from ortools.constraint_solver import pywrapcp
        if status in [pywrapcp.RoutingModel.ROUTING_SUCCESS, pywrapcp.RoutingModel.ROUTING_OPTIMAL]:
            return ""

        time_limit = self.config['search_duration'] if 'search_duration' in self.config else 0
        error_codes = {
            pywrapcp.RoutingModel.ROUTING_NOT_SOLVED: "Problem not solved yet",
            pywrapcp.RoutingModel.ROUTING_SUCCESS: "Problem solved successfully",
            pywrapcp.RoutingModel.ROUTING_OPTIMAL: "Problem has been solved to optimality",
            pywrapcp.RoutingModel.ROUTING_PARTIAL_SUCCESS_LOCAL_OPTIMUM_NOT_REACHED:
                "Problem solved successfully after calling RoutingModel.Solve(), "
                "except that a local optimum has not been reached; "
                "Leaving more time would allow improving the solution",
            pywrapcp.RoutingModel.ROUTING_FAIL: "No solution found to the problem",
            pywrapcp.RoutingModel.ROUTING_FAIL_TIMEOUT: "Time limit reached before finding a solution;"
                                                        f"current time limit {time_limit} seconds ,"
                                                        "increase the search duration and try",
            pywrapcp.RoutingModel.ROUTING_INVALID: "model, model parameters, or flags are not valid",
            pywrapcp.RoutingModel.ROUTING_INFEASIBLE: "Problem proven to be infeasible"
        }
        return f"ISSUE IN COMPUTING ROUTES, REASON: {error_codes[status].upper()}"

    @measure_time
    def insert(
            self,
            state=None,
            approach=InsertionHeuristics.ESI.value,
            solution_limit=1,
            max_solution_limit=1
    ):
        if state is None:
            state = self.environment.current_state
        self.update_state(state, is_bulk=False)
        self._perform_insertion(
            state,
            self.environment.current_instance,
            self.environment.current_response,
            approach,
            solution_limit,
            max_solution_limit
        )
        self.environment.log_remain(state.current_time)

    def rh_insert(self, state=None):
        # use rolling horizon as insertion heuristics
        if state is None:
            state = self.environment.current_state
        self.update_state_rh(state, is_bulk=False)
        resp = self.rh_solve(self.environment.current_instance)
        if resp["success"]:
            self.environment.current_state.current_requests = self.environment.current_instance["requests"]
            self.environment.current_state.accepted_requests += 1
            self.environment.last_insertion_success = True

    @measure_time
    def offline_insert(
            self,
            state=None,
            approach=InsertionHeuristics.ESI.value,
            solution_limit=1,
            max_solution_limit=1
    ):
        if state is None:
            state = self.environment.current_state
        prev_config_allow_dropping = self.config['allow_dropping']
        self.update_state(state, is_bulk=True)
        self.environment.current_instance["insertion_request_idx"] = len(state.new_request) * -1
        self._perform_insertion(
            state,
            self.environment.current_instance,
            self.environment.current_response,
            approach,
            solution_limit,
            max_solution_limit
        )
        self.config['allow_dropping'] = prev_config_allow_dropping

    @measure_time
    def search(
            self,
            state=None,
            search_approach=MetaHeuristics.SimulatedAnnealing.value,
            fixed_search=False,
            fixed_search_duration=5,
            use_target=False,
            change_environment=True
    ):
        from common.general import convert_sec_to_hh_mm_ss
        from common.logger import logger
        from common.types import MetaHeuristics

        if state is None:
            state = self.environment.current_state

        logger.info(f"SEARCH OPERATION AT TIME {convert_sec_to_hh_mm_ss(state.current_time)}")
        config = self._get_search_config(state, change_environment)

        prev_dropping_config = self.config['allow_dropping']

        solver_config = copy.deepcopy(self.config)
        solver_config["solution_constrained"] = False
        solver_config["allow_dropping"] = False  # during search no request cannot be removed
        if len(state.current_requests) > 0:
            max_search_duration = state.current_requests[-1].max_search_duration
            if fixed_search:
                max_search_duration = min(max_search_duration, fixed_search_duration)

            solver_config["search_duration"] = max_search_duration
            solver_config["max_search_duration"] = max_search_duration

        self.config = copy.deepcopy(solver_config)
        if state.value is not None:
            config["current_cost"] = -1 * state.value
        if search_approach == MetaHeuristics.RoutingAPI.value:
            resp = self.solve(config)
        elif search_approach == MetaHeuristics.SimulatedAnnealing.value:
            config["use_target"] = use_target
            config["state"] = state
            resp = self.simulated_annealing(config)
        else:
            raise ValueError(f"UNKNOWN SEARCH APPROACH {search_approach}")

        solver_config["allow_dropping"] = prev_dropping_config
        self.config = copy.deepcopy(solver_config)

        if resp["success"]:
            if change_environment:
                self.environment.current_routing_instance = None
                if "instance" in resp:
                    self.environment.current_routing_instance = resp["instance"]
                self.environment.current_solution = resp["computed_solution"]
                if resp["start_cost"] is not None and resp["end_cost"] is not None:
                    self.environment.search_summary.add_entry(
                        start_cost=resp["start_cost"],
                        end_cost=resp["end_cost"],
                        search_duration=solver_config["search_duration"],
                        operation_stats=resp["operations"] if "operations" in resp else []
                    )

            state.process_action(resp)

        if change_environment:
            self.environment.log_remain(state.current_time)
        return resp["final_action"] if 'final_action' in resp else {}, state

    def write_stats(self, state=None, execution_mode=ExecutionModes.Eval.value, model_idx=-1):
        """
        :param state: state of the environment
        :param execution_mode: whether to the stats is written in train mode or not
        :param model_idx: model index (only applicable for RL based approach)
        write overall summaries into list of CSV files
        """

        from common.general import create_dir
        from common.types import ExecutionModes, ObjectiveTypes
        if state is None:
            state = self.environment.current_state

        state.log_summary()
        if not self.environment.has_more_requests():
            if len(self.environment.requests) > 0:
                state.current_time = self.config["day_max_time"]
                self._update()  # update the solution
                self.environment.current_solution.summarize()  # compute the summaries

                # only add summary for the problem instance with at-least one request
                summary_dict = copy.deepcopy(state.instance_info)
                summary_dict.update(copy.deepcopy(state.service_summary()))
                summary_dict.update(copy.deepcopy(self.environment.search_summary.summary()))

                readable_summary_dict = copy.deepcopy(summary_dict)
                summary_dict.update(copy.deepcopy(self.environment.current_solution.get_summary()))
                readable_summary_dict.update(copy.deepcopy(self.environment.current_solution.get_readable_summary()))

                self.environment.summaries.append(summary_dict)
                self.environment.readable_summaries.append(readable_summary_dict)

                summary_dir = f"{self.environment.output_dir}_SUMMARY"
                if str(model_idx) != "-1":
                    if (execution_mode == ExecutionModes.Eval.value and
                            self.config["objective"] != ObjectiveTypes.CustomObjectiveByRL.value):
                        summary_dir = (f"{self.environment.output_dir}_SUMMARY/"
                                       f"{state.instance_info['sample_idx']}")
                    else:
                        summary_dir = (f"{self.environment.output_dir}_SUMMARY/{model_idx}/"
                                       f"{state.instance_info['sample_idx']}")

                statistics_file = f"{summary_dir}/statistics.csv"
                readable_statistics_file = f"{summary_dir}/readable_statistics.csv"
                improvement_file = f"{summary_dir}/improvement_summary.csv"
                compute_times_file = f"{summary_dir}/compute_times.csv"

                from common.general import get_df_from_dict

                create_dir(summary_dir)
                get_df_from_dict(self.environment.summaries).to_csv(statistics_file, index=False)
                get_df_from_dict(self.environment.readable_summaries).to_csv(readable_statistics_file, index=False)
                improvement_df = get_df_from_dict(self.environment.search_summary.improvement_statistics())
                if len(improvement_df) > 0:
                    improvement_df.to_csv(improvement_file, index=False)

                if self.environment.store_compute_times:
                    compute_times_dicts = []
                    for function_name, compute_time in self.environment.compute_times:
                        compute_times_dicts.append(
                            {
                                "function_name": function_name,
                                "compute_time": compute_time
                            }
                        )

                    get_df_from_dict(compute_times_dicts).to_csv(compute_times_file, index=False)

            # reset summaries
            self.environment.clear()

    def _perform_insertion(
            self,
            state,
            config,
            update_response,
            approach,
            solution_limit=1,
            max_solution_limit=1,
    ):
        from common.types import InsertionModes

        from ortools.constraint_solver import routing_enums_pb2

        from common.logger import logger
        from common.types import InsertionHeuristics
        from env.data.TimeMatrix import MatrixManager
        if approach == InsertionHeuristics.RoutingAPI.value:
            # update the solver configuration to perform only one insertion solution
            solver_config = copy.deepcopy(self.config)
            solver_config["solution_constrained"] = True
            solver_config["solution_limit"] = solution_limit
            solver_config["max_solution_limit"] = max_solution_limit
            solver_config["first_solution_strategy"] = routing_enums_pb2.FirstSolutionStrategy.AUTOMATIC
            if config["insert_mode"] == InsertionModes.SINGLE:
                # make it custom
                lcci = routing_enums_pb2.FirstSolutionStrategy.LOCAL_CHEAPEST_COST_INSERTION
                solver_config["first_solution_strategy"] = lcci

            config["verify"] = True
            self.config = copy.deepcopy(solver_config)
            resp = self.solve(config)
        elif approach == InsertionHeuristics.ESIVerified.value:
            config["verify"] = True
            if config["insert_mode"] == InsertionModes.SINGLE:
                resp = self.exhaustive_search_insert(config)
            else:
                resp = self.exhaustive_search_bulk_insert(config)
        elif approach == InsertionHeuristics.ESI.value:
            # simplified greedy with only online support (no-longer support google or-tools)
            config["verify"] = False
            if config["insert_mode"] == InsertionModes.SINGLE:
                resp = self.exhaustive_search_insert(config)
            else:
                resp = self.exhaustive_search_bulk_insert(config)
        else:
            raise ValueError(f"INVALID INSERTION APPROACH {approach}")

        # always reset routing instance and add it if it actually presents for current state
        self.environment.current_routing_instance = None
        if resp["success"]:
            if "instance" in resp.keys():
                self.environment.current_routing_instance = resp["instance"]
            self.environment.last_insertion_success = True
            self.environment.current_solution = resp["computed_solution"]

            state.process_action(resp)
            if update_response["new_request_size"] == 1:
                if len(resp["dropped_request_ids"]) == 0:
                    state.current_requests = update_response["all_requests"]
                    state.accepted_requests += 1
            else:
                newly_added_requests = [
                    request for request in update_response["all_requests"]
                    if request.request_id in resp["assigned_request_ids"]
                ]
                state.current_requests = newly_added_requests
                state.accepted_requests += len(newly_added_requests)

            self.environment.last_save_state = update_response
            MatrixManager.instance().update()
            logger.info("SUCCESSFULLY INSERTED THE NEW REQUEST")
        else:
            self.environment.last_insertion_success = False
            self.environment.last_save_state = None
            logger.error("UNABLE TO INSERT THE NEW REQUEST")
        state.log_summary()
        MatrixManager.instance().reset_temp()
        return resp

    @measure_time
    def update_state(self, state=None, is_bulk=False):
        from common.general import convert_sec_to_hh_mm_ss
        from common.logger import logger
        from common.types import SolvingModes, InsertionModes

        if state is None:
            state = self.environment.current_state

        if is_bulk:
            new_request = self.environment.get_remainder()
            state.current_time = new_request[0].arrival_time
            state.new_request = new_request
        else:
            new_request = self.environment.get_next()
            state.current_time = new_request.arrival_time
            state.new_request = new_request

        logger.info(f"INSERT OPERATION AT TIME {convert_sec_to_hh_mm_ss(state.current_time)}")
        current_instance, updated_response = self._get_insertion_config(state, new_request)
        current_instance["mode"] = SolvingModes.BULK_INSERT if is_bulk else SolvingModes.INSERT
        current_instance["insert_mode"] = InsertionModes.BULK if is_bulk else InsertionModes.SINGLE
        current_instance["allow_dropping"] = True  # if is for performing insertion via dropping
        current_instance["insertion_request_idx"] = len(new_request) * -1 if is_bulk else -1
        current_instance["verify"] = False  # verify against google or-tools
        current_instance["state"] = state
        self.environment.current_instance = current_instance
        self.environment.current_response = updated_response
        return current_instance

    @measure_time
    def update_state_rh(self, state=None, is_bulk=False):
        from common.general import convert_sec_to_hh_mm_ss
        from common.logger import logger
        from common.types import SolvingModes, InsertionModes

        if state is None:
            state = self.environment.current_state

        new_request = self.environment.get_next()
        state.current_time = 0
        state.new_request = new_request

        logger.info(f"INSERT OPERATION AT TIME {convert_sec_to_hh_mm_ss(state.current_time)}")
        current_instance, updated_response = self._get_insertion_config(state, new_request)
        current_instance["mode"] = SolvingModes.BULK_INSERT if is_bulk else SolvingModes.INSERT
        current_instance["insert_mode"] = InsertionModes.BULK if is_bulk else InsertionModes.SINGLE
        current_instance["allow_dropping"] = True  # if is for performing insertion via dropping
        current_instance["insertion_request_idx"] = len(new_request) * -1 if is_bulk else -1
        current_instance["verify"] = False  # verify against google or-tools
        current_instance["state"] = state
        self.environment.current_instance = current_instance
        self.environment.current_response = updated_response
        return current_instance

    @measure_time
    def update_state_train(self, state=None, new_request=None):
        from common.types import SolvingModes, InsertionModes

        state.current_time = new_request.arrival_time
        state.new_request = new_request
        if state.accepted_requests == 0:
            state.current_time -= self.config["depot_location"].duration(new_request.origin)

        current_instance, updated_response = self._get_insertion_config(state, new_request)
        current_instance["mode"] = SolvingModes.INSERT
        current_instance["insert_mode"] = InsertionModes.SINGLE
        current_instance["allow_dropping"] = True  # if is for performing insertion via dropping
        current_instance["insertion_request_idx"] = -1
        current_instance["verify"] = False  # verify against google or-tools
        current_instance["state"] = state
        return current_instance

    def _get_insertion_config(self, state, new_request, change_environment=True):
        raise NotImplementedError

    def _get_search_config(self, state, change_environment=True):
        raise NotImplementedError

    def _update(self, state=None, new_request=None, change_environment=True):
        raise NotImplementedError
