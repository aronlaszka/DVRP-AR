from common.general import measure_time
from common.types import ExecutionModes
from env.base.wrapper.Environment import Environment


class OnlineEnvironment(Environment):

    def __init__(self, objective, output_dir):
        super(OnlineEnvironment, self).__init__(objective, output_dir)
        from env.base.online.State import OnlineState
        self.state_class = OnlineState

    @measure_time
    def update(self, requests, routes, completed_routes, current_time, action_map, new_request=None):
        """
        :param requests: requests that are received upto now
        :param routes: routes generated based on the requests arrived so-far
        :param completed_routes: routes that are already completed
        :param current_time: current system time
        :param action_map: mapping of all pickup and dropoff action
        :param new_request: incoming request
        """
        from common.types import RequestActions
        from env.solution.RouteNode import generate_pick_up_node, generate_drop_off_node

        execution_mode = routes[0].execution_mode

        locations = []
        updated_routes = []
        updated_completed_routes = []
        for i, route in enumerate(routes):
            break_response = route.interpolate_routes(completed_routes[i], current_time)
            locations.extend(
                [break_response["current"]["route"].start_pos, break_response["current"]["route"].end_pos]
            )
            updated_routes.append(break_response["current"]["route"])
            updated_completed_routes.append(break_response["completed"]["route"])

        start_index = 2 * len(updated_routes)
        node_map = {}
        updated_requests = []
        node_conversion_map = {}
        completed_node_map = {}

        for request in requests:
            if request.pickup_idx in action_map.keys() or request.dropoff_idx in action_map.keys():
                pickup_entry = action_map[request.pickup_idx] if request.pickup_idx in action_map else None
                dropoff_entry = action_map[request.dropoff_idx] if request.dropoff_idx in action_map else None
                request = request.update(
                    pickup_entry=pickup_entry,
                    dropoff_entry=dropoff_entry,
                    fixed_timeahead=self.solver.config["fixed_timeahead"],
                    current_time=current_time
                )
                if request.is_served():
                    request.relative_pickup_idx = None
                    request.relative_dropoff_idx = None
                    pick_up_node = generate_pick_up_node(request, current_time=current_time)
                    drop_off_node = generate_drop_off_node(request, current_time=current_time)
                    completed_node_map[request.request_id, RequestActions.Pickup] = pick_up_node
                    completed_node_map[request.request_id, RequestActions.Dropoff] = drop_off_node
                else:
                    if request.is_picked():
                        node_conversion_map[request.request_id, RequestActions.Dropoff] = start_index
                        request.set_relative_index(start_index, RequestActions.Dropoff)
                        pick_up_node = generate_pick_up_node(request, current_time=current_time)
                        completed_node_map[request.request_id, RequestActions.Pickup] = pick_up_node
                        node_map[start_index] = generate_drop_off_node(request, current_time=current_time)
                        locations.append(request.destination)
                        start_index += 1
                    else:
                        node_conversion_map[request.request_id, RequestActions.Pickup] = start_index
                        node_conversion_map[request.request_id, RequestActions.Dropoff] = start_index + 1
                        request.set_relative_index(start_index, RequestActions.Pickup)
                        node_map[start_index] = generate_pick_up_node(request, current_time=current_time)
                        node_map[start_index + 1] = generate_drop_off_node(request, current_time=current_time)
                        locations.append(request.origin)
                        locations.append(request.destination)
                        start_index += 2

            if not request.is_served():
                updated_requests.append(request)

        if new_request:
            if not isinstance(new_request, list):
                new_request = [new_request]

            for new_req in new_request:
                new_req.set_relative_index(start_index, RequestActions.Pickup)
                node_map[start_index] = generate_pick_up_node(new_req, current_time=current_time)
                node_map[start_index + 1] = generate_drop_off_node(new_req, current_time=current_time)
                locations.append(new_req.origin)
                locations.append(new_req.destination)
                start_index += 2

        if not (len(node_map) + 2 * len(updated_routes) == len(locations)):
            raise AssertionError(
                f"Expected Total Locations {len(node_map) + 2 * len(updated_routes)}, "
                f"Actual Total Locations {len(locations)}"
            )

        updated_routes_fixed = []
        for updated_route in updated_routes:
            updated_nodes = []
            for node in updated_route.nodes:
                if (node.request_id, node.node_type) in node_conversion_map.keys():
                    request_id = node.request_id
                    node_type = node.node_type
                    updated_node = node_map[node_conversion_map[(node.request_id, node.node_type)]]
                    if not (updated_node.node_type == node_type):
                        raise AssertionError(
                            f"Expected Node Type {node_type}, Actual Node Type {updated_node.node_type}"
                        )
                    if not (updated_node.request_id == request_id):
                        raise AssertionError(
                            f"Expected Request ID {request_id}, Actual Request ID {updated_node.request_id}"
                        )
                    updated_nodes.append(updated_node)
                else:
                    raise ValueError(f"Current Node {node.__dict__} is missing from records !!!")

            if len(updated_nodes) > 0:
                updated_route = updated_route.add(updated_nodes, updated_route.times, verify=True)
            else:
                updated_route = updated_route.reset()
            updated_routes_fixed.append(updated_route)

        updated_routes = updated_routes_fixed

        if execution_mode != ExecutionModes.GatherExperience.value:

            updated_completed_routes_fixed = []
            for completed_route in completed_routes:
                completed_nodes = []
                for node in completed_route.nodes:
                    if (node.request_id, node.node_type) in completed_node_map.keys():
                        request_id = node.request_id
                        node_type = node.node_type
                        updated_node = completed_node_map[node.request_id, node.node_type]
                        if not (updated_node.node_type == node_type):
                            raise AssertionError(
                                f"Expected Node Type {node_type}, Actual Node Type {updated_node.node_type}"
                            )
                        if not (updated_node.request_id == request_id):
                            raise AssertionError(
                                f"Expected Request ID {request_id}, Actual Request ID {updated_node.request_id}"
                            )
                        completed_nodes.append(updated_node)
                    else:
                        completed_nodes.append(node)

                if len(completed_nodes) > 0:
                    completed_route = completed_route.add(completed_nodes, completed_route.times, verify=True)
                else:
                    completed_route = completed_route.reset()
                updated_completed_routes_fixed.append(completed_route)

            updated_completed_routes = updated_completed_routes_fixed

        all_requests = updated_requests
        all_requests += new_request if new_request else []
        new_request_size = len(new_request) if new_request else 0

        return {
            "requests": updated_requests,
            "routes": updated_routes,
            "locations": locations,
            "completed_routes": updated_completed_routes,
            "new_request": new_request,
            "new_request_size": new_request_size,
            "all_requests": all_requests,
            "node_map": node_map
        }

    def get_next(self, counter=0):
        """
        :param counter: optional parameter used in the get_next API
        :return: next request in the line
        """
        request = self.requests[self.current_state.current_idx]
        request = request.update_id(self.current_state.accepted_requests + counter, offset=self.get_offset())
        self.current_state.current_idx += 1
        return request

    def get_next_train(self, counter=0):
        """
        :param counter: optional parameter used in the get_next_train API
        :return: next request in the line
        (this call to get the request only if there is any request remains, and this will not update the request index)
        """
        request = self.requests[self.current_state.current_idx]
        request = request.update_id(self.current_state.accepted_requests + counter, offset=self.get_offset())
        return request

    def get_remainder(self):
        """
        :return: get all the remainder of requests
        """
        requests = []
        counter = 0
        while self.has_more_requests():
            requests.append(self.get_next(counter))
            counter += 1
        return requests

    def get_offset(self):
        return len(self.current_solution.routes)
