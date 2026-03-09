import copy


class ComprehensiveAction:
    def __init__(self, initial_routes, changed_routes=None, **kwargs):
        """
        :param initial_routes: initial manifest corresponding to each vehicles
        :param changed_routes: change in manifest once the action is applied
        :param kwargs: optional keyword arguments that passed overtime when capturing additional statistics
        """
        if changed_routes is None:
            changed_routes = {}

        self.changed_route_ids = set(changed_routes.keys())
        self.initial_routes = initial_routes
        self.changed_routes = changed_routes
        # routes_after could contain temporary statistics pertaining to changing the route
        self.routes_after = [
            changed_routes[route.route_id] if route.route_id in changed_routes else route
            for route in initial_routes
        ]
        assert [route.route_id for route in initial_routes] == [route.route_id for route in self.routes_after]
        self.__dict__.update(**kwargs)
        self.added_keys = set(kwargs.keys())

    def _get_value_and_cost(self, **kwargs):
        """
        :param kwargs: keyword arguments such as
            consider_threshold (used only when the objective is maximizing idle time)
            selected_route (used only when the objective is from basic route statistics)
        :return: dictionary objects that contains two keys "value" representing the value of the action described
        and "cost" representing the cost of the action (basically -value)
        """
        wait_time_threshold = kwargs["wait_time_threshold"] if "wait_time_threshold" in kwargs else 0
        cost = 0
        for route in self.initial_routes:
            if route.route_id in self.changed_route_ids:
                if isinstance(self.routes_after[route.route_id], type(route)):
                    sub_kwargs = copy.deepcopy(kwargs)
                    sub_kwargs['wait_time_threshold'] = wait_time_threshold
                    if 'action' in sub_kwargs.keys():
                        sub_kwargs.pop('action')
                    cost += self.routes_after[route.route_id].get_current_cost(**sub_kwargs)
                else:
                    sub_kwargs = copy.deepcopy(kwargs)
                    sub_kwargs['wait_time_threshold'] = wait_time_threshold
                    sub_kwargs['action'] = self.routes_after[route.route_id]
                    cost += route.get_current_cost(**sub_kwargs)
            else:
                sub_kwargs = copy.deepcopy(kwargs)
                sub_kwargs['wait_time_threshold'] = wait_time_threshold
                cost += route.get_current_cost(**sub_kwargs)
        return {
            "cost": cost,
            "value": -1 * cost
        }

    def get_value(self, **kwargs):
        """
        :param kwargs: keyword arguments such as
            consider_threshold (used only when the objective is maximizing idle time)
            selected_route (used only when the objective is from basic route statistics)
        :return: the objective cost
        """
        return self._get_value_and_cost(**kwargs)["value"]

    def get_cost(self, **kwargs):
        """
        :param kwargs: keyword arguments such as
            consider_threshold (used only when the objective is maximizing idle time)
            selected_route (used only when the objective is from basic route statistics)
        :return: the objective cost
        """
        return self._get_value_and_cost(**kwargs)["cost"]

    def clear(self):
        # clear added keys
        for key in self.added_keys:
            del self.__dict__[key]
        del self.added_keys
