from learn.util_func import convert_to_numeral
from model.base.KerasBaseModel import KerasBaseModel


def get_k_consecutive_minimum(availabilities, k):
    """
    :param availabilities: list of the number of availabilities
    :param k:  of consecutive numbers to be considered for finding minimum
    :return: list of minimum of consecutive k-numbers
    """
    from collections import deque
    result = []
    minimums = deque()  # will store indices of array elements

    for i in range(len(availabilities)):
        # remove elements not within the window
        if minimums and minimums[0] < i - k + 1:
            minimums.popleft()

        # remove elements larger than or equal to the current element
        while minimums and availabilities[minimums[-1]] >= availabilities[i]:
            minimums.pop()

        # add the current element index
        minimums.append(i)

        # append the minimum of the current window to the result
        if i >= k - 1:
            result.append(availabilities[minimums[0]])
    return result


class MasterModel(KerasBaseModel):
    def _get_total_idle_time(self, raw_input):
        """
        :param raw_input: single experience represent in the dictionary format, where the key 'route' contains
        the list of route plans representing all the routes at the current time, and the key 'current_time' represents
        the current time of the system

        compute the total idle time for all the route plans from the current time to current time + look_ahead_horizon
        """
        return sum(
            [
                route.get_wait_time(
                    current_time=raw_input['current_time'],
                    horizon=self._model_controls["look_ahead_horizon"],
                    threshold=0,
                    addition=0,
                    consider_threshold=False
                )
                for route in raw_input["routes"]
            ]
        ) * self.get_norm("wait_time")

    def _get_availabilities(self, raw_input, **kwargs):
        """
        :param raw_input: single experience represents in the dictionary format, where the key 'route' contains
        the list of route plans representing all the routes at the current time, and the key 'current_time' represents
        the current time of the system

        generate the list of availability where each entry represents the number of / fraction of vehicles
        that are idle in the specific time slot, where each slot has a fixed duration of
        `self._model_controls["look_ahead_window_size"]`
        """
        return [
            route.get_availabilities(
                current_time=raw_input["current_time"],
                horizon=self._model_controls["look_ahead_horizon"],
                window_size=self._model_controls["look_ahead_window_size"],
                allow_fraction=kwargs["allow_fraction"]
            )
            for route in raw_input["routes"]
        ]

    def _get_slot_level_availabilities(self, raw_input):
        """
        :param raw_input: single experience represents in the dictionary format, where the key 'route' contains
        the list of route plans representing all the routes at the current time, and the key 'current_time' represents
        the current time of the system

        generate the list of availability where each entry represents the number of / fraction of vehicles
        that are idle in the specific time slot, where each slot has a fixed duration of
        `self._model_controls["look_ahead_window_size"]`

        and transform into meaningful feature that compute slot level availabilities
        """
        import keras

        raw_availabilities = self._get_availabilities(raw_input, allow_fraction=False)
        number_of_routes_to_consider = self._model_controls["number_of_routes_to_consider"]

        route_availabilities = convert_to_numeral(
            keras.ops.clip(
                keras.ops.sum(self.tensor(raw_availabilities), axis=0), 0.0, number_of_routes_to_consider * 1.0
            ), dtype=int
        )
        return self.__get_slot_level_availabilities_internal(route_availabilities)

    def __get_slot_level_availabilities_internal(self, route_availabilities):
        """
        :param route_availabilities: list of availability where each entry represents
        the number of / fraction of vehicles that are idle in the specific time slot,
        where each slot has a fixed duration of `self._model_controls["look_ahead_window_size"]`

        and transform into meaningful feature that compute slot level availabilities
        """
        import numpy as np

        number_of_routes_to_consider = self._model_controls["number_of_routes_to_consider"]
        look_ahead_slot_count = self._model_controls["look_ahead_slot_count"]
        look_ahead_slot_step_size = self._model_controls["look_ahead_slot_step_size"]

        values = np.zeros((look_ahead_slot_count, number_of_routes_to_consider))

        if look_ahead_slot_step_size == 1:
            for x, min_value in enumerate(route_availabilities):
                curr_min_value = min_value
                if curr_min_value > 0:
                    # when considering only one slot
                    values[0, :curr_min_value] += 1.0
                else:
                    # continue to the next slot
                    continue

                # remainder of the array that needs to be processed
                remainder_array = route_availabilities[x + 1:x + look_ahead_slot_count]
                for idx, next_value in enumerate(remainder_array):
                    # compute the minimum from the position x till x + next_index
                    curr_min_value = min(next_value, curr_min_value)
                    if curr_min_value == 0:
                        break

                    values[idx + 1, :curr_min_value] += 1.0
        else:
            # little bit efficient code
            minimum_availabilities = get_k_consecutive_minimum(route_availabilities, look_ahead_slot_step_size)
            for x, min_value in enumerate(minimum_availabilities):
                curr_min_value = min_value
                if curr_min_value == 0:
                    # continue to the next slot
                    continue

                ptr = 0
                next_idx = x + look_ahead_slot_step_size
                max_position = min(x + look_ahead_slot_count * look_ahead_slot_step_size, len(minimum_availabilities))
                while next_idx < max_position:
                    # compute the minimum from the position x till x + next_index
                    next_value = minimum_availabilities[next_idx]
                    curr_min_value = min(next_value, curr_min_value)
                    if curr_min_value == 0:
                        break

                    # only consider features at step_size, 2 * step_size, ..., (T - 1) * step_size
                    # when considering slots i - T, i - T + 1, ... , i
                    values[ptr, :curr_min_value] += 1.0
                    next_idx += look_ahead_slot_step_size
                    ptr += 1

        return values

    def _get_slot_level_positions(self, raw_input):
        """
        :param raw_input: single experience represents in the dictionary format, where the key 'route' contains
        the list of route plans representing all the routes at the current time, and the key 'current_time' represents
        the current time of the system

        generate the list of positions where each entry represents the approximate position of the vehicles
        during in the specific time slot, where each slot has a fixed duration of
        `self._model_controls["look_ahead_window_size"]`
        """
        return [
            route.get_approx_positions(
                current_time=raw_input["current_time"],
                horizon=self._model_controls["look_ahead_horizon"],
                window_size=self._model_controls["look_ahead_window_size"],
                grid_size=self._model_controls["look_ahead_grid_size"],
            )
            for route in raw_input["routes"]
        ]

    def _get_slot_level_positions_counts(self, raw_input):
        """
        :param raw_input: single experience represent in the dictionary format, where the key 'route' contains
        the list of route plans representing all the routes at the current time, and the key 'current_time' represents
        the current time of the system

        generate the list of positions where each entry represents the approximate position of the vehicles
        during in the specific time slot, where each slot has a fixed duration of
        `self._model_controls["look_ahead_window_size"]`

        convert the raw positions to grid positions and compute the count
        """
        from env.data.Location import Location
        approx_positions = self._get_slot_level_positions(raw_input)

        time_steps = len(approx_positions[0])
        grid_size = self._model_controls.get('look_ahead_grid_size', 2)  # set the default grid size to 2
        counts = [[0 for _ in range(grid_size * grid_size)] for _ in range(time_steps)]
        for approx_position_for_route in approx_positions:
            for t, pos in enumerate(approx_position_for_route):
                if pos and isinstance(pos, Location):
                    x, y = pos.get_grid_position(grid_size=grid_size)
                    counts[t][x * grid_size + y] += 1
        return counts

    def _get_grid_specific_slot_level_availabilities(self, raw_input):
        """
        :param raw_input: single experience represent in the dictionary format, where the key 'route' contains
        the list of route plans representing all the routes at the current time, and the key 'current_time' represents
        the current time of the system

        generate the 2D-list of availability where each entry in low-dimension list represents the number of /
        fraction of vehicles that are idle in the specific time slot, where each slot has a fixed duration of
        `self._model_controls["look_ahead_window_size"]`, where the upper dimension used to identify the different grid
        positions where the grid size is defined by `self._model_controls["look_ahead_grid_size"]`

        For an example: For grid size of 2 and total number slot of 4
        the availability can be expressed as (grid_size ** 2, number_of_slots) = (2 ** 2, 4) = (4, 4)

        and transform into meaningful feature that compute slot level availabilities
        """
        import numpy as np

        look_ahead_total_slots = self._model_controls["look_ahead_total_slots"]

        grid_size = self._model_controls["look_ahead_grid_size"]
        approx_grid_positions = self._get_slot_level_positions(raw_input)
        values = np.zeros((grid_size * grid_size, look_ahead_total_slots), dtype=np.int32)
        for route_positions in approx_grid_positions:
            for t, position in enumerate(route_positions):
                if position:
                    x, y = position
                    values[x * grid_size + y][t] += 1

        computed_values = []
        for row_value in values:
            computed_values.extend(self.__get_slot_level_availabilities_internal(row_value))
        return computed_values

    def _generate_fv(self, raw_input):
        """
        :param raw_input: single experience represents in the dictionary format, where the key 'route' contains
        the list of route plans representing all the routes at the current time, and the key 'current_time' represents
        the current time of the system

        :return feature vector based on the raw input and the feature codes specified in the self._feature_code
        """
        from common.types import FeatureCodes
        from learn.util_func import polynomial_feature
        feature_map = {code for code in self._feature_code.split("-")}

        fvs = []

        if FeatureCodes.CappedIdleTime.value in feature_map:
            fvs.append(self._get_total_idle_time(raw_input))

        if FeatureCodes.CappedAvailability.value in feature_map:
            grid_size = self._model_controls["look_ahead_grid_size"]
            if grid_size == 1:
                fv_matrix = self._get_slot_level_availabilities(raw_input)
                for fv_row in fv_matrix:
                    fvs.extend(fv_row / self._model_controls["look_ahead_total_slots"])
            else:
                fv_matrix = self._get_grid_specific_slot_level_availabilities(raw_input)
                for fv_row in fv_matrix:
                    fvs.extend(fv_row / self._model_controls["look_ahead_total_slots"])

        # for keras based
        if self._polynomial_dim > 1 and not hasattr(self, '_scikit_mode'):
            # generate the polynomial features
            fvs = polynomial_feature(fvs, self._polynomial_dim)

        return fvs

    def _generate_fv_parallel(self, raw_input):
        return self._generate_fv(raw_input=raw_input[0])

    def generate_feature_vectors(self, raw_inputs):
        """
        :param raw_inputs: list of raw inputs (where each input is represented as a dictionary format,
        with two keys: current_time - the current time of the experience, routes - route plans at the current time
        :return: list of feature vectors
                (with respect to each possible pair of state and action (for each action in actions))
        """
        if self._threads == 1:
            return self.tensor(
                [
                    self._generate_fv(raw_input=raw_input) for raw_input in raw_inputs
                ]
            )

        # perform multi-threading if self._threads > 1
        # due to performance inefficiencies currently setting it to 1
        from common.logger import logger
        from common.multi_proc import thread_pool_executor_wrapper
        args = [(raw_input,) for raw_input in raw_inputs]

        if len(args) > 1:
            logger.info(f"Executing in Multi-threaded mode, number of threads: {min(self._threads, len(args))}")

        return self.tensor(
            [
                fv for fv in
                thread_pool_executor_wrapper(
                    self._generate_fv_parallel,
                    args,
                    self._threads,
                    wait_for_response=True
                )
            ]
        )
