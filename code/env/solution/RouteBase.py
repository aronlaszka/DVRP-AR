from common.types import TimeUnits


class RouteBase:

    def minimal(self):
        """
        generate minimal representation of routes that could be saved as experiences and later used to evaluate
        """
        if isinstance(self, MinimalRoute):
            return self
        if hasattr(self, 'start_pos') and hasattr(self, 'end_pos') and hasattr(self, 'nodes') and \
                hasattr(self, 'start_time') and hasattr(self, 'original_end_time') and hasattr(self, 'times'):
            locations = [self.start_pos] + [node.loc for node in self.nodes] + [self.end_pos]
            times = [self.start_time] + self.times + [self.original_end_time]
            dwell_times = [self._get_dwell_time(loc) for loc in locations]
            drive_times = [loc.duration(locations[i + 1]) for i, loc in enumerate(locations[:-1])]
            return MinimalRoute(times, drive_times, dwell_times, [loc.get_raw() for loc in locations])
        raise ValueError("Unable to generate minimal")

    @staticmethod
    def _get_divide_factor(convert_to=TimeUnits.Hour.value):
        """
        :param convert_to: whether to convert the wait-time from seconds to specific unit
        """
        match convert_to:
            case TimeUnits.Hour.value:  # hours
                divide_factor = 3600
            case TimeUnits.HectoSecond.value:  # hecto-second
                divide_factor = 100
            case TimeUnits.DecaSecond.value:  # deca-second
                divide_factor = 10
            case TimeUnits.Minute.value:  # minutes
                divide_factor = 60
            case _:
                divide_factor = 1
        return divide_factor

    @staticmethod
    def _get_dwell_time(loc):
        """
        :param loc: Location Instance
        :return: sum of total possible dwell time at the given for location (if the location represents either
        middle of pickup or dropoff then the dwell time is computed based on the captured dwell time)
        """
        from env.data.Location import LocationTypes
        from env.solution.RouteNode import RouteNode
        if isinstance(loc, RouteNode):
            loc = loc.loc
        dwell_time = 0
        if loc.loc_type in [LocationTypes.PICKUP, LocationTypes.DROPOFF]:
            dwell_time += loc.dwell_time
        elif loc.loc_type in [LocationTypes.PICKUP_MIDDLE, LocationTypes.DROPOFF_MIDDLE]:
            dwell_time += loc.captured_dwell_time
        return dwell_time

    @staticmethod
    def _get_adjusted_times(minimal, current_time):
        """
        :param minimal: minimal representation of routes that could be saved as experience
        :param current_time: current time
        :return: adjusted times (where the first value is set to zero)
        """
        times = minimal.times
        dwell_times = minimal.dwell_times

        time_values = (
                [time_value - dwell_times[k] for k, time_value in enumerate(times[:-1])] +
                [times[-1]]
        )

        # apply maximization to avoid some corner case error where interpolation need to shift
        # start-time earlier than current time (more perfect time matrix will not let this happen)
        return [
            # else condition when you have shift starting at different time (in future)
            # generally time_values[0] == current_time, and first value should be 0
            max(time_val - current_time, 0) if time_values[0] <= current_time else time_val - current_time
            for time_val in time_values
        ]


    @staticmethod
    def _get_idle_and_drive_times(minimal, horizon, adjusted_curr_time, adjusted_next_curr_time , i):
        """
        :param minimal: minimal representation of routes that could be saved as experience
        :param horizon: look_ahead horizon for considering the wait time
        :param adjusted_curr_time: zero-offset time value for i-th node
        :param adjusted_next_curr_time: zero-offset time value for (i+1)-th node
        :param i: current index
        :return: idle time between node[i] and node[i + 1], and drive time between node[i] and node[i + 1]
        """
        drive_times = minimal.drive_times
        dwell_times = minimal.dwell_times

        total_travel_time = drive_times[i] + dwell_times[i]

        # if the next time point is beyond the horizon, then we need to compute the partial idle time
        if adjusted_next_curr_time > horizon:
            available_additional = adjusted_next_curr_time - horizon

            # if the travel time between node[i] and node[i + 1] is greater than available additional time
            # then leave the traveling part at the end
            if available_additional > total_travel_time:
                idle_time = horizon - adjusted_curr_time
                drive_time = 0
            else:
                # otherwise deduct the traveling part required from total idle time
                idle_time = horizon - adjusted_curr_time - (total_travel_time - available_additional)
                drive_time = total_travel_time - available_additional
        else:
            # this is base case when the next time point is within the horizon
            idle_time = adjusted_next_curr_time - adjusted_curr_time - total_travel_time
            drive_time = total_travel_time
        return idle_time, drive_time

    def get_wait_time(self, current_time, horizon=14400, threshold=0, addition=-1, consider_threshold=False):
        """
        :param current_time: current time of the system
        :param horizon: look_ahead horizon for considering the wait time
        :param threshold: threshold for wait-time to be considered in the summation
        :param addition: correction added for wait-time when subtracting the wait-time beyond the threshold
        :param consider_threshold: whether to consider the threshold or not
        :return: sum of wait-times for the Route Instance of InsertStat Instance
        """
        minimal = self.minimal()
        if addition < 0:
            addition = threshold

        if not consider_threshold:
            threshold = 0
            addition = 0

        change = addition - threshold

        adjusted_time_values = self._get_adjusted_times(minimal, current_time)

        total_idle_time = 0
        for i, adjusted_curr_time in enumerate(adjusted_time_values[:-1]):
            if adjusted_curr_time >= horizon:
                break
            idle_time, _ = self._get_idle_and_drive_times(
                minimal, horizon, adjusted_time_values[i], adjusted_time_values[i + 1], i
            )
            if idle_time > threshold:
                total_idle_time += idle_time + change

        return total_idle_time

    def get_availabilities(self, current_time, horizon=14400, window_size=300, allow_fraction=True):
        """
        :param current_time: current time of the system
        :param horizon: look ahead horizon for considering the wait time (in seconds)
        :param window_size: size of each disjoint windows that divides the entire look ahead horizon (in seconds)
        :param allow_fraction: whether to consider each slot is fully available for the vehicle or not
        :return: list of availabilities where the size of the list equal to horizon/window_size
        """
        import math
        max_ptr = math.ceil(horizon / window_size)
        availabilities = [0.0 for _ in range(max_ptr)]

        minimal = self.minimal()
        adjusted_time_values = self._get_adjusted_times(minimal, current_time)

        total_idle_time = 0
        availability_sum = 0
        for i, adjusted_curr_time in enumerate(adjusted_time_values[:-1]):
            if adjusted_curr_time >= horizon:
                break

            idle_time, drive_time = self._get_idle_and_drive_times(
                minimal, horizon, adjusted_time_values[i], adjusted_time_values[i + 1], i
            )

            adjusted_next_curr_time = min(adjusted_time_values[i + 1], horizon)

            # only if there is an idle time
            if idle_time > 0:
                total_idle_time += idle_time
                deduct_idle_time = idle_time
                start_of_free_time = adjusted_curr_time + drive_time
                # assert start_of_free_time <= adjusted_next_curr_time
                start_idx = math.ceil(start_of_free_time / window_size)
                start_time_adjust = start_idx * window_size
                if start_time_adjust <= adjusted_next_curr_time:
                    prev_remainder = start_time_adjust - start_of_free_time
                    end_idx = int(adjusted_next_curr_time / window_size)
                    next_remainder = adjusted_next_curr_time - end_idx * window_size
                    for k in range(start_idx, min(end_idx, max_ptr)):
                        # assert availabilities[k] == 0.0
                        availabilities[k] = 1.0
                        availability_sum += 1.0
                        deduct_idle_time -= window_size
                    if 1 <= start_idx <= max_ptr:
                        availabilities[start_idx - 1] += prev_remainder / window_size
                        availability_sum += prev_remainder / window_size
                        deduct_idle_time -= prev_remainder
                    if 0 <= end_idx <= max_ptr - 1:
                        availabilities[end_idx] += next_remainder / window_size
                        availability_sum += next_remainder / window_size
                        deduct_idle_time -= next_remainder

                else:
                    if 1 <= start_idx <= max_ptr:
                        availabilities[start_idx - 1] += idle_time / window_size
                        availability_sum += idle_time / window_size
                        deduct_idle_time -= idle_time

        # if availability_sum > 0:
        #     # only if the availabilities are greater than zero
        #     assert math.floor(total_idle_time / availability_sum) <= window_size

        if not allow_fraction:
            for k, avail in enumerate(availabilities):
                if avail < 1:
                    availabilities[k] = 0.0
        return availabilities

    def get_approx_positions(self, current_time, horizon=14400, window_size=300, grid_size=-1):
        """
        :param current_time: current time of the system
        :param horizon: look ahead horizon for considering the wait time (in seconds)
        :param window_size: size of each disjoint windows that divides the entire look ahead horizon (in seconds)
        :param grid_size: size of each grid
        :return: list of positions when the vehicle serving the route is available where the
         size of the list equal to horizon/window_size
        """
        import math
        from env.data.Location import Location
        max_ptr = math.ceil(horizon / window_size)
        approx_locations = [None for _ in range(max_ptr)]

        minimal = self.minimal()
        locations = minimal.locations

        adjusted_time_values = self._get_adjusted_times(minimal, current_time)

        for i, adjusted_curr_time in enumerate(adjusted_time_values[:-1]):
            if adjusted_curr_time >= horizon:
                break

            idle_time, drive_time = self._get_idle_and_drive_times(
                minimal, horizon, adjusted_time_values[i], adjusted_time_values[i + 1], i
            )

            adjusted_next_curr_time = min(adjusted_time_values[i + 1], horizon)

            # generally the vehicle move to next location, except for the last node (in case of infinite horizon)
            next_location = locations[i + 1] if drive_time > 0 else locations[i]

            # only if there is an idle time
            if idle_time > 0:
                start_of_free_time = adjusted_curr_time + drive_time
                # assert start_of_free_time <= adjusted_next_curr_time
                start_idx = math.ceil(start_of_free_time / window_size)
                start_time_adjust = start_idx * window_size
                if start_time_adjust <= adjusted_next_curr_time:
                    end_idx = int(adjusted_next_curr_time / window_size)
                    next_location_inst = Location(next_location[0], next_location[1])
                    grid_pos = next_location_inst.get_grid_position(grid_size=grid_size)
                    for k in range(start_idx, min(end_idx, max_ptr)):
                        approx_locations[k] = grid_pos
        return approx_locations

    def get_sum_of_dwell_times(self):
        minimal = self.minimal()
        return sum(minimal.dwell_times)


# since experience are based on minimal route, the implementation is left
# as it is for minimal route
class MinimalRoute(RouteBase):
    def __init__(self, times, drive_times, dwell_times, locations=None):
        self.times = times
        self.drive_times = drive_times
        self.dwell_times = dwell_times
        self.locations = locations
