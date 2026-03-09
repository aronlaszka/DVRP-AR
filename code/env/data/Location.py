import math
from enum import Enum


class LocationTypes(Enum):
    DEPOT = "Depot"
    PICKUP = "Pickup"
    PICKUP_MIDDLE = "MiddleOfPickup"
    DROPOFF = "Dropoff"
    DROPOFF_MIDDLE = "MiddleOfDropoff"
    INTERPOLATED = "Interpolated"
    OTHER = "Other"


LOCATION_PRECISION = 6


class Location:
    def __init__(
            self,
            latitude,
            longitude,
            dwell_time=0,
            loc_type=LocationTypes.DEPOT
    ):
        latitude = round(latitude, LOCATION_PRECISION)
        longitude = round(longitude, LOCATION_PRECISION)

        self.loc = (latitude, longitude)
        self.dwell_time = dwell_time
        # during interpolation, sometimes we can't estimate location from given set of the nodes
        # in the OpenStreetMap despite the travel time source in those cases we could use some adjustments
        self.time_adjustment = 0
        self.captured_dwell_time = 0
        self.latitude = latitude
        self.longitude = longitude
        self.loc_type = loc_type

    def as_location(self, latitude, longitude, dwell_time):
        return Location(
            latitude=latitude,
            longitude=longitude,
            dwell_time=dwell_time,
            loc_type=self.loc_type
        )

    def __str__(self):
        return f"{self.latitude, self.longitude}"

    def get_raw(self):
        """
        :return: row tuple of latitude and longitude
        """
        return self.latitude, self.longitude

    def get_euclidean(self, other):
        import math
        return math.sqrt((self.latitude - other.latitude) ** 2 + (self.longitude - other.longitude) ** 2)

    def interpolate(self, other, numerator, denominator):
        """
        :param other: destination location
        :param numerator: traveled journey as time at which the function invoked
        :param denominator: total journey from the origin (i.e., location described by self to location described
                            by other)
        :return: interpolated location (if we use OSRM the interpolated location is accurate enough, as it is from
        real route between location described by self and other)
        """
        from copy import deepcopy
        if not isinstance(other, Location):
            other = other.loc

        fraction = numerator / denominator
        assert 0.0 <= fraction <= 1.0, f"Fraction value must be within 0.0 and 1.0, {fraction}"
        fraction = round(fraction, LOCATION_PRECISION)
        if fraction == 0.0:
            return deepcopy(self)
        elif fraction == 1.0:
            return deepcopy(other)
        else:
            from common.arg_parser import get_parsed_args
            from common.types import TravelTimeSources
            args = get_parsed_args()
            if args.travel_time_source == TravelTimeSources.OSMnx.value:
                return self._route_based_interpolation(other, numerator)
            elif args.travel_time_source == TravelTimeSources.Euclidean.value:
                return self._linear_interpolation(other, numerator / denominator)
            else:
                raise ValueError(f"Interpolation not yet supported for {args.travel_time_source}")

    def _route_based_interpolation(self, other, completed_duration):
        from env.data.TimeMatrix import MatrixManager
        coordinates, durations = MatrixManager.instance().get_routes(self, other)
        redux_completed_duration = completed_duration
        time_adjustment = 0
        interpolated_loc_coord = None
        for i, duration in enumerate(durations):
            redux_completed_duration -= duration
            if redux_completed_duration == 0:
                interpolated_loc_coord = coordinates[i + 1]
                break
            elif redux_completed_duration < 0:
                interpolated_loc_coord = coordinates[i + 1]
                time_adjustment = abs(redux_completed_duration)
                break

        if interpolated_loc_coord is None:
            interpolated_loc_coord = coordinates[-1]

        loc = Location(
            latitude=interpolated_loc_coord[1],
            longitude=interpolated_loc_coord[0],
            dwell_time=0,
            loc_type=LocationTypes.INTERPOLATED
        )
        loc.time_adjustment = time_adjustment
        return loc

    def _linear_interpolation(self, other, fraction):
        interpolated_latitude = (1 - fraction) * self.latitude + fraction * other.latitude
        interpolated_longitude = (1 - fraction) * self.longitude + fraction * other.longitude

        interpolated_loc = Location(
            latitude=interpolated_latitude,
            longitude=interpolated_longitude,
            dwell_time=0,
            loc_type=LocationTypes.INTERPOLATED,
        )

        expected_interpolated_time = math.ceil(fraction * self.duration(other, no_ceil=True))
        actual_interpolated_time = self.duration(interpolated_loc)

        error_msg = f"Expected (FRA) {expected_interpolated_time} +/-1, Actual {actual_interpolated_time}"
        assert actual_interpolated_time - 1 <= expected_interpolated_time <= actual_interpolated_time + 1, error_msg

        first_part_time = self.duration(interpolated_loc)
        last_part_time = interpolated_loc.duration(other)
        total_time = self.duration(other)
        error_msg = (f"Expected (DIV) {total_time}, "
                     f"Actual {first_part_time} + {last_part_time} == {first_part_time + last_part_time}")
        assert total_time - 1 <= first_part_time + last_part_time <= total_time + 1, error_msg
        return interpolated_loc

    def equals(self, other):
        """
        :param other: other Location object
        :return: true if both latitude and longitude values follows same precision
        """
        assert isinstance(other, type(self)), f"comparison expects to be of type {type(self)}, but found {type(other)}"
        return (self.latitude, self.longitude) == (other.latitude, other.longitude)

    def get_grid_position(self, grid_size=2):
        """
        :param grid_size: size of grid
        :return: tuple of latitude and longitude as the grid coordinates
        """
        from env.data.TimeMatrix import MatrixManager
        return MatrixManager.instance().get_grid_coordinates(self.get_raw(), grid_size=grid_size)

    def distance(self, other):
        """
        :param other: point as Location object either a geo-coordinates
        :return: the distance to move from point_a to point_b in miles
        """
        if not isinstance(other, Location):
            other = other.loc

        if self.equals(other):
            # if both of them are same location
            return 0
        else:
            from env.data.TimeMatrix import MatrixManager
            return MatrixManager.instance().get_travel_distance(self, other)

    def duration(self, other, no_ceil=False):
        """
        :param other: point as Location object includes latitude and longitude values
        :param no_ceil: skip ceiling
        :return: the time to move from point_a to point_b at the speed in seconds
        """
        if not isinstance(other, Location):
            other = other.loc

        if self.equals(other):
            # if both of them are same location
            return 0
        else:
            from env.data.TimeMatrix import MatrixManager
            return MatrixManager.instance().get_travel_time(self, other, no_ceil=no_ceil)
