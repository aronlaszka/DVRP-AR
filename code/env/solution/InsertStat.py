from env.solution.RouteBase import RouteBase


class InsertStat(RouteBase):
    def __init__(
            self,
            route_id,
            capacity,
            start_pos,
            end_pos,
            nodes,
            times,
            capacities,
            total_travel_time,
            total_drive_time,
            start_time,
            original_start_time,
            original_end_time
    ):
        self.route_id = route_id
        self.capacity = capacity
        self.start_pos = start_pos
        self.end_pos = end_pos
        self.nodes = nodes
        self.times = times
        self.capacities = capacities
        self.total_travel_time = total_travel_time
        self.total_drive_time = total_drive_time
        self.start_time = start_time
        self.original_start_time = original_start_time
        self.original_end_time = original_end_time
