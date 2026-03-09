from common.types import RequestActions


class RouteNode:
    def __init__(
            self,
            idx,
            relative_idx,
            request_id,
            node_type,
            loc,
            earliest_arrival,
            latest_arrival,
            capacity,
            expected_status,
            current_time
    ):
        self.idx = idx
        self.relative_idx = relative_idx
        self.request_id = request_id
        self.node_type = node_type
        self.loc = loc
        self.earliest_arrival = earliest_arrival
        self.latest_arrival = latest_arrival
        self.expected_status = expected_status
        self.current_time = current_time

        if self.node_type == RequestActions.Pickup:
            for key, value in capacity.items():
                assert value >= 0

        if self.node_type == RequestActions.Dropoff:
            for key, value in capacity.items():
                assert value <= 0

        self.capacity = capacity

    def duration(self, other):
        if isinstance(other, RouteNode):
            other = other.loc
        return self.loc.duration(other)

    def distance(self, other):
        if isinstance(other, RouteNode):
            other = other.loc
        return self.loc.distance(other)

    def interpolate(self, other, numerator, denominator):
        if isinstance(other, RouteNode):
            other = other.loc
        return self.loc.interpolate(other, numerator, denominator)

    def reachable(self, node_y, dwell_time):
        """
        :param node_y: node to which the current node reach
        :param dwell_time: dwell time
        :return: whether this node can reach the node-y
        """
        return (self.earliest_arrival + self.duration(node_y) + dwell_time <= node_y.latest_arrival) and (
                self != node_y)


def generate_pick_up_node(request, current_time):
    return RouteNode(
        idx=request.pickup_idx,
        relative_idx=request.relative_pickup_idx,
        request_id=request.request_id,
        node_type=RequestActions.Pickup,
        loc=request.origin,
        earliest_arrival=request.get_earliest_pickup(current_time=current_time),
        latest_arrival=request.get_latest_pickup(current_time=current_time),
        capacity=request.capacity,
        expected_status=request.expected_status,
        current_time=current_time
    )


def generate_drop_off_node(request, current_time):
    return RouteNode(
        idx=request.dropoff_idx,
        relative_idx=request.relative_dropoff_idx,
        request_id=request.request_id,
        node_type=RequestActions.Dropoff,
        loc=request.destination,
        earliest_arrival=request.get_earliest_dropoff(current_time=current_time),
        latest_arrival=request.get_latest_dropoff(current_time=current_time),
        capacity={key: -value for key, value in request.capacity.items()},
        expected_status=request.expected_status,
        current_time=current_time
    )
