##################################################################
#              PDPTW MUTATION OPERATOR IMPLEMENTATION            #
##################################################################

from enum import Enum

from common.general import measure_time


class MutationTypes(Enum):
    Reverse = "Reverse"  # inspired from 2-opt
    TwoOptReverse = "2-Opt"  # custom 2-opt to support the PDPTW problem [!Not actual 2-opt]
    Shift = "Shift"  # Relocate
    Reposition = "Reposition"  # Relocate Pair
    Move = "Move"
    Swap = "Swap"  # Exchange Pair, Note: Exchange operator is not appropriate for our problem instance


@measure_time
def reverse(route, number_of_pod_to_consider):
    """
    :param route: route instance
    :param number_of_pod_to_consider: number of nodes that need to be reversed
    :return: permuted route and information of the reverse operation
    """
    import math
    from copy import deepcopy
    reverse_info = {
        "operation": MutationTypes.Reverse.value,
        "success": False
    }
    copy_inst = deepcopy(route)
    nodes = copy_inst.nodes

    sample_sets = [k for k in range(route.movable_idx, len(nodes) - number_of_pod_to_consider)]
    if len(sample_sets) > 0:
        # choose one position and do the permutation
        stat = None
        import random
        k = random.choice(sample_sets)

        prev_part = nodes[:k]

        selected_part = nodes[k:k + number_of_pod_to_consider]
        selected_part_indices = [node.relative_idx for node in selected_part]
        selected_times = deepcopy(route.times[k:k + number_of_pod_to_consider])
        reversed_part = selected_part[::-1]
        reverse_part_indices = [node.relative_idx for node in reversed_part]
        reversed_times = selected_times[::-1]

        remain_part = nodes[k + number_of_pod_to_consider:]

        # skip if the reverse is same as original (in practice it will never happen when number_of_pod_to_consider >= 2)
        if selected_part_indices != reverse_part_indices:

            if route.is_pick_up_and_drop_off_inorder(reversed_part):
                updated_nodes = prev_part + reversed_part + remain_part
                stat = route.adjust(updated_nodes)

        if stat:
            route.add(stat.nodes, stat.times, verify=True)
            reverse_info.update(
                {
                    "success": True,
                    "modified_routes": [route],
                    "route": route.route_id,
                    "position_at_reverse": k,
                    "nodes_to_reverse": number_of_pod_to_consider,
                    "initial_info": [
                        (
                            node.request_id,
                            node.node_type.value,
                            math.floor(node.earliest_arrival),
                            math.ceil(selected_times[i]),
                            math.ceil(node.latest_arrival)
                        )
                        for i, node in enumerate(selected_part)
                    ],
                    "reverse_info": [
                        (
                            node.request_id,
                            node.node_type.value,
                            math.floor(node.earliest_arrival),
                            math.ceil(reversed_times[i]),
                            math.ceil(node.latest_arrival)
                        )
                        for i, node in enumerate(reversed_part)
                    ]
                }
            )
    return reverse_info


@measure_time
def two_opt_reverse(route):
    """
    :param route: route instance
    :return: permuted route and information of the 2-opt reverse operation
    """
    import math
    from copy import deepcopy
    two_opt_reverse_info = {
        "operation": MutationTypes.TwoOptReverse.value,
        "success": False
    }
    copy_inst = deepcopy(route)
    nodes = copy_inst.nodes

    reverse_able_indices = []
    for i, node in enumerate(nodes[route.movable_idx:-1]):
        if node.request_id != nodes[route.movable_idx + i + 1].request_id:
            reverse_able_indices.append(i)

    possible_combinations = []
    for i, k_i in enumerate(reverse_able_indices):
        for k_j in reverse_able_indices[i + 1:]:
            if k_j > k_i + 1:
                possible_combinations.append((k_i, k_j))

    if len(possible_combinations) > 0:
        # choose one position and do the permutation
        stat = None

        import random
        i, j = random.choice(possible_combinations)

        prev_part = nodes[:i + 1]
        selected_part = nodes[i + 1:j + 2]
        reversed_part = selected_part[::-1]
        selected_times = deepcopy(copy_inst.times[i + 1:j + 2])
        reversed_selected_times = deepcopy(copy_inst.times[i + 1:j + 2][::-1])
        remain_part = nodes[j + 2:]

        if route.is_pick_up_and_drop_off_inorder(reversed_part):
            updated_nodes = prev_part + reversed_part + remain_part
            stat = route.adjust(updated_nodes)

        if stat:
            route.add(stat.nodes, stat.times, verify=True)

            two_opt_reverse_info.update(
                {
                    "success": True,
                    "modified_routes": [route],
                    "route": route.route_id,
                    "position_at_2_opt": (i, j),
                    "initial_info": [
                        (
                            node.request_id,
                            node.node_type.value,
                            math.floor(node.earliest_arrival),
                            math.ceil(selected_times[i]),
                            math.ceil(node.latest_arrival)
                        )
                        for i, node in enumerate(selected_part)
                    ],
                    "2_opt_reverse_info": [
                        (
                            node.request_id,
                            node.node_type.value,
                            math.floor(node.earliest_arrival),
                            math.ceil(reversed_selected_times[i]),
                            math.ceil(node.latest_arrival)
                        )
                        for i, node in enumerate(reversed_part)
                    ]
                }
            )
    return two_opt_reverse_info


@measure_time
def shift(instance, route):
    """
    :param instance: complete instance descriptions
    :param route: route instance
    :return: permuted route and information of the shift operation
    """
    import math
    from copy import deepcopy
    from common.types import ObjectiveTypes
    from learn.agent.AgentManager import AgentManager
    from env.solution.Action import ComprehensiveAction
    shifted_info = {
        "operation": MutationTypes.Shift.value,
        "success": False
    }
    copy_inst = deepcopy(route)
    nodes = copy_inst.nodes

    sample_sets = [k for k in range(max(route.movable_idx, 1), len(nodes) - 1)]
    if len(sample_sets) > 0:
        # choose one position and do the permutation
        stat = None
        min_cost = math.inf
        best_shifted_part = None
        best_shifted_position = None

        import random
        k = random.choice(sample_sets)

        prev_part = nodes[:k]
        selected_node = nodes[k]
        remain_part = nodes[k + 1:]
        remain_times = copy_inst.times[k:]
        current_remain_indices = [selected_node.relative_idx] + [node.relative_idx for node in remain_part]

        max_position = len(remain_part)
        if selected_node.request_id in copy_inst.un_served_request_ids.keys():
            # restrict the pickup position to search until dropoff index
            p_idx, d_idx = copy_inst.un_served_request_ids[selected_node.request_id]
            if p_idx == k:
                max_position = d_idx - p_idx

        actions = []
        for p in range(max(route.movable_idx, 1), max_position):
            updated_remain = remain_part[:p] + [selected_node] + remain_part[p:]
            updated_remain_indices = [node.relative_idx for node in updated_remain]

            # skip if the shift is same as original
            if current_remain_indices != updated_remain_indices:
                if route.is_pick_up_and_drop_off_inorder(updated_remain):
                    updated_nodes = prev_part + updated_remain
                    temp_stat = route.adjust(updated_nodes)
                    if temp_stat:
                        if instance["objective"] == ObjectiveTypes.CustomObjectiveByRL.value:
                            actions.append(
                                ComprehensiveAction(
                                    initial_routes=instance["routes"],
                                    changed_routes={route.route_id: temp_stat},
                                    shifted_part=updated_remain,
                                    shifted_position=k + p
                                )
                            )
                        else:
                            if route.get_cost(
                                    objective=instance["objective"],
                                    state=instance,
                                    action=ComprehensiveAction(
                                        initial_routes=instance["routes"],
                                        changed_routes={route.route_id: temp_stat}
                                    )
                            ) < min_cost:
                                stat = temp_stat
                                best_shifted_part = updated_remain
                                best_shifted_position = k + p

        if len(actions) > 0:
            best_action, value = AgentManager.instance().get_best_action(instance, actions)
            stat = best_action.changed_routes[route.route_id]
            best_shifted_part = best_action.shifted_part
            best_shifted_position = best_action.shifted_position

        if stat:
            route.add(stat.nodes, stat.times, verify=True)
            best_times = deepcopy(stat.times[k:])
            shifted_info.update(
                {
                    "success": True,
                    "modified_routes": [route],
                    "route": route.route_id,
                    "position_to_shift": k,
                    "position_at_which_node_shifted": best_shifted_position,
                    "initial_info": [
                        (
                            node.request_id,
                            node.node_type.value,
                            math.floor(node.earliest_arrival),
                            math.ceil(remain_times[i]),
                            math.ceil(node.latest_arrival)
                        )
                        for i, node in enumerate([selected_node] + remain_part)
                    ],
                    "shifted_info": [
                        (
                            node.request_id,
                            node.node_type.value,
                            math.floor(node.earliest_arrival),
                            math.ceil(best_times[i]),
                            math.ceil(node.latest_arrival)
                        )
                        for i, node in enumerate(best_shifted_part)
                    ]
                }
            )
    return shifted_info


@measure_time
def reposition(instance, route):
    """
    :param instance: complete instance descriptions
    :param route: route instance
    :return: permuted route and information of the reposition operation
    """
    import random
    from copy import deepcopy
    reposition_info = {
        "operation": MutationTypes.Reposition.value,
        "success": False
    }
    if len(route.un_served_request_ids) == 0:
        return reposition_info

    copy_inst = deepcopy(route)
    nodes = copy_inst.nodes

    pick_up_idx, drop_off_idx = random.choice(list(copy_inst.un_served_request_ids.values()))
    pick_up_node = nodes[pick_up_idx]
    drop_off_node = nodes[drop_off_idx]
    stat = route.remove_and_place_new_request(
        instance, pick_up_idx, drop_off_idx, pick_up_node, drop_off_node
    )
    if stat:
        route.add(stat.nodes, stat.times, verify=True)
        reposition_info.update(
            {
                "success": True,
                "modified_routes": [route],
                "route_01": route.route_id,
                "request_repositioned_from_route_01": pick_up_node.request_id,
            }
        )
    return reposition_info


@measure_time
def move(instance, route, other_route):
    """
    :param instance: complete instance descriptions
    :param route: route instance
    :param other_route: other route from which the requests move to the current route
    :return: routes with moved request and information of the permutation operation
    """
    from copy import deepcopy
    move_info = {
        "operation": MutationTypes.Move.value,
        "success": False
    }
    if len(other_route.un_served_request_ids) == 0:
        return move_info

    import random

    copy_inst = deepcopy(route)
    copy_other_inst = deepcopy(other_route)

    o_pick_up_idx, o_drop_off_idx = random.choice(list(copy_other_inst.un_served_request_ids.values()))
    pick_up_node_other = copy_other_inst.nodes[o_pick_up_idx]
    drop_off_node_other = copy_other_inst.nodes[o_drop_off_idx]

    # get the position to insert the request such a way that the insertion provide effective utilization
    stat = route.place_new_request(
        instance, copy_inst.nodes, pick_up_node_other, drop_off_node_other
    )
    if stat:
        route.add(stat.nodes, stat.times, verify=True)
        other_route = other_route.remove(o_pick_up_idx, o_drop_off_idx, verify=True)
        move_info.update(
            {
                "success": True,
                "modified_routes": [route, other_route],
                "route_01": route.route_id,
                "route_02": other_route.route_id,
                "request_moved_from_route_02": pick_up_node_other.request_id
            }
        )
    return move_info


@measure_time
def swap(instance, route, other_route):
    """
    :param instance: complete instance descriptions
    :param route: route instance
    :param other_route: other route from which the requests can be swapped
    :return: routes with swapped and information of the permutation operation
    """
    from copy import deepcopy
    from common.types import ObjectiveTypes
    from learn.agent.AgentManager import AgentManager
    from env.solution.Action import ComprehensiveAction
    swap_info = {
        "operation": MutationTypes.Swap.value,
        "success": False
    }
    if len(route.un_served_request_ids) == 0 or len(other_route.un_served_request_ids) == 0:
        return swap_info

    import random
    import math

    copy_inst = deepcopy(route)
    copy_other_inst = deepcopy(other_route)

    pick_up_idx, drop_off_idx = random.choice(list(copy_inst.un_served_request_ids.values()))
    o_pick_up_idx, o_drop_off_idx = random.choice(list(copy_other_inst.un_served_request_ids.values()))

    pick_up_node = copy_inst.nodes[pick_up_idx]
    drop_off_node = copy_inst.nodes[drop_off_idx]
    pick_up_node_other = copy_other_inst.nodes[o_pick_up_idx]
    drop_off_node_other = copy_other_inst.nodes[o_drop_off_idx]

    stat = None
    other_stat = None
    stats = route.get_feasible_actions_after_removal(
        pick_up_idx, drop_off_idx, pick_up_node_other, drop_off_node_other
    )
    if len(stats) > 0:
        other_stats = other_route.get_feasible_actions_after_removal(
            o_pick_up_idx, o_drop_off_idx, pick_up_node, drop_off_node
        )
        actions = []
        min_cost = math.inf
        for stat_s, other_stat_s in zip(stats, other_stats):
            if instance["objective"] == ObjectiveTypes.CustomObjectiveByRL.value:
                actions.append(
                    ComprehensiveAction(
                        initial_routes=instance["routes"],
                        changed_routes={
                            route.route_id: stat_s, other_route.route_id: other_stat_s
                        }
                    )
                )
            else:
                mutation_cost = route.get_cost(
                    objective=instance["objective"],
                    state=instance,
                    action=ComprehensiveAction(
                        initial_routes=instance["routes"],
                        changed_routes={route.route_id: stat_s, other_route.route_id: other_stat_s}
                    )
                )
                if mutation_cost < min_cost:
                    min_cost = mutation_cost
                    stat = stat_s
                    other_stat = other_stat_s

        if len(actions) > 0:
            best_action, _ = AgentManager.instance().get_best_action(instance, actions)
            stat = best_action.changed_routes[route.route_id]
            other_stat = best_action.changed_routes[other_route.route_id]

    if stat and other_stat:
        route.add(stat.nodes, stat.times, verify=True)
        other_route.add(other_stat.nodes, other_stat.times, verify=True)
        swap_info.update(
            {
                "success": True,
                "modified_routes": [route, other_route],
                "route_01": route.route_id,
                "request_swapped_from_route_01": pick_up_node.request_id,
                "route_02": other_route.route_id,
                "request_swapped_from_route_02": pick_up_node_other.request_id
            }
        )
    return swap_info


def nearest_neighbour(instance):
    """
    :param instance: current solution and associated configurations
    :return: solution that is mutated from current routes, and operations that leads to the mutated routes
    """
    import random
    from copy import deepcopy
    from env.solution.Action import ComprehensiveAction
    from learn.agent.AgentManager import AgentManager
    initial_routes = deepcopy(instance["routes"])
    updated_routes = deepcopy(instance["routes"])

    two_route_ops = [MutationTypes.Move, MutationTypes.Swap]
    allowed_alteration_ops = [item for item in MutationTypes if item]
    allowed_single_route_ops = [item for item in allowed_alteration_ops if item not in two_route_ops]

    base_value = AgentManager.instance().get_value(
        objective=instance["objective"],
        state=instance,
        action=ComprehensiveAction(initial_routes),
        use_target=instance["use_target"]
    )

    operation_records = {}
    operation_counts = {}
    for ops in allowed_alteration_ops:
        operation_counts[ops.value] = 0
        operation_counts[f"{ops.value}_success"] = 0
        operation_counts[f"{ops.value}_improved"] = 0
        operation_counts[f"{ops.value}_time_taken"] = 0

    changed_route_ids = set()
    random.seed(instance["iteration"])

    improvable_routes = 0
    routes_with_movable_requests = 0
    filtered_routes = []
    routes_with_far_requests = []
    allowed_alteration_ops = []
    for updated_route in updated_routes:
        if updated_route.original_end_time >= instance["current_time"]:
            filtered_routes.append(updated_route)
            improvable_routes += 1

            # check whether there is movable requests
            if len(updated_route.un_served_request_ids) > 0:
                routes_with_movable_requests += 1
                routes_with_far_requests.append(updated_route)

            if routes_with_movable_requests == 1:
                allowed_alteration_ops.append(MutationTypes.Move)

            if routes_with_movable_requests == 2:
                allowed_alteration_ops.append(MutationTypes.Swap)

    if improvable_routes >= 1:
        allowed_alteration_ops.extend(deepcopy(allowed_single_route_ops))

    # restrict the choices
    choice = random.choice(allowed_single_route_ops if improvable_routes <= 1 else allowed_alteration_ops)
    operation_counts[choice.value] += 1

    mutation_resp = None
    selected_routes = []
    match choice:
        # https://github.com/google/or-tools/blob/stable/ortools/constraint_solver/routing_parameters.proto#L140
        case MutationTypes.Reverse:
            reverse_size_choice = random.choice([2, 3, 4])
            selected_routes = random.choice(filtered_routes)
            mutation_resp = reverse(selected_routes, reverse_size_choice)
        case MutationTypes.TwoOptReverse:
            selected_routes = random.choice(filtered_routes)
            mutation_resp = two_opt_reverse(selected_routes)
        case MutationTypes.Shift:
            selected_routes = random.choice(filtered_routes)
            mutation_resp = shift(instance, selected_routes)
        case MutationTypes.Reposition:
            selected_routes = random.choice(filtered_routes)
            mutation_resp = reposition(instance, selected_routes)
        case MutationTypes.Swap:
            selected_routes = random.sample(routes_with_far_requests, 2)
            mutation_resp = swap(instance, selected_routes[0], selected_routes[1])
        case MutationTypes.Move:
            route_with_mv_req = random.choice(routes_with_far_requests)
            main_route = random.choice(
                [route for route in filtered_routes if route.route_id != route_with_mv_req.route_id]
            )
            selected_routes = [main_route, route_with_mv_req]
            mutation_resp = move(instance, main_route, route_with_mv_req)

    if mutation_resp:
        operation_counts[choice.value + "_time_taken"] += mutation_resp["compute_time"]

        if mutation_resp["success"]:
            if not isinstance(selected_routes, list):
                selected_routes = [selected_routes]

            for k, modified_route in enumerate(mutation_resp["modified_routes"]):
                if not selected_routes[k].route_id == modified_route.route_id:
                    raise AssertionError(
                        f"Expected route ID {selected_routes[k].route_id}, actual route ID {modified_route.route_id}"
                    )
                updated_routes[selected_routes[k].route_id] = modified_route
                changed_route_ids.add(modified_route.route_id)

            current_value = AgentManager.instance().get_value(
                objective=instance["objective"],
                state=instance,
                action=ComprehensiveAction(
                    initial_routes, {route.route_id: route for route in updated_routes}
                ),
                use_target=instance["use_target"]
            )

            mutation_resp.pop("modified_routes")
            operation_records = [deepcopy(mutation_resp)]
            operation_counts[choice.value + "_success"] += 1
            if current_value > base_value:
                operation_counts[choice.value + "_improved"] += 1
            instance["routes"] = updated_routes

    return ComprehensiveAction(
        initial_routes=initial_routes,
        changed_routes={route.route_id: route for route in updated_routes},
        operation_counts=operation_counts,
        operation_records=operation_records,
    )
