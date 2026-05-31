from typing import Dict, List, Tuple


def _select_pair_split_point(
    main_client,
    aux_client,
    max_split_point: int,
    forward_flops_by_splitpoint_list_self: List[float],
    backward_flops_by_splitpoint_list_self: List[float],
    forward_flops_by_splitpoint_list_helper: List[float],
    backward_flops_by_splitpoint_list_helper: List[float],
    valid_split_points: List[int] = None,
) -> int:
    """Select split point that balances per-step compute time within a pair.

    Criterion:
    - aux lower load: (forward_self + backward_self) / aux.computing
    - main load: ((forward_helper + backward_helper) + full_model_forward + full_model_backward) / main.computing
    - choose split point minimizing absolute gap.
    """
    best_split = 1
    best_gap = float("inf")
    full_model_flops = (
        forward_flops_by_splitpoint_list_self[-1]
        + backward_flops_by_splitpoint_list_self[-1]
    )

    if valid_split_points is None:
        candidate_split_points = list(range(1, max_split_point))
    else:
        candidate_split_points = [
            int(sp) for sp in valid_split_points if 1 <= int(sp) < max_split_point
        ]
        if not candidate_split_points:
            raise ValueError("valid_split_points must contain values in [1, max_split_point-1].")

    for split_point in candidate_split_points:
        aux_time = (
            forward_flops_by_splitpoint_list_self[split_point]
            + backward_flops_by_splitpoint_list_self[split_point]
        ) / aux_client.computing
        main_time = (
            forward_flops_by_splitpoint_list_helper[split_point]
            + backward_flops_by_splitpoint_list_helper[split_point]
            + full_model_flops
        ) / main_client.computing
        gap = abs(main_time - aux_time)

        if gap < best_gap:
            best_gap = gap
            best_split = split_point

    return best_split


def build_sflmac_pairs(
    clients: List,
    max_split_point: int,
    forward_flops_by_splitpoint_list_self: List[float],
    backward_flops_by_splitpoint_list_self: List[float],
    forward_flops_by_splitpoint_list_helper: List[float],
    backward_flops_by_splitpoint_list_helper: List[float],
    shared_split_point: int = None,
    debug: bool = False,
    valid_split_points: List[int] = None,
) -> Dict[str, object]:
    """Build SFLMAC pairing and pair-wise split points.

    Pairing strategy:
    - sort clients by computing in descending order
    - pair largest with smallest sequentially
    - role: larger computing client is main, smaller is aux

    Split-point strategy:
    - if shared_split_point is provided, use it for all pairs
    - otherwise, choose split point per pair by minimizing
            | (upper_flops + full_model_flops)/main.computing - lower_flops/aux.computing |
    """
    num_client = len(clients)
    sorted_ids = sorted(range(num_client), key=lambda idx: clients[idx].computing, reverse=True)

    pairs: List[Tuple[int, int]] = []
    pair_split_points: Dict[Tuple[int, int], int] = {}
    pair_map = {i: None for i in range(num_client)}
    role_map = {i: "normal" for i in range(num_client)}
    client_loads = {i: 0.0 for i in range(num_client)}

    full_model_flops = (
        forward_flops_by_splitpoint_list_self[-1]
        + backward_flops_by_splitpoint_list_self[-1]
    )

    if debug:
        print("[SFLMAC][Debug] Initial loads before split assignment (full_model_flops / computing):")
        for client_id in range(num_client):
            init_load = full_model_flops / clients[client_id].computing
            print(
                f"[SFLMAC][Debug] client={client_id} "
                f"computing={clients[client_id].computing:.6e} "
                f"initial_load={init_load:.6e}"
            )

    if valid_split_points is None:
        candidate_split_points = list(range(1, max_split_point))
    else:
        candidate_split_points = [
            int(sp) for sp in valid_split_points if 1 <= int(sp) < max_split_point
        ]
        if not candidate_split_points:
            raise ValueError("valid_split_points must contain values in [1, max_split_point-1].")

    if shared_split_point is not None:
        shared_split_point = int(shared_split_point)
        if shared_split_point not in candidate_split_points:
            raise ValueError(
                f"shared_split_point={shared_split_point} is invalid. "
                f"Allowed split points: {candidate_split_points}"
            )

    left = 0
    right = num_client - 1
    while left < right:
        main_id = sorted_ids[left]
        aux_id = sorted_ids[right]
        left += 1
        right -= 1

        if shared_split_point is not None:
            split_point = shared_split_point
        else:
            split_point = _select_pair_split_point(
                main_client=clients[main_id],
                aux_client=clients[aux_id],
                max_split_point=max_split_point,
                forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
                backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
                forward_flops_by_splitpoint_list_helper=forward_flops_by_splitpoint_list_helper,
                backward_flops_by_splitpoint_list_helper=backward_flops_by_splitpoint_list_helper,
                valid_split_points=candidate_split_points,
            )

        pair = (main_id, aux_id)
        pairs.append(pair)
        pair_split_points[pair] = split_point
        pair_map[main_id] = pair
        pair_map[aux_id] = pair
        role_map[main_id] = "main"
        role_map[aux_id] = "aux"

        aux_load = (
            forward_flops_by_splitpoint_list_self[split_point]
            + backward_flops_by_splitpoint_list_self[split_point]
        ) / clients[aux_id].computing
        main_load = (
            forward_flops_by_splitpoint_list_helper[split_point]
            + backward_flops_by_splitpoint_list_helper[split_point]
            + full_model_flops
        ) / clients[main_id].computing
        client_loads[aux_id] = aux_load
        client_loads[main_id] = main_load

        if debug:
            print(
                f"[SFLMAC][Debug] pair(main={main_id}, aux={aux_id}) "
                f"split_point={split_point} "
                f"main_load={main_load:.6e} aux_load={aux_load:.6e}"
            )

    normal = []
    if left == right:
        normal_id = sorted_ids[left]
        normal.append(normal_id)

    for normal_id in normal:
        pair_map[normal_id] = None
        role_map[normal_id] = "normal"
        client_loads[normal_id] = full_model_flops / clients[normal_id].computing

    if debug:
        print("[SFLMAC][Debug] Final client loads after pair split selection (flops / computing):")
        for client_id in range(num_client):
            print(
                f"[SFLMAC][Debug] client={client_id} "
                f"role={role_map[client_id]} "
                f"load={client_loads[client_id]:.6e}"
            )

    return {
        "pairs": pairs,
        "pair_split_points": pair_split_points,
        "pair_map": pair_map,
        "role_map": role_map,
        "normal": normal,
        "client_loads": client_loads,
    }
