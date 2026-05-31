from utils.time.time_engine import (
    TimeDebugClientInfo,
    TimeDebugTrainingInfo,
    print_debug_time_info,
)


# 贪心算法
## 选择最佳协同训练节点
def LBFSL_load_balance_by_tx(clients: list, user_num: int,
                       forward_flops_by_splitpoint_list_self: list,
                       backward_flops_by_splitpoint_list_self: list,
                       parameter_bits_by_splitpoint_list_self: list,
                       forward_flops_by_splitpoint_list_helper: list,
                       backward_flops_by_splitpoint_list_helper: list,
                       parameter_bits_by_splitpoint_list_helper: list,
                       activate_shape_list: list):
    ## 函数结束条件（怎么判断负载实现了均衡）
    ##### 可以设置一个阈值，当所有客户端的负载时间与平均负载时间的差值小于该阈值时，认为负载均衡达成。比如说选初始时50%客户端能完成的时间作为阈值
    ##### 可以设置阈值，user_num中负载最高的k个客户端进行调整，k可以是固定值，也可以是user_num的某个比例
    ##### 可以计算负载均值，然后调整负载高于均值的客户端（尽可能减少高于此值的客户端的负载，且保证低于此值的客户端负载不会超过此值）

    ## 对负载进行排序，（选择负载最低的节点,后续可以考虑数据分布的影响）)
    ### 获取每个客户端的负载，这部分可以复用
    loads = []
    

    be_helped_list = [-1 for _ in range(user_num)]  # 记录每个客户端需要帮助的设备的列表

    for i in range(user_num):
        loads.append(
            clients[i].calculate_loads_by_splitpoint(
                clients[i].split_point,
                forward_flops_by_splitpoint_list_self,
                backward_flops_by_splitpoint_list_self,
                parameter_bits_by_splitpoint_list_self,
                activate_shape_list,
            )
        )

    print("初始各客户端负载时间：")
    for i in range(user_num):
        # 根据分割点获得前向传播负载、传输激活值、反向传播负载
        ## 需要参数：分割点，前向FLOPs列表，反向FLOPs列表，激活值形状列表，带宽列表
        print(f"客户端{i}的负载时间：{loads[i]}秒")
    average_load = sum(loads) / user_num
    print(f"调整前平均负载时间：{average_load}秒")
    print("----------------------------")

    ## 每一个分割点寻找可行的卸载设备，在这些设备中找到一个最优的设备（自己负载（固定）+他的增加负载最小）
    adjusted_clients = set()
    for _ in range(user_num):
        # 获取当前未调整客户端中的负载最高者
        remaining_indices = [idx for idx in range(user_num) if idx not in adjusted_clients]
        if not remaining_indices:
            break
        max_loads_index = max(remaining_indices, key=lambda idx: loads[idx])
        max_loads = loads[max_loads_index]
        i = max_loads_index
        # print(f"当前选择调整的客户端是{i}，其负载为{loads[i]}秒")
        aux_loads = [[] for _ in range(8)]  # 记录每个分割点对应的负载时间变化
        local_loads = [0 for _ in range(8)]  # 记录每个分割点对应的本地负载时间
        for j in range(1,8): # 分割点从1开始到7，以后设置为模型对应的最大分割点 TODO
            local_loads[j] = clients[i].calculate_loads_by_splitpoint(
                j,
                forward_flops_by_splitpoint_list_self,
                backward_flops_by_splitpoint_list_self,
                parameter_bits_by_splitpoint_list_self,
                activate_shape_list
            )
            # print(f"客户端{i}在分割点{j}时，自身负载为{local_loads[j]}秒")
            for k in range(user_num): # 寻找协同训练节点
                # 计算客户端i在分割点j，选择客户端k作为协同训练节点时的负载时间
                tmp_loads = clients[k].calculate_auxiliary_training_load(
                    j,
                    forward_flops_by_splitpoint_list_helper,
                    backward_flops_by_splitpoint_list_helper,
                    parameter_bits_by_splitpoint_list_helper,
                    activate_shape_list
                )
                # print(f"客户端{i}在分割点{j}选择客户端{k}作为协同训练节点时，自身负载为{local_loads[j]}，协同设备负载时间为{loads[k]+tmp_loads}秒")
                if loads[k] + tmp_loads <= max_loads and loads[k] + tmp_loads <= local_loads[j] + tmp_loads: # 协同设备负载不超过最大负载且不超过被协同训练设备的本地负载
                    aux_loads[j].append(tmp_loads)
                else :
                    aux_loads[j].append(float('inf'))  # 超过负载则不考虑
                    
        # 确定合适的分割点：要求是 local_loads <= average_load
        suitable_split_points = [j for j in range(1,8) if local_loads[j] <= max_loads] # 这个合适负载点很难处理,改8 TODO
        # 在合适的分割点中，找增加负载最小的协同训练节点，以确定最终的分割点和协同训练节点
        if suitable_split_points:
            min_merge_loads = float('inf')
            best_split_point = -1
            best_helper_id = -1
            for j in suitable_split_points:
                for k in range(user_num):
                    if k != i:
                        # if aux_loads[j][k] < min_increase: # 这个最优结果很难确定
                        if loads[k] + aux_loads[j][k] + local_loads[j] + aux_loads[j][k] < min_merge_loads:
                            min_merge_loads = loads[k] + aux_loads[j][k] + local_loads[j] + aux_loads[j][k]
                            best_split_point = j
                            best_helper_id = k
            # 更新客户端i的分割点和协同训练节点
            if best_split_point != -1 and best_helper_id != -1:
                clients[i].split_point = best_split_point
                be_helped_list[i] = best_helper_id
                loads[i] = local_loads[best_split_point] + aux_loads[best_split_point][best_helper_id]
                loads[best_helper_id] += aux_loads[best_split_point][best_helper_id]
                clients[i].helper_client_id = best_helper_id
                print(f"客户端{i}选择的分割点是{clients[i].split_point}，协同训练节点是客户端{clients[i].helper_client_id}，负载为{loads[i]}秒, 协同节点负载为{loads[best_helper_id]}秒")
        adjusted_clients.add(i)

    ## 在不同分割点中，找合适范围内的，他的增加负载最小的分割点。
    ## 合适范围：对初始负载考虑其为正态分布，在miu +- sigma范围内的自身负载都属于合适范围。

    ## 找到合适的分割点和合适的协助设备
    #### 计算平均负载时间
    #### 寻找负载高于平均负载时间的客户端
    #### 寻找负载最低的客户端作为协同训练节点
    #### 不断下移分割点，直至负载低于平均负载时间或者分割点到达最低
    

    print("调整后各客户端负载时间：")
    for i in range(user_num):
        # 根据分割点获得前向传播负载、传输激活值、反向传播负载
        # 需要参数：分割点，前向FLOPs列表，反向FLOPs列表，激活值形状列表，带宽列表
        print(f"客户端{i}的负载时间：{loads[i]}秒")
    average_load = sum(loads) / user_num
    print(f"调整后平均负载时间：{average_load}秒")

    ## 根据分割点选择调整负载
    for i in range(user_num):
        # 调整自己的负载
        clients[i].forward_flops = forward_flops_by_splitpoint_list_self[clients[i].split_point]
        clients[i].backward_flops = backward_flops_by_splitpoint_list_self[clients[i].split_point]
        clients[i].parameter_bits = parameter_bits_by_splitpoint_list_self[clients[i].split_point]
        # 调整协同者的负载
        if be_helped_list[i] != -1:
            clients[be_helped_list[i]].parameter_bits += parameter_bits_by_splitpoint_list_helper[clients[i].split_point]


def _activation_transfer_time_by_split(split_point: int, activate_shape_list: list, up_rate: float, down_rate: float):
    if split_point <= 0 or split_point >= len(activate_shape_list):
        return 0.0, 0.0
    shape = activate_shape_list[split_point]
    if shape is None or len(shape) < 4:
        return 0.0, 0.0
    activation_bits = shape[0] * shape[1] * shape[2] * shape[3] * 32
    upload_time = activation_bits / up_rate if up_rate > 0 else float("inf")
    download_time = activation_bits / down_rate if down_rate > 0 else float("inf")
    return upload_time, download_time


def _full_model_step_time(client, split_point_full: int, forward_flops_by_splitpoint_list_self: list, backward_flops_by_splitpoint_list_self: list):
    forward_time = forward_flops_by_splitpoint_list_self[split_point_full] / client.computing
    backward_time = backward_flops_by_splitpoint_list_self[split_point_full] / client.computing
    return forward_time + backward_time


def _full_model_round_time(
    client,
    local_epoch: int,
    split_point_full: int,
    forward_flops_by_splitpoint_list_self: list,
    backward_flops_by_splitpoint_list_self: list,
    parameter_bits_by_splitpoint_list_self: list,
):
    model_download_time = client.compute_model_download_time_by_splitpoint(
        split_point=split_point_full,
        parameter_bits_by_splitpoint=parameter_bits_by_splitpoint_list_self,
    )
    model_upload_time = client.compute_model_upload_time_by_splitpoint(
        split_point=split_point_full,
        parameter_bits_by_splitpoint=parameter_bits_by_splitpoint_list_self,
    )
    step_time = _full_model_step_time(
        client=client,
        split_point_full=split_point_full,
        forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
        backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
    )
    return model_download_time + local_epoch * step_time + model_upload_time


def _pair_one_step_time(
    main_client,
    aux_client,
    split_point: int,
    forward_flops_by_splitpoint_list_self: list,
    backward_flops_by_splitpoint_list_self: list,
    forward_flops_by_splitpoint_list_helper: list,
    backward_flops_by_splitpoint_list_helper: list,
    activate_shape_list: list,
    split_point_full: int,
):
    aux_forward_lower = forward_flops_by_splitpoint_list_self[split_point] / aux_client.computing
    aux_backward_lower = backward_flops_by_splitpoint_list_self[split_point] / aux_client.computing

    activation_upload_time, gradient_download_time = _activation_transfer_time_by_split(
        split_point=split_point,
        activate_shape_list=activate_shape_list,
        up_rate=aux_client.up_rate,
        down_rate=aux_client.down_rate,
    )

    main_forward_upper = forward_flops_by_splitpoint_list_helper[split_point] / main_client.computing
    main_backward_upper = backward_flops_by_splitpoint_list_helper[split_point] / main_client.computing

    aux_split_training_time = (
        aux_forward_lower
        + activation_upload_time
        + main_forward_upper
        + main_backward_upper
        + gradient_download_time
        + aux_backward_lower
    )

    main_full_training_time = _full_model_step_time(
        client=main_client,
        split_point_full=split_point_full,
        forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
        backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
    )

    return aux_split_training_time + main_full_training_time


def _pair_time_breakdown(
    main_client,
    aux_client,
    split_point: int,
    local_epoch: int,
    forward_flops_by_splitpoint_list_self: list,
    backward_flops_by_splitpoint_list_self: list,
    forward_flops_by_splitpoint_list_helper: list,
    backward_flops_by_splitpoint_list_helper: list,
    activate_shape_list: list,
    split_point_full: int,
):
    aux_forward_lower = forward_flops_by_splitpoint_list_self[split_point] / aux_client.computing
    aux_backward_lower = backward_flops_by_splitpoint_list_self[split_point] / aux_client.computing

    activation_upload_time, gradient_download_time = _activation_transfer_time_by_split(
        split_point=split_point,
        activate_shape_list=activate_shape_list,
        up_rate=aux_client.up_rate,
        down_rate=aux_client.down_rate,
    )

    main_forward_upper = forward_flops_by_splitpoint_list_helper[split_point] / main_client.computing
    main_backward_upper = backward_flops_by_splitpoint_list_helper[split_point] / main_client.computing

    main_assist_one_step = main_forward_upper + main_backward_upper + gradient_download_time
    aux_side_one_step = aux_forward_lower + activation_upload_time + aux_backward_lower

    main_full_training_one_step = _full_model_step_time(
        client=main_client,
        split_point_full=split_point_full,
        forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
        backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
    )

    one_step_total = aux_side_one_step + main_assist_one_step + main_full_training_one_step

    return {
        "aux_side_one_step": aux_side_one_step,
        "main_assist_one_step": main_assist_one_step,
        "main_forward_upper_one_step": main_forward_upper,
        "main_backward_upper_one_step": main_backward_upper,
        "gradient_download_one_step": gradient_download_time,
        "main_full_training_one_step": main_full_training_one_step,
        "one_step_total": one_step_total,
        "aux_side_round": local_epoch * aux_side_one_step,
        "main_assist_round": local_epoch * main_assist_one_step,
        "main_full_training_round": local_epoch * main_full_training_one_step,
        "round_total": local_epoch * one_step_total,
    }


def _build_time_debug_training_info(
    forward_flops_list: list,
    backward_flops_list: list,
    parameter_bits_list: list,
    activate_shape_list: list,
    forward_flops_by_splitpoint_list_self: list,
    backward_flops_by_splitpoint_list_self: list,
    parameter_bits_by_splitpoint_list_self: list,
):
    return TimeDebugTrainingInfo(
        forward_flops_list=forward_flops_list,
        backward_flops_list=backward_flops_list,
        parameter_bits_list=parameter_bits_list,
        activate_shape_list=activate_shape_list,
        forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
        backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
        parameter_bits_by_splitpoint_list_self=parameter_bits_by_splitpoint_list_self,
        total_forward_flops=sum(forward_flops_list),
        total_backward_flops=sum(backward_flops_list),
        total_parameter_bits=sum(parameter_bits_list),
    )


def _build_time_debug_client_infos(clients: list):
    return [TimeDebugClientInfo(client_idx=i, client=c) for i, c in enumerate(clients)]


def FSCL_load_balance_by_time(
    clients: list,
    local_epoch: int,
    max_split_point: int,
    forward_flops_by_splitpoint_list_self: list,
    backward_flops_by_splitpoint_list_self: list,
    parameter_bits_by_splitpoint_list_self: list,
    forward_flops_by_splitpoint_list_helper: list,
    backward_flops_by_splitpoint_list_helper: list,
    activate_shape_list: list,
    debug: bool = False,
):
    user_num = len(clients)
    if user_num % 2 != 0:
        raise ValueError("SFLMAC requires an even number of clients for forced pairing.")
    split_point_full = max_split_point

    forward_flops_list = [
        forward_flops_by_splitpoint_list_self[i] - forward_flops_by_splitpoint_list_self[i - 1]
        for i in range(1, len(forward_flops_by_splitpoint_list_self))
    ]
    backward_flops_list = [
        backward_flops_by_splitpoint_list_self[i] - backward_flops_by_splitpoint_list_self[i - 1]
        for i in range(1, len(backward_flops_by_splitpoint_list_self))
    ]
    parameter_bits_list = [
        parameter_bits_by_splitpoint_list_self[i] - parameter_bits_by_splitpoint_list_self[i - 1]
        for i in range(1, len(parameter_bits_by_splitpoint_list_self))
    ]

    solo_round_times = []
    for client in clients:
        full_round_time = _full_model_round_time(
            client=client,
            local_epoch=local_epoch,
            split_point_full=split_point_full,
            forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
            backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
            parameter_bits_by_splitpoint_list_self=parameter_bits_by_splitpoint_list_self,
        )
        solo_round_times.append(full_round_time)

    if debug:
        print("\n" + "=" * 80)
        print("DEBUG: FSCL负载均衡前(含模型下载/上传的一轮全局迭代时间)")
        print("=" * 80)
        for idx, t in enumerate(solo_round_times):
            print(f"client {idx}: {t:.6f}s")
        sorted_by_time = sorted(range(user_num), key=lambda idx: solo_round_times[idx], reverse=True)
        print(f"排序后的客户端序列(从慢到快): {sorted_by_time}")
        training_info = _build_time_debug_training_info(
            forward_flops_list=forward_flops_list,
            backward_flops_list=backward_flops_list,
            parameter_bits_list=parameter_bits_list,
            activate_shape_list=activate_shape_list,
            forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
            backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
            parameter_bits_by_splitpoint_list_self=parameter_bits_by_splitpoint_list_self,
        )
        client_infos = _build_time_debug_client_infos(clients)
        print_debug_time_info(training_info=training_info, client_infos=client_infos)

    sorted_clients = sorted(range(user_num), key=lambda idx: solo_round_times[idx], reverse=True)

    pairs = []
    pair_split_points = {}
    pair_round_times = {}
    pair_map = {i: None for i in range(user_num)}
    role_map = {i: "normal" for i in range(user_num)}
    normal = []
    balanced_round_times = [0.0 for _ in range(user_num)]

    left = 0
    right = user_num - 1
    while left < right:
        aux_id = sorted_clients[left]
        main_id = sorted_clients[right]
        left += 1
        right -= 1

        best_split = None
        best_pair_time = float("inf")

        for split_point in range(1, max_split_point):
            one_step_time = _pair_one_step_time(
                main_client=clients[main_id],
                aux_client=clients[aux_id],
                split_point=split_point,
                forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
                backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
                forward_flops_by_splitpoint_list_helper=forward_flops_by_splitpoint_list_helper,
                backward_flops_by_splitpoint_list_helper=backward_flops_by_splitpoint_list_helper,
                activate_shape_list=activate_shape_list,
                split_point_full=split_point_full,
            )
            round_time = local_epoch * one_step_time

            if round_time < best_pair_time:
                best_pair_time = round_time
                best_split = split_point

        if best_split is None:
            raise ValueError("No valid split point found for forced pairing.")

        pair = (main_id, aux_id)
        pairs.append(pair)
        pair_split_points[pair] = best_split
        clients[aux_id].split_point = best_split
        clients[aux_id].helper_client_id = main_id
        clients[main_id].split_point = split_point_full
        clients[main_id].helper_client_id = aux_id
        pair_map[main_id] = pair
        pair_map[aux_id] = pair
        role_map[main_id] = "main"
        role_map[aux_id] = "aux"
        pair_round_times[pair] = best_pair_time

    if left == right:
        raise ValueError("Unpaired client remains; SFLMAC requires even client count.")

    normal = []

    for main_id, aux_id in pairs:
        pair_time = pair_round_times[(main_id, aux_id)]
        balanced_round_times[main_id] = pair_time
        balanced_round_times[aux_id] = pair_time

    # No normal clients in forced pairing mode.

    if debug:
        print("\n" + "=" * 80)
        print("DEBUG: FSCL负载均衡后")
        print("=" * 80)
        print("配对与分割点信息:")
        for main_id, aux_id in pairs:
            split_point = pair_split_points[(main_id, aux_id)]
            breakdown = _pair_time_breakdown(
                main_client=clients[main_id],
                aux_client=clients[aux_id],
                split_point=split_point,
                local_epoch=local_epoch,
                forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
                backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
                forward_flops_by_splitpoint_list_helper=forward_flops_by_splitpoint_list_helper,
                backward_flops_by_splitpoint_list_helper=backward_flops_by_splitpoint_list_helper,
                activate_shape_list=activate_shape_list,
                split_point_full=split_point_full,
            )
            print(f"pair(main={main_id}, aux={aux_id}), split_point={split_point}")
            print(
                f"  main辅助(前向+反向+梯度传输): {breakdown['main_assist_round']:.6f}s "
                f"(per-step={breakdown['main_assist_one_step']:.6f}s, "
                f"fwd={breakdown['main_forward_upper_one_step']:.6f}s, "
                f"bwd={breakdown['main_backward_upper_one_step']:.6f}s, "
                f"grad={breakdown['gradient_download_one_step']:.6f}s)"
            )
            print(
                f"  aux侧(下层前反+激活上传): {breakdown['aux_side_round']:.6f}s "
                f"(per-step={breakdown['aux_side_one_step']:.6f}s)"
            )
            print(
                f"  main独立训练(完整模型步): {breakdown['main_full_training_round']:.6f}s "
                f"(per-step={breakdown['main_full_training_one_step']:.6f}s)"
            )
            print(
                f"  pair一轮总时间: {breakdown['round_total']:.6f}s "
                f"(= local_epoch * {breakdown['one_step_total']:.6f}s)"
            )
        print(f"normal clients: {normal}")
        for idx, t in enumerate(balanced_round_times):
            print(f"client {idx}: {t:.6f}s")
        print_debug_time_info(training_info=training_info, client_infos=client_infos)

    return {
        "pairs": pairs,
        "pair_split_points": pair_split_points,
        "pair_round_times": pair_round_times,
        "pair_map": pair_map,
        "role_map": role_map,
        "normal": normal,
        "solo_round_times": solo_round_times,
        "balanced_round_times": balanced_round_times,
    }


def FSCL_load_balance_by_time_shared_split(
    clients: list,
    local_epoch: int,
    max_split_point: int,
    forward_flops_by_splitpoint_list_self: list,
    backward_flops_by_splitpoint_list_self: list,
    parameter_bits_by_splitpoint_list_self: list,
    forward_flops_by_splitpoint_list_helper: list,
    backward_flops_by_splitpoint_list_helper: list,
    activate_shape_list: list,
    shared_split_point: int,
    debug: bool = False,
):
    user_num = len(clients)
    if user_num % 2 != 0:
        raise ValueError("SFLMAC requires an even number of clients for forced pairing.")
    split_point_full = max_split_point
    shared_split_point = int(shared_split_point)
    if shared_split_point <= 0 or shared_split_point >= max_split_point:
        raise ValueError("shared_split_point must be in [1, max_split_point-1].")

    forward_flops_list = [
        forward_flops_by_splitpoint_list_self[i] - forward_flops_by_splitpoint_list_self[i - 1]
        for i in range(1, len(forward_flops_by_splitpoint_list_self))
    ]
    backward_flops_list = [
        backward_flops_by_splitpoint_list_self[i] - backward_flops_by_splitpoint_list_self[i - 1]
        for i in range(1, len(backward_flops_by_splitpoint_list_self))
    ]
    parameter_bits_list = [
        parameter_bits_by_splitpoint_list_self[i] - parameter_bits_by_splitpoint_list_self[i - 1]
        for i in range(1, len(parameter_bits_by_splitpoint_list_self))
    ]

    solo_round_times = []
    for client in clients:
        full_round_time = _full_model_round_time(
            client=client,
            local_epoch=local_epoch,
            split_point_full=split_point_full,
            forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
            backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
            parameter_bits_by_splitpoint_list_self=parameter_bits_by_splitpoint_list_self,
        )
        solo_round_times.append(full_round_time)

    if debug:
        print("\n" + "=" * 80)
        print("DEBUG: FSCL负载均衡前(含模型下载/上传的一轮全局迭代时间)")
        print("=" * 80)
        for idx, t in enumerate(solo_round_times):
            print(f"client {idx}: {t:.6f}s")
        sorted_by_time = sorted(range(user_num), key=lambda idx: solo_round_times[idx], reverse=True)
        print(f"排序后的客户端序列(从慢到快): {sorted_by_time}")
        training_info = _build_time_debug_training_info(
            forward_flops_list=forward_flops_list,
            backward_flops_list=backward_flops_list,
            parameter_bits_list=parameter_bits_list,
            activate_shape_list=activate_shape_list,
            forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
            backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
            parameter_bits_by_splitpoint_list_self=parameter_bits_by_splitpoint_list_self,
        )
        client_infos = _build_time_debug_client_infos(clients)
        print_debug_time_info(training_info=training_info, client_infos=client_infos)

    sorted_clients = sorted(range(user_num), key=lambda idx: solo_round_times[idx], reverse=True)

    pairs = []
    pair_split_points = {}
    pair_round_times = {}
    pair_map = {i: None for i in range(user_num)}
    role_map = {i: "normal" for i in range(user_num)}
    normal = []
    balanced_round_times = [0.0 for _ in range(user_num)]

    left = 0
    right = user_num - 1
    while left < right:
        aux_id = sorted_clients[left]
        main_id = sorted_clients[right]
        left += 1
        right -= 1

        one_step_time = _pair_one_step_time(
            main_client=clients[main_id],
            aux_client=clients[aux_id],
            split_point=shared_split_point,
            forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
            backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
            forward_flops_by_splitpoint_list_helper=forward_flops_by_splitpoint_list_helper,
            backward_flops_by_splitpoint_list_helper=backward_flops_by_splitpoint_list_helper,
            activate_shape_list=activate_shape_list,
            split_point_full=split_point_full,
        )
        best_pair_time = local_epoch * one_step_time

        pair = (main_id, aux_id)
        pairs.append(pair)
        pair_split_points[pair] = shared_split_point
        clients[aux_id].split_point = shared_split_point
        clients[aux_id].helper_client_id = main_id
        clients[main_id].split_point = split_point_full
        clients[main_id].helper_client_id = aux_id
        pair_map[main_id] = pair
        pair_map[aux_id] = pair
        role_map[main_id] = "main"
        role_map[aux_id] = "aux"
        pair_round_times[pair] = best_pair_time

    if left == right:
        raise ValueError("Unpaired client remains; SFLMAC requires even client count.")

    normal = []

    for main_id, aux_id in pairs:
        pair_time = pair_round_times[(main_id, aux_id)]
        balanced_round_times[main_id] = pair_time
        balanced_round_times[aux_id] = pair_time

    # No normal clients in forced pairing mode.

    if debug:
        print("\n" + "=" * 80)
        print("DEBUG: FSCL负载均衡后")
        print("=" * 80)
        print("配对与分割点信息:")
        for main_id, aux_id in pairs:
            split_point = pair_split_points[(main_id, aux_id)]
            breakdown = _pair_time_breakdown(
                main_client=clients[main_id],
                aux_client=clients[aux_id],
                split_point=split_point,
                local_epoch=local_epoch,
                forward_flops_by_splitpoint_list_self=forward_flops_by_splitpoint_list_self,
                backward_flops_by_splitpoint_list_self=backward_flops_by_splitpoint_list_self,
                forward_flops_by_splitpoint_list_helper=forward_flops_by_splitpoint_list_helper,
                backward_flops_by_splitpoint_list_helper=backward_flops_by_splitpoint_list_helper,
                activate_shape_list=activate_shape_list,
                split_point_full=split_point_full,
            )
            print(f"pair(main={main_id}, aux={aux_id}), split_point={split_point}")
            print(
                f"  main辅助(前向+反向+梯度传输): {breakdown['main_assist_round']:.6f}s "
                f"(per-step={breakdown['main_assist_one_step']:.6f}s, "
                f"fwd={breakdown['main_forward_upper_one_step']:.6f}s, "
                f"bwd={breakdown['main_backward_upper_one_step']:.6f}s, "
                f"grad={breakdown['gradient_download_one_step']:.6f}s)"
            )
            print(
                f"  aux侧(下层前反+激活上传): {breakdown['aux_side_round']:.6f}s "
                f"(per-step={breakdown['aux_side_one_step']:.6f}s)"
            )
            print(
                f"  main独立训练(完整模型步): {breakdown['main_full_training_round']:.6f}s "
                f"(per-step={breakdown['main_full_training_one_step']:.6f}s)"
            )
            print(
                f"  pair一轮总时间: {breakdown['round_total']:.6f}s "
                f"(= local_epoch * {breakdown['one_step_total']:.6f}s)"
            )
        print(f"normal clients: {normal}")
        for idx, t in enumerate(balanced_round_times):
            print(f"client {idx}: {t:.6f}s")
        print_debug_time_info(training_info=training_info, client_infos=client_infos)

    return {
        "pairs": pairs,
        "pair_split_points": pair_split_points,
        "pair_round_times": pair_round_times,
        "pair_map": pair_map,
        "role_map": role_map,
        "normal": normal,
        "solo_round_times": solo_round_times,
        "balanced_round_times": balanced_round_times,
    }


def FSL_load_balance_by_time(
    clients: list,
    server,
    local_steps: int,
    max_split_point: int,
    forward_flops_by_splitpoint_list_self: list,
    backward_flops_by_splitpoint_list_self: list,
    forward_flops_by_splitpoint_list_helper: list,
    backward_flops_by_splitpoint_list_helper: list,
    activate_shape_list: list,
    optimize_method: str = "alternating",
    alt_max_iter: int = 30,
    alt_tol: float = 1e-6,
    debug: bool = False,
    valid_split_points: list | None = None,
):
    """FSL分割点与服务器算力联合优化。

    目标：min \sum_i t_i^2
    其中 t_i = local_steps * (a_i(split_i) + b_i(split_i) / c_i)
    - a_i: 客户端本地计算 + 激活上行 + 梯度下行（与服务器算力无关）
    - b_i: 服务器负责的上层前反向 FLOPs
    - c_i: 分配给客户端 i 的服务器算力，\sum_i c_i = server.total_computing
    """
    num_client = len(clients)
    if local_steps <= 0:
        raise ValueError("local_steps must be > 0")
    if num_client == 0:
        raise ValueError("clients cannot be empty")

    eps = 1e-12
    total_server_computing = float(server.total_computing)
    if total_server_computing <= 0:
        raise ValueError("server.total_computing must be > 0")

    def _compute_ab(client, split_point: int):
        client_forward = forward_flops_by_splitpoint_list_self[split_point] / client.computing
        client_backward = backward_flops_by_splitpoint_list_self[split_point] / client.computing

        if split_point == max_split_point:
            activation_upload = 0.0
            gradient_download = 0.0
            server_forward_flops = 0.0
            server_backward_flops = 0.0
        else:
            activation_upload, gradient_download = _activation_transfer_time_by_split(
                split_point=split_point,
                activate_shape_list=activate_shape_list,
                up_rate=client.up_rate,
                down_rate=client.down_rate,
            )
            server_forward_flops = float(forward_flops_by_splitpoint_list_helper[split_point])
            server_backward_flops = float(backward_flops_by_splitpoint_list_helper[split_point])

        a_term = client_forward + activation_upload + gradient_download + client_backward
        b_term = server_forward_flops + server_backward_flops
        detail = {
            "client_forward": client_forward,
            "activation_upload": activation_upload,
            "gradient_download": gradient_download,
            "client_backward": client_backward,
            "server_forward_flops": server_forward_flops,
            "server_backward_flops": server_backward_flops,
        }
        return a_term, b_term, detail

    def _objective_by_alloc(a_terms, b_terms, alloc):
        value = 0.0
        for i in range(num_client):
            c_i = max(alloc[i], eps)
            one_step = a_terms[i] + b_terms[i] / c_i
            round_t = local_steps * one_step
            value += round_t * round_t
        return value

    def _projected_server_alloc(a_terms, b_terms, init_alloc=None):
        if init_alloc is None:
            p = [1.0 / num_client for _ in range(num_client)]
        else:
            alloc = [max(float(x), eps) for x in init_alloc]
            s = sum(alloc)
            alloc = [x * total_server_computing / s for x in alloc]
            p = [x / total_server_computing for x in alloc]

        def p_to_alloc(p_vec):
            return [max(x, eps) * total_server_computing for x in p_vec]

        alloc = p_to_alloc(p)
        obj = _objective_by_alloc(a_terms, b_terms, alloc)

        for _ in range(400):
            c_alloc = p_to_alloc(p)

            grad_c = []
            for i in range(num_client):
                c_i = max(c_alloc[i], eps)
                a_i = a_terms[i]
                b_i = b_terms[i]
                if b_i <= 0:
                    grad_c.append(0.0)
                    continue
                one_step = a_i + b_i / c_i
                grad_c.append(2.0 * (local_steps ** 2) * one_step * (-b_i / (c_i ** 2)))

            grad_p = [total_server_computing * g for g in grad_c]
            grad_scale = sum(abs(g) for g in grad_p)
            if grad_scale < 1e-12:
                break

            step = 0.05
            accepted = False
            for _ in range(20):
                p_candidate = [max(p[i] - step * grad_p[i], eps) for i in range(num_client)]
                s = sum(p_candidate)
                p_candidate = [x / s for x in p_candidate]
                alloc_candidate = p_to_alloc(p_candidate)
                obj_candidate = _objective_by_alloc(a_terms, b_terms, alloc_candidate)

                if obj_candidate <= obj:
                    delta = sum(abs(p_candidate[i] - p[i]) for i in range(num_client))
                    p = p_candidate
                    obj = obj_candidate
                    accepted = True
                    if delta < 1e-10:
                        return p_to_alloc(p)
                    break
                step *= 0.5

            if not accepted:
                break

        return p_to_alloc(p)

    def _build_outputs(split_points, server_alloc):
        one_step_times = [0.0 for _ in range(num_client)]
        round_times = [0.0 for _ in range(num_client)]
        details = []
        for i, client in enumerate(clients):
            split_point = split_points[i]
            a_i, b_i, detail = _compute_ab(client, split_point)
            c_i = max(server_alloc[i], eps)
            server_forward = detail["server_forward_flops"] / c_i if detail["server_forward_flops"] > 0 else 0.0
            server_backward = detail["server_backward_flops"] / c_i if detail["server_backward_flops"] > 0 else 0.0
            one_step = a_i + b_i / c_i
            round_t = local_steps * one_step

            detail.update(
                {
                    "server_forward": server_forward,
                    "server_backward": server_backward,
                    "server_alloc": c_i,
                    "one_step": one_step,
                    "round": round_t,
                }
            )
            details.append(detail)
            one_step_times[i] = one_step
            round_times[i] = round_t

        objective = sum(t * t for t in round_times)
        return one_step_times, round_times, details, objective

    if optimize_method != "alternating":
        raise ValueError(f"Unsupported optimize_method: {optimize_method}")

    if valid_split_points is None:
        candidate_split_points = list(range(1, max_split_point + 1))
    else:
        candidate_split_points = sorted(
            {int(x) for x in valid_split_points if 1 <= int(x) <= max_split_point}
        )
        if not candidate_split_points:
            raise ValueError("valid_split_points must contain at least one value in [1, max_split_point]")

    split_points = [max_split_point for _ in range(num_client)]
    server_alloc = [total_server_computing / num_client for _ in range(num_client)]

    prev_obj = float("inf")
    for _ in range(max(1, alt_max_iter)):
        # Step A: fix server alloc, optimize split for each client
        for i, client in enumerate(clients):
            c_i = max(server_alloc[i], eps)
            best_split = split_points[i]
            best_one_step = float("inf")
            for split_point in candidate_split_points:
                a_i, b_i, _ = _compute_ab(client, split_point)
                one_step = a_i + b_i / c_i
                if one_step < best_one_step:
                    best_one_step = one_step
                    best_split = split_point
            split_points[i] = best_split

        # Step B: fix split, optimize server alloc
        a_terms = []
        b_terms = []
        for i, client in enumerate(clients):
            a_i, b_i, _ = _compute_ab(client, split_points[i])
            a_terms.append(a_i)
            b_terms.append(b_i)
        server_alloc = _projected_server_alloc(a_terms, b_terms, init_alloc=server_alloc)

        _, round_times_tmp, _, obj = _build_outputs(split_points, server_alloc)
        if abs(prev_obj - obj) < alt_tol:
            break
        prev_obj = obj

    one_step_times, round_times, details, objective = _build_outputs(split_points, server_alloc)
    equal_alloc = [total_server_computing / num_client for _ in range(num_client)]
    _, _, _, equal_objective = _build_outputs(split_points, equal_alloc)
    if equal_objective > 0:
        improve_pct = (equal_objective - objective) / equal_objective * 100.0
    else:
        improve_pct = 0.0

    if debug:
        print("\n" + "=" * 80)
        print("DEBUG: FSL分割点+服务器算力联合优化结果")
        print("=" * 80)
        print(f"optimize_method={optimize_method}")
        print(f"server_total_computing={server.total_computing:.2e}")
        print(f"objective_equal_share={equal_objective:.6f}")
        print(f"objective(sum round_time^2)={objective:.6f}")
        print(f"improvement_vs_equal_share={improve_pct:.2f}%")
        print(f"candidate_split_points={candidate_split_points}")
        for client_idx in range(num_client):
            d = details[client_idx]
            print(
                f"client {client_idx}: split_point={split_points[client_idx]}, "
                f"server_alloc={d['server_alloc']:.2e}, "
                f"one_step={one_step_times[client_idx]:.6f}s, round={round_times[client_idx]:.6f}s"
            )
            print(
                f"  client_fwd={d['client_forward']:.6f}s, act_up={d['activation_upload']:.6f}s, "
                f"server_fwd={d['server_forward']:.6f}s, server_bwd={d['server_backward']:.6f}s, "
                f"grad_down={d['gradient_download']:.6f}s, client_bwd={d['client_backward']:.6f}s"
            )

    return {
        "split_points": split_points,
        "server_allocations": server_alloc,
        "one_step_times": one_step_times,
        "round_times": round_times,
        "details": details,
        "objective": objective,
        "optimize_method": optimize_method,
    }