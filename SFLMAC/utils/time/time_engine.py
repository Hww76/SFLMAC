from dataclasses import dataclass


@dataclass
class TimeDebugTrainingInfo:
    forward_flops_list: list
    backward_flops_list: list
    parameter_bits_list: list
    activate_shape_list: list
    forward_flops_by_splitpoint_list_self: list
    backward_flops_by_splitpoint_list_self: list
    parameter_bits_by_splitpoint_list_self: list
    total_forward_flops: int
    total_backward_flops: int
    total_parameter_bits: int


@dataclass
class TimeDebugClientInfo:
    client_idx: int
    client: object


def print_debug_time_info(training_info: TimeDebugTrainingInfo, client_infos, logger=print):
    logger("\n" + "=" * 80)
    logger("DEBUG: 时间计算相关信息")
    logger("=" * 80)

    logger("\n模型各层负载信息:")
    logger("-" * 80)
    logger(f"{'层号':<5} {'前向FLOPs':<15} {'反向FLOPs':<15} {'参数量(bits)':<15} {'激活值形状':<30}")
    logger("-" * 80)
    for i in range(len(training_info.forward_flops_list)):
        activate_shape = str(training_info.activate_shape_list[i]) if i < len(training_info.activate_shape_list) else "N/A"
        logger(
            f"{i:<5} {training_info.forward_flops_list[i]:<15} "
            f"{training_info.backward_flops_list[i]:<15} "
            f"{training_info.parameter_bits_list[i]:<15} {activate_shape:<30}"
        )

    logger(f"总前向FLOPs: {training_info.total_forward_flops}")
    logger(f"总反向FLOPs: {training_info.total_backward_flops}")
    logger(f"总参数量(bits): {training_info.total_parameter_bits}")

    logger("\n\n客户端信息和时间计算:")
    logger("-" * 80)
    for client in client_infos:
        c = client.client
        split_point = c.split_point
        breakdown = c.compute_load_breakdown_by_splitpoint(
            split_point=split_point,
            forward_flops_by_splitpoint_self=training_info.forward_flops_by_splitpoint_list_self,
            backward_flops_by_splitpoint_self=training_info.backward_flops_by_splitpoint_list_self,
            parameter_bits_by_splitpoint_self=training_info.parameter_bits_by_splitpoint_list_self,
            activate_shape_list=training_info.activate_shape_list,
        )

        total_load_time = breakdown["total_load_time"]
        model_dl_time = breakdown["model_dl_time"]
        forward_time = breakdown["forward_time"]
        backward_time = breakdown["backward_time"]
        activation_time = breakdown["activation_time"]
        model_ul_time = breakdown["model_ul_time"]

        logger(f"\n客户端 {client.client_idx}:")
        logger(f"  数据大小: {c.data_sizes}")
        logger(f"  本地轮数: {c.local_epoch}")
        logger(f"  小批量大小: {c.minibatch}")
        logger(f"  总批次数: {c.batch_total_cnt}")
        logger(f"  分割点: {split_point}")
        logger(f"  计算能力(FLOPs): {c.computing:.2e}")
        logger(f"  上传速率: {c.up_rate:.2e}")
        logger(f"  下载速率: {c.down_rate:.2e}")
        logger(f"  总负载时间(分割点{split_point}): {total_load_time:.2f}秒")
        logger(f"    - 模型下载: {model_dl_time:.4f}秒")
        logger(f"    - 前向传播: {forward_time:.4f}秒")
        logger(f"    - 反向传播: {backward_time:.4f}秒")
        logger(f"    - 激活值传输: {activation_time:.4f}秒")
        logger(f"    - 模型上传: {model_ul_time:.4f}秒")

    logger("\n" + "=" * 80 + "\n")


class TimeEngine:
    """Reusable round-level time simulator for federated training.

    The engine records:
    - per-client cumulative finish time list (`clients_time_lst`)
    - global cumulative time list by round max (`time_lst`)
    """

    def __init__(
        self,
        num_client: int,
        initial_time: float = 0.0,
        debug: bool = False,
        name: str = "TimeEngine",
        logger=None,
    ):
        self.num_client = num_client
        self.global_time = float(initial_time)
        self.time_lst = [self.global_time]
        self.clients_time_lst = [[self.global_time] for _ in range(num_client)]
        self.debug = debug
        self.name = name
        self._logger = logger if logger is not None else print

        if self.debug:
            self._log(
                f"init: num_client={self.num_client}, initial_time={self.global_time:.6f}"
            )

    def _log(self, msg: str):
        self._logger(f"[{self.name}] {msg}")

    def record_round(self, client_round_durations):
        """Record one global round by client simulated durations.

        Args:
            client_round_durations: iterable with length == num_client,
                each item is the simulated duration of one client in this round.
        Returns:
            epoch_max_time: max cumulative finish time in this round.
            client_end_times: per-client cumulative finish time in this round.
        """
        if len(client_round_durations) != self.num_client:
            raise ValueError(
                f"client_round_durations length ({len(client_round_durations)}) "
                f"!= num_client ({self.num_client})"
            )

        for idx, duration in enumerate(client_round_durations):
            if float(duration) < 0:
                raise ValueError(
                    f"client_round_durations[{idx}] is negative ({duration}), "
                    "which is invalid for simulated time."
                )

        client_end_times = [self.global_time + float(duration) for duration in client_round_durations]
        for client_idx, end_time in enumerate(client_end_times):
            self.clients_time_lst[client_idx].append(end_time)

        epoch_max_time = max(client_end_times)
        self.global_time = epoch_max_time
        self.time_lst.append(self.global_time)

        if self.debug:
            durations_fmt = ", ".join([f"c{idx}={float(d):.6f}" for idx, d in enumerate(client_round_durations)])
            self._log(
                "round: "
                f"durations[{durations_fmt}] -> epoch_max={epoch_max_time:.6f}, "
                f"global_cumulative={self.global_time:.6f}"
            )

        return epoch_max_time, client_end_times
