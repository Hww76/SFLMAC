import numpy as np
import math
from config import training as training_config

class Client_FedAvg:
    """简单的联邦客户端定义，通信时间估算。"""
    def __init__(self, data_sizes, local_epoch, minibatch=0, flops=69833728, parameter_bits = 123799872,
                  computing=0, rate=0, time=0):
        # 数据相关
        self.data_sizes = data_sizes
        self.local_epoch = local_epoch
        self.minibatch = minibatch
        
        # 时间统计相关
        self.computing = computing # 客户端计算能力
        self.flops = flops  # 客户端负载
        self.parameter_bits = parameter_bits  # 客户端参数量
        self.rate = rate
        self.time = time

        self.batch_total_cnt = ((self.data_sizes + self.minibatch - 1) // self.minibatch ) * self.local_epoch  # 计算总批次数
    
    # 计算前向传播
    def compute_forward_time(self):
        pass

    # 计算反向传播
    def compute_backward_time(self):
        pass

    # 计算激活值上传
    def compute_activation_upload_time(self):
        pass

    # 计算梯度下载
    def compute_gradient_download_time(self):
        pass

    # 计算模型下载
    def compute_model_download_time(self):
        pass

    # 计算模型上传
    def compute_model_upload_time(self):
        pass

    # 计算总负载
    def compute_total_time(self):
        return self.compute_model_download_time() + self.compute_forward_time() + self.compute_activation_upload_time() +\
               self.compute_gradient_download_time() + self.compute_backward_time() + self.compute_model_upload_time()

class Client_LBFSL:
    """半异步分割联邦客户端定义，包含本地迭代计数与通信时间估算。"""
    def __init__(self, data_sizes, local_epoch, minibatch=0, forward_flops=69833728, backward_flops=139667456,
                 parameter_bits = 123799872, computing=0, up_rate=0, down_rate=0, time=0,
                 split_point=training_config.ALEXNET_MAX_SPLIT_POINT,
                 helper_client_id=None, activate_shape=None):
        # 数据相关
        self.data_sizes = data_sizes
        self.local_epoch = local_epoch
        self.minibatch = minibatch

        # 模型分割相关
        self.split_point = split_point # 当分割点等于ALEXNET_MAX_SPLIT_POINT时，表示不进行分割
        self.helper_client_id = helper_client_id  # 分割后的上层模型由哪个客户端承担（upper）
        self.activate_shape = activate_shape  # 分割点激活值形状
        
        # 时间统计相关
        self.computing = computing # 客户端计算能力
        self.forward_flops = forward_flops  # 客户端前向传播负载
        self.backward_flops = backward_flops  # 客户端反向传播负载
        self.parameter_bits = parameter_bits  # 客户端参数量
        self.up_rate = up_rate # 上传速率
        self.down_rate = down_rate # 下载速率
        self.time = time

        self.batch_total_cnt = self.local_epoch  # 计算总批次数
    
    # 计算前向传播
    def compute_forward_time(self):
        return self.forward_flops / self.computing

    # 计算反向传播
    def compute_backward_time(self):
        return self.backward_flops / self.computing
    
    # 计算激活值上传
    def compute_activation_upload_time(self):
        if self.activate_shape is not None:
            return (self.activate_shape[0] * self.activate_shape[1] * self.activate_shape[2] * self.activate_shape[3] * 32) / self.up_rate  # 激活值为32位浮点数
        else:
            return 0

    # 计算梯度下载
    def compute_gradient_download_time(self):
        if self.activate_shape is not None:
            return (self.activate_shape[0] * self.activate_shape[1] * self.activate_shape[2] * self.activate_shape[3] * 32) / self.down_rate  # 激活值为32位浮点数
        else:
            return 0
    # 计算模型下载
    def compute_model_download_time(self):
        return self.parameter_bits / self.down_rate

    # 计算模型上传
    def compute_model_upload_time(self):
        return self.parameter_bits / self.up_rate

    def compute_model_download_time_by_splitpoint(self, split_point: int, parameter_bits_by_splitpoint: list):
        return parameter_bits_by_splitpoint[split_point] / self.down_rate

    def compute_model_upload_time_by_splitpoint(self, split_point: int, parameter_bits_by_splitpoint: list):
        return parameter_bits_by_splitpoint[split_point] / self.up_rate

    def compute_forward_time_by_splitpoint(self, split_point: int, forward_flops_by_splitpoint: list, batch_cnt: int = None):
        if batch_cnt is None:
            batch_cnt = self.compute_batch_count()
        return (forward_flops_by_splitpoint[split_point] * batch_cnt) / self.computing

    def compute_backward_time_by_splitpoint(self, split_point: int, backward_flops_by_splitpoint: list, batch_cnt: int = None):
        if batch_cnt is None:
            batch_cnt = self.compute_batch_count()
        return (backward_flops_by_splitpoint[split_point] * batch_cnt) / self.computing

    def compute_activation_upload_time_by_splitpoint(self, split_point: int, activate_shape_list: list, batch_cnt: int = None):
        if batch_cnt is None:
            batch_cnt = self.compute_batch_count()

        if split_point == training_config.ALEXNET_MAX_SPLIT_POINT:
            return 0

        activate_shape = activate_shape_list[split_point]
        activate_bit_size = activate_shape[0] * activate_shape[1] * activate_shape[2] * activate_shape[3] * 32
        activate_bit_size *= batch_cnt
        return activate_bit_size / self.down_rate if activate_bit_size > 0 else 0

    def compute_load_breakdown_by_splitpoint(
        self,
        split_point: int,
        forward_flops_by_splitpoint_self: list,
        backward_flops_by_splitpoint_self: list,
        parameter_bits_by_splitpoint_self: list,
        activate_shape_list: list,
    ):
        model_dl_time = self.compute_model_download_time_by_splitpoint(split_point, parameter_bits_by_splitpoint_self)
        forward_time = self.compute_forward_time_by_splitpoint(split_point, forward_flops_by_splitpoint_self, self.batch_total_cnt)
        activation_time = self.compute_activation_upload_time_by_splitpoint(split_point, activate_shape_list, self.batch_total_cnt)
        backward_time = self.compute_backward_time_by_splitpoint(split_point, backward_flops_by_splitpoint_self, self.batch_total_cnt)
        model_ul_time = self.compute_model_upload_time_by_splitpoint(split_point, parameter_bits_by_splitpoint_self)
        total_load_time = model_dl_time + forward_time + activation_time + backward_time + model_ul_time

        return {
            "model_dl_time": model_dl_time,
            "forward_time": forward_time,
            "activation_time": activation_time,
            "backward_time": backward_time,
            "model_ul_time": model_ul_time,
            "total_load_time": total_load_time,
        }

    def calculate_loads_by_splitpoint(
        self,
        split_point: int,
        forward_flops_by_splitpoint_self: list,
        backward_flops_by_splitpoint_self: list,
        parameter_bits_by_splitpoint_self: list,
        activate_shape_list: list,
    ):
        return self.compute_load_breakdown_by_splitpoint(
            split_point,
            forward_flops_by_splitpoint_self,
            backward_flops_by_splitpoint_self,
            parameter_bits_by_splitpoint_self,
            activate_shape_list,
        )["total_load_time"]

    def calculate_auxiliary_training_load(
        self,
        split_point: int,
        forward_flops_by_splitpoint_helper: list,
        backward_flops_by_splitpoint_helper: list,
        parameter_bits_by_splitpoint_helper: list,
        activate_shape_list: list,
    ):
        batch_cnt = self.compute_batch_count()
        forward_time = self.compute_forward_time_by_splitpoint(split_point, forward_flops_by_splitpoint_helper, batch_cnt)
        backward_time = self.compute_backward_time_by_splitpoint(split_point, backward_flops_by_splitpoint_helper, batch_cnt)
        activation_time = self.compute_activation_upload_time_by_splitpoint(split_point, activate_shape_list, batch_cnt)
        model_ul_time = self.compute_model_upload_time_by_splitpoint(split_point, parameter_bits_by_splitpoint_helper)
        return forward_time + backward_time + activation_time + model_ul_time

    # 计算总负载
    def compute_total_time(self):
        return self.compute_model_download_time() + self.batch_total_cnt * (self.compute_forward_time() + self.compute_activation_upload_time() +\
               self.compute_gradient_download_time() + self.compute_backward_time()) + self.compute_model_upload_time()


class FSL_Server:
    """FSL 服务器资源模型。

    - 总算力固定为 5e8 FLOPs
    - 按客户端数量均分
    - 客户端视角下每个客户端独占其均分后的服务器算力
    """

    def __init__(self, num_client: int, total_computing: float = 5e8):
        if num_client <= 0:
            raise ValueError("num_client must be > 0")
        if total_computing <= 0:
            raise ValueError("total_computing must be > 0")
        self.num_client = int(num_client)
        self.total_computing = float(total_computing)

    @property
    def per_client_computing(self) -> float:
        return self.total_computing / self.num_client

    def get_dedicated_computing(self, client_id: int) -> float:
        if client_id < 0 or client_id >= self.num_client:
            raise IndexError(f"client_id out of range: {client_id}")
        return self.per_client_computing
