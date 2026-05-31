import torch
import time
import sys

import numpy as np
from copy import deepcopy

from config.client import Client_LBFSL
from utils.time.time_ctrl import set_client_LBFSL
from utils.time.time_engine import TimeDebugTrainingInfo, TimeDebugClientInfo, print_debug_time_info
from utils.time.load_calculate import (
    calculate_model_loads_by_layer,
)

class FedAvg:
    # 初始化联邦学习环境
    # 参数: global_model(全局模型), client_models(客户端模型列表), criterions(损失函数列表)
    #      optimizers(优化器列表), schedulers(学习率调度器列表)
    #      global_criterion(全局损失函数), global_optimizer(全局优化器), global_scheduler(全局调度器)
    #      dataloaders(客户端数据加载器列表), valloader(验证数据加载器), completeloader(完整训练集加载器)
    #      device(设备类型), global_epoch(全局轮数), local_epoch(本地轮数)
    #      beta(动量系数), nlp(是否是NLP任务)
    #      benchmark(基准名称), bs(批次大小), alpha(数据分布参数)
    def __init__(self, train_ctx):
        ctx_keys = (
            "global_model", "client_models", "criterions", "optimizers", "schedulers",
            "global_criterion", "global_optimizer", "global_scheduler",
            "dataloaders", "valloader", "completeloader",
            "device", "global_epoch", "local_epoch", "beta", "benchmark", "bs", "alpha", "debug"
        )
        for key in ctx_keys:
            setattr(self, key, getattr(train_ctx, key))

        private_data = getattr(train_ctx, "private_data", {}) or {}
        algo_private_data = private_data.get(self.__class__.__name__, private_data)
        self.kwargs = algo_private_data if isinstance(algo_private_data, dict) else {}

        self.num_client = len(self.client_models)
        self.model_bits = int(sum(param.numel() for param in self.global_model.state_dict().values()) * 32)
        self.round_model_bits = []
        self.round_activation_bits = []

        self.names = self.global_model.state_dict().keys()
        self.tilde_gradients = {name: torch.zeros_like(param).to(self.device) for name, param in self.global_model.state_dict().items()}

        # Persistent dataloader iterators for one-mini-batch per local epoch.
        self._dl_iters = {i: iter(self.dataloaders[i]) for i in range(self.num_client)}

        # 时间模拟设置
        # 计算模型负载信息
        self.forward_flops_list, self.backward_flops_list, self.parameter_bits_list, self.activate_shape_list \
        = calculate_model_loads_by_layer(benchmark=self.benchmark, batch_size=self.bs)

        self.forward_flops_by_splitpoint_list_self = [0] # 各分割点自己前向FLOPs累计
        self.backward_flops_by_splitpoint_list_self = [0] # 各分割点自己反向FLOPs累计
        self.forward_flops_by_splitpoint_list_helper = [0] # 各分割点助手前向FLOPs累计
        self.backward_flops_by_splitpoint_list_helper = [0] # 各分割点助手反向FLOPs累计
        self.parameter_bits_by_splitpoint_list_self = [0] # 各分割点自己参数量累计
        self.parameter_bits_by_splitpoint_list_helper = [0] # 各分割点助手
        for i in range(1,len(self.forward_flops_list)+1):
            self.forward_flops_by_splitpoint_list_self.append(self.forward_flops_by_splitpoint_list_self[i-1] + self.forward_flops_list[i-1])
            self.backward_flops_by_splitpoint_list_self.append(self.backward_flops_by_splitpoint_list_self[i-1] + self.backward_flops_list[i-1])
            self.parameter_bits_by_splitpoint_list_self.append(self.parameter_bits_by_splitpoint_list_self[i-1] + self.parameter_bits_list[i-1])

        for i in range(1,len(self.forward_flops_by_splitpoint_list_self)):
            self.forward_flops_by_splitpoint_list_helper.append(self.forward_flops_by_splitpoint_list_self[len(self.forward_flops_by_splitpoint_list_self)-1] - self.forward_flops_by_splitpoint_list_self[i])
            self.backward_flops_by_splitpoint_list_helper.append(self.backward_flops_by_splitpoint_list_self[len(self.backward_flops_by_splitpoint_list_self)-1] - self.backward_flops_by_splitpoint_list_self[i])
            self.parameter_bits_by_splitpoint_list_helper.append(self.parameter_bits_by_splitpoint_list_self[len(self.parameter_bits_by_splitpoint_list_self)-1] - self.parameter_bits_by_splitpoint_list_self[i])

        # 客户端信息统计
        # 根据数据加载器计算每个客户端的数据大小
        data_sizes = [len(loader.dataset) for loader in self.dataloaders]
        # 调用set_client_LBFSL获取客户端信息
        self.clients = set_client_LBFSL(
            num_client=self.num_client, 
            data_sizes=data_sizes, 
            local_epoch=self.local_epoch, 
            minibatch=self.bs,
            benchmark=self.benchmark,
            forward_flops_by_splitpoint_list_self=self.forward_flops_by_splitpoint_list_self,
            backward_flops_by_splitpoint_list_self=self.backward_flops_by_splitpoint_list_self,
            forward_flops_by_splitpoint_list_helper=self.forward_flops_by_splitpoint_list_helper,
            backward_flops_by_splitpoint_list_helper=self.backward_flops_by_splitpoint_list_helper,
            parameter_bits_by_splitpoint_list_self=self.parameter_bits_by_splitpoint_list_self,
            parameter_bits_by_splitpoint_list_helper=self.parameter_bits_by_splitpoint_list_helper
        )
        

 
    # 执行联邦学习的主循环
    # 返回: (loss_lst, acc1_lst, acc5_lst) - 分别为每个全局轮的平均损失和Top1、Top5准确率列表
    def run(self):
        loss_lst = []
        acc1_lst = []
        acc5_lst = []
        time_lst = [0]
        client_time_lst = [[0] for _ in range(self.num_client)]

        for epoch in range(self.global_epoch):
            start_time = time.time()
            all_client_gradients = {}
            avg_loss_lst = []
            epoch_max_time = time_lst[-1]

            # 在第一轮进行debug打印
            if epoch == 0 and self.debug:
                self._print_debug_time_info()

            for client_idx in range(self.num_client):
                gradient, avg_loss, end_time = self.train(client_idx, start_time=time_lst[-1])
                avg_loss_lst.append(avg_loss)
                client_time_lst[client_idx].append(end_time)

                for i, v in gradient.items():
                    if i in all_client_gradients:
                        all_client_gradients[i].append(v)
                    else:
                        all_client_gradients[i] = [v]
                
                epoch_max_time = max(epoch_max_time, end_time)
            
            
            self.aggregate(all_client_gradients, epoch)

            model_bits_this_round = 2 * self.num_client * self.model_bits
            activation_bits_this_round = 0
            self.round_model_bits.append(int(model_bits_this_round))
            self.round_activation_bits.append(int(activation_bits_this_round))

            loss_avg = sum(avg_loss_lst) / len(avg_loss_lst)
            acc1, acc5 = self.validate()

            loss_lst.append(loss_avg)
            acc1_lst.append(acc1)
            acc5_lst.append(acc5)
            time_lst.append(epoch_max_time)

            loss_avg = round(loss_avg, 4)
            acc1 = round(acc1, 4)
            acc5 = round(acc5, 4)
            t = round(time.time() - start_time, 1)
            print(f'{epoch=}  {loss_avg=}  {acc1=} {acc5=} {t=}')

        return loss_lst, acc1_lst, acc5_lst, time_lst, client_time_lst

    def get_communication_stats(self):
        total_model_bits = int(sum(self.round_model_bits))
        total_activation_bits = int(sum(self.round_activation_bits))
        total_bits = total_model_bits + total_activation_bits
        return {
            "round_model_bits": list(self.round_model_bits),
            "round_activation_bits": list(self.round_activation_bits),
            "total_model_bits": total_model_bits,
            "total_activation_bits": total_activation_bits,
            "total_bits": total_bits,
            "total_bytes": total_bits / 8.0,
        }

    # 训练单个客户端的本地模型
    # 参数: client_idx(客户端索引)
    # 返回: (gradient, running_loss/len(dataloader)) - 客户端模型更新的梯度和平均损失
    def train(self, client_idx, start_time=0):
        client = self.client_models[client_idx]
        dataloader = self.dataloaders[client_idx]
        criterion = self.criterions[client_idx]
        scheduler = self.schedulers[client_idx]
        optimizer = self.optimizers[client_idx]
        client.train()
        end_time = start_time

        initial_weights = deepcopy(client.state_dict())

        end_time += self.clients[client_idx].compute_model_download_time() # 下载模型用时
        for epoch in range(self.local_epoch):
            running_loss = 0.0
            try:
                data, target = next(self._dl_iters[client_idx])
            except StopIteration:
                self._dl_iters[client_idx] = iter(dataloader)
                data, target = next(self._dl_iters[client_idx])
            data, target = data.to(self.device), target.to(self.device)
            optimizer.zero_grad()
            output = client(data)
            end_time += self.clients[client_idx].compute_forward_time()  # 前向传播用时
            loss = criterion(output, target)
            loss.backward()
            end_time += self.clients[client_idx].compute_backward_time()  # 反向传播用时
            torch.nn.utils.clip_grad_norm_(parameters=client.parameters(), max_norm=10) # Gradient clipping，加了这句才正常，loss_avg不会爆炸，但没有减小，acc仍然0.1
            running_loss += loss.item()
            optimizer.step()
                
            if scheduler is not None:
                scheduler.step()
            # print(f"Client@{client_idx}: Local epoch={epoch} loss={running_loss / len(dataloader)} time={time.time()-start_time}")

        gradient = {name: client.state_dict()[name] - initial_weights[name] for name in self.names}

        end_time += self.clients[client_idx].compute_model_upload_time() # 上传模型用时

        del initial_weights
        return gradient, running_loss, end_time
    
    # 在全局验证集上评估全局模型的性能
    # 返回: (acc1, acc5) - Top1准确率和Top5准确率
    def validate(self):
        with torch.no_grad():
            self.global_model.eval()
            total, correct1, correct5 = 0, 0, 0
            for data, target in self.valloader:
                data, target = data.to(self.device), target.to(self.device)
                total += len(data)
                output = self.global_model(data)
                predict = output.argmax(dim=1)
                correct1 += torch.eq(predict, target).sum().float().item()
                target_resize = target.view(-1, 1)
                _, predict = output.topk(5)
                correct5 += torch.eq(predict, target_resize).sum().float().item()
            acc1 = correct1 / total
            acc5 = correct5 / total
            return acc1, acc5
    
    # 聚合所有客户端的梯度并更新全局模型
    # 参数: gradients(所有客户端梯度字典), epoch(当前全局轮数)
    # 返回: 无返回值，直接更新self.global_model和所有客户端模型
    def aggregate(self, gradients, epoch):
        with torch.no_grad():
            aggregate_gradient = {}
            aggregate_param = {}
            for name, param in self.global_model.state_dict().items():
                aggregate_gradient[name] = sum(gradients[name]) / self.num_client
                self.tilde_gradients[name] = aggregate_gradient[name]
                aggregate_param[name] = param.to(self.device) + aggregate_gradient[name]

            self.global_model.load_state_dict(aggregate_param)
            for i in range(self.num_client):
                self.client_models[i].load_state_dict(aggregate_param)
    
    # Debug模式下打印时间计算信息
    def _print_debug_time_info(self):
        """在第一轮全局迭代时打印时间计算相关信息"""
        training_info = TimeDebugTrainingInfo(
            forward_flops_list=self.forward_flops_list,
            backward_flops_list=self.backward_flops_list,
            parameter_bits_list=self.parameter_bits_list,
            activate_shape_list=self.activate_shape_list,
            forward_flops_by_splitpoint_list_self=self.forward_flops_by_splitpoint_list_self,
            backward_flops_by_splitpoint_list_self=self.backward_flops_by_splitpoint_list_self,
            parameter_bits_by_splitpoint_list_self=self.parameter_bits_by_splitpoint_list_self,
            total_forward_flops=sum(self.forward_flops_list),
            total_backward_flops=sum(self.backward_flops_list),
            total_parameter_bits=sum(self.parameter_bits_list),
        )

        client_infos = []
        for client_idx, client in enumerate(self.clients):
            client_infos.append(
                TimeDebugClientInfo(
                    client_idx=client_idx,
                    client=client,
                )
            )

        print_debug_time_info(training_info=training_info, client_infos=client_infos)