import time
from copy import deepcopy

import torch
import torch.nn as nn

from config import training as training_config
from utils.time.time_ctrl import set_client_LBFSL

from .client import RingClientMeta, RingSFLClient
from .offloader import RingTaskOffloader
from .trainer import RingSFLTrainer


class RingSFL:
    def __init__(self, train_ctx):
        ctx_keys = (
            "global_model", "client_models", "criterions", "optimizers", "schedulers",
            "global_criterion", "global_optimizer", "global_scheduler",
            "dataloaders", "valloader", "completeloader",
            "device", "global_epoch", "local_epoch", "benchmark", "bs", "debug"
        )
        for key in ctx_keys:
            setattr(self, key, getattr(train_ctx, key))

        private_data = getattr(train_ctx, "private_data", {}) or {}
        algo_private_data = private_data.get(self.__class__.__name__, private_data)
        kwargs = algo_private_data if isinstance(algo_private_data, dict) else {}

        self.global_model = self.global_model.to(self.device)
        self.client_models = [m.to(self.device) for m in self.client_models]
        self.num_client = len(self.client_models)
        self.model_bits = int(sum(param.numel() for param in self.global_model.state_dict().values()) * 32)
        self.round_model_bits = []
        self.round_activation_bits = []

        self.ring_size = int(kwargs.get("ring_size", 5))

        self._disable_inplace_relu(self.global_model)
        for m in self.client_models:
            self._disable_inplace_relu(m)

        self.layer_param_names, self.conv_layer_count = self._get_layer_param_names(self.global_model)
        self.num_blocks = len(self.layer_param_names)

        self.client_system = self._build_client_system()
        self.ring_clients = self._build_ring_clients()

        client_compute = {cid: self.ring_clients[cid].meta.computing for cid in range(self.num_client)}
        self.offloader = RingTaskOffloader(client_compute=client_compute, ring_size=self.ring_size)
        self.ring_size = self.offloader.ring_size
        self.trainer = RingSFLTrainer(
            ring_clients=self.ring_clients,
            layer_param_names=self.layer_param_names,
            conv_layer_count=self.conv_layer_count,
        )

        for cid in range(self.num_client):
            meta = self.ring_clients[cid].meta
            print(
                f"[RingSFL] client {cid}: computing={meta.computing:.2e}, "
                f"up_rate={meta.up_rate:.2e}, down_rate={meta.down_rate:.2e}"
            )

    def run(self):
        loss_lst = []
        acc1_lst = []
        acc5_lst = []
        time_lst = [0.0]
        clients_time_lst = [[0.0] for _ in range(self.num_client)]

        all_client_ids = list(range(self.num_client))

        for epoch in range(self.global_epoch):
            start_t = time.time()

            selected = self.offloader.select_clients_for_round(all_client_ids)
            if self.debug:
                selected_set = set(selected)
                left_pool = [cid for cid in all_client_ids if cid not in selected_set]
                print(
                    f"[RingSFL][Debug] epoch={epoch} selected={selected} "
                    f"left_in_pool={left_pool} ring_size={self.ring_size}"
                )
            if len(selected) < self.ring_size:
                acc1, acc5 = self.validate()
                loss_lst.append(0.0)
                acc1_lst.append(acc1)
                acc5_lst.append(acc5)
                time_lst.append(time_lst[-1])
                for cid in range(self.num_client):
                    clients_time_lst[cid].append(clients_time_lst[cid][-1])
                print(f"[RingSFL] epoch={epoch} skip: selected_clients={len(selected)} < ring_size={self.ring_size}")
                continue

            self._sync_selected_with_global(selected)

            rings = self.offloader.build_rings(selected)
            ring_plans = []
            for ring_id, ring in enumerate(rings):
                prop_lengths = self.offloader.assign_ring_propagation_lengths(ring=ring, num_blocks=self.num_blocks)
                self._normalize_lengths_to_num_blocks(prop_lengths)
                ring_plans.append(
                    {
                        "ring_id": ring_id,
                        "ring_clients": ring,
                        "propagation_lengths": prop_lengths,
                    }
                )
                for cid in ring:
                    self.ring_clients[cid].ring_id = ring_id
                    self.ring_clients[cid].propagation_length = int(prop_lengths[cid])

                print(
                    f"[RingSFL] epoch={epoch} ring={ring_id} members={ring} "
                    f"prop_lengths={{{', '.join([str(k)+':'+str(v) for k,v in prop_lengths.items()])}}}"
                )

            train_info = self.trainer.train_round(
                ring_plans=ring_plans,
                local_epoch=self.local_epoch,
                debug=self.debug,
            )
            participants = train_info["participants"]
            avg_loss = float(train_info["avg_loss"])

            if self.debug:
                self._print_debug_ring_stats(epoch=epoch, ring_plans=ring_plans, train_info=train_info)

            if participants:
                self.aggregate_selected_clients(participants)
            self.broadcast_global_to_all_clients()

            model_bits_this_round = len(selected) * self.model_bits + self.num_client * self.model_bits
            activation_bits_this_round = int(train_info.get("total_activation_bits", 0))
            self.round_model_bits.append(int(model_bits_this_round))
            self.round_activation_bits.append(int(activation_bits_this_round))

            acc1, acc5 = self.validate()

            elapsed = time.time() - start_t
            epoch_end_time = time_lst[-1] + elapsed
            time_lst.append(epoch_end_time)

            participant_set = set(participants)
            for cid in range(self.num_client):
                if cid in participant_set:
                    clients_time_lst[cid].append(epoch_end_time)
                else:
                    clients_time_lst[cid].append(clients_time_lst[cid][-1])

            loss_lst.append(avg_loss)
            acc1_lst.append(acc1)
            acc5_lst.append(acc5)

            print(
                f"epoch={epoch} loss_avg={round(avg_loss,4)} acc1={round(acc1,4)} "
                f"acc5={round(acc5,4)} t={round(elapsed,1)}"
            )

        return loss_lst, acc1_lst, acc5_lst, time_lst, clients_time_lst

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

    def aggregate_selected_clients(self, selected_client_ids):
        with torch.no_grad():
            selected = list(selected_client_ids)
            if not selected:
                return

            data_sizes = {cid: len(self.dataloaders[cid].dataset) for cid in selected}
            total = float(sum(data_sizes.values()))
            if total <= 0:
                total = float(len(selected))
                data_sizes = {cid: 1.0 for cid in selected}

            state_dicts = {cid: self.client_models[cid].state_dict() for cid in selected}
            new_global = {}
            for name, param in self.global_model.state_dict().items():
                weighted = None
                for cid in selected:
                    w = float(data_sizes[cid]) / total
                    contrib = state_dicts[cid][name].to(self.device) * w
                    weighted = contrib if weighted is None else weighted + contrib
                new_global[name] = weighted if weighted is not None else param

            self.global_model.load_state_dict(new_global)

    def broadcast_global_to_all_clients(self):
        global_state = self.global_model.state_dict()
        for cid in range(self.num_client):
            self.client_models[cid].load_state_dict(global_state)

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

    def _build_ring_clients(self):
        ring_clients = {}
        for cid in range(self.num_client):
            sys_client = self.client_system[cid]
            ring_clients[cid] = RingSFLClient(
                client_id=cid,
                model=self.client_models[cid],
                dataloader=self.dataloaders[cid],
                criterion=self.criterions[cid],
                optimizer=self.optimizers[cid],
                scheduler=self.schedulers[cid],
                device=self.device,
                meta=RingClientMeta(
                    client_id=cid,
                    computing=float(sys_client.computing),
                    up_rate=float(sys_client.up_rate),
                    down_rate=float(sys_client.down_rate),
                ),
            )
        return ring_clients

    def _build_client_system(self):
        data_sizes = [len(loader.dataset) for loader in self.dataloaders]
        max_split = self._get_max_split_point(self.benchmark)
        zeros = [0 for _ in range(max_split + 1)]
        return set_client_LBFSL(
            num_client=self.num_client,
            data_sizes=data_sizes,
            local_epoch=self.local_epoch,
            minibatch=self.bs,
            benchmark=self.benchmark,
            forward_flops_by_splitpoint_list_self=zeros,
            backward_flops_by_splitpoint_list_self=zeros,
            forward_flops_by_splitpoint_list_helper=zeros,
            backward_flops_by_splitpoint_list_helper=zeros,
            parameter_bits_by_splitpoint_list_self=zeros,
            parameter_bits_by_splitpoint_list_helper=zeros,
            split_point=max_split,
        )

    @staticmethod
    def _disable_inplace_relu(model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    @staticmethod
    def _get_layer_param_names(model):
        if not hasattr(model, "get_conv_layers_list"):
            raise ValueError("Model must provide get_conv_layers_list/get_fc_layers_list for RingSFL.")

        conv_layers = model.get_conv_layers_list()
        fc_layers = model.get_fc_layers_list()
        layers = conv_layers + fc_layers

        name_by_id = {id(p): name for name, p in model.named_parameters()}
        layer_param_names = []
        for layer in layers:
            names = []
            for p in layer.parameters():
                name = name_by_id.get(id(p))
                if name:
                    names.append(name)
            layer_param_names.append(names)

        return layer_param_names, len(conv_layers)

    def _sync_selected_with_global(self, selected_client_ids):
        global_state = deepcopy(self.global_model.state_dict())
        for cid in selected_client_ids:
            self.client_models[cid].load_state_dict(global_state)

    def _normalize_lengths_to_num_blocks(self, prop_lengths):
        if not prop_lengths:
            return
        keys = list(prop_lengths.keys())
        values = [max(1, int(prop_lengths[k])) for k in keys]
        total = sum(values)

        if total > self.num_blocks:
            need_remove = total - self.num_blocks
            idx = 0
            while need_remove > 0:
                k = keys[idx % len(keys)]
                if prop_lengths[k] > 1:
                    prop_lengths[k] -= 1
                    need_remove -= 1
                idx += 1
        elif total < self.num_blocks:
            need_add = self.num_blocks - total
            idx = 0
            while need_add > 0:
                k = keys[idx % len(keys)]
                prop_lengths[k] += 1
                need_add -= 1
                idx += 1

    @staticmethod
    def _get_max_split_point(benchmark: str) -> int:
        if "vgg11" in benchmark:
            return training_config.VGG11_MAX_SPLIT_POINT
        if "vgg16" in benchmark:
            return training_config.VGG16_MAX_SPLIT_POINT
        if "resnet18" in benchmark:
            return training_config.RESNET18_MAX_SPLIT_POINT
        return training_config.ALEXNET_MAX_SPLIT_POINT

    def _print_debug_ring_stats(self, epoch: int, ring_plans: list, train_info: dict):
        ring_debug = train_info.get("ring_debug", [])
        print(f"[RingSFL][Debug] epoch={epoch} local_epoch={self.local_epoch} num_rings={len(ring_plans)}")

        for ring_info in ring_debug:
            ring_id = ring_info["ring_id"]
            members = ring_info["members"]
            prop_lengths = ring_info["propagation_lengths"]
            client_stats = ring_info["client_stats"]

            print(
                f"[RingSFL][Debug] ring={ring_id} members={members} "
                f"propagation_lengths={prop_lengths}"
            )

            for cid in members:
                stat = client_stats[cid]
                total_time = stat["forward_time"] + stat["backward_time"] + stat["step_time"]
                print(
                    f"[RingSFL][Debug] ring={ring_id} client={cid} "
                    f"source_pass={stat['source_pass_count']} owner_seg={stat['owner_segment_count']} "
                    f"zero_grad={stat['zero_grad_count']} step={stat['optimizer_step_count']} "
                    f"act_tx/rx={stat['activation_send_count']}/{stat['activation_recv_count']} "
                    f"grad_tx/rx={stat['grad_send_count']}/{stat['grad_recv_count']} "
                    f"time(fwd/bwd/step/total)="
                    f"{stat['forward_time']:.4f}/{stat['backward_time']:.4f}/{stat['step_time']:.4f}/{total_time:.4f}s"
                )
