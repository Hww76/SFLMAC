import time
from copy import deepcopy
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from config import training as training_config
from utils.time.load_calculate import calculate_model_loads_by_layer
from utils.time.time_ctrl import set_client_LBFSL

from .strategy import build_sflmac_pairs


class SFLMAC:
    """Clean SFLMAC.

    - main/aux collaborative alternating training in each pair
    - pairing: largest computing with smallest computing
    - split point: per-pair balance by flops/computing, or shared fixed split point
    """

    def __init__(self, train_ctx):
        ctx_keys = (
            "global_model", "client_models", "criterions", "optimizers", "schedulers",
            "dataloaders", "valloader",
            "device", "global_epoch", "local_epoch", "benchmark", "debug", "bs"
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
        self.round_model_bits_by_role = []
        self.round_activation_bits = []

        self.local_epoch = int(kwargs.get("local_epoch", self.local_epoch))
        self.local_pair_epoch = 2 * self.local_epoch

        self._disable_inplace_relu(self.global_model)
        for m in self.client_models:
            self._disable_inplace_relu(m)

        (
            self.forward_flops_list,
            self.backward_flops_list,
            self.parameter_bits_list,
            self.activate_shape_list,
        ) = calculate_model_loads_by_layer(benchmark=self.benchmark, batch_size=self.bs)

        # Align split-point semantics to real model layers.
        # load_calculate may return a leading placeholder (index 0 == 0).
        real_forward_flops_list = self.forward_flops_list
        real_backward_flops_list = self.backward_flops_list
        real_parameter_bits_list = self.parameter_bits_list
        if len(self.forward_flops_list) > 1 and self.forward_flops_list[0] == 0:
            real_forward_flops_list = self.forward_flops_list[1:]
            real_backward_flops_list = self.backward_flops_list[1:]
            real_parameter_bits_list = self.parameter_bits_list[1:]

        max_split_point = min(self._get_max_split_point(self.benchmark), len(real_forward_flops_list))
        self.default_split_point = int(max(1, max_split_point // 2))
        if self.default_split_point >= max_split_point:
            self.default_split_point = max(1, max_split_point - 1)

        self.shared_split_point = kwargs.get("split_point")
        if self.shared_split_point is not None:
            self.shared_split_point = int(self.shared_split_point)
            if not (1 <= self.shared_split_point < max_split_point):
                raise ValueError("split_point must be in [1, max_split_point-1].")

        valid_split_points = None
        if "resnet18" in self.benchmark:
            valid_split_points = list(getattr(training_config, "RESNET18_VALID_SPLIT_POINTS", []))
            if self.shared_split_point is not None and valid_split_points:
                if self.shared_split_point not in valid_split_points:
                    nearest = min(
                        valid_split_points,
                        key=lambda sp: (abs(sp - self.shared_split_point), sp),
                    )
                    if self.debug:
                        print(
                            f"[SFLMAC][Debug] split_point={self.shared_split_point} is invalid for resnet18; "
                            f"auto-adjusted to nearest valid split_point={nearest}. "
                            f"allowed={valid_split_points}"
                        )
                    self.shared_split_point = nearest

        self.forward_flops_by_splitpoint_list_self = [0]
        self.backward_flops_by_splitpoint_list_self = [0]
        self.forward_flops_by_splitpoint_list_helper = [0]
        self.backward_flops_by_splitpoint_list_helper = [0]
        self.parameter_bits_by_splitpoint_list_self = [0]
        self.parameter_bits_by_splitpoint_list_helper = [0]

        for split_point in range(1, len(real_forward_flops_list) + 1):
            self.forward_flops_by_splitpoint_list_self.append(
                self.forward_flops_by_splitpoint_list_self[-1] + real_forward_flops_list[split_point - 1]
            )
            self.backward_flops_by_splitpoint_list_self.append(
                self.backward_flops_by_splitpoint_list_self[-1] + real_backward_flops_list[split_point - 1]
            )
            self.parameter_bits_by_splitpoint_list_self.append(
                self.parameter_bits_by_splitpoint_list_self[-1] + real_parameter_bits_list[split_point - 1]
            )

        for i in range(1, len(self.forward_flops_by_splitpoint_list_self)):
            self.forward_flops_by_splitpoint_list_helper.append(
                self.forward_flops_by_splitpoint_list_self[-1] - self.forward_flops_by_splitpoint_list_self[i]
            )
            self.backward_flops_by_splitpoint_list_helper.append(
                self.backward_flops_by_splitpoint_list_self[-1] - self.backward_flops_by_splitpoint_list_self[i]
            )
            self.parameter_bits_by_splitpoint_list_helper.append(
                self.parameter_bits_by_splitpoint_list_self[-1] - self.parameter_bits_by_splitpoint_list_self[i]
            )

        data_sizes = [len(loader.dataset) for loader in self.dataloaders]
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
            parameter_bits_by_splitpoint_list_helper=self.parameter_bits_by_splitpoint_list_helper,
            split_point=max_split_point,
        )

        self.pair_info = build_sflmac_pairs(
            clients=self.clients,
            max_split_point=max_split_point,
            forward_flops_by_splitpoint_list_self=self.forward_flops_by_splitpoint_list_self,
            backward_flops_by_splitpoint_list_self=self.backward_flops_by_splitpoint_list_self,
            forward_flops_by_splitpoint_list_helper=self.forward_flops_by_splitpoint_list_helper,
            backward_flops_by_splitpoint_list_helper=self.backward_flops_by_splitpoint_list_helper,
            shared_split_point=self.shared_split_point,
            debug=self.debug,
            valid_split_points=valid_split_points,
        )
        self.pair_split_points = self.pair_info.get("pair_split_points", {})
        self.main_split_points = {main_id: split for (main_id, _), split in self.pair_split_points.items()}
        self.aux_split_points = {aux_id: split for (main_id, aux_id), split in self.pair_split_points.items()}

        self._lower_state_bits_cache: Dict[int, int] = {}
        self.layer_param_names = self._get_layer_param_names(self.global_model)
        self.num_layers = len(self.layer_param_names)

        self._dl_iters: Dict[int, iter] = {i: iter(self.dataloaders[i]) for i in range(self.num_client)}

        if self.debug:
            for main_id, aux_id in self.pair_info["pairs"]:
                split = self.pair_split_points[(main_id, aux_id)]
                print(
                    f"[SFLMAC] pair(main={main_id}, aux={aux_id}), split_point={split}, "
                    f"main_compute={self.clients[main_id].computing:.2e}, aux_compute={self.clients[aux_id].computing:.2e}"
                )
            if self.pair_info["normal"]:
                print(f"[SFLMAC] normal clients: {self.pair_info['normal']}")

    def run(self):
        loss_lst = []
        acc1_lst = []
        acc5_lst = []
        time_lst = [0.0]
        clients_time_lst = [[0.0] for _ in range(self.num_client)]

        for epoch in range(self.global_epoch):
            start_time = time.time()
            running_loss = {i: 0.0 for i in range(self.num_client)}
            step_count = {i: 0 for i in range(self.num_client)}
            round_activation_bits = 0

            for main_id, aux_id in self.pair_info["pairs"]:
                pair_split_point = self.pair_split_points.get((main_id, aux_id), self.default_split_point)
                for step in range(self.local_pair_epoch):
                    if step % 2 == 0:
                        batch = self.get_minibatch(aux_id)
                        loss, activation_bits = self.split_learning_one_step(main_id, aux_id, pair_split_point, batch)
                        round_activation_bits += int(activation_bits)
                        running_loss[aux_id] += loss
                        step_count[aux_id] += 1
                    else:
                        batch = self.get_minibatch(main_id)
                        loss = self.full_model_one_step(main_id, batch)
                        running_loss[main_id] += loss
                        step_count[main_id] += 1

            for client_id in self.pair_info["normal"]:
                for _ in range(self.local_epoch):
                    batch = self.get_minibatch(client_id)
                    loss = self.full_model_one_step(client_id, batch)
                    running_loss[client_id] += loss
                    step_count[client_id] += 1

            for i in range(self.num_client):
                if self.schedulers[i] is not None:
                    self.schedulers[i].step()

            uploads = {i: deepcopy(self.client_models[i].state_dict()) for i in range(self.num_client)}
            new_global = self.aggregate(uploads)
            self.global_model.load_state_dict(new_global)
            for i in range(self.num_client):
                self.client_models[i].load_state_dict(new_global)

            split_client_ids = set()
            for main_id, aux_id in self.pair_info["pairs"]:
                split_client_ids.add(main_id)
                split_client_ids.add(aux_id)
            for cid in split_client_ids:
                self.optimizers[cid].state.clear()

            round_model_bits_by_role = self._calculate_round_model_bits_by_role()
            model_bits_this_round = sum(round_model_bits_by_role.values())
            self.round_model_bits.append(int(model_bits_this_round))
            self.round_model_bits_by_role.append({k: int(v) for k, v in round_model_bits_by_role.items()})
            self.round_activation_bits.append(int(round_activation_bits))

            acc1, acc5 = self.validate()

            losses = []
            for i in range(self.num_client):
                if step_count[i] > 0:
                    losses.append(running_loss[i] / step_count[i])
            loss_avg = sum(losses) / len(losses) if losses else 0.0

            elapsed = time.time() - start_time
            epoch_end_time = time_lst[-1] + elapsed

            loss_lst.append(float(loss_avg))
            acc1_lst.append(float(acc1))
            acc5_lst.append(float(acc5))
            time_lst.append(epoch_end_time)
            for client_idx in range(self.num_client):
                clients_time_lst[client_idx].append(epoch_end_time)

            print(f"epoch={epoch} loss_avg={loss_avg:.4f} acc1={acc1:.4f} acc5={acc5:.4f} t={elapsed:.1f}")

        return loss_lst, acc1_lst, acc5_lst, time_lst, clients_time_lst

    def get_communication_stats(self):
        total_model_bits = int(sum(self.round_model_bits))
        total_activation_bits = int(sum(self.round_activation_bits))
        total_model_bits_by_role = {
            "normal": int(sum(item.get("normal", 0) for item in self.round_model_bits_by_role)),
            "main": int(sum(item.get("main", 0) for item in self.round_model_bits_by_role)),
            "aux": int(sum(item.get("aux", 0) for item in self.round_model_bits_by_role)),
        }
        total_bits = total_model_bits + total_activation_bits
        return {
            "round_model_bits": list(self.round_model_bits),
            "round_model_bits_by_role": list(self.round_model_bits_by_role),
            "round_activation_bits": list(self.round_activation_bits),
            "total_model_bits": total_model_bits,
            "total_model_bits_by_role": total_model_bits_by_role,
            "total_activation_bits": total_activation_bits,
            "total_bits": total_bits,
            "total_bytes": total_bits / 8.0,
        }

    def _calculate_round_model_bits_by_role(self) -> Dict[str, int]:
        role_map = self.pair_info.get("role_map", {})
        round_bits = {"normal": 0, "main": 0, "aux": 0}
        for client_id in range(self.num_client):
            role = role_map.get(client_id, "normal")
            if role == "aux":
                split_point = self.aux_split_points.get(client_id, self.default_split_point)
                client_model_bits = self._get_lower_state_bits(split_point)
            elif role == "main":
                client_model_bits = self.model_bits
            else:
                role = "normal"
                client_model_bits = self.model_bits
            round_bits[role] += int(2 * client_model_bits)
        return round_bits

    def _get_lower_state_bits(self, split_point: int) -> int:
        split_point = int(split_point)
        if split_point in self._lower_state_bits_cache:
            return self._lower_state_bits_cache[split_point]

        reference_model = self.client_models[0] if self.client_models else self.global_model
        reference_state = self.global_model.state_dict()
        lower_keys = self.get_lower_state_keys(reference_model, split_point)
        lower_bits = int(sum(reference_state[key].numel() * 32 for key in lower_keys if key in reference_state))
        self._lower_state_bits_cache[split_point] = lower_bits
        return lower_bits

    def get_minibatch(self, client_id: int):
        try:
            batch = next(self._dl_iters[client_id])
        except StopIteration:
            self._dl_iters[client_id] = iter(self.dataloaders[client_id])
            batch = next(self._dl_iters[client_id])
        return batch

    def split_learning_one_step(self, main_id: int, aux_id: int, split_point: int, batch) -> Tuple[float, int]:
        aux_model = self.client_models[aux_id]
        main_model = self.client_models[main_id]
        aux_model.train()
        main_model.train()

        if not (hasattr(aux_model, "lower_forward") and hasattr(main_model, "upper_forward")):
            raise ValueError("Model must implement lower_forward/upper_forward for split learning.")

        aux_opt = self.optimizers[aux_id]
        main_opt = self.optimizers[main_id]
        criterion = self.criterions[aux_id]

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        aux_opt.zero_grad(set_to_none=True)
        main_opt.zero_grad(set_to_none=True)

        h = aux_model.lower_forward(x, split_point=split_point)
        h_detached = h.detach().requires_grad_(True)
        logits = main_model.upper_forward(h_detached, split_point=split_point)

        loss = criterion(logits, y)
        grad_logits = torch.autograd.grad(loss, logits, retain_graph=False, create_graph=False)[0]

        upper_params = self._params_for_layers(main_model, split_point, self.num_layers)
        grads = torch.autograd.grad(
            outputs=logits,
            inputs=[h_detached] + upper_params,
            grad_outputs=grad_logits,
            retain_graph=False,
            create_graph=False,
        )
        grad_h = grads[0]
        upper_param_grads = grads[1:]
        for p, g in zip(upper_params, upper_param_grads):
            p.grad = g

        lower_params = self._params_for_layers(aux_model, 0, split_point)
        lower_param_grads = torch.autograd.grad(
            outputs=h,
            inputs=lower_params,
            grad_outputs=grad_h,
            retain_graph=False,
            create_graph=False,
        )
        for p, g in zip(lower_params, lower_param_grads):
            p.grad = g

        torch.nn.utils.clip_grad_norm_(parameters=main_model.parameters(), max_norm=10)
        torch.nn.utils.clip_grad_norm_(parameters=aux_model.parameters(), max_norm=10)

        aux_opt.step()
        main_opt.step()
        activation_bits = int(h_detached.numel() * 32)
        if grad_h is not None:
            activation_bits += int(grad_h.numel() * 32)
        return float(loss.item()), activation_bits

    def full_model_one_step(self, client_id: int, batch) -> float:
        model = self.client_models[client_id]
        model.train()
        optimizer = self.optimizers[client_id]
        criterion = self.criterions[client_id]
        role = self.pair_info.get("role_map", {}).get(client_id, "normal")

        x, y = batch
        x = x.to(self.device)
        y = y.to(self.device)

        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=10)
        optimizer.step()
        return float(loss.item())

    def aggregate(self, uploads: Dict[int, Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        global_state = deepcopy(self.global_model.state_dict())
        if not uploads:
            return global_state

        role_map = self.pair_info.get("role_map", {})
        lower_keys_cache: Dict[int, set] = {}

        for key in global_state.keys():
            vals = []
            for client_id, state in uploads.items():
                role = role_map.get(client_id, "normal")
                if role == "aux":
                    split_point = self.aux_split_points.get(client_id, self.default_split_point)
                    if client_id not in lower_keys_cache:
                        lower_keys_cache[client_id] = self.get_lower_state_keys(self.client_models[client_id], split_point)
                    if key not in lower_keys_cache[client_id]:
                        continue
                if key in state:
                    vals.append(state[key])
            if not vals:
                continue
            global_state[key] = sum(vals) / len(vals)

        return global_state

    def get_lower_state_keys(self, model, split_point: int) -> set:
        layer_modules = self._get_layer_modules(model)
        lower_modules = layer_modules[:max(0, min(split_point, len(layer_modules)))]
        name_by_module_id = {id(m): name for name, m in model.named_modules()}
        lower_keys = set()
        for module in lower_modules:
            prefix = name_by_module_id.get(id(module), "")
            for name, _ in module.named_parameters(recurse=True):
                full_name = f"{prefix}.{name}" if prefix else name
                lower_keys.add(full_name)
            for name, _ in module.named_buffers(recurse=True):
                full_name = f"{prefix}.{name}" if prefix else name
                lower_keys.add(full_name)
        return lower_keys

    def _get_layer_modules(self, model) -> List[nn.Module]:
        if not hasattr(model, "get_conv_layers_list") or not hasattr(model, "get_fc_layers_list"):
            raise ValueError("Model must provide get_conv_layers_list/get_fc_layers_list for SFLMAC.")
        conv_layers = model.get_conv_layers_list()
        fc_layers = model.get_fc_layers_list()
        return list(conv_layers) + list(fc_layers)

    def validate(self) -> Tuple[float, float]:
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
            acc1 = correct1 / total if total > 0 else 0.0
            acc5 = correct5 / total if total > 0 else 0.0
            return acc1, acc5

    def _params_for_layers(self, model: nn.Module, start: int, end: int) -> List[nn.Parameter]:
        start = max(0, min(start, self.num_layers))
        end = max(0, min(end, self.num_layers))
        name_by_id = {id(p): name for name, p in model.named_parameters()}
        allowed_names = set()
        for layer_names in self.layer_param_names[start:end]:
            allowed_names.update(layer_names)
        params = []
        for p in model.parameters():
            name = name_by_id.get(id(p))
            if name in allowed_names:
                params.append(p)
        return params

    @staticmethod
    def _disable_inplace_relu(model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    @staticmethod
    def _get_layer_param_names(model: nn.Module) -> List[List[str]]:
        if not hasattr(model, "get_conv_layers_list") or not hasattr(model, "get_fc_layers_list"):
            raise ValueError("Model must provide get_conv_layers_list/get_fc_layers_list for SFLMAC.")
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
        return layer_param_names

    @staticmethod
    def _get_max_split_point(benchmark: str) -> int:
        if "vgg11" in benchmark:
            return training_config.VGG11_MAX_SPLIT_POINT
        if "vgg16" in benchmark:
            return training_config.VGG16_MAX_SPLIT_POINT
        if "resnet18" in benchmark:
            return training_config.RESNET18_MAX_SPLIT_POINT
        return training_config.ALEXNET_MAX_SPLIT_POINT
