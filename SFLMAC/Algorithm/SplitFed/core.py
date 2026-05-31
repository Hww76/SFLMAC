import time
from copy import deepcopy
from typing import Dict, List

import torch
import torch.nn as nn

from config import training as training_config


class SplitFed:
    """Split Federated Learning (full-client participation).

    Semantics aligned with requested migration:
    - all clients participate in every global round
    - fixed shared split_point
    - each local epoch consumes exactly one mini-batch per client
    - client phase: lower -> upper -> loss -> backward, but update lower only
    - server phase: concatenate detached activations from all clients, then update upper only
    - aggregate only lower parameters by simple average
    - rebuild global_model = averaged lower + current server upper
    - reuse existing validate(global_model)
    """

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

        self.num_client = len(self.client_models)
        self.names = self.global_model.state_dict().keys()
        self._dl_iters = {i: iter(self.dataloaders[i]) for i in range(self.num_client)}

        # fixed split point required
        self.max_split_point = self._get_max_split_point(self.benchmark)
        self.shared_split_point = kwargs.get("split_point")
        if self.shared_split_point is None:
            raise ValueError("SplitFed requires a fixed split_point.")
        self.shared_split_point = int(self.shared_split_point)
        if not (1 <= self.shared_split_point <= self.max_split_point):
            raise ValueError(
                f"split_point must be in [1, {self.max_split_point}], got {self.shared_split_point}."
            )

        # local_epoch == local mini-batch steps per global round
        self.local_steps = int(kwargs.get("local_steps", self.local_epoch))

        # central upper model lives on server
        self.server_model = deepcopy(self.global_model).to(self.device)

        # split safety
        self._disable_inplace_relu(self.server_model)
        self._disable_inplace_relu(self.global_model)
        for m in self.client_models:
            self._disable_inplace_relu(m)

        # logical layer mapping from full model
        self.layer_param_names = self._get_layer_param_names(self.global_model)
        self.num_layers = len(self.layer_param_names)
        self.client_lower_keys = {
            client_idx: self.get_lower_state_keys(self.client_models[client_idx], self.shared_split_point)
            for client_idx in range(self.num_client)
        }
        self.server_upper_keys = self.get_upper_state_keys(self.server_model, self.shared_split_point)

        # strict optimizer split
        self.client_lower_param_names = {
            client_idx: self._get_parameter_names_from_state_keys(self.client_models[client_idx], self.client_lower_keys[client_idx])
            for client_idx in range(self.num_client)
        }
        self.server_upper_param_names = self._get_parameter_names_from_state_keys(self.server_model, self.server_upper_keys)

        self.client_lower_params = {
            client_idx: self._get_named_params(self.client_models[client_idx], self.client_lower_param_names[client_idx])
            for client_idx in range(self.num_client)
        }
        self.server_upper_params = self._get_named_params(self.server_model, self.server_upper_param_names)

        if any(len(v) == 0 for v in self.client_lower_params.values()):
            raise ValueError("SplitFed found empty lower parameter set for at least one client. Check split_point.")
        if len(self.server_upper_params) == 0:
            raise ValueError("SplitFed found empty server upper parameter set. Check split_point.")

        self.client_lower_optimizers = [
            self._build_optimizer_from_param_list(self.client_lower_params[i], self.optimizers[i])
            for i in range(self.num_client)
        ]
        self.server_upper_optimizer = self._build_optimizer_from_param_list(
            self.server_upper_params,
            self.global_optimizer,
        )

        self.client_lower_schedulers = [
            self._build_scheduler_from_optimizer(self.schedulers[i], self.client_lower_optimizers[i])
            for i in range(self.num_client)
        ]
        self.server_upper_scheduler = self._build_scheduler_from_optimizer(
            self.global_scheduler,
            self.server_upper_optimizer,
        )

        # communication stats placeholders (optional, currently lightweight)
        self.round_model_bits = []
        self.round_activation_bits = []

        # initial sync: all clients start from the same global lower, server from global full model
        self.server_model.load_state_dict(self.global_model.state_dict())
        for client_idx in range(self.num_client):
            self._load_client_lower_state_from_global(client_idx)

    def run(self):
        loss_lst = []
        acc1_lst = []
        acc5_lst = []
        time_lst = [0.0]
        clients_time_lst = [[0.0] for _ in range(self.num_client)]

        for epoch in range(self.global_epoch):
            start_time = time.time()
            avg_loss = self.train_one_round()
            self.aggregate_lower_and_update_global()

            # scheduler stepping after one global round
            for scheduler in self.client_lower_schedulers:
                if scheduler is not None:
                    scheduler.step()
            if self.server_upper_scheduler is not None:
                self.server_upper_scheduler.step()

            acc1, acc5 = self.validate()
            elapsed = time.time() - start_time
            time_lst.append(time_lst[-1] + elapsed)
            for client_idx in range(self.num_client):
                clients_time_lst[client_idx].append(time_lst[-1])

            loss_lst.append(float(avg_loss))
            acc1_lst.append(float(acc1))
            acc5_lst.append(float(acc5))

            if self.debug:
                print(
                    f"[SplitFed] epoch={epoch + 1}/{self.global_epoch} "
                    f"loss={avg_loss:.6f} acc1={acc1:.6f} acc5={acc5:.6f}"
                )

        return loss_lst, acc1_lst, acc5_lst, time_lst, clients_time_lst

    def train_one_round(self) -> float:
        """One global round.

        Each local step consumes exactly one mini-batch per client.
        """
        round_loss_sum = 0.0
        round_loss_count = 0

        # keep all clients aligned with latest averaged lower before this round starts
        for client_idx in range(self.num_client):
            self._load_client_lower_state_from_global(client_idx)

        for _local_step in range(self.local_steps):
            detached_features: List[torch.Tensor] = []
            detached_targets: List[torch.Tensor] = []

            # ---- client phase: update each client's lower using one mini-batch ----
            for client_idx in range(self.num_client):
                loss_value, feat_detached, target_detached = self.train_client_one_minibatch(client_idx)
                round_loss_sum += float(loss_value)
                round_loss_count += 1
                detached_features.append(feat_detached)
                detached_targets.append(target_detached)

            # ---- server phase: concatenate detached activations, update upper once ----
            self.train_server_one_minibatch(detached_features, detached_targets)

        if round_loss_count == 0:
            return 0.0
        return round_loss_sum / float(round_loss_count)

    def train_client_one_minibatch(self, client_idx: int):
        client_model = self.client_models[client_idx]
        optimizer = self.client_lower_optimizers[client_idx]
        criterion = self.criterions[client_idx]

        client_model.train()
        self.server_model.train()

        data, target = self._next_batch(client_idx)
        data = data.to(self.device)
        target = target.to(self.device)

        optimizer.zero_grad(set_to_none=True)
        self.server_upper_optimizer.zero_grad(set_to_none=True)
        self.server_model.zero_grad(set_to_none=True)

        # strict reproduction:
        # lower -> upper -> loss -> backward, but step lower only
        h = client_model.lower_forward(data, split_point=self.shared_split_point)
        logits = self.server_model.upper_forward(h, split_point=self.shared_split_point)
        loss = criterion(logits, target)
        loss.backward()

        # IMPORTANT: client phase must not update / accumulate server upper grads
        self._clear_upper_grads()

        torch.nn.utils.clip_grad_norm_(self.client_lower_params[client_idx], max_norm=10.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        feat_detached = h.detach()
        target_detached = target.detach()
        return float(loss.item()), feat_detached, target_detached

    def train_server_one_minibatch(self, detached_features: List[torch.Tensor], detached_targets: List[torch.Tensor]):
        if len(detached_features) == 0:
            return 0.0

        self.server_model.train()
        self.server_upper_optimizer.zero_grad(set_to_none=True)
        self.server_model.zero_grad(set_to_none=True)

        concat_h = torch.cat(detached_features, dim=0)
        concat_y = torch.cat(detached_targets, dim=0)

        logits = self.server_model.upper_forward(concat_h, split_point=self.shared_split_point)
        loss = self.global_criterion(logits, concat_y)
        loss.backward()

        # server phase updates upper only
        self._clear_lower_grads_on_server()
        torch.nn.utils.clip_grad_norm_(self.server_upper_params, max_norm=10.0)
        self.server_upper_optimizer.step()
        self.server_upper_optimizer.zero_grad(set_to_none=True)
        return float(loss.item())

    def aggregate_lower_and_update_global(self):
        """Average lower states across all clients, keep server upper unchanged.

        Result:
            global_model = averaged_lower + current_server_upper
            server_model  = averaged_lower + current_server_upper
            each client lower <- averaged_lower
        """
        with torch.no_grad():
            client_states = [m.state_dict() for m in self.client_models]
            server_state = self.server_model.state_dict()
            global_state = self.global_model.state_dict()
            new_global = {}

            lower_keys = self.client_lower_keys[0]
            for name in global_state.keys():
                if name in lower_keys:
                    vals = [state[name].detach().to(self.device) for state in client_states if name in state]
                    new_global[name] = sum(vals) / len(vals)
                else:
                    new_global[name] = server_state[name].detach().clone()

            self.global_model.load_state_dict(new_global)
            self.server_model.load_state_dict(new_global)
            for client_idx in range(self.num_client):
                self._load_client_lower_state_from_global(client_idx)

    def _next_batch(self, client_idx: int):
        try:
            data, target = next(self._dl_iters[client_idx])
        except StopIteration:
            self._dl_iters[client_idx] = iter(self.dataloaders[client_idx])
            data, target = next(self._dl_iters[client_idx])
        return data, target

    def _load_client_lower_state_from_global(self, client_idx: int):
        global_state = self.global_model.state_dict()
        client_state = self.client_models[client_idx].state_dict()
        lower_keys = self.client_lower_keys[client_idx]
        for key in lower_keys:
            if key in global_state and key in client_state:
                client_state[key] = global_state[key].detach().clone()
        self.client_models[client_idx].load_state_dict(client_state)

    def _clear_upper_grads(self):
        for name, param in self.server_model.named_parameters():
            if name in self.server_upper_param_names:
                param.grad = None

    def _clear_lower_grads_on_server(self):
        lower_names = self.client_lower_param_names[0]
        for name, param in self.server_model.named_parameters():
            if name in lower_names:
                param.grad = None

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

    def get_upper_state_keys(self, model, split_point: int) -> set:
        all_keys = set(model.state_dict().keys())
        lower_keys = self.get_lower_state_keys(model, split_point)
        return all_keys - lower_keys

    def _get_layer_modules(self, model):
        if not hasattr(model, "get_conv_layers_list") or not hasattr(model, "get_fc_layers_list"):
            raise ValueError("Model must provide get_conv_layers_list/get_fc_layers_list for SplitFed.")
        conv_layers = model.get_conv_layers_list()
        fc_layers = model.get_fc_layers_list()
        return list(conv_layers) + list(fc_layers)

    @staticmethod
    def _disable_inplace_relu(model: nn.Module):
        for module in model.modules():
            if isinstance(module, nn.ReLU):
                module.inplace = False

    @staticmethod
    def _get_layer_param_names(model):
        if not hasattr(model, "get_conv_layers_list"):
            raise ValueError("Model must provide get_conv_layers_list/get_fc_layers_list.")
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
    def _get_parameter_names_from_state_keys(model, state_keys: set) -> set:
        named_param_keys = {name for name, _ in model.named_parameters()}
        return {k for k in state_keys if k in named_param_keys}

    @staticmethod
    def _get_named_params(model, param_names: set) -> List[torch.nn.Parameter]:
        return [param for name, param in model.named_parameters() if name in param_names]

    @staticmethod
    def _build_optimizer_from_param_list(params, base_optimizer):
        group = base_optimizer.param_groups[0]
        params = list(params)
        if isinstance(base_optimizer, torch.optim.SGD):
            return torch.optim.SGD(
                params,
                lr=group.get("lr", 0.1),
                momentum=group.get("momentum", 0.9),
                weight_decay=group.get("weight_decay", 0.0),
                dampening=group.get("dampening", 0.0),
                nesterov=group.get("nesterov", False),
            )
        if isinstance(base_optimizer, torch.optim.Adam):
            return torch.optim.Adam(
                params,
                lr=group.get("lr", 1e-3),
                betas=group.get("betas", (0.9, 0.999)),
                eps=group.get("eps", 1e-8),
                weight_decay=group.get("weight_decay", 0.0),
                amsgrad=group.get("amsgrad", False),
            )
        return torch.optim.SGD(params, lr=group.get("lr", 0.1))

    @staticmethod
    def _build_scheduler_from_optimizer(base_scheduler, optimizer):
        if base_scheduler is None:
            return None
        if isinstance(base_scheduler, torch.optim.lr_scheduler.CosineAnnealingLR):
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=getattr(base_scheduler, "T_max", 1),
                eta_min=getattr(base_scheduler, "eta_min", 0.0),
                last_epoch=-1,
            )
        return None

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

    @staticmethod
    def _get_max_split_point(benchmark: str) -> int:
        if "vgg11" in benchmark:
            return training_config.VGG11_MAX_SPLIT_POINT
        if "vgg16" in benchmark:
            return training_config.VGG16_MAX_SPLIT_POINT
        if "resnet18" in benchmark:
            return training_config.RESNET18_MAX_SPLIT_POINT
        return training_config.ALEXNET_MAX_SPLIT_POINT
