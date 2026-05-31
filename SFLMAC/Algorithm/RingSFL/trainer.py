import time
from typing import Dict, List, Tuple

import torch

from .client import RingSFLClient


class RingSFLTrainer:
    def __init__(
        self,
        ring_clients: Dict[int, RingSFLClient],
        layer_param_names: List[List[str]],
        conv_layer_count: int,
    ):
        self.ring_clients = ring_clients
        self.layer_param_names = layer_param_names
        self.num_blocks = len(layer_param_names)
        self.conv_layer_count = conv_layer_count
        self._execution_layers_cache = {}
        self._is_resnet_layout = self._detect_resnet_layout()

    def train_round(self, ring_plans: List[dict], local_epoch: int, debug: bool = False) -> Dict[str, object]:
        ring_loss = []
        touched_clients = set()
        ring_debug = []
        total_activation_bits = 0

        for ring_plan in ring_plans:
            ring_id = ring_plan["ring_id"]
            ring_clients = ring_plan["ring_clients"]
            propagation_lengths = ring_plan["propagation_lengths"]
            touched_clients.update(ring_clients)

            per_client_stats = {
                cid: {
                    "source_pass_count": 0,
                    "owner_segment_count": 0,
                    "zero_grad_count": 0,
                    "optimizer_step_count": 0,
                    "activation_send_count": 0,
                    "activation_recv_count": 0,
                    "grad_send_count": 0,
                    "grad_recv_count": 0,
                    "forward_time": 0.0,
                    "backward_time": 0.0,
                    "step_time": 0.0,
                }
                for cid in ring_clients
            }

            one_ring_losses = []
            for _ in range(local_epoch):
                for source_id in ring_clients:
                    per_client_stats[source_id]["source_pass_count"] += 1
                    loss_val = self._run_one_ring_pass(
                        ring_clients=ring_clients,
                        propagation_lengths=propagation_lengths,
                        source_id=source_id,
                        per_client_stats=per_client_stats,
                    )
                    total_activation_bits += int(loss_val[1])
                    loss_val = loss_val[0]
                    one_ring_losses.append(loss_val)

            ring_loss.append(sum(one_ring_losses) / len(one_ring_losses) if one_ring_losses else 0.0)

            if debug:
                ring_debug.append(
                    {
                        "ring_id": ring_id,
                        "members": list(ring_clients),
                        "propagation_lengths": dict(propagation_lengths),
                        "client_stats": per_client_stats,
                    }
                )

        for cid in touched_clients:
            self.ring_clients[cid].step_scheduler()

        result = {
            "avg_loss": (sum(ring_loss) / len(ring_loss)) if ring_loss else 0.0,
            "participants": sorted(list(touched_clients)),
            "total_activation_bits": int(total_activation_bits),
        }
        if debug:
            result["ring_debug"] = ring_debug
        return result

    def _run_one_ring_pass(
        self,
        ring_clients: List[int],
        propagation_lengths: Dict[int, int],
        source_id: int,
        per_client_stats: Dict[int, dict],
    ) -> tuple:
        for cid in ring_clients:
            self.ring_clients[cid].zero_grad()
            per_client_stats[cid]["zero_grad_count"] += 1

        seq = self._rotate_ring(ring_clients, source_id)
        ownership = self._build_ownership_for_sequence(seq, propagation_lengths)

        for owner_idx in range(len(ownership) - 1):
            cur_owner = ownership[owner_idx][0]
            next_owner = ownership[owner_idx + 1][0]
            per_client_stats[cur_owner]["activation_send_count"] += 1
            per_client_stats[next_owner]["activation_recv_count"] += 1

        source_client = self.ring_clients[source_id]
        data, target = source_client.next_batch()

        current = data
        records = []
        for owner_id, start_block, end_block in ownership:
            owner_client = self.ring_clients[owner_id]
            owner_model = owner_client.model
            per_client_stats[owner_id]["owner_segment_count"] += 1

            segment_input = current.detach().requires_grad_(True)
            forward_start = time.perf_counter()
            segment_output = self._forward_block_range(
                model=owner_model,
                x=segment_input,
                start_block=start_block,
                end_block=end_block,
            )
            per_client_stats[owner_id]["forward_time"] += time.perf_counter() - forward_start

            segment_params = self._params_for_blocks(
                model=owner_model,
                start=start_block,
                end=end_block,
            )
            records.append(
                {
                    "owner_id": owner_id,
                    "segment_input": segment_input,
                    "segment_output": segment_output,
                    "segment_params": segment_params,
                }
            )
            current = segment_output

        activation_bits = 0
        for i in range(len(records) - 1):
            activation_bits += int(records[i]["segment_output"].numel() * 32)

        logits = current
        loss = source_client.criterion(logits, target)

        grad_out = torch.autograd.grad(loss, records[-1]["segment_output"], retain_graph=False, create_graph=False)[0]
        for idx in range(len(records) - 1, -1, -1):
            rec = records[idx]
            owner_id = rec["owner_id"]
            owner_client = self.ring_clients[owner_id]
            segment_params = rec["segment_params"]

            if idx > 0:
                prev_owner = records[idx - 1]["owner_id"]
                per_client_stats[owner_id]["grad_send_count"] += 1
                per_client_stats[prev_owner]["grad_recv_count"] += 1

            backward_start = time.perf_counter()
            grads = torch.autograd.grad(
                outputs=rec["segment_output"],
                inputs=[rec["segment_input"]] + segment_params,
                grad_outputs=grad_out,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            per_client_stats[owner_id]["backward_time"] += time.perf_counter() - backward_start

            grad_out = grads[0]
            if idx > 0 and grad_out is not None:
                activation_bits += int(grad_out.numel() * 32)
            param_grads = grads[1:]
            for p, g in zip(segment_params, param_grads):
                if g is None:
                    continue
                if p.grad is None:
                    p.grad = g
                else:
                    p.grad = p.grad + g

        for cid in ring_clients:
            step_start = time.perf_counter()
            self.ring_clients[cid].step()
            per_client_stats[cid]["step_time"] += time.perf_counter() - step_start
            per_client_stats[cid]["optimizer_step_count"] += 1

        return float(loss.item()), int(activation_bits)

    @staticmethod
    def _rotate_ring(ring_clients: List[int], source_id: int) -> List[int]:
        pos = ring_clients.index(source_id)
        return ring_clients[pos:] + ring_clients[:pos]

    def _build_ownership_for_sequence(
        self,
        sequence: List[int],
        propagation_lengths: Dict[int, int],
    ) -> List[Tuple[int, int, int]]:
        ownership = []
        cursor = 0
        for cid in sequence:
            length = int(propagation_lengths[cid])
            start_block = cursor
            end_block = min(self.num_blocks, cursor + length)
            end_block = self._adjust_cut_for_model(cursor, end_block)
            if end_block <= start_block:
                continue
            ownership.append((cid, start_block, end_block))
            cursor = end_block
            if cursor >= self.num_blocks:
                break

        if cursor < self.num_blocks:
            last_cid = sequence[-1]
            ownership.append((last_cid, cursor, self.num_blocks))

        return ownership

    def _adjust_cut_for_model(self, start_block: int, end_block: int) -> int:
        if not self._is_resnet_layout:
            return end_block
        if end_block >= self.num_blocks:
            return self.num_blocks
        if self._is_valid_resnet_cut(end_block):
            return end_block

        up = self._next_valid_resnet_cut(end_block)
        down = self._prev_valid_resnet_cut(end_block)
        candidates = [c for c in (down, up) if c is not None and c > start_block]
        if not candidates:
            return end_block

        return min(candidates, key=lambda c: (abs(c - end_block), c))

    def _detect_resnet_layout(self) -> bool:
        if not self.ring_clients:
            return False
        any_client = next(iter(self.ring_clients.values()))
        model = any_client.model
        return self._is_resnet18_param_layer_layout(model)

    def _is_valid_resnet_cut(self, cut: int) -> bool:
        return cut == self.num_blocks or (cut >= 1 and cut % 2 == 1)

    def _next_valid_resnet_cut(self, cut: int):
        for c in range(cut + 1, self.num_blocks + 1):
            if self._is_valid_resnet_cut(c):
                return c
        return None

    def _prev_valid_resnet_cut(self, cut: int):
        for c in range(cut - 1, 0, -1):
            if self._is_valid_resnet_cut(c):
                return c
        return None

    def _forward_block_range(self, model, x: torch.Tensor, start_block: int, end_block: int) -> torch.Tensor:
        if start_block >= end_block:
            return x

        layers = self._get_execution_layers(model)

        out = x
        for block_idx in range(start_block, end_block):
            if (not self._is_resnet_layout) and block_idx == self.conv_layer_count and out.dim() > 2:
                out = torch.flatten(out, 1)
            out = layers[block_idx](out)
        return out

    def _get_execution_layers(self, model):
        cache_key = id(model)
        if cache_key in self._execution_layers_cache:
            return self._execution_layers_cache[cache_key]

        conv_layers = model.get_conv_layers_list()
        fc_layers = model.get_fc_layers_list()
        raw_layers = conv_layers + fc_layers

        if all(callable(layer) for layer in raw_layers):
            self._execution_layers_cache[cache_key] = raw_layers
            return raw_layers

        if self._is_resnet18_param_layer_layout(model):
            layers = self._build_resnet18_execution_layers(model)
            if len(layers) != self.num_blocks:
                raise ValueError(
                    f"ResNet18 execution layer count mismatch: expected {self.num_blocks}, got {len(layers)}"
                )
            self._execution_layers_cache[cache_key] = layers
            return layers

        raise TypeError(
            "Model provides non-callable split layers and no compatible execution-layer adapter was found."
        )

    @staticmethod
    def _is_resnet18_param_layer_layout(model) -> bool:
        return hasattr(model, "_blocks") and hasattr(model, "stem") and hasattr(model, "avg_pool") and hasattr(model, "fc")

    @staticmethod
    def _build_resnet18_execution_layers(model):
        class _StemLayer:
            def __call__(self, inp):
                return model.stem(inp)

        class _BlockPart1:
            def __init__(self, block):
                self.block = block

            def __call__(self, inp):
                identity = inp
                out = self.block.conv1(inp)
                out = self.block.bn1(out)
                out = self.block.relu(out)
                return out, identity

        class _BlockPart2:
            def __init__(self, block):
                self.block = block

            def __call__(self, inp):
                out, identity = inp
                out = self.block.conv2(out)
                out = self.block.bn2(out)
                out = out + self.block.shortcut(identity)
                out = self.block.relu(out)
                return out

        class _FCLayer:
            def __call__(self, inp):
                out = model.avg_pool(inp)
                out = torch.flatten(out, 1)
                out = model.fc(out)
                return out

        layers = [_StemLayer()]
        for block in model._blocks():
            layers.append(_BlockPart1(block))
            layers.append(_BlockPart2(block))
        layers.append(_FCLayer())
        return layers

    def _params_for_blocks(self, model, start: int, end: int):
        start = max(0, min(start, self.num_blocks))
        end = max(0, min(end, self.num_blocks))
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
