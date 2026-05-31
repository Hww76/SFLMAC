from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class RingClientMeta:
    client_id: int
    computing: float
    up_rate: float
    down_rate: float


class RingSFLClient:
    def __init__(
        self,
        client_id: int,
        model: nn.Module,
        dataloader,
        criterion,
        optimizer,
        scheduler,
        device: str,
        meta: RingClientMeta,
    ):
        self.client_id = client_id
        self.model = model
        self.dataloader = dataloader
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device

        self.meta = meta
        self.ring_id: Optional[int] = None
        self.propagation_length: int = 0

        self._dl_iter = iter(self.dataloader)

    def next_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            data, target = next(self._dl_iter)
        except StopIteration:
            self._dl_iter = iter(self.dataloader)
            data, target = next(self._dl_iter)
        return data.to(self.device), target.to(self.device)

    def zero_grad(self):
        self.optimizer.zero_grad(set_to_none=True)

    def step(self):
        torch.nn.utils.clip_grad_norm_(parameters=self.model.parameters(), max_norm=10)
        self.optimizer.step()

    def step_scheduler(self):
        if self.scheduler is not None:
            self.scheduler.step()
