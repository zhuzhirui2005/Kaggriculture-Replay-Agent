"""Memory policy with shared-worker attention: inventories and jobs interact."""

from dataclasses import dataclass

import torch
from torch import nn

from farm_policy import encoding as E
from farm_policy.recurrent import MemoryConfig, MemoryPolicy


@dataclass
class CoordinatedConfig(MemoryConfig):
    attention_layers: int = 2


class CoordinatedPolicy(MemoryPolicy):
    def __init__(self, config=None):
        super().__init__(config or CoordinatedConfig())
        h = self.config.hidden
        layer = nn.TransformerEncoderLayer(
            h, 4, 2 * h, dropout=0, batch_first=True, norm_first=True, activation="gelu"
        )
        self.coordinator = nn.TransformerEncoder(
            layer, self.config.attention_layers, enable_nested_tensor=False
        )
        self.coordination_residual = nn.Linear(h, h)
        nn.init.zeros_(self.coordination_residual.weight)
        nn.init.zeros_(self.coordination_residual.bias)

    def core(self, batch):
        context, actors, logits = super().core(batch)
        # Sequence memories are serial, but worker communication is batched
        # across all frames in the final resolved pass. Never attends to time.
        if "_base" in batch:
            return context, actors, logits
        count = batch["count"] if "count" in batch else self._sequence_counts
        legal = batch["legal"] if "legal" in batch else self._sequence_legal
        absent = (
            torch.arange(E.MAX_ACTORS + 1, device=context.device)[None] > count[:, None]
        )
        joined = torch.cat((context[:, None], actors), 1)
        changed = self.coordination_residual(
            self.coordinator(joined, src_key_padding_mask=absent)
        )
        context = context + changed[:, 0]
        actors = actors + changed[:, 1:]
        logits = self.actor_head(actors).float().masked_fill(~legal.bool(), -1e9)
        return context, actors, logits

    def sequence(self, batch, targets=True):
        # Base sequence dispatches Policy.forward with a resolved feature set;
        # it also needs actor counts and legality for coordination here.
        self._sequence_counts = batch["count"].flatten(0, 1)
        self._sequence_legal = batch["legal"].flatten(0, 1)
        try:
            return super().sequence(batch, targets)
        finally:
            self._sequence_counts = None
            self._sequence_legal = None

    def forward(self, batch, targets=None, deterministic=False):
        return super().forward(batch, targets, deterministic)
