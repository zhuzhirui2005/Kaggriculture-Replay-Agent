"""Day-memory policy with a fixed, learnable demonstration-style condition.

Style is our own declared policy mode, NOT an inferred opponent identity. It is
fixed before a game and never contains the hidden seed or future town schedule.
Both actor and shared memory reset daily when farm hands are replaced.
"""

from dataclasses import dataclass

import torch
from torch import nn

from farm_policy import encoding as E
from farm_policy.model import Config, Policy


@dataclass
class MemoryConfig(Config):
    memory: int = 256
    styles: tuple = ()


class MemoryPolicy(Policy):
    def __init__(self, config=None):
        super().__init__(config or MemoryConfig())
        c = self.config
        h = c.hidden
        m = c.memory
        self.style = nn.Embedding(max(1, len(c.styles)), h)
        self.global_memory = nn.GRUCell(h, m)
        self.actor_memory = nn.GRUCell(h, m)
        self.context_residual = nn.Linear(m, h)
        self.actor_residual = nn.Linear(2 * m + h, h)
        nn.init.zeros_(self.context_residual.weight)
        nn.init.zeros_(self.context_residual.bias)
        nn.init.zeros_(self.actor_residual.weight)
        nn.init.zeros_(self.actor_residual.bias)
        nn.init.normal_(self.style.weight, std=0.02)

    def core(self, batch):
        if "_resolved" in batch:
            return batch["_resolved"]
        if "_base" in batch:
            context, actors = batch["_base"]
        else:
            context, actors, _ = Policy.core(self, batch)
        n = len(context)
        m = self.config.memory
        h = self.config.hidden
        shared = batch.get("memory_global", context.new_zeros(n, m))
        local = batch.get("memory_actor", context.new_zeros(n, E.MAX_ACTORS, m))
        style = self.style(
            batch.get("style", torch.zeros(n, dtype=torch.long, device=context.device))
        )
        shared = self.global_memory(context + style, shared)
        local = self.actor_memory(actors.reshape(-1, h), local.reshape(-1, m)).reshape(
            n, E.MAX_ACTORS, m
        )
        present = (
            torch.arange(E.MAX_ACTORS, device=context.device)[None]
            < batch["count"][:, None]
        )
        local = local * present[:, :, None]
        context = context + self.context_residual(shared) + style
        actors = actors + self.actor_residual(
            torch.cat(
                (
                    local,
                    shared[:, None].expand(-1, E.MAX_ACTORS, -1),
                    context[:, None].expand(-1, E.MAX_ACTORS, -1),
                ),
                -1,
            )
        )
        self._next_memory = (shared, local)
        logits = (
            self.actor_head(actors).float().masked_fill(~batch["legal"].bool(), -1e9)
        )
        return context, actors, logits

    def forward(self, batch, targets=None, deterministic=False):
        output = super().forward(batch, targets, deterministic)
        output["memory_global"], output["memory_actor"] = self._next_memory
        return output

    def sequence(self, batch, targets=True):
        b, t = batch["unit"].shape[:2]
        style = batch["style"]
        flat = {
            k: v.reshape(b * t, *v.shape[2:]) for k, v in batch.items() if k != "style"
        }
        context, actors, _ = Policy.core(self, flat)
        context = context.reshape(b, t, -1)
        actors = actors.reshape(b, t, E.MAX_ACTORS, -1)
        resolved = []
        memory = {}
        for i in range(t):
            frame = {k: v[:, i] for k, v in batch.items() if k != "style"}
            frame.update(style=style, _base=(context[:, i], actors[:, i]), **memory)
            resolved.append(self.core(frame))
            memory = dict(zip(("memory_global", "memory_actor"), self._next_memory))
        # Only the memories require serial time steps. Decode every frame's
        # actor and market heads together on GPU instead of240 tiny GRU calls.
        resolved = tuple(
            torch.stack([row[i] for row in resolved], 1).flatten(0, 1) for i in range(3)
        )
        output = Policy.forward(
            self,
            {"_resolved": resolved},
            targets=flat if targets else None,
            deterministic=not targets,
        )
        return output, flat
