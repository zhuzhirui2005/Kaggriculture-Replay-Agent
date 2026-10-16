"""Spatial actor policy, ordered market decoder and a shared value function."""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from farm_policy import encoding as E


@dataclass
class Config:
    width: int = 192
    hidden: int = 768
    blocks: int = 3


class Residual(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
        )

    def forward(self, x):
        return F.silu(x + self.net(x))


class Policy(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.config = config or Config()
        c = self.config
        h = c.hidden
        self.spatial = nn.Sequential(
            nn.Conv2d(E.BOARD_CHANNELS, c.width, 3, padding=1),
            nn.GroupNorm(8, c.width),
            nn.SiLU(),
            *[Residual(c.width) for _ in range(c.blocks)],
        )
        self.pool = nn.Sequential(
            nn.Conv2d(c.width, 32, 1), nn.Flatten(), nn.Linear(3200, h), nn.SiLU()
        )
        self.unit_embedding = nn.Embedding(len(E.UNIT), 32)
        self.market_embedding = nn.Embedding(len(E.MARKET), 32)
        self.quantity_embedding = nn.Embedding(101, 16)
        self.actor_embedding = nn.Embedding(E.MAX_ACTORS, 16)
        self.context = nn.Sequential(
            nn.Linear(h + E.GLOBAL_FEATURES + 48, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.actor = nn.Sequential(
            nn.Linear(c.width + h + E.ACTOR_FEATURES + 64, h),
            nn.LayerNorm(h),
            nn.SiLU(),
            nn.Linear(h, h),
            nn.SiLU(),
        )
        self.actor_head = nn.Linear(h, len(E.UNIT))
        self.actor_quantity = nn.Linear(h + 32, 101)
        self.goal_cell = nn.Linear(h, 100)
        self.goal_command = nn.Linear(h, len(E.UNIT))
        self.market_cell = nn.GRUCell(48, h)
        self.market_head = nn.Linear(h, len(E.MARKET))
        self.market_quantity = nn.Linear(h + 32, 101)
        self.value = nn.Linear(h, 1)

    def core(self, batch):
        board = self.spatial(batch["board"].float())
        b = board.shape[0]
        market_history = torch.cat(
            (
                self.market_embedding(batch["previous_market"].long()),
                self.quantity_embedding(batch["previous_market_quantity"].long()),
            ),
            dim=-1,
        ).mean(1)
        context = self.context(
            torch.cat(
                (self.pool(board), batch["global_features"].float(), market_history), -1
            )
        )
        local = (
            board.flatten(2)
            .transpose(1, 2)
            .gather(
                1,
                batch["positions"].long()[..., None].expand(-1, -1, self.config.width),
            )
        )
        ids = self.actor_embedding(torch.arange(E.MAX_ACTORS, device=board.device))[
            None
        ].expand(b, -1, -1)
        history = torch.cat(
            (
                self.unit_embedding(batch["previous_unit"].long()),
                self.quantity_embedding(batch["previous_quantity"].long()),
                ids,
            ),
            -1,
        )
        actors = self.actor(
            torch.cat(
                (
                    local,
                    context[:, None].expand(-1, E.MAX_ACTORS, -1),
                    batch["actors"].float(),
                    history,
                ),
                -1,
            )
        )
        logits = (
            self.actor_head(actors).float().masked_fill(~batch["legal"].bool(), -1e9)
        )
        return context, actors, logits

    def forward(self, batch, targets=None, deterministic=False):
        context, actors, unit_logits = self.core(batch)
        device = context.device
        b = len(context)
        unit = (
            targets["unit"].clamp_min(0).long()
            if targets is not None
            else unit_logits.argmax(-1)
            if deterministic
            else torch.distributions.Categorical(logits=unit_logits).sample()
        )
        quantity_logits = self.actor_quantity(
            torch.cat((actors, self.unit_embedding(unit)), -1)
        ).float()
        quantity = (
            targets["quantity"].clamp_min(0).long()
            if targets is not None
            else quantity_logits.argmax(-1)
            if deterministic
            else torch.distributions.Categorical(logits=quantity_logits).sample()
        )
        market = []
        market_q = []
        market_logits = []
        market_quantity_logits = []
        previous_m = torch.zeros(b, dtype=torch.long, device=device)
        previous_q = previous_m.clone()
        hidden = context
        for i in range(E.MAX_MARKET):
            hidden = self.market_cell(
                torch.cat(
                    (
                        self.market_embedding(previous_m),
                        self.quantity_embedding(previous_q),
                    ),
                    -1,
                ),
                hidden,
            )
            ml = self.market_head(hidden).float()
            m = (
                targets["market"][:, i].clamp_min(0).long()
                if targets is not None
                else ml.argmax(-1)
                if deterministic
                else torch.distributions.Categorical(logits=ml).sample()
            )
            ql = self.market_quantity(
                torch.cat((hidden, self.market_embedding(m)), -1)
            ).float()
            q = (
                targets["market_quantity"][:, i].clamp_min(0).long()
                if targets is not None
                else ql.argmax(-1)
                if deterministic
                else torch.distributions.Categorical(logits=ql).sample()
            )
            # Non-quantity orders must have one canonical recurrent history.
            # Otherwise HIRE/IGNORED carries an unused quantity that disappears
            # in the executed action but changes the next decoder state.
            has_quantity = torch.tensor(
                [len(command) > 1 for command in E.MARKET], device=device
            )[m]
            q = torch.where(has_quantity, q, torch.zeros_like(q))
            market.append(m)
            market_q.append(q)
            market_logits.append(ml)
            market_quantity_logits.append(ql)
            previous_m = m
            previous_q = q
        return dict(
            unit=unit,
            quantity=quantity,
            market=torch.stack(market, 1),
            market_quantity=torch.stack(market_q, 1),
            unit_logits=unit_logits,
            quantity_logits=quantity_logits,
            market_logits=torch.stack(market_logits, 1),
            market_quantity_logits=torch.stack(market_quantity_logits, 1),
            value=self.value(context).squeeze(-1).float(),
            goal_cell_logits=self.goal_cell(actors).float(),
            goal_command_logits=self.goal_command(actors).float(),
        )


def _ce(logits, targets, weight=None):
    flat = targets.reshape(-1).long()
    valid = flat != -100
    if not valid.any():
        return logits.sum() * 0
    return F.cross_entropy(
        logits.reshape(-1, logits.shape[-1])[valid], flat[valid], weight=weight
    )


def supervised_loss(output, batch):
    # Weight productive decisions; do not inflate scores with padded/pass actors.
    weight = torch.ones(len(E.UNIT), device=output["value"].device)
    weight[0] = 0.5
    for i, cmd in enumerate(E.UNIT):
        if cmd[0] in ("PLANT", "BUILD_COOP", "BUILD_PASTURE", "PLACE", "PICKUP", "DIG"):
            weight[i] = 2
    losses = dict(
        unit=_ce(output["unit_logits"], batch["unit"], weight),
        quantity=_ce(output["quantity_logits"], batch["quantity"]),
        market=_ce(output["market_logits"], batch["market"]),
        market_quantity=_ce(output["market_quantity_logits"], batch["market_quantity"]),
        goal_cell=_ce(output["goal_cell_logits"], batch["goal_cell"]),
        goal_command=_ce(output["goal_command_logits"], batch["goal_command"]),
        value=F.smooth_l1_loss(output["value"], batch["return"].float()),
    )
    total = (
        losses["unit"]
        + 0.5 * losses["quantity"]
        + losses["market"]
        + 0.5 * losses["market_quantity"]
    )
    total += (
        0.2 * (losses["goal_cell"] + losses["goal_command"]) + 0.1 * losses["value"]
    )
    return total, losses
