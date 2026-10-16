"""Worker action decoding with sequential shared-seed quotas."""

from dataclasses import asdict

import torch

from farm_policy import encoding as E
from farm_policy.coordinated import CoordinatedConfig, CoordinatedPolicy

PLANT_START = E.UI["PLANT", "WHEAT"]


def seed_counts(batch):
    # The existing feature schema stores asinh(seed_count/10). Rounding is exact
    # for every binding quota (0..31), including compressed float16 shards.
    return (
        (10 * batch["global_features"][..., -14:-9].float().sinh())
        .round()
        .clamp(0, E.MAX_ACTORS)
        .long()
    )


def masked_logits(logits, units, batch):
    choices = units.long() - PLANT_START
    planted = torch.stack([(choices == i) & (units >= 0) for i in range(5)], -1).long()
    before = planted.cumsum(-2) - planted
    available = seed_counts(batch)[..., None, :] - before
    result = logits.clone()
    result[..., PLANT_START : PLANT_START + 5] = result[
        ..., PLANT_START : PLANT_START + 5
    ].masked_fill(available <= 0, -1e9)
    return result


def valid_demonstrations(batch):
    # All over-requested PLANT actions are no-ops in the real game. Exclude
    # these ineffective labels instead of forcing an impossible joint target.
    result = dict(batch)
    unit = batch["unit"].clone()
    seeds = seed_counts(batch)
    for crop in range(5):
        selected = unit == PLANT_START + crop
        excessive = selected.sum(-1) > seeds[..., crop]
        unit = unit.masked_fill(selected & excessive[..., None], -100)
    result["unit"] = unit
    return result


class QuotaPolicy(CoordinatedPolicy):
    def forward(self, batch, targets=None, deterministic=False):
        context, actors, base_logits = self.core(batch)
        device = context.device
        b = len(context)
        if targets is not None:
            unit = targets["unit"].clamp_min(0).long()
            unit_logits = masked_logits(base_logits, unit, batch)
        else:
            remaining = seed_counts(batch)
            selected = []
            logits = []
            for actor in range(E.MAX_ACTORS):
                current = base_logits[:, actor].clone()
                current[:, PLANT_START : PLANT_START + 5] = current[
                    :, PLANT_START : PLANT_START + 5
                ].masked_fill(remaining <= 0, -1e9)
                chosen = (
                    current.argmax(-1)
                    if deterministic
                    else torch.distributions.Categorical(logits=current).sample()
                )
                selected.append(chosen)
                logits.append(current)
                remaining = (
                    remaining
                    - torch.stack(
                        [chosen == PLANT_START + i for i in range(5)], -1
                    ).long()
                )
            unit = torch.stack(selected, 1)
            unit_logits = torch.stack(logits, 1)
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
        has_quantity = torch.tensor([len(c) > 1 for c in E.MARKET], device=device)
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
            q = torch.where(has_quantity[m], q, torch.zeros_like(q))
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
            memory_global=self._next_memory[0],
            memory_actor=self._next_memory[1],
        )

    def sequence(self, batch, targets=True):
        if not targets:
            raise ValueError("Free-running quota sequence must be stepped online")
        clean = valid_demonstrations(batch)
        output, flat = super().sequence(clean, targets=True)
        output["unit_logits"] = masked_logits(output["unit_logits"], flat["unit"], flat)
        return output, flat


def save(path, model, **metadata):
    torch.save(
        dict(
            architecture="quota-coordinated-v1",
            schema=E.SCHEMA,
            config=asdict(model.config),
            state_dict=model.state_dict(),
            **metadata,
        ),
        path,
    )


def load(path, device="cpu", warmstart=False):
    state = torch.load(path, map_location=device, weights_only=True)
    if state["schema"] != E.SCHEMA:
        raise ValueError("Incompatible observation schema")
    if not warmstart and state.get("architecture") != "quota-coordinated-v1":
        raise ValueError("Not a quota-policy checkpoint")
    model = QuotaPolicy(CoordinatedConfig(**state["config"])).to(device)
    model.load_state_dict(state["state_dict"])
    return model, state
