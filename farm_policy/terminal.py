"""A last-turn inventory liquidation rule; never changes worker actions."""

from collections import Counter
from copy import deepcopy

from .worker_rules import _apply_unit_action

PRODUCTS = (
    "CARROT",
    "TOMATO",
    "STRAWBERRY",
    "MELON",
    "EGG",
    "MILK",
    "WOOL",
    "WHEAT",
    "FERTILIZER",
)


def liquidate(observation, configuration, action):
    config = configuration or {}
    if int(observation["step"]) != config.get("episodeSteps", 720) - 2:
        return action
    farm = deepcopy(observation["farms"][observation["player"]])
    private = deepcopy(observation["private"])
    commands = [action.get("farmer") or ["PASS"], *action.get("hands", [])]
    asked = Counter(c[1] for c in commands if len(c) > 1 and c[0] == "PLANT")
    blocked = {item for item, n in asked.items() if n > private["seeds"].get(item, 0)}
    for actor, command in enumerate(commands[: len(private["inventories"])]):
        if len(command) > 1 and command[0] == "PLANT" and command[1] in blocked:
            continue
        _apply_unit_action(
            farm,
            private,
            actor,
            command,
            config.get("boardSize", 10),
            observation["day"],
            config.get("turnsPerDay", 24),
            config.get("shedCapacity", 100),
        )
    slots = config.get("maxMarketOrdersPerTurn", 10)
    market = deepcopy((action.get("market") or [])[:slots])
    left = Counter(private["shed"])
    for order in market:
        if len(order) >= 3 and order[0] == "SELL":
            left[order[1]] -= min(max(0, int(order[2])), max(0, left[order[1]]))
    prices = observation["market"]["prices"]
    candidates = sorted(
        (item for item in PRODUCTS if left[item] > 0),
        key=lambda item: (-prices.get(item, 1) * left[item], item),
    )
    changed = False
    for item in candidates:
        existing = next(
            (o for o in market if len(o) >= 3 and o[:2] == ["SELL", item]), None
        )
        if existing is not None:
            existing[2] = int(existing[2]) + left[item]
        elif len(market) < slots:
            market.append(["SELL", item, left[item]])
        else:
            continue
        changed = True
    return dict(action, market=market) if changed else action
