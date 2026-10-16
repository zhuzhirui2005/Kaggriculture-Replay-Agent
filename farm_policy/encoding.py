"""Observation-only features and an executable structured action vocabulary."""

import math
from collections import Counter

import numpy as np

CROPS = ("WHEAT", "CARROT", "TOMATO", "STRAWBERRY", "MELON")
ANIMALS = ("GOOSE", "COW", "SHEEP")
PRODUCTS = (*CROPS, "EGG", "MILK", "WOOL", "FERTILIZER")
ITEMS = (*PRODUCTS, *ANIMALS)
SHOPS = (
    "BAKERY",
    "PIZZA_SHOP",
    "BRUNCH_SPOT",
    "YARN_STORE",
    "ICE_CREAM_SHOP",
    "PET_CAFE",
    "SMOOTHIE_SHOP",
    "FARMERS_MARKET",
)
UNIT = (
    ("PASS",),
    ("NORTH",),
    ("SOUTH",),
    ("EAST",),
    ("WEST",),
    ("DROP",),
    ("WATER",),
    ("HARVEST",),
    ("FERTILIZE",),
    ("DIG",),
    ("BUILD_COOP",),
    ("BUILD_PASTURE",),
    ("FEED",),
    ("CARE",),
    ("COLLECT_FERTILIZER",),
    *(("PLANT", c) for c in CROPS),
    *(("PICKUP", i) for i in ITEMS),
    *(("PLACE", i) for i in ITEMS),
)
MARKET = (
    ("END",),
    ("HIRE",),
    ("BUY_LAND",),
    ("IGNORED",),
    *(("BUY_SEED", c) for c in CROPS),
    *(("BUY_ANIMAL", a) for a in ANIMALS),
    ("BUY_PRODUCT", "WHEAT"),
    ("BUY_PRODUCT", "FERTILIZER"),
    *(("SELL", i) for i in PRODUCTS),
)
UI = {v: i for i, v in enumerate(UNIT)}
MI = {v: i for i, v in enumerate(MARKET)}
MAX_ACTORS = 32
MAX_MARKET = 10
MAX_QUANTITY = 100
TILES = ("EMPTY", "LOCKED", "WEED", "COOP", "PASTURE", *CROPS, *ANIMALS)
TI = {v: i for i, v in enumerate(TILES)}
TILE_FEATURES = len(TILES) + 12
BOARD_CHANNELS = 2 * TILE_FEATURES + 2
BASE = (25, 35, 60, 120, 250, 50, 160, 200, 100)
THROUGHPUT = (400, 450, 200, 100, 300, 332, 122, 105, 200)
ACTOR_FEATURES = 4 + len(ITEMS)
# Configuration is observable; overrides cannot silently share standard features.
CONFIG_KEYS = (
    "turnsPerDay",
    "episodeSteps",
    "shedCapacity",
    "farmHandCostMult",
    "maxMarketOrdersPerTurn",
    "weedSpawnChance",
    "townShopUnlockInterval",
    "townShopSellInterval",
    "townCenterSellInterval",
)
CONFIG_DEFAULT = (24, 720, 100, 1, 10, 0.005, 3, 4, 24)
GLOBAL_FEATURES = (
    4
    + 24
    + 2 * 8
    + len(SHOPS)
    + len(PRODUCTS) * 4
    + len(ITEMS)
    + len(CROPS)
    + len(CONFIG_KEYS)
)
SCHEMA = "spatial-actions-v3"


def _log(n, scale=100):
    return math.asinh(float(n) / scale)


def token(command, market=False):
    vocabulary = MI if market else UI
    if not isinstance(command, (list, tuple)) or not command:
        return (MI[("IGNORED",)] if market else 0), 0
    key = (
        tuple(command[:2])
        if len(command) > 1 and tuple(command[:2]) in vocabulary
        else (command[0],)
    )
    if key not in vocabulary:
        return (MI[("IGNORED",)] if market else 0), 0
    quantity = int(command[2]) if len(command) > 2 else 1
    if key[0] in ("PICKUP", "PLACE", "BUY_SEED", "BUY_ANIMAL", "BUY_PRODUCT", "SELL"):
        # Seed/animal over100 purchases cannot be represented losslessly.
        if key[0] in ("BUY_SEED", "BUY_ANIMAL") and quantity > MAX_QUANTITY:
            raise ValueError(f"Out-of-vocabulary quantity: {command!r}")
        quantity = max(0, min(MAX_QUANTITY, quantity))
    else:
        quantity = 0
    return vocabulary[key], quantity


def decode(unit, quantity, market, market_quantity, count):
    commands = []
    for u, q in zip(unit[:count], quantity[:count]):
        command = list(UNIT[int(u)])
        if command[0] in ("PICKUP", "PLACE"):
            command.append(max(0, int(q)))
        commands.append(command)
    orders = []
    for m, q in zip(market, market_quantity):
        command = list(MARKET[int(m)])
        if command[0] == "END":
            break
        if len(command) > 1:
            command.append(max(0, int(q)))
        orders.append(command)
    return dict(farmer=commands[0], hands=commands[1:], market=orders)


def labels(obs, action):
    count = 1 + len(obs["farms"][obs["player"]]["hands"])
    if count > MAX_ACTORS:
        raise ValueError(f"{count} actors exceeds model capacity")
    units = np.full(MAX_ACTORS, -100, np.int16)
    quantities = units.copy()
    commands = [action.get("farmer", ["PASS"]), *(action.get("hands") or [])]
    for i in range(count):
        u, q = token(commands[i] if i < len(commands) else ["PASS"])
        units[i] = u
        if UNIT[u][0] in ("PICKUP", "PLACE"):
            quantities[i] = q
    markets = np.full(MAX_MARKET, -100, np.int16)
    market_q = markets.copy()
    for i, command in enumerate((action.get("market") or [])[:MAX_MARKET]):
        m, q = token(command, True)
        markets[i] = m
        if len(MARKET[m]) > 1:
            market_q[i] = q
    end = min(len(action.get("market") or []), MAX_MARKET)
    if end < MAX_MARKET:
        markets[end] = 0
    return dict(
        unit=units, quantity=quantities, market=markets, market_quantity=market_q
    )


def legal_units(obs):
    # Conservative feasibility: earlier actors can change shared tiles/shed.
    # Do not forbid valid build/place or plant/water chains within one turn.
    farm = obs["farms"][obs["player"]]
    private = obs["private"]
    day = obs["day"]
    positions = [farm["farmer"], *farm["hands"]]
    inventories = private["inventories"]
    mask = np.zeros((MAX_ACTORS, len(UNIT)), np.bool_)
    mask[:, 0] = True
    for actor, (x, y) in enumerate(positions):
        inv = inventories[actor]
        tile = farm["tiles"][y][x]
        shed = x in (4, 5) and y in (4, 5)
        for action, valid in (
            ("NORTH", y > 0),
            ("SOUTH", y < 9),
            ("EAST", x < 9),
            ("WEST", x > 0),
        ):
            mask[actor, UI[(action,)]] = valid
        mask[actor, UI[("DROP",)]] = shed and bool(inv)
        for item in ITEMS:
            transfer = any(
                positions[j][0] in (4, 5)
                and positions[j][1] in (4, 5)
                and inventories[j].get(item, 0) > 0
                for j in range(actor)
            )
            mask[actor, UI["PICKUP", item]] = shed and (
                private["shed"].get(item, 0) > 0 or transfer
            )
            animal_place = (
                item in ANIMALS
                and isinstance(tile, dict)
                and "animal" not in tile
                and tile.get("kind") == ("COOP" if item == "GOOSE" else "PASTURE")
            )
            mask[actor, UI["PLACE", item]] = inv.get(item, 0) > 0 and (
                shed or animal_place
            )
        if tile == "LOCKED":
            continue
        if tile is None:
            mask[actor, UI[("BUILD_COOP",)]] = True
            mask[actor, UI[("BUILD_PASTURE",)]] = True
            for crop in CROPS:
                mask[actor, UI["PLANT", crop]] = private["seeds"].get(crop, 0) > 0
        elif isinstance(tile, dict):
            animal = "animal" in tile
            mask[actor, UI[("DIG",)]] = not animal
            if tile.get("kind") == "PLANT":
                mask[actor, UI[("WATER",)]] = not tile["watered_today"]
                mask[actor, UI[("FERTILIZE",)]] = inv.get("FERTILIZER", 0) > 0
                first = dict(zip(CROPS, (2, 2, 8, 10, 10)))[tile["crop"]]
                mask[actor, UI[("HARVEST",)]] = (
                    tile.get("yield_units", 0) > 0
                    and day - tile["planted_day"] >= first
                )
            elif animal:
                mask[actor, UI[("HARVEST",)]] = tile.get("yield_units", 0) > 0
                mask[actor, UI[("FEED",)]] = (
                    not tile["fed_today"] and inv.get("WHEAT", 0) > 0
                )
                mask[actor, UI[("CARE",)]] = not tile["cared_today"]
                mask[actor, UI[("COLLECT_FERTILIZER",)]] = tile["fertilizer_available"]
        if any(positions[j] == positions[actor] for j in range(actor)):
            occupied = isinstance(tile, dict) and "animal" in tile
            if not occupied:
                for op in (
                    "DIG",
                    "BUILD_COOP",
                    "BUILD_PASTURE",
                    "WATER",
                    "CARE",
                    "COLLECT_FERTILIZER",
                ):
                    mask[actor, UI[(op,)]] = True
                mask[actor, UI[("FERTILIZE",)]] = inv.get("FERTILIZER", 0) > 0
                mask[actor, UI[("FEED",)]] = inv.get("WHEAT", 0) > 0
                for crop in CROPS:
                    mask[actor, UI["PLANT", crop]] = private["seeds"].get(crop, 0) > 0
                for animal in ANIMALS:
                    mask[actor, UI["PLACE", animal]] = inv.get(animal, 0) > 0
                if isinstance(tile, dict) and tile.get("crop") in CROPS:
                    first = dict(zip(CROPS, (2, 2, 8, 10, 10)))[tile["crop"]]
                    mask[actor, UI[("HARVEST",)]] = day - tile["planted_day"] >= first
    return mask


def encode(obs, config=None, previous=None, previous_observation=None):
    config = config or {}
    seat = obs["player"]
    step = int(obs.get("step", 0))
    day = int(obs["day"])
    hour = int(obs["hour"])
    if config.get("marketParams"):
        raise ValueError("Custom market curves need an extended feature schema")
    farms = [obs["farms"][seat], obs["farms"][1 - seat]]
    if any(len(f["tiles"]) != 10 for f in farms):
        raise ValueError("This model schema is for10x10 boards")
    private = obs["private"]
    positions = [farms[0]["farmer"], *farms[0]["hands"]]
    count = len(positions)
    if count > MAX_ACTORS:
        raise ValueError(f"{count} actors exceeds model capacity")
    board = np.zeros((BOARD_CHANNELS, 10, 10), np.float32)
    for owner, farm in enumerate(farms):
        offset = owner * TILE_FEATURES
        for y, row in enumerate(farm["tiles"]):
            for x, tile in enumerate(row):
                kind = (
                    "EMPTY"
                    if tile is None
                    else tile
                    if isinstance(tile, str)
                    else tile.get("animal") or tile.get("crop") or tile["kind"]
                )
                board[offset + TI[kind], y, x] = 1
                values = [0.0] * 12
                if isinstance(tile, dict):
                    values = [
                        (day - tile.get("planted_day", tile.get("placed_day", day)))
                        / 30,
                        tile.get("yield_units", 0) / 6,
                        float(tile.get("watered_today", False)),
                        tile.get("consecutive_unwatered", 0) / 2,
                        max(-1, min(3, tile.get("fertilized_until_day", -1) - day)) / 3,
                        max(-1, min(48, tile.get("max_lifespan_step", -1) - step)) / 48,
                        float(tile.get("fed_today", False)),
                        tile.get("consecutive_unfed", 0) / 2,
                        float(tile.get("cared_today", False)),
                        float(tile.get("fertilizer_available", False)),
                        tile.get("pending_care_bonus", 0) / 6,
                        0.0,
                    ]
                board[offset + len(TILES) : offset + TILE_FEATURES, y, x] = values
        for x, y in [farm["farmer"], *farm["hands"]]:
            board[offset + TILE_FEATURES - 1, y, x] += 1 / 4
    board[-2] = np.arange(10, dtype=np.float32)[None, :] / 9
    board[-1] = np.arange(10, dtype=np.float32)[:, None] / 9
    global_values = [step / 719, day / 29, hour / 23, (719 - step) / 719] + [
        float(hour == i) for i in range(24)
    ]
    for farm in farms:
        global_values.extend(
            [
                _log(farm["money"], 10000),
                farm["hires_today"] / 20,
                len(farm["hands"]) / 20,
                len(farm["unlocked_quadrants"]) / 4,
                *[
                    float(q in farm["unlocked_quadrants"])
                    for q in ("NW", "NE", "SW", "SE")
                ],
            ]
        )
    shops = Counter(obs["town"]["unlocked_shops"])
    global_values.extend(shops[s] / 8 for s in SHOPS)
    for item, base, throughput in zip(PRODUCTS, BASE, THROUGHPUT):
        market = obs["market"]
        prior = previous_observation["market"] if previous_observation else market
        global_values.extend(
            [
                (10000 - market["inventory"][item]) / throughput,
                market["prices"][item] / base,
                (market["inventory"][item] - prior["inventory"][item]) / throughput,
                (market["prices"][item] - prior["prices"][item]) / base,
            ]
        )
    global_values.extend(_log(private["shed"].get(item, 0), 10) for item in ITEMS)
    global_values.extend(_log(private["seeds"].get(c, 0), 10) for c in CROPS)
    global_values.extend(
        float(config.get(k, d)) / d for k, d in zip(CONFIG_KEYS, CONFIG_DEFAULT)
    )
    assert len(global_values) == GLOBAL_FEATURES
    actors = np.zeros((MAX_ACTORS, ACTOR_FEATURES), np.float32)
    pos = np.zeros(MAX_ACTORS, np.int64)
    for i, (x, y) in enumerate(positions):
        inv = private["inventories"][i]
        pos[i] = y * 10 + x
        actors[i] = [
            x / 9,
            y / 9,
            float(i == 0),
            i / 31,
            *[_log(inv.get(item, 0), 10) for item in ITEMS],
        ]
    history = dict(
        unit=np.zeros(MAX_ACTORS, np.int64),
        quantity=np.zeros(MAX_ACTORS, np.int64),
        market=np.zeros(MAX_MARKET, np.int64),
        market_quantity=np.zeros(MAX_MARKET, np.int64),
    )
    if previous is not None:
        old = labels(previous_observation or obs, previous)
        history = {k: np.maximum(0, v).astype(np.int64) for k, v in old.items()}
        if hour == 0:
            history["unit"][1:] = 0
            history["quantity"][1:] = 0
    return dict(
        board=board,
        global_features=np.asarray(global_values, np.float32),
        actors=actors,
        positions=pos,
        count=np.asarray(count, np.int64),
        legal=legal_units(obs),
        **{"previous_" + k: v for k, v in history.items()},
    )
