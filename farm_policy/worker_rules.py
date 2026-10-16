"""Own-worker mechanics adapted from kaggle-environments 1.32.7"""

CROPS = {
    "WHEAT": {
        "seed": 10,
        "first_yield_day": 2,
        "max_yield_day": 4,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
    "CARROT": {
        "seed": 20,
        "first_yield_day": 2,
        "max_yield_day": 3,
        "interval": 0,
        "max_yield": 4,
        "ongoing": False,
    },
    "TOMATO": {
        "seed": 50,
        "first_yield_day": 8,
        "max_yield_day": 8,
        "interval": 1,
        "max_yield": 4,
        "ongoing": True,
    },
    "STRAWBERRY": {
        "seed": 100,
        "first_yield_day": 10,
        "max_yield_day": 10,
        "interval": 2,
        "max_yield": 4,
        "ongoing": True,
    },
    "MELON": {
        "seed": 80,
        "first_yield_day": 10,
        "max_yield_day": 12,
        "interval": 0,
        "max_yield": 6,
        "ongoing": False,
    },
}


ANIMALS = {
    "GOOSE": {
        "cost": 300,
        "structure": "COOP",
        "first_yield_day": 4,
        "interval": 1,
        "max_held": 4,
        "product": "EGG",
    },
    "COW": {
        "cost": 400,
        "structure": "PASTURE",
        "first_yield_day": 8,
        "interval": 2,
        "max_held": 6,
        "product": "MILK",
    },
    "SHEEP": {
        "cost": 500,
        "structure": "PASTURE",
        "first_yield_day": 6,
        "interval": 3,
        "max_held": 6,
        "product": "WOOL",
    },
}


FARMER_MOVES = {
    "NORTH": (0, -1),
    "SOUTH": (0, 1),
    "EAST": (1, 0),
    "WEST": (-1, 0),
}


def _shed_access_tiles(board_size):
    """Four inner-corner tiles around the shed, in NWSE order."""
    half = board_size // 2
    return [(half - 1, half - 1), (half, half - 1), (half - 1, half), (half, half)]


def _is_shed_adjacent(pos, board_size):
    return tuple(pos) in {(x, y) for (x, y) in _shed_access_tiles(board_size)}


def _new_plant(crop, day, turns_per_day):
    cd = CROPS[crop]
    return {
        "kind": "PLANT",
        "crop": crop,
        "planted_day": day,
        "watered_today": False,
        "consecutive_unwatered": 1,  # planting day counts as unwatered
        "yield_units": 0 if cd["ongoing"] else 1,
        "max_lifespan_step": (
            -1 if cd["ongoing"] else (day + cd["max_yield_day"] + 1) * turns_per_day
        ),
        "fertilized_until_day": -1,
    }


def _new_animal(animal, day):
    a = ANIMALS[animal]
    return {
        "kind": a["structure"],
        "animal": animal,
        "placed_day": day,
        "yield_units": 0,
        "consecutive_unfed": 0,
        "fed_today": False,
        "cared_today": False,
        "fertilizer_available": False,
        "pending_care_bonus": 0,
    }


def _farmer_position(farm, idx):
    """idx 0 = main farmer, 1+ = hand index."""
    if idx == 0:
        return farm["farmer"]
    return farm["hands"][idx - 1] if idx - 1 < len(farm["hands"]) else None


def _set_farmer_position(farm, idx, pos):
    if idx == 0:
        farm["farmer"] = list(pos)
    else:
        farm["hands"][idx - 1] = list(pos)


def _farmer_inventory(private, idx):
    """Inventories list is [main_farmer, *hands]; grow it if idx is past the end."""
    while len(private["inventories"]) <= idx:
        private["inventories"].append({})
    return private["inventories"][idx]


def _inv_add(inv, item, n=1):
    inv[item] = inv.get(item, 0) + n


def _inv_take(inv, item, n=1):
    if inv.get(item, 0) < n:
        return False
    inv[item] -= n
    if inv[item] == 0:
        del inv[item]
    return True


def _apply_unit_action(
    farm, private, idx, action, board_size, day, turns_per_day, shed_capacity=100
):
    """Process one farmer/hand's action. Invalid / illegal actions are silent no-ops."""
    if not isinstance(action, list) or not action:
        return
    op = action[0]
    pos = _farmer_position(farm, idx)
    if pos is None:
        return
    fx, fy = pos[0], pos[1]
    inv = _farmer_inventory(private, idx)

    if op in FARMER_MOVES:
        dx, dy = FARMER_MOVES[op]
        nx, ny = fx + dx, fy + dy
        if not (0 <= nx < board_size and 0 <= ny < board_size):
            return
        # Movement onto LOCKED tiles is allowed: a hand can spawn on a locked
        # shed-access tile, and blocking movement would strand it there forever.
        # Tile operations (PLANT, WATER, etc.) still no-op on LOCKED tiles.
        _set_farmer_position(farm, idx, (nx, ny))
        return

    if op == "PASS":
        return

    tile = farm["tiles"][fy][fx]

    # Shed operations resolve before the LOCKED guard. They use the tile only as
    # a standing position -- the shed itself is always owned -- and three of the
    # four shed-access tiles start LOCKED, so guarding them first would make the
    # shed unreachable from those tiles.
    if op == "DROP":
        if not _is_shed_adjacent((fx, fy), board_size):
            return
        shed = private["shed"]
        for item, n in list(inv.items()):
            if n <= 0:
                del inv[item]
                continue
            room = max(0, shed_capacity - sum(shed.values()))
            take = min(n, room)
            if take > 0:
                shed[item] = shed.get(item, 0) + take
            del inv[item]
        return

    if op == "PICKUP":
        if not _is_shed_adjacent((fx, fy), board_size):
            return
        if len(action) < 2:
            return
        item = action[1]
        n = int(action[2]) if len(action) >= 3 else 1
        if n <= 0:
            return
        # Seeds live in private["seeds"] and are consumed directly by PLANT;
        # they never pass through farmer inventory or the shed.
        available = private["shed"].get(item, 0)
        n = min(n, available)
        if n <= 0:
            return
        private["shed"][item] -= n
        _inv_add(inv, item, n)
        return

    if op == "PLACE":
        if len(action) < 2:
            return
        item = action[1]
        # Animal placement: standing on a matching unoccupied structure. A LOCKED
        # tile is the string "LOCKED", never a dict, so this branch cannot match
        # there and PLACE falls through to the shed path below.
        if (
            item in ANIMALS
            and isinstance(tile, dict)
            and tile.get("kind") == ANIMALS[item]["structure"]
            and "animal" not in tile
        ):
            if _inv_take(inv, item, 1):
                farm["tiles"][fy][fx] = _new_animal(item, day)
            return
        # Shed drop: orthogonally adjacent to the shed; obeys shedCapacity.
        if _is_shed_adjacent((fx, fy), board_size):
            n = int(action[2]) if len(action) >= 3 else 1
            if n <= 0:
                return
            n = min(n, inv.get(item, 0))
            if n <= 0:
                return
            current = sum(private["shed"].values())
            room = max(0, shed_capacity - current)
            n = min(n, room)
            if n <= 0:
                return
            inv[item] -= n
            if inv[item] == 0:
                del inv[item]
            private["shed"][item] = private["shed"].get(item, 0) + n
        return

    # Everything below mutates the tile the unit stands on, so it requires that
    # tile to be owned.
    if tile == "LOCKED":
        return

    if op == "PLANT":
        if len(action) < 2:
            return
        crop = action[1]
        if crop not in CROPS:
            return
        if tile is not None:
            return
        if private["seeds"].get(crop, 0) <= 0:
            return
        private["seeds"][crop] -= 1
        farm["tiles"][fy][fx] = _new_plant(crop, day, turns_per_day)
        return

    if op == "WATER":
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            return
        if tile["watered_today"]:
            return
        tile["watered_today"] = True
        crop_data = CROPS[tile["crop"]]
        if not crop_data["ongoing"]:
            age_days = day - tile["planted_day"]
            window_start = (crop_data["max_yield_day"] + 1) // 2
            if window_start <= age_days <= crop_data["max_yield_day"]:
                bonus = 2 if tile["fertilized_until_day"] >= day else 1
                tile["yield_units"] = min(
                    crop_data["max_yield"], tile["yield_units"] + bonus
                )
        return

    if op == "HARVEST":
        if not isinstance(tile, dict):
            return
        if tile.get("yield_units", 0) <= 0:
            return
        if tile.get("kind") == "PLANT":
            crop_data = CROPS[tile["crop"]]
            if day - tile["planted_day"] < crop_data["first_yield_day"]:
                # Ongoing crops only accumulate yield_units after first_yield_day,
                # so reaching here with yield_units > 0 indicates a bug.
                if crop_data["ongoing"]:
                    print(
                        f"WARNING: HARVEST on immature ongoing {tile['crop']} "
                        f"(planted day {tile['planted_day']}, current day {day}, "
                        f"first_yield_day {crop_data['first_yield_day']}, "
                        f"yield_units {tile['yield_units']}); should never happen"
                    )
                return
            units = tile["yield_units"]
            tile["yield_units"] = 0
            _inv_add(inv, tile["crop"], units)
            if not crop_data["ongoing"]:
                farm["tiles"][fy][fx] = None
        elif "animal" in tile:
            units = tile["yield_units"]
            tile["yield_units"] = 0
            _inv_add(inv, ANIMALS[tile["animal"]]["product"], units)
        return

    if op == "FERTILIZE":
        if not (isinstance(tile, dict) and tile.get("kind") == "PLANT"):
            return
        if not _inv_take(inv, "FERTILIZER", 1):
            return
        # Active for `day`, `day+1`, `day+2` (3 days inclusive).
        tile["fertilized_until_day"] = max(
            tile.get("fertilized_until_day", -1), day + 2
        )
        return

    if op == "DIG":
        if tile is None:
            return
        # Removes plants, weeds, empty coop/pasture. Does NOT remove a placed animal.
        if isinstance(tile, dict) and "animal" in tile:
            return
        farm["tiles"][fy][fx] = None
        return

    if op == "BUILD_COOP":
        if tile is not None:
            return
        farm["tiles"][fy][fx] = {"kind": "COOP"}
        return

    if op == "BUILD_PASTURE":
        if tile is not None:
            return
        farm["tiles"][fy][fx] = {"kind": "PASTURE"}
        return

    if op == "FEED":
        if not (isinstance(tile, dict) and "animal" in tile):
            return
        if tile["fed_today"]:
            return
        if not _inv_take(inv, "WHEAT", 1):
            return
        tile["fed_today"] = True
        return

    if op == "COLLECT_FERTILIZER":
        if not (isinstance(tile, dict) and "animal" in tile):
            return
        if not tile["fertilizer_available"]:
            return
        tile["fertilizer_available"] = False
        _inv_add(inv, "FERTILIZER", 1)
        return

    if op == "CARE":
        if not (isinstance(tile, dict) and "animal" in tile):
            return
        if tile["cared_today"]:
            return
        tile["cared_today"] = True
        return
