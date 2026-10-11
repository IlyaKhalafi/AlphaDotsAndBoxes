"""Pool independent duel receipts without mixing models or search treatments."""

from collections import defaultdict


def pool_duels(receipts: list[dict]) -> list[dict]:
    if not receipts:
        raise ValueError("Provide at least one duel receipt.")
    groups, seen = defaultdict(list), set()
    reference = None
    for receipt in receipts:
        treatment = (
            receipt["checkpoint_sha256"],
            receipt["opponent_sha256"],
            receipt["opening_moves"],
            tuple(
                sorted((k, v) for k, v in receipt["search_config"].items() if k != "simulations")
            ),
            tuple(
                sorted(
                    (k, v)
                    for k, v in receipt["opponent_search_config"].items()
                    if k != "simulations"
                )
            ),
        )
        if reference is not None and treatment != reference:
            raise ValueError("Cannot pool different checkpoints or search treatments.")
        reference = treatment
        budget = receipt["simulations"]
        if any(
            receipt[key]["simulations"] != budget
            for key in ("search_config", "opponent_search_config")
        ):
            raise ValueError("Both players must have the recorded fresh simulation budget.")
        for row in receipt["results"]:
            size = tuple(row["size"])
            key = (receipt["seed"], budget, size)
            if key in seen:
                raise ValueError("Duplicate seed, budget and board would double-count games.")
            seen.add(key)
            groups[budget, size].append((receipt["seed"], row))
    result = []
    for (budget, size), entries in sorted(groups.items()):
        rows = [row for _, row in entries]
        games = sum(row["games"] for row in rows)
        wins, draws, losses = (
            sum(row[name] for row in rows) for name in ("wins", "draws", "losses")
        )
        if games != wins + draws + losses:
            raise ValueError("Game count does not match wins, draws and losses.")
        counts = [sum(row["move_counts"][i] for row in rows) for i in (0, 1)]
        times = [sum(row["search_seconds"][i] for row in rows) for i in (0, 1)]
        means = [seconds / max(count, 1) for seconds, count in zip(times, counts, strict=True)]
        inherited = (
            [
                sum(row["mean_inherited_visits"][i] * row["move_counts"][i] for row in rows)
                / max(counts[i], 1)
                for i in (0, 1)
            ]
            if all("mean_inherited_visits" in row for row in rows)
            else None
        )
        result.append(
            {
                "simulations": budget,
                "size": list(size),
                "seeds": sorted(seed for seed, _ in entries),
                "games": games,
                "wins": wins,
                "draws": draws,
                "losses": losses,
                "score_rate": (wins + draws / 2) / games,
                "mean_move_seconds": means,
                "latency_ratio": means[0] / means[1] if means[1] > 0 else None,
                "mean_inherited_visits": inherited,
            }
        )
    return result


def deployment_gate(rows: list[dict], minimum_games: int = 80) -> dict:
    """A limited practical regression gate, not a statistical superiority test."""
    required = {(budget, size) for budget in (128, 512) for size in ((4, 4), (5, 5))}
    indexed = {(row["simulations"], tuple(row["size"])): row for row in rows}
    failures = []
    for key in sorted(required):
        row = indexed.get(key)
        label = f"{key[1][0]}x{key[1][1]} / {key[0]} simulations"
        if row is None:
            failures.append(f"Missing {label}")
            continue
        if row["games"] < minimum_games:
            failures.append(f"Too few games for {label}")
        threshold = row["score_rate"] > 0.5 if key == (512, (5, 5)) else row["score_rate"] >= 0.5
        if not threshold:
            failures.append(f"Score below gate for {label}")
        if row["latency_ratio"] is None or row["latency_ratio"] > 2:
            failures.append(f"Thinking latency exceeds gate for {label}")
    return {
        "passed": not failures,
        "failures": failures,
        "limitation": "Limited seeded regression check; no expert-human strength claim.",
    }
