from copy import deepcopy

import pytest

from alphaboxes.experiments import deployment_gate, pool_duels


def receipt(seed, count, seconds):
    return {
        "seed": seed,
        "checkpoint_sha256": "a",
        "opponent_sha256": "b",
        "opening_moves": 6,
        "simulations": 128,
        "search_config": {"simulations": 128, "reuse_tree": True},
        "opponent_search_config": {"simulations": 128, "reuse_tree": False},
        "results": [
            {
                "size": [5, 5],
                "games": 2,
                "wins": 1,
                "draws": 0,
                "losses": 1,
                "move_counts": [count, count],
                "search_seconds": [seconds, seconds * 2],
            }
        ],
    }


def test_pooling_weights_move_latency_and_rejects_duplicate_or_mixed_treatments():
    first, second = receipt(1, 1, 1), receipt(2, 9, 27)
    row = pool_duels([first, second])[0]
    assert row["games"] == 4 and row["score_rate"] == 0.5
    assert row["mean_move_seconds"] == [2.8, 5.6]
    assert row["mean_inherited_visits"] is None
    with pytest.raises(ValueError, match="Duplicate"):
        pool_duels([first, first])
    mixed = deepcopy(second)
    mixed["search_config"]["reuse_tree"] = False
    with pytest.raises(ValueError, match="treatments"):
        pool_duels([first, mixed])


def test_gate_requires_all_boards_budgets_and_more_than_a_tie_on_deep_5x5():
    rows = [
        {
            "size": list(size),
            "simulations": budget,
            "games": 80,
            "score_rate": 0.6,
            "latency_ratio": 1.2,
        }
        for budget in (128, 512)
        for size in ((4, 4), (5, 5))
    ]
    assert deployment_gate(rows)["passed"]
    rows[-1]["score_rate"] = 0.5
    assert not deployment_gate(rows)["passed"]
    rows[-1]["score_rate"] = 0.6
    rows[0]["latency_ratio"] = 2.1
    assert not deployment_gate(rows)["passed"]
    assert not deployment_gate(rows[1:])["passed"]
