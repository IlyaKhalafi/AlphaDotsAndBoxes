import numpy as np
import pytest

from alphaboxes.curriculum import StartPositionArchive, sample_endgame
from alphaboxes.endgame import position
from alphaboxes.game import State
from alphaboxes.search import MCTS, SearchConfig
from alphaboxes.selfplay import play_episode, play_episodes


def test_endgame_sampler_generates_repeatable_unfinished_structural_positions():
    first, second = np.random.default_rng(741), np.random.default_rng(741)
    for size in ((1, 1), (1, 2), (2, 3), (3, 3), (5, 5), (8, 8), (10, 10), (12, 12)):
        for _ in range(10):
            state = sample_endgame(size, first)
            assert state == sample_endgame(size, second)
            assert not state.terminal and position(state) is not None
            for owner, edges in zip(state.owners, state.board.boxes, strict=True):
                free = sum(state.edges[e] < 0 for e in edges)
                assert free == (2 if owner < 0 else 0)


def test_endgame_curriculum_counts_starts_separately():
    archive = StartPositionArchive(endgame_fraction=1)
    rng = np.random.default_rng(742)
    for _ in range(4):
        state = archive.choose((5, 5), rng)
        assert position(state) is not None
    stats = archive.statistics()
    assert stats["endgame_starts"] == stats["sampled_games"] == 4
    assert stats["midgame_starts"] == 0 and stats["initial_moves_total"] > 0


def test_disabled_curriculum_preserves_random_stream():
    starts = StartPositionArchive()
    rng, expected = np.random.default_rng(8), np.random.default_rng(8)
    assert starts.choose((2, 2), rng) == State.new(2, 2)
    assert rng.random() == expected.random()


def test_archive_starts_are_reachable_bounded_and_board_specific():
    starts = StartPositionArchive(fraction=1, min_moves=2, max_moves=4, capacity=2)
    rng = np.random.default_rng(9)
    initial = starts.choose((2, 2), rng)
    assert 2 <= initial.board.num_edges - len(initial.legal_actions) <= 4
    state, positions = State.new(2, 2), []
    for action in range(6):
        state = state.play(action)
        positions.append(state)
    starts.observe(positions)
    assert list(starts.positions[2, 2]) == positions[2:4]
    assert starts.choose((2, 2), rng) in positions[2:4]
    assert starts.choose((1, 2), rng).board.cols == 2
    assert starts.statistics()["midgame_starts"] == 3


def test_midgame_targets_include_existing_scores_and_extra_turns():
    # Player one already owns a box and keeps the turn; completing the second wins.
    state = State.new(1, 2)
    for action in (0, 1, 2, 3, 4, 5):
        state = state.play(action)
    assert state.scores == (0, 1) and state.player == 1
    search = MCTS(lambda s: (np.ones(s.board.num_edges), 0), SearchConfig(simulations=8))
    examples = play_episode(search, (1, 2), np.random.default_rng(1), initial_state=state)
    assert len(examples) == 1 and examples[0].state == state
    assert examples[0].value == 1
    batched = play_episodes(
        search, [(1, 2), (2, 2)], np.random.default_rng(1), initial_states=[state, State.new(2, 2)]
    )
    assert batched[0].value == 1 and len(batched) == 13
    # Capturing the last box still loses when the opponent owns the first two.
    losing = State.new(1, 3)
    for action in (7, 9, 8, 6, 2, 4, 5, 1, 3):
        losing = losing.play(action)
    assert losing.scores == (2, 0) and losing.player == 1
    last = play_episode(search, (1, 3), np.random.default_rng(1), initial_state=losing)[0]
    assert last.value == -1
    with pytest.raises(ValueError, match="match"):
        play_episode(search, (2, 2), np.random.default_rng(1), initial_state=state)


@pytest.mark.parametrize("fraction", [-0.1, 1.1, float("nan")])
def test_invalid_start_fraction(fraction):
    with pytest.raises(ValueError, match="fraction"):
        StartPositionArchive(fraction=fraction)
