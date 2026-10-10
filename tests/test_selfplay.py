import numpy as np

from alphaboxes.game import State
from alphaboxes.search import MCTS, SearchConfig
from alphaboxes.selfplay import play_episodes


def test_batched_episodes_keep_independent_results_and_legal_targets():
    sizes = [(1, 1), (1, 2), (2, 2)]
    search = MCTS(lambda state: (np.ones(state.board.num_edges), 0), SearchConfig(simulations=8))
    examples = play_episodes(search, sizes, np.random.default_rng(17), temperature_moves=3)
    offset = 0
    for size in sizes:
        initial = State.new(*size)
        episode = examples[offset : offset + initial.board.num_edges]
        assert episode[0].state == initial
        last = episode[-1].state
        final = last.play(last.legal_actions[0])
        assert final.terminal
        for example in episode:
            assert example.value == final.outcome(example.state.player)
            assert np.isclose(example.policy.sum(), 1)
            assert not example.policy[np.array(example.state.edges) >= 0].any()
        offset += initial.board.num_edges
    assert offset == len(examples)
