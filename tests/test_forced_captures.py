import numpy as np

from alphaboxes.chain_moves import allowed_actions
from alphaboxes.forced_captures import ForcedCaptureEvaluator
from alphaboxes.game import State


class RecordingEvaluator:
    def __init__(self):
        self.states = []
        self.clears = 0

    def __call__(self, state):
        self.states.append(state)
        policy = np.zeros(state.board.num_edges, np.float32)
        policy[list(state.legal_actions)] = 1 / len(state.legal_actions)
        return policy, (state.scores[state.player] - state.scores[1 - state.player]) / 20

    def clear_cache(self):
        self.clears += 1


def test_ordinary_choices_retain_neural_policy_and_value():
    neural = RecordingEvaluator()
    evaluator = ForcedCaptureEvaluator(neural)
    state = State.new(4, 4)
    policy, value = evaluator(state)
    assert neural.states == [state] and value == 0
    assert evaluator.moves == 0
    np.testing.assert_allclose(policy[list(state.legal_actions)], 1 / len(state.legal_actions))


def test_capture_extension_stops_before_the_two_box_handout():
    state = State.new(2, 6)
    for edge, (r1, _, r2, _) in enumerate(state.board.edges):
        if r1 == r2:
            state = state.play(edge)
    state = state.play(state.board.boxes[0][2])
    choices = allowed_actions(state)
    assert len(choices) == 1
    neural = RecordingEvaluator()
    evaluator = ForcedCaptureEvaluator(neural)
    policy, value = evaluator(state)
    assert np.flatnonzero(policy).tolist() == [choices[0]]
    assert evaluator.moves == 4 and evaluator.positions == 1
    assert len(neural.states) == 1
    continuation = neural.states[0]
    assert continuation.player == state.player
    assert continuation.scores[state.player] == 4 and value == 0.2
    options = allowed_actions(continuation)
    assert len(options) == 2
    assert sum(bool(continuation.captures(action)) for action in options) == 1
    evaluator(state)
    assert evaluator.moves == 4 and len(neural.states) == 1
    evaluator.clear_cache()
    evaluator(state)
    assert evaluator.moves == 8 and neural.clears == 1


def test_terminal_capture_returns_exact_initial_player_outcome():
    state = State.new(1, 1)
    for action in (0, 1, 2):
        state = state.play(action)
    neural = RecordingEvaluator()
    evaluator = ForcedCaptureEvaluator(neural)
    policy, value = evaluator(state)
    assert policy.argmax() == state.legal_actions[0] and value == 1
    assert not neural.states and evaluator.terminal_finishes == 1
