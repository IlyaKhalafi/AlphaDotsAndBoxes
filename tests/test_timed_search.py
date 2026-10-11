from time import monotonic, sleep

import numpy as np
import pytest

from alphaboxes.game import State
from alphaboxes.search import SearchConfig
from alphaboxes.timed_search import BoundedSearch


class Evaluator:
    def __init__(self, delay=0):
        self.delay = delay

    def __call__(self, state):
        sleep(self.delay)
        return np.ones(state.board.num_edges), 0.0

    def clear_cache(self):
        pass


def test_worker_enforces_deadline_even_when_evaluator_stalls_and_restarts():
    with BoundedSearch(Evaluator(delay=5), timeout=0.15) as search:
        state = State.new(3, 3)
        started = monotonic()
        policy, _ = search.policy(state)
        assert monotonic() - started < 0.75
        assert search.timed_out and policy.argmax() in state.legal_actions
        # An exact root bypasses the stalled evaluator after worker replacement.
        search.config = SearchConfig(exact_threshold=12)
        search.timeout = 1.5
        policy, value = search.policy(State.new(1, 1))
        assert not search.timed_out and policy.sum() == 1 and value == -1


def test_queue_wait_is_bounded_and_normal_result_preserves_statistics():
    with BoundedSearch(Evaluator(), SearchConfig(simulations=8), timeout=0.2) as search:
        state = State.new(2, 2)
        policy, _ = search.policy(state)
        assert search.completed_simulations == 8 and policy.sum() == pytest.approx(1)
        search._lock.acquire()
        try:
            started = monotonic()
            policy, _ = search.policy(state)
            assert monotonic() - started < 0.75
            assert policy.argmax() in state.legal_actions
        finally:
            search._lock.release()


@pytest.mark.parametrize("timeout", [0, -1, 2, np.inf, np.nan])
def test_invalid_deadlines(timeout):
    with pytest.raises(ValueError):
        BoundedSearch(Evaluator(), timeout=timeout)
