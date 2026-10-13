from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from alphaboxes.game import State
from alphaboxes.web.app import create_app


class Worker:
    instances = []

    def __init__(self, evaluator, config=None, timeout=1.9, **kwargs):
        self.timeout = timeout
        self.calls = []
        self.closed = False
        self.__class__.instances.append(self)

    def policy(self, state, *, config):
        self.calls.append(config)
        self.internal_seconds = config.time_limit or self.timeout - 0.1
        self.completed_simulations = config.simulations
        self.timed_out = False
        policy = np.zeros(state.board.num_edges)
        policy[list(state.legal_actions)] = 1 / len(state.legal_actions)
        return policy, 0.0

    def close(self):
        self.closed = True


def setup_workers(monkeypatch):
    Worker.instances = []
    monkeypatch.setattr("alphaboxes.web.app.load_evaluator", lambda path: (object(), {}))
    monkeypatch.setattr("alphaboxes.web.app.BoundedSearch", Worker)


def test_extended_is_opt_in_and_does_not_change_ordinary_requests(monkeypatch):
    setup_workers(monkeypatch)
    with TestClient(create_app(Path("test.npz"))) as client:
        assert not client.get("/api/health").json()["search"]["extended_large_boards"]
        game = client.post("/api/games", json={"rows": 10, "cols": 10, "demo": True}).json()
        path = f"/api/games/{game['id']}/agent"
        assert client.post(path, json={"revision": 0, "extended": True}).status_code == 422
        result = client.post(path, json={"revision": 0, "time_limit": 2}).json()
        assert result["analysis"]["response_deadline"] == 1.9
        assert not result["analysis"]["extended"]
        assert Worker.instances[0].calls[-1].time_limit == 1.75
    assert len(Worker.instances) == 1 and Worker.instances[0].closed


def test_extended_uses_separate_worker_only_on_large_graph_boards(monkeypatch):
    setup_workers(monkeypatch)
    with TestClient(create_app(Path("test.npz"), extended_search=True)) as client:
        health = client.get("/api/health").json()["search"]
        assert health["response_deadline"] == 1.9 and health["extended_large_boards"]
        assert health["extended_response_deadline"] == 60
        small = client.post("/api/games", json={"rows": 5, "cols": 5, "demo": True}).json()
        path = f"/api/games/{small['id']}/agent"
        assert client.post(path, json={"revision": 0, "extended": True}).status_code == 422
        big = client.post("/api/games", json={"rows": 10, "cols": 10, "demo": True}).json()
        path = f"/api/games/{big['id']}/agent"
        assert (
            client.post(
                path, json={"revision": 0, "extended": True, "engine": "alpha_beta"}
            ).status_code
            == 422
        )
        result = client.post(path, json={"revision": 0, "extended": True, "time_limit": 2}).json()
        assert result["analysis"]["response_deadline"] == 60
        assert result["analysis"]["extended"] and result["analysis"]["full_budget_completed"]
        fast, extended = Worker.instances
        assert not fast.calls
        config = extended.calls[-1]
        assert config.simulations == 8192 and config.time_limit is None
        assert config.sparse_selection and config.exact_threshold == 12
        assert config.fpu_reduction == 0.33 and config.root_safety_scale == 1
        assert not config.safe_moves and not config.prove_results and not config.reuse_tree
        client.post(path, json={"revision": 1, "time_limit": 2})
        assert fast.calls[-1].time_limit == 1.75
    assert all(worker.closed for worker in Worker.instances)


def test_extended_timeout_cannot_claim_exact_or_full_completion(monkeypatch):
    setup_workers(monkeypatch)

    class StalledWorker(Worker):
        def policy(self, state, *, config):
            result = super().policy(state, config=config)
            self.timed_out = True
            self.completed_simulations = 0
            return result

    state = State.new(10, 10)
    while len(state.legal_actions) > 12:
        state = state.play(state.legal_actions[0])
    monkeypatch.setattr(State, "new", lambda rows=3, cols=3: state)
    monkeypatch.setattr("alphaboxes.web.app.BoundedSearch", StalledWorker)
    with TestClient(create_app(Path("test.npz"), extended_search=True)) as client:
        game = client.post("/api/games", json={"rows": 10, "cols": 10, "demo": True}).json()
        result = client.post(
            f"/api/games/{game['id']}/agent", json={"revision": 0, "extended": True}
        ).json()
        analysis = result["analysis"]
        assert analysis["method"] == "deadline_fallback"
        assert analysis["timed_out"] and not analysis["full_budget_completed"]
        assert analysis["value"] is None


def test_extended_startup_failure_closes_the_fast_worker(monkeypatch):
    setup_workers(monkeypatch)

    class FailingWorker(Worker):
        def __init__(self, evaluator, config=None, timeout=1.9, **kwargs):
            if timeout > 2:
                raise RuntimeError("Extended worker failed")
            super().__init__(evaluator, config, timeout, **kwargs)

    monkeypatch.setattr("alphaboxes.web.app.BoundedSearch", FailingWorker)
    with pytest.raises(RuntimeError, match="Extended worker failed"):
        with TestClient(create_app(Path("test.npz"), extended_search=True)):
            pass
    assert len(Worker.instances) == 1 and Worker.instances[0].closed
