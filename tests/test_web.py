from pathlib import Path

from fastapi.testclient import TestClient

from alphaboxes.web.app import create_app


def test_playable_api_and_stale_move_protection():
    client = TestClient(create_app())
    assert client.get("/").status_code == 200
    assert client.get("/api/health").json()["agent"] == "tactical"
    game = client.post("/api/games", json={"rows": 1, "cols": 1}).json()
    path = f"/api/games/{game['id']}"
    hint = client.post(path + "/hint", json={"revision": 0, "simulations": 8})
    assert hint.status_code == 200
    assert client.get(path).json()["revision"] == 0
    move = client.post(path + "/move", json={"revision": 0, "action": 0})
    assert move.status_code == 200
    assert client.post(path + "/move", json={"revision": 0, "action": 1}).status_code == 409
    assert client.post(path + "/agent", json={"revision": 1, "simulations": 8}).status_code == 200
    assert client.post(path + "/move", json={"revision": 2, "action": 0}).status_code == 422
    undone = client.post(path + "/undo", json={"revision": 2, "action": 0}).json()
    assert undone["edges"] == [-1] * 4
    assert undone["revision"] == 3


def test_demo_can_finish_and_sessions_can_be_deleted():
    client = TestClient(create_app())
    game = client.post("/api/games", json={"rows": 1, "cols": 1, "demo": True}).json()
    path = f"/api/games/{game['id']}"
    while not game["terminal"]:
        game = client.post(
            path + "/agent", json={"revision": game["revision"], "simulations": 8}
        ).json()
    assert sum(game["scores"]) == 1
    assert client.delete(path).status_code == 204
    assert client.get(path).status_code == 404


def test_replay_preserves_turns_and_tracks_undo():
    import json

    from alphaboxes.game import State

    client = TestClient(create_app())
    game = client.post("/api/games", json={"rows": 1, "cols": 2, "demo": True}).json()
    path = f"/api/games/{game['id']}"
    while not game["terminal"]:
        game = client.post(
            path + "/agent", json={"revision": game["revision"], "simulations": 8}
        ).json()
    for undo in (False, True):
        if undo:
            game = client.post(path + "/undo", json={"revision": game["revision"]}).json()
        replay = client.get(path + "/replay").json()
        state = State.new(replay["rows"], replay["cols"])
        players = []
        for move in replay["moves"]:
            assert move["player"] == state.player
            assert move["analysis"]["action"] == move["action"]
            players.append(state.player)
            state = state.play(move["action"])
        assert json.loads(json.dumps(state.as_dict())) == replay["final_state"]
        assert state.edges == tuple(game["edges"])
        if not undo:
            assert any(a == b for a, b in zip(players, players[1:], strict=False))
    client.delete(path)
    assert client.get(path + "/replay").status_code == 404


def test_invalid_dimensions_and_out_of_turn():
    client = TestClient(create_app())
    assert client.post("/api/games", json={"rows": 0}).status_code == 422
    game = client.post("/api/games", json={"human_player": 1}).json()
    response = client.post(f"/api/games/{game['id']}/move", json={"revision": 0, "action": 0})
    assert response.status_code == 409


def test_solved_endgame_does_not_report_neural_simulations(tmp_path):
    from alphaboxes.checkpoint import save_agent
    from alphaboxes.network import module_spec

    model = module_spec(width=16, depth=2).build()
    checkpoint = tmp_path / "agent.pt"
    save_agent(checkpoint, model.state_dict(), 16, 2, {"iteration": 0})
    client = TestClient(create_app(checkpoint))
    game = client.post("/api/games", json={"rows": 1, "cols": 1, "demo": True}).json()
    result = client.post(
        f"/api/games/{game['id']}/agent", json={"revision": 0, "simulations": 128}
    ).json()
    assert result["analysis"]["method"] == "exact_endgame"
    assert result["analysis"]["simulations"] == 0


def test_timed_engines_and_replay_report_actual_work(monkeypatch):
    import numpy as np

    monkeypatch.setattr(
        "alphaboxes.web.app.load_evaluator",
        lambda path: (lambda state: (np.ones(state.board.num_edges), 0.0), {}),
    )
    client = TestClient(create_app(Path("test.npz")))
    for engine in ("mcts", "alpha_beta"):
        game = client.post("/api/games", json={"rows": 4, "cols": 4, "demo": True}).json()
        path = f"/api/games/{game['id']}"
        result = client.post(
            path + "/agent", json={"revision": 0, "engine": engine, "time_limit": 0.05}
        )
        assert result.status_code == 200
        analysis = result.json()["analysis"]
        assert analysis["engine"] == engine and analysis["time_limit"] == 0.05
        if engine == "mcts":
            assert 0 < analysis["simulations"] <= 8192
        else:
            assert analysis["simulations"] == 0 and analysis["nodes"] > 0
        assert client.get(path + "/replay").json()["moves"][0]["analysis"] == analysis
        for invalid in ({"time_limit": 0}, {"time_limit": 11}, {"engine": "chess"}):
            assert client.post(path + "/hint", json={"revision": 1, **invalid}).status_code == 422

