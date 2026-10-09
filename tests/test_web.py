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
