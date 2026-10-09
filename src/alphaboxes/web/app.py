"""Small FastAPI game service with bounded sessions and serialized CPU search."""

import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from alphaboxes.checkpoint import load_evaluator
from alphaboxes.game import State
from alphaboxes.opponents import tactical_action
from alphaboxes.search import MCTS, SearchConfig

STATIC = Path(__file__).parent / "static"


class NewGame(BaseModel):
    rows: int = Field(default=3, ge=1, le=12)
    cols: int = Field(default=3, ge=1, le=12)
    human_player: int = Field(default=0, ge=0, le=1)
    demo: bool = False


class Revision(BaseModel):
    revision: int = Field(ge=0)


class Move(Revision):
    action: int = Field(ge=0)


class SearchRequest(Revision):
    simulations: int = Field(default=128, ge=8, le=512)


@dataclass
class Session:
    state: State
    human: int
    demo: bool
    revision: int = 0
    history: list[State] = field(default_factory=list)
    last_action: int | None = None
    touched: float = field(default_factory=time.monotonic)
    lock: threading.Lock = field(default_factory=threading.Lock)

    def snapshot(self) -> dict:
        return self.state.as_dict() | {
            "revision": self.revision,
            "human_player": self.human,
            "demo": self.demo,
            "last_action": self.last_action,
            "can_undo": bool(self.history),
        }

    def play(self, action: int):
        next_state = self.state.play(action)
        self.history.append(self.state)
        self.state = next_state
        self.last_action = action
        self.revision += 1


def create_app(checkpoint: Path | None = None) -> FastAPI:
    evaluator, metadata = load_evaluator(checkpoint) if checkpoint else (None, {})
    sessions: dict[str, Session] = {}
    sessions_lock = threading.Lock()
    search_lock = threading.Lock()
    rng = np.random.default_rng(2026)
    app = FastAPI(title="Alpha Dots & Boxes", docs_url=None, redoc_url=None)
    app.mount("/static", StaticFiles(directory=STATIC), name="static")

    def get_session(game_id: str) -> Session:
        with sessions_lock:
            session = sessions.get(game_id)
            if session is None:
                raise HTTPException(404, "Game expired. Start a new board.")
            session.touched = time.monotonic()
            return session

    def check_revision(session: Session, revision: int):
        if session.revision != revision:
            raise HTTPException(409, "The board has changed. Refresh this game.")

    def search(session: Session, simulations: int) -> tuple[int, float | None]:
        if session.state.terminal:
            raise HTTPException(409, "The game has finished.")
        with search_lock:
            if evaluator is None:
                return tactical_action(session.state, rng), None
            # Exact search is a clearly identified playing aid, never training data.
            planner = MCTS(evaluator, SearchConfig(simulations=simulations, exact_threshold=12))
            policy, value = planner.policy(session.state)
            action = int(rng.choice(np.flatnonzero(policy == policy.max())))
            return action, value

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html")

    @app.get("/api/health")
    def health():
        return {
            "status": "ok",
            "agent": "graph" if evaluator else "tactical",
            "inference": "numpy"
            if checkpoint and checkpoint.suffix == ".npz"
            else "torch"
            if evaluator
            else None,
            "training": metadata,
            "exact_endgame_edges": 12 if evaluator else 0,
        }

    @app.post("/api/games", status_code=201)
    def new_game(request: NewGame):
        with sessions_lock:
            now = time.monotonic()
            expired = [key for key, session in sessions.items() if now - session.touched > 3600]
            for key in expired:
                del sessions[key]
            if len(sessions) >= 128:
                raise HTTPException(503, "Too many active games. Try again later.")
            game_id = str(uuid.uuid4())
            session = Session(
                State.new(request.rows, request.cols), request.human_player, request.demo
            )
            sessions[game_id] = session
        return {"id": game_id, **session.snapshot()}

    @app.get("/api/games/{game_id}")
    def get_game(game_id: str):
        session = get_session(game_id)
        with session.lock:
            return session.snapshot()

    @app.delete("/api/games/{game_id}", status_code=204)
    def delete_game(game_id: str):
        with sessions_lock:
            sessions.pop(game_id, None)

    @app.post("/api/games/{game_id}/move")
    def human_move(game_id: str, request: Move):
        session = get_session(game_id)
        with session.lock:
            check_revision(session, request.revision)
            if session.demo or session.state.player != session.human:
                raise HTTPException(409, "It is the agent's turn.")
            try:
                session.play(request.action)
            except ValueError as error:
                raise HTTPException(422, str(error)) from error
            return session.snapshot()

    @app.post("/api/games/{game_id}/agent")
    def agent_move(game_id: str, request: SearchRequest):
        session = get_session(game_id)
        with session.lock:
            check_revision(session, request.revision)
            if not session.demo and session.state.player == session.human:
                raise HTTPException(409, "It is your turn.")
            started = time.monotonic()
            exact = evaluator is not None and len(session.state.legal_actions) <= 12
            action, value = search(session, request.simulations)
            session.play(action)
            return session.snapshot() | {
                "analysis": {
                    "action": action,
                    "value": value,
                    "seconds": time.monotonic() - started,
                    "simulations": request.simulations if evaluator and not exact else 0,
                    "method": "exact_endgame"
                    if exact
                    else "graph_search"
                    if evaluator
                    else "tactical",
                }
            }

    @app.post("/api/games/{game_id}/hint")
    def hint(game_id: str, request: SearchRequest):
        session = get_session(game_id)
        with session.lock:
            check_revision(session, request.revision)
            action, value = search(session, request.simulations)
            return {"action": action, "value": value, "revision": session.revision}

    @app.post("/api/games/{game_id}/undo")
    def undo(game_id: str, request: Revision):
        session = get_session(game_id)
        with session.lock:
            check_revision(session, request.revision)
            if not session.history:
                raise HTTPException(409, "No move to undo.")
            # Return to before the most recent human move, including subsequent agent moves.
            while session.history:
                previous = session.history.pop()
                session.state = previous
                if session.demo or previous.player == session.human:
                    break
            session.revision += 1
            session.last_action = None
            return session.snapshot()

    return app
