"""A supervised NumPy search process with a caller-enforced response deadline.

The worker is warmed before play. A stalled evaluator or exact solver can be
terminated without blocking the game server. Only this instance's child process
is managed; external training and GPU processes are never inspected or touched.
"""

import multiprocessing as mp
import threading
from time import monotonic

import numpy as np

from alphaboxes.engine import AlphaBeta, EngineConfig
from alphaboxes.opponents import tactical_action
from alphaboxes.search import MCTS, SearchConfig


def _worker(connection, evaluator):
    try:
        connection.send("ready")
        while True:
            task = connection.recv()
            if task is None:
                return
            if task == "clear":
                evaluator.clear_cache()
                continue
            state, config = task
            search = (
                MCTS(evaluator, config)
                if isinstance(config, SearchConfig)
                else AlphaBeta(evaluator, config)
            )
            policy, value = search.policy(state)
            stats = {
                name: getattr(search, name, 0 if name in (
                    "completed_simulations", "endgame_hits", "proven_nodes"
                ) else None)
                for name in (
                    "completed_simulations", "endgame_hits", "proven_nodes",
                    "depth", "nodes", "cutoffs", "table_hits",
                )
            }
            connection.send((policy, value, stats))
    except (EOFError, BrokenPipeError):
        pass
    finally:
        connection.close()


class BoundedSearch:
    """Persistent worker; each policy call includes queueing in its deadline.

    Keep the search's internal thinking time below the response timeout. The
    latter defaults to 1.9 seconds, leaving room before a two-second API limit.
    Wall-clock measurements remain necessary under OS scheduling contention.
    """

    def __init__(self, evaluator, config=None, timeout: float = 1.9):
        if not np.isfinite(timeout) or not 0 < timeout < 2:
            raise ValueError("Response timeout must be positive and below two seconds.")
        self.evaluator = evaluator
        self.config = config or SearchConfig(simulations=8192, time_limit=1.75, exact_threshold=12)
        self.timeout = timeout
        self._context = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._process = self._connection = None
        self._retired = []
        self._last = threading.local()
        self._start()
        if not self._connection.poll(10) or self._connection.recv() != "ready":
            self.close()
            raise RuntimeError("Search worker did not start.")

    @property
    def stats(self):
        return getattr(self._last, "stats", {})

    def __getattr__(self, name):
        if name in ("timed_out", "completed_simulations", "endgame_hits", "proven_nodes",
                    "depth", "nodes", "cutoffs", "table_hits"):
            return self.stats.get(name)
        raise AttributeError(name)

    def _start(self):
        parent, child = self._context.Pipe()
        self._process = self._context.Process(target=_worker, args=(child, self.evaluator), daemon=True)
        self._process.start()
        child.close()
        self._connection = parent

    def _discard(self):
        if self._process is not None:
            self._process.terminate()
            self._retired.append(self._process)
            self._process = None
        if self._connection is not None:
            self._connection.close()
            self._connection = None

    def clear_cache(self):
        with self._lock:
            if self._connection is not None:
                self._connection.send("clear")

    def policy(self, state, *, config=None):
        if state.terminal:
            raise ValueError("Cannot search a finished game.")
        deadline = monotonic() + self.timeout
        # Compute the legal fallback before waiting on any other request.
        fallback = np.zeros(state.board.num_edges, np.float32)
        fallback[tactical_action(state, np.random.default_rng(0))] = 1.0
        self._last.stats = {
            "timed_out": False, "completed_simulations": 0, "endgame_hits": 0, "proven_nodes": 0,
            "depth": None, "nodes": None, "cutoffs": None, "table_hits": None,
        }
        if not self._lock.acquire(timeout=max(0.0, deadline - monotonic())):
            self._last.stats["timed_out"] = True
            return fallback, 0.0
        try:
            if self._process is None:
                self._start()
            self._connection.send((state, config or self.config))
            while self._connection.poll(max(0.0, deadline - monotonic())):
                result = self._connection.recv()
                if isinstance(result, str) and result == "ready":
                    continue
                policy, value, stats = result
                self._last.stats.update(stats)
                return policy, value
            self._last.stats["timed_out"] = True
            self._discard()
            return fallback, 0.0
        except (EOFError, BrokenPipeError, OSError):
            self._last.stats["timed_out"] = True
            self._discard()
            return fallback, 0.0
        finally:
            self._lock.release()

    def close(self):
        self._discard()
        for process in self._retired:
            process.join(timeout=1)
        self._retired.clear()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()
