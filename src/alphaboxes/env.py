"""RLlib multi-agent environment. A capture keeps the same agent active."""

import gymnasium as gym
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv

from alphaboxes.game import State, board
from alphaboxes.graph import encode, spaces


class LegalDiscrete(gym.spaces.Discrete):
    """Fixed action space whose default random sampling respects the current mask."""

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self.legal_mask = np.ones(capacity, dtype=np.int8)

    def sample(self, mask=None, probability=None):
        if mask is None and probability is None:
            mask = self.legal_mask
        return super().sample(mask=mask, probability=probability)


class DotsAndBoxesEnv(MultiAgentEnv):
    def __init__(self, config: dict | None = None):
        super().__init__()
        config = config or {}
        self.sizes = [tuple(size) for size in config.get("sizes", [(3, 3)])]
        self.capacity = max(board(*size).num_nodes for size in self.sizes)
        observation_space, _ = spaces(self.capacity)
        action_space = LegalDiscrete(self.capacity)
        self.possible_agents = ["player_0", "player_1"]
        self.agents = self.possible_agents.copy()
        self.observation_spaces = dict.fromkeys(self.agents, observation_space)
        self.action_spaces = dict.fromkeys(self.agents, action_space)
        self.rng = np.random.default_rng(config.get("seed"))
        self.state = State.new(*self.sizes[0])
        self._update_mask()

    def _update_mask(self):
        mask = np.zeros(self.capacity, dtype=np.int8)
        mask[list(self.state.legal_actions)] = 1
        for action_space in self.action_spaces.values():
            action_space.legal_mask = mask

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.agents = self.possible_agents.copy()
        self.state = State.new(*self.sizes[int(self.rng.integers(len(self.sizes)))])
        self._update_mask()
        return {f"player_{self.state.player}": encode(self.state, self.capacity)}, {}

    def step(self, action_dict):
        active = f"player_{self.state.player}"
        if set(action_dict) != {active}:
            raise ValueError(f"Only {active} may act on this turn.")
        self.state = self.state.play(int(action_dict[active]))
        self._update_mask()
        terminal = self.state.terminal
        rewards = {
            agent: self.state.outcome(i) if terminal else 0.0
            for i, agent in enumerate(self.possible_agents)
        }
        terminated = {agent: terminal for agent in self.possible_agents} | {"__all__": terminal}
        truncated = dict.fromkeys([*self.possible_agents, "__all__"], False)
        obs = {} if terminal else {f"player_{self.state.player}": encode(self.state, self.capacity)}
        if terminal:
            self.agents = []
        return obs, rewards, terminated, truncated, {}
