from ray.rllib.utils.pre_checks.env import check_multiagent_environments

from alphaboxes.env import DotsAndBoxesEnv


def test_rllib_environment_contract():
    check_multiagent_environments(DotsAndBoxesEnv({"sizes": [(1, 1), (2, 3)]}))


def test_extra_turn_and_zero_sum_terminal_reward():
    env = DotsAndBoxesEnv({"sizes": [(1, 2)]})
    obs, _ = env.reset(seed=5)
    for action in [0, 2, 4, 1]:
        active = next(iter(obs))
        obs, rewards, done, _, _ = env.step({active: action})
        assert not done["__all__"]
        assert all(reward == 0 for reward in rewards.values())
    # Shared edge completes the first box; the same player gets the next move.
    active = next(iter(obs))
    obs, _, done, _, _ = env.step({active: 5})
    assert next(iter(obs)) == active
    assert not done["__all__"]
    for action in [3, 6]:
        active = next(iter(obs))
        obs, rewards, done, truncated, _ = env.step({active: action})
    assert done["__all__"]
    assert not truncated["__all__"]
    assert sum(rewards.values()) == 0
    assert env.state.scores == (1, 1)
