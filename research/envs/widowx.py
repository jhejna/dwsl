import copy

import gym
import numpy as np


class GoalConditionedWidowX200Env(gym.Env):
    def __init__(self, obs_keys=["images0", "state"]):
        super().__init__()
        obs_spaces = {}
        self._obs_keys = obs_keys
        for key in obs_keys:
            if "image" in key:
                obs_spaces[key] = gym.spaces.Box(low=0, high=255, shape=(3, 128, 128), dtype=np.uint8)
            elif key == "state":
                obs_spaces[key] = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(4,), dtype=np.float32)
            else:
                raise ValueError("Did not find observation key")

        if len(obs_spaces) == 1:
            obs_space = next(iter(obs_spaces.values()))
        else:
            obs_space = gym.spaces.Dict(obs_spaces)

        self.observation_space = gym.spaces.Dict(
            {"achieved_goal": copy.deepcopy(obs_space), "desired_goal": copy.deepcopy(obs_space)}
        )
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(4,), dtype=np.float32)

    def step(self, action):
        raise NotImplementedError

    def reset(self, **kwargs):
        raise NotImplementedError
