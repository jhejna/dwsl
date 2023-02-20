import gym
import numpy as np

THRESHOLD = 0.5


def ant_sparse(achieved, desired, info=None):
    # Vectorized reward function.
    # Returns -1 we are not at the goal, and zero otherwise
    assert achieved.shape == desired.shape
    d = np.linalg.norm(achieved - desired, axis=-1)
    return (d < THRESHOLD).astype(np.float32)


def negative_ant_sparse(achieved, desired, info=None):
    # Vectorized reward function.
    # Returns -1 we are not at the goal, and zero otherwise
    assert achieved.shape == desired.shape
    d = np.linalg.norm(achieved - desired, axis=-1)
    return -(d > THRESHOLD).astype(np.float32)


class AntMazeGoalConditionedWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Expose the TimeLimit wrapper if one exists
        if hasattr(self.env, "_max_episode_steps"):
            self._max_episode_steps = self.env._max_episode_steps

        obs_low = self.observation_space.low
        obs_high = self.observation_space.high
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(low=obs_low[2:], high=obs_high[2:], dtype=np.float32),
                "achieved_goal": gym.spaces.Box(low=obs_low[:2], high=obs_high[:2], dtype=np.float32),
                "desired_goal": gym.spaces.Box(low=obs_low[:2], high=obs_high[:2], dtype=np.float32),
            }
        )

    def _get_dict_obs(self, obs):
        achieved_goal = obs[:2]  # The first two components of the obs
        observation = obs[2:]  # everything else
        desired_goal = np.array(self.env.target_goal, dtype=np.float32)
        return dict(observation=observation, achieved_goal=achieved_goal, desired_goal=desired_goal)

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_dict_obs(obs)
        reward = ant_sparse(obs["achieved_goal"], obs["desired_goal"]).item()
        d = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        info["goal_distance"] = d
        info["success"] = d < THRESHOLD
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        return self._get_dict_obs(obs)
