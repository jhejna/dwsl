# Register environment classes here
from .base import Empty
import d4rl  # registers the d4rl envs

# Import wrappers
from .gym_robotics import GymSparseRewardWrapper, FetchImageWrapper
from .antmaze import AntMazeGoalConditionedWrapper
from .widowx import GoalConditionedWidowXEnv

import gym
from gym.envs import register
from gym.spaces import Box
import numpy as np


try:
    from .rmimic import GoalConditionedRoboMimicEnv, get_robomimc_concat_keys
except:
    print("[research] Could not import Robomimic envs.")

register(id="FetchPush-v2", entry_point="research.envs.gym_robotics:ModifiedFetchPushEnv", max_episode_steps=50)
register(id="FetchReach-v2", entry_point="research.envs.gym_robotics:ModifiedFetchReachEnv", max_episode_steps=50)
register(id="FetchPick-v2", entry_point="research.envs.gym_robotics:ModifiedFetchPickAndPlaceEnv", max_episode_steps=50)
register(id="FetchSlide-v2", entry_point="research.envs.gym_robotics:ModifiedFetchSlideEnv", max_episode_steps=50)
register(id="HandReachImage-v0", entry_point="research.envs.gym_robotics:HandReachImage", max_episode_steps=50)

try:
    from .kitchen import KitchenGoalConditionedWrapper
    import adept_envs

    register(
        id="kitchen-all-v0",
        entry_point="research.envs.kitchen:KitchenAllV0",
        max_episode_steps=280,
        reward_threshold=1.0,
    )

except ImportError:
    print(
        "[research] Could not import Franka Kitchen envs. Please install the Adept Envs via the instructions in Play to"
        " Policy"
    )
