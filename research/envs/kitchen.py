"""Environments using kitchen and Franka robot."""
import os

import gym
import numpy as np
import torch
from adept_envs.franka.kitchen_multitask_v0 import KitchenTaskRelaxV1
from dm_control.mujoco import engine

OBS_ELEMENT_INDICES = {
    "bottom burner": np.array([11, 12]),
    "top burner": np.array([15, 16]),
    "light switch": np.array([17, 18]),
    "slide cabinet": np.array([19]),
    "hinge cabinet": np.array([20, 21]),
    "microwave": np.array([22]),
    "kettle": np.array([23, 24, 25, 26, 27, 28, 29]),
}
OBS_ELEMENT_GOALS = {
    "bottom burner": np.array([-0.88, -0.01]),
    "top burner": np.array([-0.92, -0.01]),
    "light switch": np.array([-0.69, -0.05]),
    "slide cabinet": np.array([0.37]),
    "hinge cabinet": np.array([0.0, 1.45]),
    "microwave": np.array([-0.75]),
    "kettle": np.array([-0.23, 0.75, 1.62, 0.99, 0.0, 0.0, -0.06]),
}
BONUS_THRESH = 0.3


class KitchenBase(KitchenTaskRelaxV1):
    # A string of element names. The robot's task is then to modify each of
    # these elements appropriately.
    TASK_ELEMENTS = []
    ALL_TASKS = [
        "bottom burner",
        "top burner",
        "light switch",
        "slide cabinet",
        "hinge cabinet",
        "microwave",
        "kettle",
    ]
    REMOVE_TASKS_WHEN_COMPLETE = True
    TERMINATE_ON_TASK_COMPLETE = True
    TERMINATE_ON_WRONG_COMPLETE = False
    COMPLETE_IN_ANY_ORDER = True  # This allows for the tasks to be completed in arbitrary order.

    def __init__(self, dataset_url=None, ref_max_score=None, ref_min_score=None, **kwargs):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        self.all_completions = []
        self.goal_masking = True
        super(KitchenBase, self).__init__(**kwargs)

    def set_goal_masking(self, goal_masking=True):
        """Sets goal masking for goal-conditioned approaches (like RPL)."""
        self.goal_masking = goal_masking

    def _get_task_goal(self, task=None, actually_return_goal=False):
        if task is None:
            task = ["microwave", "kettle", "bottom burner", "light switch"]
        new_goal = np.zeros_like(self.goal)
        if self.goal_masking and not actually_return_goal:
            return new_goal
        for element in task:
            element_idx = OBS_ELEMENT_INDICES[element]
            element_goal = OBS_ELEMENT_GOALS[element]
            new_goal[element_idx] = element_goal

        return new_goal

    def reset_model(self):
        self.tasks_to_complete = list(self.TASK_ELEMENTS)
        self.all_completions = []
        return super(KitchenBase, self).reset_model()

    def _get_reward_n_score(self, obs_dict):
        reward_dict, score = super(KitchenBase, self)._get_reward_n_score(obs_dict)
        next_q_obs = obs_dict["qp"]
        next_obj_obs = obs_dict["obj_qp"]
        next_goal = self._get_task_goal(task=self.TASK_ELEMENTS, actually_return_goal=True)  # obs_dict['goal']
        idx_offset = len(next_q_obs)
        completions = []
        all_completed_so_far = True
        for element in self.tasks_to_complete:
            element_idx = OBS_ELEMENT_INDICES[element]
            distance = np.linalg.norm(next_obj_obs[..., element_idx - idx_offset] - next_goal[element_idx])
            complete = distance < BONUS_THRESH
            condition = complete and all_completed_so_far if not self.COMPLETE_IN_ANY_ORDER else complete
            if condition:  # element == self.tasks_to_complete[0]:
                completions.append(element)
                self.all_completions.append(element)
            all_completed_so_far = all_completed_so_far and complete
        if self.REMOVE_TASKS_WHEN_COMPLETE:
            [self.tasks_to_complete.remove(element) for element in completions]
        bonus = float(len(completions))
        reward_dict["bonus"] = bonus
        reward_dict["r_total"] = bonus
        score = bonus
        return reward_dict, score

    def step(self, a, b=None):
        obs, reward, done, env_info = super(KitchenBase, self).step(a, b=b)
        if self.TERMINATE_ON_TASK_COMPLETE:
            done = not self.tasks_to_complete
        if self.TERMINATE_ON_WRONG_COMPLETE:
            all_goal = self._get_task_goal(task=self.ALL_TASKS)
            for wrong_task in list(set(self.ALL_TASKS) - set(self.TASK_ELEMENTS)):
                element_idx = OBS_ELEMENT_INDICES[wrong_task]
                distance = np.linalg.norm(obs[..., element_idx] - all_goal[element_idx])
                complete = distance < BONUS_THRESH
                if complete:
                    done = True
                    break
        env_info["all_completions"] = self.all_completions
        return obs, reward, done, env_info

    def get_goal(self):
        """Loads goal state from dataset for goal-conditioned approaches (like RPL)."""
        raise NotImplementedError


class KitchenMicrowaveKettleBottomBurnerLightV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "bottom burner", "light switch"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenMicrowaveKettleLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ["microwave", "kettle", "light switch", "slide cabinet"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenKettleMicrowaveLightSliderV0(KitchenBase):
    TASK_ELEMENTS = ["kettle", "microwave", "light switch", "slide cabinet"]
    COMPLETE_IN_ANY_ORDER = False


class KitchenMicrowaveV0(KitchenBase):
    TASK_ELEMENTS = ["microwave"]
    COMPLETE_IN_ANY_ORDER = True


class KitchenAllV0(KitchenBase):
    TASK_ELEMENTS = KitchenBase.ALL_TASKS


def kitchen_sparse(achieved, desired, info=None):
    # Vectorized reward function.
    # Returns -1 we are not at the goal, and zero otherwise
    assert achieved.shape == desired.shape
    d = np.linalg.norm(achieved - desired, axis=-1)
    return (d < BONUS_THRESH).astype(np.float32)


def negative_kitchen_sparse(achieved, desired, info=None):
    # Vectorized reward function.
    # Returns -1 we are not at the goal, and zero otherwise
    assert achieved.shape == desired.shape
    d = np.linalg.norm(achieved - desired, axis=-1)
    return -(d > BONUS_THRESH).astype(np.float32)


class KitchenGoalConditionedWrapper(gym.Wrapper):
    """
    This is an evaluation wrapper that is written to match the CBeT paper.
    """

    def __init__(self, *args, path=None, test_fraction=0.05, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Expose the TimeLimit wrapper if one exists
        if hasattr(self.env, "_max_episode_steps"):
            self._max_episode_steps = self.env._max_episode_steps

        obs_low = self.observation_space.low
        obs_high = self.observation_space.high
        self.observation_space = gym.spaces.Dict(
            {
                "achieved_goal": gym.spaces.Box(low=obs_low[:30], high=obs_high[:30], dtype=np.float32),
                "desired_goal": gym.spaces.Box(low=obs_low[30:], high=obs_high[30:], dtype=np.float32),
            }
        )

        observations = np.load(os.path.join(path, "observations_seq.npy")).astype(np.float32)  # Shape (T, B, D)
        masks = np.load(os.path.join(path, "existence_mask.npy"))  # Shape (T, B)
        self.end_points = masks.sum(axis=0).astype(np.int)
        self.goals = observations.transpose((1, 0, 2))[:, :, :30]  # (B, T, D)
        self.onehot_labels = (
            torch.load(os.path.join(path, "onehot_goals.pth")).permute(1, 0, 2).numpy().astype(np.bool_)
        )
        # Get the indexes to use for validation
        # this is done exactly as in Play-to-Policy
        # https://github.com/jeffacce/play-to-policy/blob/master/utils/__init__.py#L32
        l_all = self.goals.shape[0]
        idx = torch.randperm(l_all, generator=torch.Generator().manual_seed(42)).tolist()
        l_train = int((1 - test_fraction) * l_all)
        idx = idx[l_train:]

        # Trim everything to be a part of the test set.
        self.end_points = self.end_points[idx]
        self.goals = self.goals[idx]
        self.onehot_labels = self.onehot_labels[idx]

        self._sample_goal()

    def _sample_goal(self):
        goal_idx = np.random.randint(len(self.end_points))
        self.desired_goal = self.goals[goal_idx, self.end_points[goal_idx] - 1, :]
        self.expected_mask = self.onehot_labels[goal_idx].max(0)

    def _get_obs(self, obs):
        obs = {
            "achieved_goal": obs[:30],
            "desired_goal": self.desired_goal,
        }
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        obs = self._get_obs(obs)
        # Compute the distance
        d = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        info["goal_distance"] = d
        # Compute task success metrics
        tasks = np.array(self.env.ALL_TASKS)
        expected_tasks = tasks[self.expected_mask].tolist()
        conditional_done = set(self.env.all_completions).intersection(expected_tasks)
        info["success"] = len(conditional_done) / np.sum(self.expected_mask)
        info["completions"] = len(conditional_done)
        del info["all_completions"]
        return obs, reward, done, info

    def reset(self, *args, **kwargs):
        obs = self.env.reset(*args, **kwargs)
        self._sample_goal()
        return self._get_obs(obs)
