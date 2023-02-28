import os

import gym
import h5py
import numpy as np
import torch

from research.envs.rmimic import get_robomimc_concat_keys
from research.utils.utils import remove_float64

from .replay_buffer import HindsightReplayBuffer


class GoalConditionedRobomimicDataset(HindsightReplayBuffer):
    """
    Simple Class that writes the data from the GoalConditionedRobomimicDatasets into a HindsightReplayBuffer
    """

    def __init__(self, *args, robomimic_path=None, **kwargs):
        self.robomimic_path = robomimic_path
        super().__init__(*args, **kwargs)

    def _data_generator(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Goal Conditioned Robomimic Dataset does not support sharded loading."
        if self.robomimic_path is not None:
            dataset_path = os.path.join(self.robomimic_path, self.path)
        else:
            dataset_path = self.path
        # Get the keys to use
        obs_keys, goal_keys = get_robomimc_concat_keys(dataset_path)

        # Compute the worker info
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        f = h5py.File(dataset_path, "r")
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/train"][:])]  # Extract the training demonstrations

        for i, demo in enumerate(demos):
            if i % num_workers != worker_id and len(demos) > num_workers:
                continue
            obs = dict()
            if len(obs_keys) > 0:
                observation = np.concatenate(
                    [
                        (
                            f["data"][demo]["obs"][k[0]][:, k[1] : k[2]]
                            if isinstance(k, tuple)
                            else f["data"][demo]["obs"][k]
                        )
                        for k in obs_keys
                    ],
                    axis=1,
                )
                final_observation = np.concatenate(
                    [
                        (
                            f["data"][demo]["next_obs"][k[0]][-1:, k[1] : k[2]]
                            if isinstance(k, tuple)
                            else f["data"][demo]["next_obs"][k][-1:]
                        )
                        for k in obs_keys
                    ],
                    axis=1,
                )
                obs["observation"] = np.concatenate((observation, final_observation), axis=0)  # Concat on time axis
            achieved_goal = np.concatenate(
                [
                    f["data"][demo]["obs"][k[0]][:, k[1] : k[2]] if isinstance(k, tuple) else f["data"][demo]["obs"][k]
                    for k in goal_keys
                ],
                axis=1,
            )
            final_achieved_goal = np.concatenate(
                [
                    (
                        f["data"][demo]["next_obs"][k[0]][-1:, k[1] : k[2]]
                        if isinstance(k, tuple)
                        else f["data"][demo]["next_obs"][k][-1:]
                    )
                    for k in goal_keys
                ],
                axis=1,
            )
            obs[self.achieved_key] = np.concatenate((achieved_goal, final_achieved_goal), axis=0)  # Concat on time axis
            obs = remove_float64(obs)

            dummy_action = np.expand_dims(self.dummy_action, axis=0)
            action = np.concatenate((dummy_action, f["data"][demo]["actions"]), axis=0)
            action = remove_float64(action)

            reward = np.concatenate(([0], f["data"][demo]["rewards"]), axis=0)
            reward = remove_float64(reward)

            done = np.zeros(action.shape[0], dtype=np.bool_)  # Gets recomputed with HER
            done[-1] = True
            discount = np.ones(action.shape[0])  # Gets recomputed with HER
            assert len(obs[self.achieved_key]) == len(action) == len(reward) == len(done) == len(discount)
            kwargs = dict()
            yield (obs, action, reward, done, discount, kwargs)

        f.close()  # Close the file handler.
