import os
import random

import gym
import numpy as np
import torch

from research.utils import utils

from .replay_buffer import HindsightReplayBuffer


class BridgeDataset(HindsightReplayBuffer):
    """
    Class for loading the bridge dataset.
    It is constructed by simply overwriting the data generator option.
    """

    def __init__(self, *args, train=True, **kwargs):
        self.directory_suffix = "train" if train else "val"
        super().__init__(*args, **kwargs)
        assert self.path is not None, "Must provide path to the Brudge dataset"
        assert (
            "relabel_fraction" in kwargs and kwargs["relabel_fraction"] == 1.0
        ), "Must use full relabeling for the bridge dataset."

    def _data_generator(self):
        # By default get all of the file names that are distributed at the correct index
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        # First, get all of the file names and sort them
        data_files = []
        for directory, subdirectory, files in os.walk(self.path):
            for filename in files:
                if filename == "out.npy" and directory.endswith(self.directory_suffix):
                    data_files.append(os.path.join(self.path, directory, filename))
        # Sort the files.
        data_files.sort()

        if num_workers > 1 and len(data_files) == 1:
            print("[BridgeDataset] Warning: using multiple workers but single replay file.")
        elif num_workers > 1 and len(data_files) < num_workers:
            print("[BridgeDataset] Warning: using more workers than dataset files.")

        # Next, assign files to each worker
        # Take every nth file
        data_files = data_files[worker_id::num_workers]

        # Shuffle the files within each worker
        random.shuffle(data_files)

        # Determine the observation keys
        if isinstance(self.observation_space[self.achieved_key], gym.spaces.Dict):
            obs_keys = list(self.observation_space.spaces.keys())
        else:
            assert isinstance(self.observation_space[self.achieved_key], gym.spaces.Box)
            if self.observation_space[self.achieved_key].dtype == np.uint8:
                obs_keys = ["images0"]
            else:
                obs_keys = ["states"]

        for data_file in data_files:
            data = np.load(data_file, allow_pickle=True)
            for ep_idx in range(len(data)):
                action = [self.dummy_action]
                action.extend(data[ep_idx]["actions"])
                action = np.array(action)

                obs = {k: [] for k in obs_keys}
                # Add the bulk of the data
                for t in range(len(data[ep_idx]["observations"])):
                    for obs_key in obs_keys:
                        obs[obs_key].append(data[ep_idx]["observations"][t][obs_key])

                for obs_key in obs_keys:
                    # Add the final observation
                    obs[obs_key].append(data[ep_idx]["next_observations"][-1][obs_key])

                # Construct the final dataset values
                for obs_key in obs_keys:
                    obs_as_arr = np.array(obs[obs_key])
                    if "image" in obs_key:
                        obs_as_arr = obs_as_arr.transpose(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
                    assert obs_as_arr.shape[0] == action.shape[0]
                    obs[obs_key] = obs_as_arr

                if len(obs_keys) == 1:
                    obs = next(iter(obs.values()))

                obs = {self.achieved_key: obs}

                reward = np.zeros(action.shape[0], dtype=np.float32)
                done = np.zeros(action.shape[0], dtype=np.bool_)
                done[-1] = True  # Make sure to set the last part to done.
                discount = np.ones(action.shape[0], dtype=np.float32)
                kwargs = {}

                yield (obs, action, reward, done, discount, kwargs)
