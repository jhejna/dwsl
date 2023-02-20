import os

import numpy as np
import torch

from .replay_buffer import HindsightReplayBuffer


class KitchenDataset(HindsightReplayBuffer):
    """
    Simple Class that writes the data from the GCSL buffers into a HindsightReplayBuffer
    """

    def __init__(self, *args, test_fraction: float = 0.05, **kwargs):
        self.test_fraction = test_fraction
        super().__init__(*args, **kwargs)
        assert self.path is not None, "Must provide path to BeT Kitchen dataset"
        assert (
            "relabel_fraction" in kwargs and kwargs["relabel_fraction"] == 1.0
        ), "Must use full relabeling for kitchen dataset."

    def _data_generator(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Kitchen Dataset does not support sharded loading."

        observations = np.load(os.path.join(self.path, "observations_seq.npy")).astype(np.float32)  # Shape (T, B, D)
        actions = np.load(os.path.join(self.path, "actions_seq.npy")).astype(np.float32)  # Shape (T, B, D)
        masks = np.load(os.path.join(self.path, "existence_mask.npy"))  # Shape (T, B)
        end_points = masks.sum(axis=0).astype(np.int)

        observations = observations.transpose((1, 0, 2))  # (B, T, D)
        actions = actions.transpose((1, 0, 2))  # (B, T, D)

        # Get the indexes to use for training
        # this is done exactly as in Play-to-Policy
        # https://github.com/jeffacce/play-to-policy/blob/master/utils/__init__.py#L32
        l_all = observations.shape[0]
        idx = torch.randperm(l_all, generator=torch.Generator().manual_seed(42)).tolist()
        l_train = int((1 - self.test_fraction) * l_all)
        idx = idx[:l_train]

        # Loop through everything by episode
        for i in idx:
            # Compute the end time with the mask
            end = end_points[i]
            if end < 3:
                continue  # skip this episode
            obs = {
                "achieved_goal": observations[i, :end, :30],
                # Do not bother allocating the desired goal, use all relabeling.
            }
            dummy_action = np.expand_dims(self.dummy_action, axis=0)
            action = np.concatenate((dummy_action, actions[i, : end - 1]), axis=0)
            discount = np.ones(action.shape[0])  # Gets recomputed with HER
            reward = np.zeros(action.shape[0])  # Gets recomputed with HER
            done = np.zeros(action.shape[0], dtype=np.bool_)
            done[-1] = True  # Add the episode delineation
            kwargs = {}
            yield (obs, action, reward, done, discount, kwargs)
