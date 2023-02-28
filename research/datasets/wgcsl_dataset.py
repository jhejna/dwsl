import gc
import os
import pickle
from typing import List, Optional

import d4rl
import gym
import numpy as np
import torch

from research.utils.utils import remove_float64

from .replay_buffer import HindsightReplayBuffer


class WGCSLDataset(HindsightReplayBuffer):
    """
    Simple Class that writes the data from the WGCSL buffers into a HindsightReplayBuffer
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        paths: List[str] = [],
        percents: Optional[List[float]] = None,
        load_from_end: bool = False,
        terminal_threshold: Optional[float] = None,
        trim_to_terminal: bool = False,
        **kwargs,
    ):
        assert len(paths) > 0, "Must provide at least one dataset path"
        percents = [1.0] * len(paths) if percents is None else percents
        self.percents = percents
        self.paths = paths
        self.load_from_end = load_from_end
        self.terminal_threshold = terminal_threshold
        self.trim_to_terminal = trim_to_terminal
        super().__init__(observation_space, action_space, **kwargs)

    def _data_generator(self):
        # Creating sharding from the paths
        sharded_paths, sharded_percents = [], []
        for path, percent in zip(self.paths, self.percents):
            if os.path.isdir(path):
                # then we have sharding
                for f in os.listdir(path):
                    if f.startswith("buffer") and f.endswith(".pkl"):
                        sharded_paths.append(os.path.join(path, f))
                        sharded_percents.append(percent)
            else:
                sharded_paths.append(path)
                sharded_percents.append(percent)

        # Get the torch worker info
        worker_info = torch.utils.data.get_worker_info()
        num_workers = 1 if worker_info is None else worker_info.num_workers
        worker_id = 0 if worker_info is None else worker_info.id

        for buffer_idx, (path, percent) in enumerate(zip(sharded_paths, sharded_percents)):
            if buffer_idx % num_workers != worker_id and len(sharded_paths) > len(self.paths):
                continue  # Skip the file if we have shards.
            with open(path, "rb") as f:
                data = pickle.load(f)
            eps, horizon = data["ag"].shape[:2]
            # Add the episodes
            ep_idxs = range(eps - int(eps * percent), eps) if self.load_from_end else range(int(eps * percent))
            for i in ep_idxs:
                # We need to make sure we appropriately handle the dummy transition
                obs = dict(achieved_goal=data["ag"][i].copy())
                if "o" in data:
                    obs["observation"] = data["o"][i].copy()
                if "g" in data:
                    goal = data["g"][i]
                    obs[self.goal_key] = np.concatenate((goal[:1], goal), axis=0)
                obs = remove_float64(obs)
                dummy_action = np.expand_dims(self.dummy_action, axis=0)
                action = np.concatenate((dummy_action, data["u"][i]), axis=0)
                action = remove_float64(action)

                kwargs = dict()
                if obs[self.achieved_key].dtype == np.float32 and self.goal_key in obs:
                    kwargs["goal_distance"] = np.linalg.norm(obs[self.goal_key] - obs[self.achieved_key], axis=-1)

                # If we have a terminal threshold compute and store the horizon
                if self.terminal_threshold is not None and "goal_distance" in kwargs:
                    terminals = kwargs["goal_distance"] < self.terminal_threshold
                    horizon = -100 * np.ones(terminals.shape, dtype=np.int)
                    (ends,) = np.where(terminals)
                    starts = np.concatenate(([0], ends[:-1] + 1))
                    for start, end in zip(starts, ends):
                        horizon[start : end + 1] = np.arange(end - start + 1, 0, -1)
                    kwargs["horizon"] = horizon
                    # If we trim to the terminal, trim the episode.
                    if self.trim_to_terminal and len(ends) > 0:
                        terminal_idx = starts[0]  # Terminate on the first
                        if terminal_idx < 3:
                            continue  # not enough for a whole episode
                        obs = {k: v[:terminal_idx] for k, v in obs.items()}
                        action = action[:terminal_idx]
                        kwargs = {k: v[:terminal_idx] for k, v in kwargs.items()}

                discount = np.ones(action.shape[0])  # Gets recomputed with HER
                reward = np.zeros(action.shape[0])  # Gets recomputed with HER
                done = np.zeros(action.shape[0], dtype=np.bool_)
                done[-1] = True  # Add the episode delineation
                assert len(obs[self.achieved_key]) == len(action) == len(reward) == len(done) == len(discount)
                yield (obs, action, reward, done, discount, kwargs)
            # Explicitly delete the data objects to save memory
            del data
            del obs
            del action
            del reward
            del done
            del discount
            gc.collect()
