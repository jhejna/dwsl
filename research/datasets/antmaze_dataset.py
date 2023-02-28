from typing import Optional

import d4rl
import gym
import numpy as np
import torch

from .replay_buffer import HindsightReplayBuffer


class GoalConditionedAntDataset(HindsightReplayBuffer):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        name: str,
        d4rl_path: Optional[str] = None,
        action_eps: float = 0.0,
        max_ep: Optional[int] = None,
        terminal_threshold: Optional[float] = None,
        trim_to_terminal: bool = False,
        **kwargs,
    ) -> None:
        assert "ant" in name
        self.env_name = name
        self.action_eps = action_eps
        self.max_ep = max_ep
        self.terminal_threshold = terminal_threshold
        self.trim_to_terminal = trim_to_terminal
        if d4rl_path is not None:
            d4rl.set_dataset_path(d4rl_path)
        super().__init__(observation_space, action_space, **kwargs)

    def _data_generator(self):
        worker_info = torch.utils.data.get_worker_info()
        assert worker_info is None, "Kitchen Dataset does not support sharded loading."

        env = gym.make(self.env_name)
        dataset = env.get_dataset()
        num_ep_added = 0

        obs_ = []
        ag_ = []
        g_ = []
        action_ = [self.dummy_action]
        reward_ = [0.0]
        done_ = [False]
        discount_ = [1.0]
        episode_step = 0

        for i in range(dataset["rewards"].shape[0]):
            obs = obs = dataset["observations"][i, 2:].astype(np.float32)
            achieved_goal = dataset["observations"][i, :2].astype(np.float32)
            desired_goal = dataset["infos/goal"][i].astype(np.float32)
            action = dataset["actions"][i].astype(np.float32)
            reward = dataset["rewards"][i].astype(np.float32)
            terminal = bool(dataset["terminals"][i])
            done = dataset["timeouts"][i]
            if self.terminal_threshold is not None and self.trim_to_terminal:
                done = done or (np.linalg.norm(achieved_goal - desired_goal) < self.terminal_threshold)

            obs_.append(obs)
            ag_.append(achieved_goal)
            g_.append(desired_goal)
            action_.append(action)
            reward_.append(reward)
            discount_.append(1 - float(terminal))
            done_.append(done)

            episode_step += 1

            if done:
                if "next_observations" in dataset:
                    obs_.append(dataset["next_observations"][i, 2:].astype(np.float32))
                    ag_.append(dataset["next_observations"][i, :2].astype(np.float32))
                    g_.append(dataset["infos/goal"][i].astype(np.float32))
                else:
                    # We need to do somethign to pad to the full length.
                    # The default solution is to get rid of this transtion
                    # but we need a transition with the terminal flag for our replay buffer
                    # implementation to work.
                    # Since we always end up masking this out anyways, it shouldn't matter and we can just repeat
                    obs_.append(dataset["observations"][i, 2:].astype(np.float32))
                    ag_.append(dataset["observations"][i, :2].astype(np.float32))
                    g_.append(dataset["infos/goal"][i].astype(np.float32))

                dict_obs = {
                    "observation": np.array(obs_),
                    self.achieved_key: np.array(ag_),
                    self.goal_key: np.array(g_),
                }
                action_ = np.array(action_)
                if self.action_eps > 0.0:
                    action_ = np.clip(action_, -1.0 + self.action_eps, 1.0 - self.action_eps)
                reward_ = np.array(reward_)
                discount_ = np.array(discount_)
                done_ = np.array(done_, dtype=np.bool_)
                dist = np.linalg.norm(dict_obs[self.goal_key] - dict_obs[self.achieved_key], axis=-1)
                kwargs = dict(goal_distance=dist)
                # Compute the ground truth horizon metrics
                if self.terminal_threshold is not None:
                    terminals = dist < self.terminal_threshold
                    horizon = -100 * np.ones(terminals.shape, dtype=np.int)
                    (ends,) = np.where(terminals)
                    starts = np.concatenate(([0], ends[:-1] + 1))
                    for start, end in zip(starts, ends):
                        horizon[start : end + 1] = np.arange(end - start + 1, 0, -1)
                    kwargs["horizon"] = horizon

                if len(reward_) > 3:
                    # print("yielded")
                    yield (dict_obs, action_, reward_, done_, discount_, kwargs)

                # reset the episode trackers
                obs_ = []
                ag_ = []
                g_ = []
                action_ = [self.dummy_action]
                reward_ = [0.0]
                done_ = [False]
                discount_ = [1.0]
                episode_step = 0

                num_ep_added += 1
                if self.max_ep is not None and num_ep_added == self.max_ep:
                    break

        # Finally clean up the environment
        del dataset
        del env
