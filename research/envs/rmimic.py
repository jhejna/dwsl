import copy
import os

import gym
import h5py
import numpy as np
from robomimic.config import config_factory
from robomimic.utils import env_utils, file_utils, obs_utils


def get_robomimc_concat_keys(dataset_path):
    env_meta = file_utils.get_env_metadata_from_dataset(dataset_path)
    use_image_obs = dataset_path.endswith("image.hdf5")  # Determine if we are using images
    env_name = env_meta["env_name"] + ("_image" if use_image_obs else "")

    env_keys = {
        "PickPlaceCan": {
            "observation": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", ("object", 3, 7)],
            "goal": [("object", 0, 3)],
        },
        "NutAssemblySquare": {
            "observation": ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", ("object", 3, 7)],
            "goal": [("object", 0, 3)],
        },
    }

    assert env_name in env_keys, "Using currently unsupported dataset."
    obs_keys, goal_keys = env_keys[env_name]["observation"], env_keys[env_name]["goal"]

    obs_keys = sorted(obs_keys, key=lambda x: x[0] if isinstance(x, tuple) else x)
    goal_keys = sorted(goal_keys, key=lambda x: x[0] if isinstance(x, tuple) else x)

    env_keys = [k[0] if isinstance(k, tuple) else k for k in obs_keys]
    env_keys.extend([k[0] if isinstance(k, tuple) else k for k in goal_keys])
    print(env_keys)
    env_keys = list(set(env_keys))
    obs_modality_specs = {"obs": {("rgb" if use_image_obs else "low_dim"): env_keys}}

    obs_utils.initialize_obs_utils_with_obs_specs(obs_modality_specs)

    return obs_keys, goal_keys


class GoalConditionedRoboMimicEnv(gym.Env):
    def __init__(self, dataset_path, robomimic_path=None, horizon=500, terminate_on_success=True):
        if robomimic_path is not None:
            dataset_path = os.path.join(robomimic_path, dataset_path)
        env_meta = file_utils.get_env_metadata_from_dataset(dataset_path=dataset_path)
        use_image_obs = dataset_path.endswith("image.hdf5")  # Determine if we are using images
        self._env = env_utils.create_env_from_metadata(
            env_meta=env_meta,
            env_name=env_meta["env_name"],
            render=False,
            render_offscreen=False,
            use_image_obs=use_image_obs,
        )
        self._obs_keys, self._goal_keys = get_robomimc_concat_keys(dataset_path)
        # Open the hdf5 files.
        f = h5py.File(dataset_path, "r")
        demo_0 = next(iter(f["data"].keys()))

        # Compute the observation and action spaces
        act_dim = f["data"][demo_0]["actions"][0].shape[0]
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(act_dim,), dtype=np.float32)

        spaces = dict()
        dtype = np.uint8 if use_image_obs else np.float32
        low = 0 if dtype == np.uint8 else -np.inf
        high = 255 if dtype == np.uint8 else np.inf

        if len(self._obs_keys) > 0:
            # Concatenate on dimension 0
            shapes = [
                (
                    f["data"][demo_0]["obs"][k[0]][0, k[1] : k[2]].shape
                    if isinstance(k, tuple)
                    else f["data"][demo_0]["obs"][k][0].shape
                )
                for k in self._obs_keys
            ]
            shape = (sum([shape[0] for shape in shapes]),) + tuple(shapes[0][1:])
            spaces["observation"] = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)

        assert len(self._goal_keys) > 0, "Must have at least one goal key"
        shapes = [
            (
                f["data"][demo_0]["obs"][k[0]][0, k[1] : k[2]].shape
                if isinstance(k, tuple)
                else f["data"][demo_0]["obs"][k][0].shape
            )
            for k in self._goal_keys
        ]
        shape = (sum([shape[0] for shape in shapes]),) + tuple(shapes[0][1:])
        spaces["achieved_goal"] = gym.spaces.Box(low=low, high=high, shape=shape, dtype=dtype)
        spaces["desired_goal"] = copy.deepcopy(spaces["achieved_goal"])
        self.observation_space = gym.spaces.Dict(spaces)

        # Load the validation demos, and use the "successful" observations as goals
        f = h5py.File(dataset_path, "r")
        demos = [elem.decode("utf-8") for elem in np.array(f["mask/valid"][:])]
        goals = []
        for demo in demos:
            successful_mask = f["data"][demo]["dones"][:] == 1  # Assume done is an indicator variable for success...
            demo_goals = np.concatenate(
                [
                    (
                        f["data"][demo]["obs"][k[0]][successful_mask][:, k[1] : k[2]]
                        if isinstance(k, tuple)
                        else f["data"][demo]["obs"][k][successful_mask]
                    )
                    for k in self._goal_keys
                ],
                axis=1,
            )
            goals.append(demo_goals)
        self.goals = np.concatenate(goals, axis=0)

        # Close the file handler.
        f.close()

        self._goal = self.goals[np.random.randint(low=0, high=len(self.goals))]
        self._max_episode_steps = horizon
        self.terminate_on_success = terminate_on_success
        self.use_image_obs = use_image_obs
        self._ep_steps = 0

    def _process_obs(self, obs):
        # Concatenate everything on dim=0 according to the obs and goal keys
        achieved_goal = np.concatenate(
            [obs[k[0]][k[1] : k[2]] if isinstance(k, tuple) else obs[k] for k in self._goal_keys], axis=0
        )
        desired_goal = self._goal.copy()
        dict_obs = dict(achieved_goal=achieved_goal, desired_goal=desired_goal)
        if len(self._obs_keys) > 0:
            dict_obs["observation"] = np.concatenate(
                [obs[k[0]][k[1] : k[2]] if isinstance(k, tuple) else obs[k] for k in self._obs_keys], axis=0
            )
        return dict_obs

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        self._ep_steps += 1
        success = self._env.is_success()
        info["success"] = success["task"]
        if self.terminate_on_success and success["task"]:
            done = True

        if self._ep_steps == self._max_episode_steps:
            done = True
            info["discount"] = 1.0
        else:
            info["discount"] = 1 - float(done)

        obs = self._process_obs(obs)
        if not self.use_image_obs:
            info["goal_distance"] = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        return obs, reward, done, info

    def reset(self, **kwargs):
        obs = self._env.reset(**kwargs)
        self._ep_steps = 0
        self._goal = self.goals[np.random.randint(low=0, high=len(self.goals))]
        return self._process_obs(obs)
