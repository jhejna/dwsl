import pickle

import gym
import mujoco_py
import numpy as np
from gym.envs.robotics import FetchPickAndPlaceEnv, FetchPushEnv, FetchReachEnv, FetchSlideEnv, HandReachEnv


def fetch_sparse(achieved, desired, info=None):
    # Vectorized reward function.
    # Returns -1 we are not at the goal, and zero otherwise
    assert achieved.shape == desired.shape
    d = np.linalg.norm(achieved - desired, axis=-1)
    return (d < 0.05).astype(np.float32)


def negative_fetch_sparse(achieved, desired, info=None):
    # Vectorized reward function.
    # Returns -1 we are not at the goal, and zero otherwise
    # assert achieved.shape == desired.shape
    d = np.linalg.norm(achieved - desired, axis=-1)
    return -(d > 0.05).astype(np.float32)


def fetch_dummy(achieved, desired, info=None):
    return np.zeros(achieved.shape[0], dtype=np.float32)


def hand_sparse(achieved, desired, info=None):
    # Vectorized reward function.
    # Returns -1 we are not at the goal, and zero otherwise
    assert achieved.shape == desired.shape
    d = np.linalg.norm(achieved - desired, axis=-1)
    return (d < 0.01).astype(np.float32)


def negative_hand_sparse(achieved, desired, info=None):
    # Vectorized reward function.
    # Returns -1 we are not at the goal, and zero otherwise
    # assert achieved.shape == desired.shape
    d = np.linalg.norm(achieved - desired, axis=-1)
    return -(d > 0.01).astype(np.float32)


class GymSparseRewardWrapper(gym.Wrapper):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Expose the TimeLimit wrapper if one exists
        if hasattr(self.env, "_max_episode_steps"):
            self._max_episode_steps = self.env._max_episode_steps
        # Get determine the reward function
        self._reward_fn = hand_sparse if "Hand" in str(type(self.env.unwrapped)) else fetch_sparse

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        reward = self._reward_fn(obs["achieved_goal"], obs["desired_goal"]).item()
        info["goal_distance"] = np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"])
        return obs, reward, done, info


# The below classes are modified versions of the Fetch environments designed to be suitable for images.


class ModifiedFetchReachEnv(FetchReachEnv):
    """Modify FetchReach environment to produce images."""

    def __init__(self, height=64, width=64, render=True):
        self.height = height
        self.width = width
        self.do_render = render  # able to disable image generation for faster training when images not needed
        super(ModifiedFetchReachEnv, self).__init__()
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset(self):
        # generate the new goal image
        obs = super(ModifiedFetchReachEnv, self).reset()
        self._goal = obs["desired_goal"].copy()

        if self.do_render:
            for _ in range(10):
                hand = obs["achieved_goal"]
                obj = obs["desired_goal"]
                delta = obj - hand
                a = np.concatenate([np.clip(10 * delta, -1, 1), [0.0]])
                obs, _, _, _ = super(ModifiedFetchReachEnv, self).step(a)

            self._goal_image = self._get_image()

        obs = super(ModifiedFetchReachEnv, self).reset()
        obs["desired_goal"] = self._goal.copy()
        self.goal = self._goal
        return obs

    def step(self, action):
        obs, _, _, _ = super(ModifiedFetchReachEnv, self).step(action)
        dist = np.linalg.norm(obs["achieved_goal"] - self._goal)
        success = dist < 0.05
        r = float(success)
        info = {"success": success, "goal_distance": dist}

        return obs, r, False, info

    def _get_image(self):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode="rgb_array", height=self.height, width=self.width)
        return img.transpose(2, 0, 1)

    def _viewer_setup(self):
        super(ModifiedFetchReachEnv, self)._viewer_setup()
        self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.75, 0.4])
        self.viewer.cam.distance = 1
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -40

    def get_image_obs(self):
        assert self.do_render
        return dict(achieved_goal=self._get_image().copy(), desired_goal=self._goal_image.copy())

    def set_render(self, render=True):
        self.do_render = render


class ModifiedFetchPushEnv(FetchPushEnv):
    """Modify FetchPush environment to produce images."""

    def __init__(self, height=64, width=64, render=True):
        self.height = height
        self.width = width
        self.do_render = render
        super(ModifiedFetchPushEnv, self).__init__()
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def _move_hand_to_obj(self):
        obs = super(ModifiedFetchPushEnv, self)._get_obs()
        for _ in range(100):
            hand = obs["observation"][:3]
            obj = obs["achieved_goal"] + np.array([-0.02, 0.0, 0.0])
            delta = obj - hand
            if np.linalg.norm(delta) < 0.06:
                break
            a = np.concatenate([np.clip(delta, -1, 1), [0.0]])
            obs, _, _, _ = super(ModifiedFetchPushEnv, self).step(a)
        return obs

    def reset(self):
        # generate the new goal image
        obs = super(ModifiedFetchPushEnv, self).reset()

        object_qpos = self.sim.data.get_joint_qpos("object0:joint")

        for _ in range(8):
            super(ModifiedFetchPushEnv, self).step(np.array([-1.0, 0.0, 0.0, 0.0]))

        self.sim.data.set_joint_qpos("object0:joint", object_qpos)
        self._move_hand_to_obj()
        block_xyz = self.sim.data.get_joint_qpos("object0:joint")[:3]
        if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
            print("Bad reset, recursing.")
            return self.reset()

        self._goal = block_xyz[:3].copy()

        if self.do_render:
            self._goal_image = self._get_image()

        while True:  # make sure goal isn't reached to start
            obs = super(ModifiedFetchPushEnv, self).reset()
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            if np.linalg.norm(object_qpos[:2] - self._goal[:2]) > 0.05:
                break

        obs["desired_goal"] = self._goal.copy()
        self.goal = self._goal

        return obs

    def step(self, action):
        obs, _, _, _ = super(ModifiedFetchPushEnv, self).step(action)
        block_xy = self.sim.data.get_joint_qpos("object0:joint")[:2]
        dist = np.linalg.norm(block_xy - self._goal[:2])
        success = dist < 0.05
        r = float(success)
        info = {"success": success, "goal_distance": dist}
        obs["desired_goal"] = self._goal.copy()

        return obs, r, False, info

    def _get_image(self):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode="rgb_array", height=self.height, width=self.width)
        return img.transpose(2, 0, 1)

    def _viewer_setup(self):
        super(ModifiedFetchPushEnv, self)._viewer_setup()
        self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.75, 0.4])
        self.viewer.cam.distance = 1
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -40

    def get_image_obs(self):
        assert self.do_render
        return dict(achieved_goal=self._get_image().copy(), desired_goal=self._goal_image.copy())

    def set_render(self, render=True):
        self.do_render = render


class ModifiedFetchPickAndPlaceEnv(FetchPickAndPlaceEnv):
    """Modify FetchPush environment to produce images."""

    def __init__(self, height=64, width=64, render=True):
        self.height = height
        self.width = width
        self.do_render = render
        super(ModifiedFetchPickAndPlaceEnv, self).__init__()
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def _reach_goal(self, last_obs):
        max_steps = 50
        timestep = 0

        goal = last_obs["desired_goal"]
        object_pos = last_obs["observation"][3:6]
        object_rel_pos = last_obs["observation"][6:9]

        object_oriented_goal = object_rel_pos.copy()
        object_oriented_goal[2] += 0.03  # first make the gripper go slightly above the object

        while np.linalg.norm(object_oriented_goal) >= 0.005 and timestep <= max_steps:
            action = [0, 0, 0, 0]
            object_oriented_goal = object_rel_pos.copy()
            object_oriented_goal[2] += 0.03

            for i in range(len(object_oriented_goal)):
                action[i] = object_oriented_goal[i] * 6

            action[len(action) - 1] = 0.05  # open

            obs_data_new, reward, done, info = super(ModifiedFetchPickAndPlaceEnv, self).step(action)
            timestep += 1

            object_pos = obs_data_new["observation"][3:6]
            object_rel_pos = obs_data_new["observation"][6:9]

        while np.linalg.norm(object_rel_pos) >= 0.005 and timestep <= max_steps:
            action = [0, 0, 0, 0]
            for i in range(len(object_rel_pos)):
                action[i] = object_rel_pos[i] * 6

            action[len(action) - 1] = -0.005

            obs_data_new, reward, done, info = super(ModifiedFetchPickAndPlaceEnv, self).step(action)
            timestep += 1

            object_pos = obs_data_new["observation"][3:6]
            object_rel_pos = obs_data_new["observation"][6:9]

        while np.linalg.norm(goal - object_pos) >= 0.01 and timestep <= max_steps:
            action = [0, 0, 0, 0]
            for i in range(len(goal - object_pos)):
                action[i] = (goal - object_pos)[i] * 6

            action[len(action) - 1] = -0.005

            obs_data_new, reward, done, info = super(ModifiedFetchPickAndPlaceEnv, self).step(action)
            timestep += 1

            object_pos = obs_data_new["observation"][3:6]
            object_rel_pos = obs_data_new["observation"][6:9]

        success = np.linalg.norm(goal - object_pos) < 0.01
        return success

    def reset(self):
        # generate the new goal image
        obs = super(ModifiedFetchPickAndPlaceEnv, self).reset()

        success = self._reach_goal(obs)
        if not success:
            print("Bad reset, recursing.")
            return self.reset()

        block_xyz = self.sim.data.get_joint_qpos("object0:joint")[:3]
        self._goal = block_xyz[:3].copy()

        if self.do_render:
            self._goal_image = self._get_image()

        while True:  # make sure goal isn't reached to start
            obs = super(ModifiedFetchPickAndPlaceEnv, self).reset()
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            if np.linalg.norm(object_qpos[:3] - self._goal[:3]) > 0.05:
                break

        obs["desired_goal"] = self._goal.copy()
        self.goal = self._goal

        return obs

    def step(self, action):
        obs, _, _, _ = super(ModifiedFetchPickAndPlaceEnv, self).step(action)
        block_xyz = self.sim.data.get_joint_qpos("object0:joint")[:3]
        dist = np.linalg.norm(block_xyz - self._goal)
        success = dist < 0.05
        r = float(success)
        info = {"success": success, "goal_distance": dist}
        obs["desired_goal"] = self._goal.copy()

        return obs, r, False, info

    def _get_image(self):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode="rgb_array", height=self.height, width=self.width)
        return img.transpose(2, 0, 1)

    def _viewer_setup(self):
        super(ModifiedFetchPickAndPlaceEnv, self)._viewer_setup()
        self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.75, 0.4])
        self.viewer.cam.distance = 1
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -40

    def get_image_obs(self):
        assert self.do_render
        return dict(achieved_goal=self._get_image().copy(), desired_goal=self._goal_image.copy())

    def set_render(self, render=True):
        self.do_render = render


class ModifiedFetchSlideEnv(FetchSlideEnv):
    """Modify FetchSlide environment to produce images."""

    def __init__(self, height=64, width=64, render=True):
        self.height = height
        self.width = width
        self.do_render = render
        super(ModifiedFetchSlideEnv, self).__init__()
        self.sim.model.geom_rgba[1:5] = 0  # Hide the lasers

    def reset(self):
        # generate the new goal image
        obs = super(ModifiedFetchSlideEnv, self).reset()

        object_qpos = self.sim.data.get_joint_qpos("object0:joint")
        object_qpos[:3] = obs["desired_goal"]
        self._goal = obs["desired_goal"].copy()

        # randomize arm
        for _ in range(20):
            super(ModifiedFetchSlideEnv, self).step(self.action_space.sample())

        self.sim.data.set_joint_qpos("object0:joint", object_qpos)

        block_xyz = self.sim.data.get_joint_qpos("object0:joint")[:3]
        if block_xyz[2] < 0.4:  # If block has fallen off the table, recurse.
            print("Bad reset, recursing.")
            return self.reset()

        if self.do_render:
            self._goal_image = self._get_image()

        while True:  # make sure goal isn't reached to start
            obs = super(ModifiedFetchSlideEnv, self).reset()
            object_qpos = self.sim.data.get_joint_qpos("object0:joint")
            if np.linalg.norm(object_qpos[:2] - self._goal[:2]) > 0.1:
                break

        obs["desired_goal"] = self._goal.copy()
        self.goal = self._goal

        return obs

    def step(self, action):
        obs, _, _, _ = super(ModifiedFetchSlideEnv, self).step(action)
        block_xy = self.sim.data.get_joint_qpos("object0:joint")[:2]
        dist = np.linalg.norm(block_xy - self._goal[:2])
        success = dist < 0.05
        r = float(success)
        info = {"success": success, "goal_distance": dist}
        obs["desired_goal"] = self._goal.copy()

        return obs, r, False, info

    def _get_image(self):
        self.sim.data.site_xpos[0] = 1_000_000
        img = self.render(mode="rgb_array", height=self.height, width=self.width)
        return img.transpose(2, 0, 1)

    def _viewer_setup(self):
        super(ModifiedFetchSlideEnv, self)._viewer_setup()
        self.viewer.cam.lookat[Ellipsis] = np.array([1.2, 0.75, 0.4])
        self.viewer.cam.distance = 1.75
        self.viewer.cam.azimuth = 180
        self.viewer.cam.elevation = -40

    def get_image_obs(self):
        assert self.do_render
        return dict(achieved_goal=self._get_image().copy(), desired_goal=self._goal_image.copy())

    def set_render(self, render=True):
        self.do_render = render


class FetchImageWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        env.set_render(True)
        shape = (3, env.height, env.width)

        self.observation_space = gym.spaces.Dict(
            {
                "achieved_goal": gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
                "desired_goal": gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
            }
        )

        # Expose the TimeLimit wrapper if one exists
        if hasattr(self.env, "_max_episode_steps"):
            self._max_episode_steps = self.env._max_episode_steps

    def step(self, action):
        _, reward, done, info = self.env.step(action)
        # Throw away the state-based observation.
        return self._get_obs(), reward, done, info

    def reset(self):
        self.env.reset()
        return self._get_obs()

    def _get_obs(self):
        return self.env.get_image_obs()


class HandReachImageIntermediate(HandReachEnv):
    def __init__(self, *args, goal_path=None, width=64, height=64, **kwargs):
        super().__init__(*args, **kwargs)
        # Check to see if we are using the Modified XML file. If not the user must replace it in their install.
        error_msg = "In order to use HandReachImage, please change the XML file in gym to the following one: "
        error_msg += "https://github.com/Farama-Foundation/Gymnasium-Robotics/blob/main/gymnasium_robotics/envs/assets"
        error_msg += "/hand/reach.xml"
        error_msg += " this fixes the rendering bug documented here: https://github.com/openai/gym/issues/2061"
        error_msg += " this can be done by cping into the current gym install. Check location with gym.__file__."
        for finger_idx in range(5):
            site_name = "target{}".format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            assert (self.sim.model.site_pos[site_id] == np.array([0.01, 0, 0])).all(), error_msg
            site_name = "finger{}".format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            assert (self.sim.model.site_pos[site_id] == np.array([0.01, 0, 0])).all(), error_msg

        # Modify the observation space
        self.width = width
        self.height = height
        shape = (3, self.height, self.width)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(low=0, high=0, shape=(1,)),  # Filler
                "achieved_goal": gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
                "desired_goal": gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
            }
        )
        # Load the goals
        if goal_path is not None:
            with open(goal_path, "rb") as f:
                goal_data = pickle.load(f)
            self._state_goals, self._image_goals = goal_data["g"], goal_data["img"]
        else:
            print("[research] Warning: goal_path for HandReachImage not provided.")

        self._goal_image = self.observation_space.sample()["desired_goal"]

    def get_image(self):
        img = self.render(mode="rgb_array", height=self.height, width=self.width)
        return img.transpose(2, 0, 1)

    def get_image_obs(self):
        return dict(achieved_goal=self.get_image().copy(), desired_goal=self._goal_image.copy())

    def step(self, action):
        _, reward, done, info = super().step(action)
        # Get our image observations
        # Invert the reward by adding one to make it positive sparse
        reward += 1
        return self.get_image_obs(), reward, done, info

    def set_state(self, obs):
        # Set the state from an observation
        achieved_goal_size = obs["achieved_goal"].shape[0]
        qpos, qvel = np.split(obs["observation"][:-achieved_goal_size], 2)
        self.goal = obs["desired_goal"].copy()  # Set the goal
        old_state = self.sim.get_state()
        new_state = mujoco_py.MjSimState(old_state.time, qpos, qvel, old_state.act, old_state.udd_state)
        self.sim.set_state(new_state)
        self.sim.forward()  # Call once to apply the new state

    def _render_callback(self):
        # Visualize targets.
        sites_offset = (self.sim.data.site_xpos - self.sim.model.site_pos).copy()

        # Move the target sites wayyyy far away
        for finger_idx in range(5):
            site_name = "target{}".format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = np.array([100, 100, 100])

        # Visualize finger positions.
        achieved_goal = self._get_achieved_goal().reshape(5, 3)
        for finger_idx in range(5):
            site_name = "finger{}".format(finger_idx)
            site_id = self.sim.model.site_name2id(site_name)
            self.sim.model.site_pos[site_id] = achieved_goal[finger_idx] - sites_offset[site_id]
        self.sim.forward()

    def reset(self, *args, **kwargs):
        assert self._image_goals is not None
        assert self._state_goals is not None
        super().reset(*args, **kwargs)

        # Set a goal
        idx = np.random.randint(self._state_goals.shape[0])
        self.goal = self._state_goals[idx].copy()
        self._goal_image = self._image_goals[idx].copy()
        return self.get_image_obs()

    def _viewer_setup(self):
        # Zoom the camera in a bit
        body_id = self.sim.model.body_name2id("robot0:palm")
        lookat = self.sim.data.body_xpos[body_id]
        for idx, value in enumerate(lookat):
            self.viewer.cam.lookat[idx] = value
        self.viewer.cam.distance = 0.45  # Reduced from 0.5
        self.viewer.cam.azimuth = 55.0
        self.viewer.cam.elevation = -25.0


class HandReachImage(gym.Env):
    # We need to wrap the hand reach env in order to avoid exceptions from
    # not having "observation" in the goal space

    def __init__(self, *args, **kwargs):
        self.env = HandReachImageIntermediate(*args, **kwargs)
        shape = (3, self.env.height, self.env.width)
        self.observation_space = gym.spaces.Dict(
            {
                "achieved_goal": gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
                "desired_goal": gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
            }
        )
        self.action_space = self.env.action_space

    def __getattr__(self, name):
        return getattr(self.env, name)

    def step(self, action):
        return self.env.step(action)

    def render(self, *args, **kwargs):
        return self.env.render(*args, **kwargs)

    def reset(self, *args, **kwargs):
        return self.env.reset(*args, **kwargs)
