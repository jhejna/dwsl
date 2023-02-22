import argparse
import datetime
import os
import time

import gym
import numpy as np

import research  # To run environment imports
from research.datasets.replay_buffer import ReplayBuffer
from research.utils.config import Config
from research.utils.evaluate import EvalMetricTracker

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--num-ep", type=int, default=np.inf)
    parser.add_argument("--num-steps", type=int, default=np.inf)
    parser.add_argument("--noise", type=float, default=0.0, help="Gaussian noise std.")
    parser.add_argument("--random-percent", type=float, default=0.0, help="percent of dataset to be purely random.")
    parser.add_argument("--checkpoint", type=str, default=None, help="Checkpoint to load")
    parser.add_argument("--device", type=str, default="auto")
    args = parser.parse_args()

    assert not (args.num_steps == np.inf and args.num_ep == np.inf), "Must set one of num-steps and num-ep"
    assert not (args.num_steps != np.inf and args.num_ep != np.inf), "Cannot set both num-steps and num-ep"
    assert args.random_percent <= 1.0 and args.random_percent >= 0.0, "Invalid random-percent"

    if os.path.exists(args.path):
        print("[research] Warning: saving dataset to an existing directory.")
    os.makedirs(args.path, exist_ok=True)

    # Load the config
    config = Config.load(os.path.dirname(args.checkpoint) if args.checkpoint.endswith(".pt") else args.checkpoint)
    config["checkpoint"] = None  # Set checkpoint to None

    # Parse the config
    config = config.parse()
    if args.random_percent < 1.0:
        assert args.checkpoint.endswith(".pt"), "Did not specify checkpoint file."
        model = config.get_model(device=args.device)
        metadata = model.load(args.checkpoint)
        env = model.env  # get the env from the model
    else:
        model = None
        env = config.get_train_env()

    if isinstance(env, research.envs.base.Empty):
        env = config.get_eval_env()  # Get the eval env instead as it actually exists.

    shape = (3, env.height, env.width)
    image_obs_space = gym.spaces.Dict(
        {
            "achieved_goal": gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
            "desired_goal": gym.spaces.Box(low=0, high=255, shape=shape, dtype=np.uint8),
        }
    )
    # Make sure we are dealing with a Fetch Image environment
    assert hasattr(env, "set_render"), "Not using an env with `set_render`"
    assert hasattr(env, "get_image_obs"), "Not using an env with `get_image_obs`"
    env.set_render(True)

    capacity = 1000  # Set some small number because
    replay_buffer = ReplayBuffer(image_obs_space, env.action_space, capacity=capacity, cleanup=False, distributed=False)

    # Track data collection
    num_steps = 0
    num_ep = 0
    finished_data_collection = False
    # Episode metrics
    metric_tracker = EvalMetricTracker()
    start_time = time.time()

    while not finished_data_collection:
        # Determine if we should use random actions or not.
        progress = num_ep / args.num_ep if args.num_ep != np.inf else num_steps / args.num_steps
        use_random_actions = progress < args.random_percent

        # Collect an episode
        done = False
        ep_length = 0
        obs = env.reset()
        image_obs = env.get_image_obs()
        metric_tracker.reset()
        replay_buffer.add(image_obs)
        while not done:
            if use_random_actions:
                action = env.action_space.sample()
            else:
                action = model.predict(dict(obs=obs))
                if args.noise > 0:
                    assert isinstance(env.action_space, gym.spaces.Box)
                    action = action + args.noise * np.random.randn(*action.shape)
                    # Step the environment with the predicted action
                    env_action = np.clip(action, env.action_space.low, env.action_space.high)

            obs, reward, done, info = env.step(action)
            image_obs = env.get_image_obs()
            metric_tracker.step(reward, info)
            ep_length += 1

            # Determine the discount factor.
            if "discount" in info:
                discount = info["discount"]
            elif hasattr(env, "_max_episode_steps") and ep_length == env._max_episode_steps:
                discount = 1.0
            else:
                discount = 1 - float(done)

            # Store the consequences.
            replay_buffer.add(image_obs, action, reward, done, discount)
            num_steps += 1

        num_ep += 1
        # Determine if we should stop data collection
        finished_data_collection = num_steps >= args.num_steps or num_ep >= args.num_ep

    end_time = time.time()
    print("Finished", num_ep, "episodes in", num_steps, "steps.")
    print("It took", (end_time - start_time) / num_steps, "seconds per step")

    # Save the dataset.
    replay_buffer.save(args.path)
    fname = datetime.datetime.now().strftime("%Y%m%dT%H%M%S")

    # Write the metrics
    metrics = metric_tracker.export()
    print("Metrics:")
    print(metrics)
    with open(os.path.join(args.path, "metrics.txt"), "a") as f:
        f.write("Collected data: " + str(fname) + "\n")
        for k, v in metrics.items():
            f.write(k + ": " + str(v) + "\n")
