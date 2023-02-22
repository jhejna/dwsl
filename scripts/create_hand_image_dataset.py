import argparse
import os
import pickle

import gym
import imageio
import numpy as np

import research

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--path",
        type=str,
        default="../datasets/offline_goal_conditioned_data/random/HandReach/buffer.pkl",
        help="path to WGCSL benchmark dataset.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="../datasets/offline_goal_conditioned_data/random/HandReachImage",
        help="Output path.",
    )
    parser.add_argument("--shards", type=int, default=64, help="Number of shards to create.")

    args = parser.parse_args()

    with open(args.path, "rb") as f:
        data = pickle.load(f)

    env = gym.make("HandReachImage-v0")

    goals, goal_images = [], []
    eps, horizon, _ = data["ag"].shape
    chunk_size = eps // args.shards

    for shard in range(args.shards):
        imgs = []
        for ep in range(shard * chunk_size, (shard + 1) * chunk_size):
            ep_imgs = []
            # set the env goal to be the thing from the dataset
            goal = data["g"][ep, 0]
            for t in range(horizon):
                obs = dict(observation=data["o"][ep][t], achieved_goal=data["ag"][ep][t], desired_goal=goal)
                env.set_state(obs)
                img = env.get_image()
                ep_imgs.append(img)
                if np.linalg.norm(obs["achieved_goal"] - obs["desired_goal"]) < 0.01:
                    # If we achieved the goal, save it
                    goals.append(obs["desired_goal"])
                    goal_images.append(img)

            imgs.append(ep_imgs)
        print("Finished shard", shard)

        img_ag = np.array(imgs, dtype=np.uint8)
        u = data["u"]

        dataset = dict(ag=img_ag, u=u)
        with open(os.path.join(args.output, "buffer_{}.pkl".format(shard)), "wb") as f:
            pickle.dump(dataset, f)

    goals = np.array(goals, dtype=np.float32)
    goal_imgs = np.array(goal_images, dtype=np.uint8)
    goal_dataset = dict(g=goals, img=goal_images)

    with open(os.path.join(args.output, "goals.pkl"), "wb") as f:
        pickle.dump(goal_dataset, f)

    # Save only the last 200 goals (should be a good random sample)
    goal_dataset_200 = dict(g=goals[-200:], img=goal_imgs[-200:])
    with open(os.path.join(args.output, "goals_200.pkl"), "wb") as f:
        pickle.dump(goal_dataset_200, f)
