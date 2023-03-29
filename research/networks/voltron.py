import gym
import numpy as np
import torch
import torchvision.transforms as T
from torch import nn
from voltron import instantiate_extractor, load


class VoltronEncoder(nn.Module):
    def __init__(
        self, observation_space: gym.Space, action_space: gym.Space, stack: int = 1, use_state_goal: bool = True
    ):
        super().__init__()
        assert len(observation_space["achieved_goal.images0"].shape) == 3

        self.stack = stack
        self.backbone, self.preprocess_fn = load("v-dual", freeze=True)
        self.extractor = instantiate_extractor(self.backbone, n_latents=1)()
        self.ln = nn.BatchNorm1d(self.extractor.embed_dim)

        state_dim = observation_space["achieved_goal.state"].shape[0]
        self.output_dim = self.extractor.embed_dim * self.stack + (self.stack + int(use_state_goal)) * state_dim
        self.use_state_goal = use_state_goal

    def forward(self, obs):
        if len(obs["achieved_goal.images0"].shape) == 5:
            imgs = torch.cat((obs["achieved_goal.images0"], obs["desired_goal.images0"]), dim=1)
        else:
            imgs = torch.stack((obs["achieved_goal.images0"], obs["desired_goal.images0"]), dim=1)
        b, s, c, h, w = imgs.shape
        imgs = imgs.view(b * s, c, h, w).to(torch.uint8)

        with torch.no_grad():
            imgs = self.preprocess_fn(imgs)
            _, c, h, w = imgs.shape
            imgs = imgs.view(b, s, c, h, w)
            # Split the images into stack
            slices = [imgs[:, i : i + 2] for i in range(self.stack)]
            imgs = torch.cat(slices, dim=0)  # Move to batch dim: (B*stack, 2, c, h, w)
            h = self.backbone(imgs, mode="visual")
        h = self.extractor(h)
        h = self.ln(h)
        h = torch.cat(torch.chunk(h, self.stack, dim=0), dim=1)  # Concatenate the hidden dim

        cat_keys = [h, obs["achieved_goal.state"].flatten(1, -1)]
        if self.use_state_goal:
            cat_keys.append(obs["desired_goal.state"].flatten(1, -1))
        return torch.cat(cat_keys, dim=1)

    @property
    def output_space(self):
        return gym.spaces.Box(shape=(self.output_dim,), low=-np.inf, high=np.inf, dtype=np.float32)
