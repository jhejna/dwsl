from typing import Dict, List, Optional, Type, Union

import gym
import numpy as np
import torch
from torch import nn

import research


class GoFarNetwork(nn.Module):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        actor_class: Union[str, Type[nn.Module]],
        value_class: Union[str, Type[nn.Module]],
        discriminator_class: Union[str, Type[nn.Module]],
        encoder_class: Optional[Union[str, Type[nn.Module]]] = None,
        actor_kwargs: Dict = {},
        value_kwargs: Dict = {},
        discriminator_kwargs: Dict = {},
        encoder_kwargs: Dict = {},
        concat_keys: List[str] = [
            "observation",
            "achieved_goal",
            "desired_goal",
        ],  # For hiding "achieved_goal" from the Q, pi networks.
        share_encoder: bool = True,
        concat_dim: Optional[int] = None,  # dimension to concatenate states on
        **kwargs,
    ):
        super().__init__()
        assert isinstance(observation_space, gym.spaces.Dict)
        concat_keys = concat_keys.copy()
        if "observation" not in observation_space.spaces:
            print("[GoFar] Overrideing hide achievied goal because no 'observation' key is present in the environment.")
            concat_keys.remove("observation")

        assert all([k in observation_space.spaces for k in concat_keys])
        assert "achieved_goal" in observation_space.spaces
        assert isinstance(action_space, gym.spaces.Box)

        # Try to determine the concatenation dimensions automatically if they were not set
        if concat_dim is None:
            space = observation_space["achieved_goal"]
            if (len(space.shape) == 3 or len(space.shape) == 4) and space.dtype == np.uint8:
                concat_dim = 0
            else:
                concat_dim = -1

        self.forward_concat_dim = concat_dim if concat_dim < 0 else concat_dim + 1

        # First create the encoder space
        low = np.concatenate([observation_space[k].low for k in concat_keys], axis=concat_dim)
        high = np.concatenate([observation_space[k].high for k in concat_keys], axis=concat_dim)
        encoder_space = gym.spaces.Box(low=low, high=high, dtype=observation_space["desired_goal"].dtype)

        encoder_class = vars(research.networks)[encoder_class] if isinstance(encoder_class, str) else encoder_class
        encoder_class = nn.Identity if encoder_class is None else encoder_class
        _encoder_kwargs = kwargs.copy()
        _encoder_kwargs.update(encoder_kwargs)
        self._encoder = encoder_class(encoder_space, action_space, **_encoder_kwargs)

        policy_space = self._encoder.output_space if hasattr(self._encoder, "output_space") else encoder_space

        # Construct the policy
        actor_class = vars(research.networks)[actor_class] if isinstance(actor_class, str) else actor_class
        _actor_kwargs = kwargs.copy()
        _actor_kwargs.update(actor_kwargs)
        self._actor = actor_class(policy_space, action_space, **_actor_kwargs)

        value_class = vars(research.networks)[value_class] if isinstance(value_class, str) else value_class
        _value_kwargs = kwargs.copy()
        _value_kwargs.update(value_kwargs)
        self._value = value_class(policy_space, action_space, **_value_kwargs)

        # Finally construct the discriminator
        if share_encoder and set(concat_keys) == {"achieved_goal", "desired_goal"}:
            # Share the encoder
            discriminator_space = policy_space
            self._discriminator_encoder = self._encoder  # Share reference!
        else:
            low = np.concatenate(
                (observation_space["achieved_goal"].low, observation_space["desired_goal"].low), axis=concat_dim
            )
            high = np.concatenate(
                (observation_space["achieved_goal"].high, observation_space["desired_goal"].high), axis=concat_dim
            )
            discriminator_space = gym.spaces.Box(low=low, high=high, dtype=observation_space["desired_goal"].dtype)
            self._discriminator_encoder = nn.Identity()

        discriminator_class = (
            vars(research.networks)[discriminator_class]
            if isinstance(discriminator_class, str)
            else discriminator_class
        )
        _discriminator_kwargs = kwargs.copy()
        _discriminator_kwargs.update(discriminator_kwargs)
        self._discriminator = discriminator_class(discriminator_space, action_space, **_discriminator_kwargs)

    def forward(self):
        raise NotImplementedError

    @property
    def encoder(self):
        return self._encoder

    @property
    def discriminator_encoder(self):
        return self._discriminator_encoder

    @property
    def discriminator(self):
        return self._discriminator

    @property
    def actor(self):
        return self._actor

    @property
    def value(self):
        return self._value

    def format_policy_obs(self, obs):
        return torch.cat([obs[k] for k in self.concat_keys], dim=self.forward_concat_dim)

    def format_discriminator_obs(self, obs):
        pos_pairs = torch.cat([obs["desired_goal"], obs["desired_goal"]], dim=self.forward_concat_dim)
        neg_pairs = torch.cat([obs["achieved_goal"], obs["desired_goal"]], dim=self.forward_concat_dim)
        return pos_pairs, neg_pairs
