import itertools
from typing import Dict, Type

import numpy as np
import torch

from .off_policy_algorithm import OffPolicyAlgorithm


class WGCSL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        tau: float = 0.1,
        target_freq: int = 1,
        drw: bool = True,
        beta: float = 1,
        clip_score: float = 100.0,
        sparse_reward: bool = False,
        encoder_gradients: str = "both",
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert encoder_gradients in ("actor", "critic", "both")
        self.encoder_gradients = encoder_gradients
        self.tau = tau
        self.target_freq = target_freq
        self.drw = drw
        self.beta = beta
        self.clip_score = clip_score
        self.sparse_reward = sparse_reward
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        self.network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network = network_class(
            self.processor.observation_space, self.processor.action_space, **network_kwargs
        ).to(self.device)
        self.target_network.load_state_dict(self.network.state_dict())
        for param in self.target_network.parameters():
            param.requires_grad = False

    def setup_optimizers(self) -> None:
        # Default optimizer initialization
        if self.encoder_gradients == "critic" or self.encoder_gradients == "both":
            critic_params = itertools.chain(self.network.critic.parameters(), self.network.encoder.parameters())
            actor_params = self.network.actor.parameters()
        elif self.encoder_gradients == "actor":
            critic_params = self.network.critic.parameters()
            actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        else:
            raise ValueError("Unsupported value of encoder_gradients")
        self.optim["actor"] = self.optim_class(actor_params, **self.optim_kwargs)
        self.optim["critic"] = self.optim_class(critic_params, **self.optim_kwargs)

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        batch["obs"] = self.network.encoder(batch["obs"])
        with torch.no_grad():
            batch["next_obs"] = self.target_network.encoder(batch["next_obs"])

        # First update the critic
        with torch.no_grad():
            next_actions = self.target_network.actor(batch["next_obs"])
            next_q = self.target_network.critic(batch["next_obs"], next_actions).min(dim=0)[0]
            if self.sparse_reward:
                reward = -(batch["horizon"] != 1).float()
            else:
                reward = batch["reward"]
            target_q = reward + batch["discount"] * next_q
        qs = self.network.critic(
            batch["obs"].detach() if self.encoder_gradients == "actor" else batch["obs"], batch["action"]
        )
        q_loss = torch.nn.functional.mse_loss(qs, target_q.expand(qs.shape[0], -1), reduction="none").mean(dim=-1).sum()

        # Compute the policy action
        dist = self.network.actor(batch["obs"].detach() if self.encoder_gradients == "critic" else batch["obs"])
        if isinstance(dist, torch.distributions.Distribution):
            action = dist.sample()
            bc_loss = -dist.log_prob(batch["action"]).sum(dim=-1)
        elif torch.is_tensor(dist):
            assert dist.shape == batch["action"].shape
            action = dist
            bc_loss = torch.nn.functional.mse_loss(action, batch["action"], reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")

        # Compute the discount relabeling weight (DRW)
        drw = torch.pow(batch["discount"], batch["horizon"]) if self.drw else torch.ones_like(batch["discount"])

        # Compute the advantage weighting
        with torch.no_grad():
            v = self.network.critic(batch["obs"], action).min(dim=0)[0]
            adv = target_q - v
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        actor_loss = (drw * exp_adv * bc_loss).mean()

        self.optim["critic"].zero_grad(set_to_none=True)
        self.optim["actor"].zero_grad(set_to_none=True)
        (q_loss + actor_loss).backward()
        self.optim["critic"].step()
        self.optim["actor"].step()

        if step % self.target_freq == 0:
            with torch.no_grad():
                for param, target_param in zip(self.network.parameters(), self.target_network.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return dict(
            q_loss=q_loss.item(),
            actor_loss=actor_loss.item(),
            q=qs.mean().item(),
            advantage=adv.mean().item(),
        )

    def _predict(self, batch: Dict, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            z = self.network.encoder(batch["obs"])
            dist = self.network.actor(z)
            if isinstance(dist, torch.distributions.Distribution):
                action = dist.sample() if sample else dist.loc
            elif torch.is_tensor(dist):
                action = dist
            else:
                raise ValueError("Invalid policy output")
            action = action.clamp(*self.action_range)

        return action

    def _get_train_action(self, step: int, total_steps: int) -> np.ndarray:
        batch = dict(obs=self._current_obs)
        with torch.no_grad():
            action = self.predict(batch, is_batched=False, sample=True)
        return action

    def validation_step(self, batch: Dict) -> Dict:
        """
        perform a validation step. Should return a dict of loggable values.
        TODO: refactor method to re-use this computation
        """
        with torch.no_grad():
            batch["obs"] = self.network.encoder(batch["obs"])
            batch["next_obs"] = self.network.encoder(batch["next_obs"])

            # First compute critic loss
            next_actions = self.target_network.actor(batch["next_obs"])
            next_q = self.target_network.critic(batch["next_obs"], next_actions).min(dim=0)[0]
            if self.sparse_reward:
                reward = -(batch["horizon"] != 1).float()
            else:
                reward = batch["reward"]
            target_q = reward + batch["discount"] * next_q
            qs = self.network.critic(
                batch["obs"].detach() if self.encoder_gradients == "actor" else batch["obs"], batch["action"]
            )
            q_loss = (
                torch.nn.functional.mse_loss(qs, target_q.expand(qs.shape[0], -1), reduction="none").mean(dim=-1).sum()
            )

            # Compute the policy action
            dist = self.network.actor(batch["obs"].detach() if self.encoder_gradients == "critic" else batch["obs"])
            if isinstance(dist, torch.distributions.Distribution):
                action = dist.sample()
                bc_loss = -dist.log_prob(batch["action"]).sum(dim=-1)
            elif torch.is_tensor(dist):
                assert dist.shape == batch["action"].shape
                action = dist
                bc_loss = torch.nn.functional.mse_loss(action, batch["action"], reduction="none").sum(dim=-1)
            else:
                raise ValueError("Invalid policy output provided")

            # Compute the discount relabeling weight (DRW)
            drw = torch.pow(batch["discount"], batch["horizon"]) if self.drw else torch.ones_like(batch["discount"])

            # Compute the advantage weighting
            with torch.no_grad():
                v = self.network.critic(batch["obs"], action).min(dim=0)[0]
                adv = target_q - v
                exp_adv = torch.exp(adv / self.beta)
                if self.clip_score is not None:
                    exp_adv = torch.clamp(exp_adv, max=self.clip_score)

            actor_loss = (drw * exp_adv * bc_loss).mean()

        return dict(
            q_loss=q_loss.item(),
            actor_loss=actor_loss.item(),
            q=qs.mean().item(),
            advantage=adv.mean().item(),
        )
