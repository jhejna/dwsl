import itertools
from typing import Dict, Type

import numpy as np
import torch

from research.networks.gofar import GoFarNetwork

from .off_policy_algorithm import OffPolicyAlgorithm


class GoFar(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        discriminator_lambda: float = 0.01,
        tau: float = 0.05,
        target_freq: int = 20,
        encoder_gradients: str = "both",
        **kwargs,
    ) -> None:
        # After determining dimension parameters, setup the network
        super().__init__(*args, **kwargs)
        assert encoder_gradients in ("actor", "value", "both")
        self.encoder_gradients = encoder_gradients
        self.discriminator_lambda = discriminator_lambda
        self.tau = tau
        self.target_freq = target_freq
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]

    def setup_network(self, network_class: Type[torch.nn.Module], network_kwargs: Dict) -> None:
        assert network_class is GoFarNetwork, "Must use GoFarNetwork with GoFar."
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
        if self.encoder_gradients == "value" or self.encoder_gradients == "both":
            value_params = itertools.chain(self.network.value.parameters(), self.network.encoder.parameters())
            actor_params = self.network.actor.parameters()
        elif self.encoder_gradients == "actor":
            value_params = self.network.value.parameters()
            actor_params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
        else:
            raise ValueError("Unsupported value of encoder_gradients")
        self.optim["actor"] = self.optim_class(actor_params, **self.optim_kwargs)
        self.optim["value"] = self.optim_class(value_params, **self.optim_kwargs)
        self.optim["discriminator"] = self.optim_class(self.network.discriminator.parameters(), **self.optim_kwargs)

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        # discriminator data
        pos_pairs, neg_pairs = self.network.format_discriminator_obs(batch["obs"])

        with torch.no_grad():
            pos_pairs = self.network.discriminator_encoder(pos_pairs)
            neg_pairs = self.network.discriminator_encoder(neg_pairs)

        expert_d = self.network.discriminator(pos_pairs)
        policy_d = self.network.discriminator(neg_pairs)

        pos_loss = torch.nn.functional.binary_cross_entropy_with_logits(expert_d, torch.ones_like(expert_d))
        neg_loss = torch.nn.functional.binary_cross_entropy_with_logits(policy_d, torch.zeros_like(policy_d))

        alpha = torch.rand_like(expert_d, device=expert_d.device).unsqueeze(-1).expand_as(pos_pairs)
        mixup_pairs = alpha * pos_pairs + (1 - alpha) * neg_pairs
        mixup_pairs.requires_grad = True

        # Compute the gradient penalty on the mixup pairs.
        mixup_d = self.network.discriminator(mixup_pairs)
        grad = torch.autograd.grad(
            outputs=mixup_d,
            inputs=mixup_pairs,
            grad_outputs=torch.ones_like(mixup_d),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        grad_pen = self.discriminator_lambda * (grad.norm(2, dim=1) - 1).pow(2).mean()
        discriminator_loss = pos_loss + neg_loss + grad_pen

        self.optim["discriminator"].zero_grad(set_to_none=True)
        discriminator_loss.backward()
        self.optim["discriminator"].step()

        # Unset the grads for desired_goal
        batch["obs"]["desired_goal"].requires_grad = False

        # Format the observations
        init_obs = self.network.format_policy_obs(batch["init_obs"])
        init_obs = self.network.encoder(init_obs)
        obs = self.network.format_policy_obs(batch["obs"])
        obs = self.network.encoder(obs)
        with torch.no_grad():
            next_obs = self.target_network.format_policy_obs(batch["next_obs"])
            next_obs = self.target_network.encoder(next_obs)

        # compute the value loss
        vs_initial = self.network.value(init_obs.detach() if self.encoder_gradients == "actor" else init_obs)
        vs = self.network.value(obs.detach() if self.encoder_gradients == "actor" else obs)
        with torch.no_grad():
            v_next = self.target_network.value(next_obs).min(dim=0)[0]  # Get the min
            # Compute the reward using the discriminator
            s = torch.sigmoid(policy_d.detach())
            reward = s.log() - (1 - s).log()
            target_v = reward + batch["discount"] * v_next

        adv = target_v.expand(vs.shape[0], -1) - vs  # Expand target_v to add ensemble dim
        v_loss_init = (1 - batch["discount"]).unsqueeze(0) * vs_initial
        v_loss_current = (adv + 1).pow(2)
        v_loss = (v_loss_init + v_loss_current).mean(dim=-1).sum()

        # Compute the actor loss
        dist = self.network.actor(obs.detach() if self.encoder_gradients == "value" else obs)
        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(batch["action"]).sum(dim=-1)
        elif torch.is_tensor(dist):
            assert dist.shape == batch["action"].shape
            bc_loss = torch.nn.functional.mse_loss(dist, batch["action"], reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")

        with torch.no_grad():
            w_e = torch.relu(adv.mean(dim=0) + 1)
        actor_loss = (w_e * bc_loss).mean()

        # Update the networks
        self.optim["value"].zero_grad(set_to_none=True)
        self.optim["actor"].zero_grad(set_to_none=True)
        (v_loss + actor_loss).backward()
        self.optim["value"].step()
        self.optim["actor"].step()

        # Apply the soft weight update to the value function
        if step % self.target_freq == 0:
            with torch.no_grad():
                # Update encoder and value parameters.
                for param, target_param in zip(
                    self.network.encoder.parameters(), self.target_network.encoder.parameters()
                ):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                for param, target_param in zip(self.network.value.parameters(), self.target_network.value.parameters()):
                    target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return dict(
            pos_loss=pos_loss.item(),
            neg_loss=neg_loss.item(),
            grad_pen=grad_pen.item(),
            v_loss=v_loss.item(),
            actor_loss=actor_loss.item(),
            v=vs.mean().item(),
            advantage=adv.mean().item(),
            reward=reward.mean().item(),
        )

    def _predict(self, batch: Dict, sample: bool = False) -> torch.Tensor:
        with torch.no_grad():
            obs = self.network.format_policy_obs(batch["obs"])
            z = self.network.encoder(obs)
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
