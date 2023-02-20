import itertools
from typing import Dict, Optional

import numpy as np
import torch

from .off_policy_algorithm import OffPolicyAlgorithm


class DWSL(OffPolicyAlgorithm):
    def __init__(
        self,
        *args,
        alpha: Optional[float] = None,
        beta: float = 1,
        clip_score: float = 100.0,
        encoder_gradients: str = "actor",
        nstep: int = 1,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        assert hasattr(self.network.value, "bins"), "Value function must have bin attr for DWSL"
        self.bins = self.network.value.bins  # Get bins from the network
        assert encoder_gradients in ("actor", "value", "both")
        self.encoder_gradients = encoder_gradients
        self.beta = beta
        self.alpha = alpha
        self.clip_score = clip_score
        self.nstep = nstep
        self.action_range = [
            float(self.processor.action_space.low.min()),
            float(self.processor.action_space.high.max()),
        ]

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

    def _get_distance(self, logits: torch.Tensor) -> torch.Tensor:
        distribution = torch.nn.functional.softmax(logits, dim=-1)  # (E, B, D)
        distances = torch.arange(start=0, end=self.bins, device=logits.device) / self.bins
        distances = distances.unsqueeze(0).unsqueeze(0)  # (E, B, D)
        if self.alpha is None:
            # Return the expectation
            predicted_distance = (distribution * distances).sum(dim=-1)
        else:
            # Return the LSE weighted by the distribution.
            exp_q = torch.exp(-distances / self.alpha)
            predicted_distance = -self.alpha * torch.log(torch.sum(distribution * exp_q, dim=-1))
        return torch.max(predicted_distance, dim=0)[0]

    def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
        batch["obs"] = self.network.encoder(batch["obs"])
        with torch.no_grad():
            batch["next_obs"] = self.network.encoder(batch["next_obs"])

        with torch.no_grad():
            empirical_targets = (batch["horizon"] - 1) // self.nstep
            empirical_targets[empirical_targets < 0] = self.bins - 1  # Set to max bin value (= horizon by default)
            empirical_targets[empirical_targets >= self.bins] = self.bins - 1
            # Now one-hot the empirical distribution
            target_distribution = torch.nn.functional.one_hot(empirical_targets, num_classes=self.bins)
            target_distribution = target_distribution.unsqueeze(0)  # (1, B, D)

        # Now train the distance function with NLL loss
        logits = self.network.value(batch["obs"].detach() if self.encoder_gradients == "actor" else batch["obs"])
        log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
        v_loss = (-target_distribution * log_probs).sum(dim=-1).mean()  # Sum over D dim, avg over B and E

        # Compute the Actor Loss
        with torch.no_grad():
            # Compute the advantage. This is equal to Q(s,a) - V(s) normally
            # But, we are using costs. Thus the advantage is V(s) - Q(s,a) = V(s) - c(s,a) - V(s')

            # First compute the cost tensor. This is zero unless the horizon is in nstep
            cost = torch.logical_or(batch["horizon"] >= self.nstep, batch["horizon"] < 0)
            cost = cost.float() / self.bins  # Cost is zero if we reach on the next state within nstep
            distance = self._get_distance(logits)
            next_distance = self._get_distance(self.network.value(batch["next_obs"]))
            adv = distance - cost - batch["discount"] * next_distance
            exp_adv = torch.exp(adv / self.beta)
            if self.clip_score is not None:
                exp_adv = torch.clamp(exp_adv, max=self.clip_score)

        dist = self.network.actor(batch["obs"].detach() if self.encoder_gradients == "value" else batch["obs"])
        if isinstance(dist, torch.distributions.Distribution):
            bc_loss = -dist.log_prob(batch["action"]).sum(dim=-1)
        elif torch.is_tensor(dist):
            assert dist.shape == batch["action"].shape
            bc_loss = torch.nn.functional.mse_loss(dist, batch["action"], reduction="none").sum(dim=-1)
        else:
            raise ValueError("Invalid policy output provided")
        assert exp_adv.shape == bc_loss.shape
        actor_loss = (exp_adv * bc_loss).mean()

        self.optim["value"].zero_grad(set_to_none=True)
        self.optim["actor"].zero_grad(set_to_none=True)
        (v_loss + actor_loss).backward()
        self.optim["value"].step()
        self.optim["actor"].step()

        return dict(
            v_loss=v_loss.item(),
            actor_loss=actor_loss.item(),
            distance=distance.mean().item(),
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
