# Register Network Classes here.
from .base import ActorCriticPolicy, ActorValuePolicy, ActorCriticValuePolicy, ActorPolicy
from .mlp import (
    ContinuousMLPActor,
    ContinuousMLPCritic,
    DiagonalGaussianMLPActor,
    MLPValue,
    MLPEncoder,
    DiscreteMLPDistance,
    MLPDiscriminator,
)
from .drqv2 import DrQv2Encoder, DrQv2Critic, DrQv2Actor, DrQv2Value, DiscreteDrQv2Distance, DrQv2Discriminator
from .rmimic import RobomimicEncoder
from .gofar import GoFarNetwork
