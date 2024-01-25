""""Script for the dual policy networks"""
from typing import Any, Dict, List, Optional, Type

import torch as th
from gym import spaces
from torch import nn

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.torch_layers import (
    BaseFeaturesExtractor,
    FlattenExtractor,
    NatureCNN,
    create_mlp,
)
from stable_baselines3.common.type_aliases import Schedule

from gym.spaces.discrete import Discrete


class QNetwork(BasePolicy):
    """
    Action-Value (Q-Value) network for DQN

    :param observation_space: Observation space
    :param action_space: Action space
    :param net_arch: The specification of the policy and value networks.
    :param activation_fn: Activation function
    :param normalize_images: Whether to normalize images or not,
         dividing by 255.0 (True by default)
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        features_extractor: nn.Module,
        features_dim: int,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        normalize_images: bool = True,
    ):
        super().__init__(
            observation_space,
            action_space,
            features_extractor=features_extractor,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            net_arch = [64, 64]

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.features_extractor = features_extractor
        self.features_dim = features_dim
        action_dim = self.action_space.n  # number of actions
        q_net = create_mlp(self.features_dim, action_dim, self.net_arch, self.activation_fn)
        self.q_net = nn.Sequential(*q_net)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        """
        Predict the q-values.

        :param obs: Observation
        :return: The estimated Q-Value for each action.
        """
        return self.q_net(self.extract_features(obs, self.features_extractor))

    def _predict(self, observation: th.Tensor, deterministic: bool = True) -> th.Tensor:
        q_values = self(observation)
        return q_values

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_arch,
                features_dim=self.features_dim,
                activation_fn=self.activation_fn,
                features_extractor=self.features_extractor,
            )
        )
        return data


class CDDQN(BasePolicy):
    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        net_arch: Optional[List[int]] = None,
        activation_fn: Type[nn.Module] = nn.ReLU,
        features_extractor_class: Type[BaseFeaturesExtractor] = FlattenExtractor,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        normalize_images: bool = True,
        optimizer_class: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Dict[str, Any]] = None,
        nr_nodes: int = None,
        obs_dim: int = None,
    ):
        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
            normalize_images=normalize_images,
        )

        if net_arch is None:
            if features_extractor_class == NatureCNN:
                net_arch = []
            else:
                net_arch = [64, 64]

        self.features_extractor_class = FlattenExtractor
        self.features_extractor_kwargs = {}

        self.net_arch = net_arch
        self.activation_fn = activation_fn
        self.nr_nodes = nr_nodes
        self.obs_dim = obs_dim
        self.act_dim_cd = 2*(self.nr_nodes**2 - self.nr_nodes) + 1
        self.act_dim_in = self.nr_nodes + 1

        self.net_args_cd = {
            "observation_space": self.observation_space,
            "action_space": Discrete(self.act_dim_cd),
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }

        self.net_args_in = {
            "observation_space": self.observation_space,
            "action_space": Discrete(self.act_dim_in),
            "net_arch": self.net_arch,
            "activation_fn": self.activation_fn,
            "normalize_images": normalize_images,
        }
        self.q_net_cd, self.q_net_cd_target = None, None
        self.q_net_in, self.q_net_in_target = None, None
        self._build(lr_schedule)

    def _build(self, lr_schedule: Schedule) -> None:

        """
            Create the network and the optimizer.

            Put the target network into evaluation mode.

            :param lr_schedule: Learning rate schedule
                lr_schedule(1) is the initial learning rate
            """

        # cd agent
        self.q_net_cd = self.make_q_net(self.net_args_cd)
        self.q_net_cd_target = self.make_q_net(self.net_args_cd)
        self.q_net_cd_target.load_state_dict(self.q_net_cd.state_dict())
        self.q_net_cd_target.set_training_mode(False)

        # int agent
        self.q_net_in = self.make_q_net(self.net_args_in)
        self.q_net_in_target = self.make_q_net(self.net_args_in)
        self.q_net_in_target.load_state_dict(self.q_net_in.state_dict())
        self.q_net_in_target.set_training_mode(False)

        # Setup optimizer with initial learning rate
        self.optimizer_cd = self.optimizer_class(self.q_net_cd.parameters(), lr=lr_schedule(1))
        self.optimizer_in = self.optimizer_class(self.q_net_in.parameters(), lr=lr_schedule(1))

    def make_q_net(self, net_args=None) -> QNetwork:
        # Make sure we always have separate networks for features extractors etc
        net_args = self._update_features_extractor(net_args, features_extractor=None)
        return QNetwork(**net_args).to(self.device)

    def forward(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self._predict(obs, deterministic=deterministic)

    def _predict(self, obs: th.Tensor, deterministic: bool = True) -> th.Tensor:
        return self.q_net._predict(obs, deterministic=deterministic)

    def _get_constructor_parameters(self) -> Dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                net_arch=self.net_args["net_arch"],
                activation_fn=self.net_args["activation_fn"],
                lr_schedule=self._dummy_schedule,  # dummy lr schedule, not needed for loading policy alone
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
            )
        )
        return data

    def set_training_mode(self, mode: bool) -> None:
        """
        Put the policy in either training or evaluation mode.

        This affects certain modules, such as batch normalisation and dropout.

        :param mode: if true, set to training mode, else set to evaluation mode
        """
        self.q_net_cd.set_training_mode(mode)
        self.q_net_in.set_training_mode(mode)
        self.training = mode


