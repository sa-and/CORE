"""Variation of the DQN algorithm that simultaneously trains two networks"""
import random

from torch.nn import functional as F

from stable_baselines3.common.off_policy_algorithm import OffPolicyAlgorithm
from stable_baselines3.common.preprocessing import maybe_transpose
from stable_baselines3.common.utils import get_linear_fn, get_parameters_by_name, is_vectorized_observation, polyak_update
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy

import warnings
from typing import Any, Dict, List, Optional, Tuple, Type, TypeVar, Union
import wandb

import numpy as np
import torch as th
from gym.spaces import Discrete, MultiDiscrete

from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.noise import ActionNoise, VectorizedActionNoise
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, RolloutReturn, Schedule, TrainFreq, TrainFrequencyUnit
from stable_baselines3.common.utils import should_collect_more_steps
from stable_baselines3.common.vec_env import VecEnv


SelfDQN = TypeVar("SelfDQN", bound="DQN")


class DualDQN(OffPolicyAlgorithm):
    """
    Deep Q-Network (DQN)

    Paper: https://arxiv.org/abs/1312.5602, https://www.nature.com/articles/nature14236
    Default hyperparameters are taken from the Nature paper,
    except for the optimizer and learning rate that were taken from Stable Baselines defaults.

    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param buffer_size: size of the replay buffer
    :param learning_starts: how many steps of the model to collect transitions for before learning starts
    :param batch_size: Minibatch size for each gradient update
    :param tau: the soft update coefficient ("Polyak update", between 0 and 1) default 1 for hard update
    :param gamma: the discount factor
    :param train_freq: Update the model every ``train_freq`` steps. Alternatively pass a tuple of frequency and unit
        like ``(5, "step")`` or ``(2, "episode")``.
    :param gradient_steps: How many gradient steps to do after each rollout (see ``train_freq``)
        Set to ``-1`` means to do as many gradient steps as steps done in the environment
        during the rollout.
    :param replay_buffer_class: Replay buffer class to use (for instance ``HerReplayBuffer``).
        If ``None``, it will be automatically selected.
    :param replay_buffer_kwargs: Keyword arguments to pass to the replay buffer on creation.
    :param optimize_memory_usage: Enable a memory efficient variant of the replay buffer
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
    :param target_update_interval: update the target network every ``target_update_interval``
        environment steps.
    :param exploration_fraction: fraction of entire training period over which the exploration rate is reduced
    :param exploration_initial_eps: initial value of random action probability
    :param exploration_final_eps: final value of random action probability
    :param max_grad_norm: The maximum value for the gradient clipping
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :param policy_kwargs: additional arguments to be passed to the policy on creation
    :param verbose: Verbosity level: 0 for no output, 1 for info messages (such as device or wrappers used), 2 for
        debug messages
    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    :param _init_setup_model: Whether or not to build the network at the creation of the instance
    """

    policy_aliases: Dict[str, Type[BasePolicy]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 2056,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            action_noise=None,  # No action noise
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            policy_kwargs=policy_kwargs,
            tensorboard_log=tensorboard_log,
            verbose=verbose,
            device=device,
            seed=seed,
            sde_support=False,
            optimize_memory_usage=optimize_memory_usage,
            supported_action_spaces=None,
            support_multi_env=True,
        )

        self.exploration_initial_eps = exploration_initial_eps
        self.exploration_final_eps = exploration_final_eps
        self.exploration_fraction = exploration_fraction
        self.target_update_interval = target_update_interval
        # For updating the target network with multiple envs:
        self._n_calls = 0
        self.max_grad_norm = max_grad_norm
        # "epsilon" for the epsilon-greedy exploration
        self.exploration_rate = 0.0
        # Linear schedule will be defined in `_setup_model()`
        self.exploration_schedule = None
        self.q_net, self.q_net_target = None, None

        if _init_setup_model:
            self._setup_model()

    def _setup_model(self) -> None:
        super()._setup_model()
        self._create_aliases()
        # Copy running stats, see GH issue #996
        self.batch_norm_stats_cd = get_parameters_by_name(self.q_net_cd, ["running_"])
        self.batch_norm_stats_cd_target = get_parameters_by_name(self.q_net_cd_target, ["running_"])
        self.batch_norm_stats_in = get_parameters_by_name(self.q_net_in, ["running_"])
        self.batch_norm_stats_in_target = get_parameters_by_name(self.q_net_in_target, ["running_"])

        self.exploration_schedule = get_linear_fn(
            self.exploration_initial_eps,
            self.exploration_final_eps,
            self.exploration_fraction,
        )
        # Account for multiple environments
        # each call to step() corresponds to n_envs transitions
        if self.n_envs > 1:
            if self.n_envs > self.target_update_interval:
                warnings.warn(
                    "The number of environments used is greater than the target network "
                    f"update interval ({self.n_envs} > {self.target_update_interval}), "
                    "therefore the target network will be updated after each call to env.step() "
                    f"which corresponds to {self.n_envs} steps."
                )

            self.target_update_interval = max(self.target_update_interval // self.n_envs, 1)

    def _create_aliases(self) -> None:
        self.q_net_cd = self.policy.q_net_cd
        self.q_net_cd_target = self.policy.q_net_cd_target
        self.q_net_in = self.policy.q_net_in
        self.q_net_in_target = self.policy.q_net_in_target

    def _on_step(self) -> None:
        """
        Update the exploration rate and target network if needed.
        This method is called in ``collect_rollouts()`` after each step in the environment.
        """
        self._n_calls += 1
        if self._n_calls % self.target_update_interval == 0:
            polyak_update(self.q_net_cd.parameters(), self.q_net_cd_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats_cd, self.batch_norm_stats_cd_target, 1.0)
            polyak_update(self.q_net_in.parameters(), self.q_net_in_target.parameters(), self.tau)
            # Copy running stats, see GH issue #996
            polyak_update(self.batch_norm_stats_in, self.batch_norm_stats_in_target, 1.0)

        self.exploration_rate = self.exploration_schedule(self._current_progress_remaining)
        wandb.log({"train/exploration_rate": self.exploration_rate}, commit=False)

    def compute_target_q_values(self, q_target, replay_data):
        # Compute the next Q-values using the target network
        next_q_values = q_target(replay_data.next_observations)
        # Follow greedy policy: use the one with the highest value
        next_q_values, _ = next_q_values.max(dim=1)
        # Avoid potential broadcast issue
        next_q_values = next_q_values.reshape(-1, 1)
        # 1-step TD target
        target_q_values = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values
        return target_q_values

    def train(self, gradient_steps: int, batch_size: int = 2056) -> None:
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer_cd)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer_in)

        losses_cd = []
        losses_in = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            replay_data = self.replay_buffer.sample(batch_size, env=self._vec_normalize_env)

            with th.no_grad():
                target_q_values_cd = self.compute_target_q_values(self.q_net_cd_target, replay_data)
                target_q_values_in = self.compute_target_q_values(self.q_net_in_target, replay_data)

            # Get current Q-values estimates
            current_q_values_cd = self.q_net_cd(replay_data.observations)
            current_q_values_in = self.q_net_in(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values_cd = th.gather(current_q_values_cd, dim=1, index=replay_data.actions[:, 1].reshape(-1, 1).long())
            current_q_values_in = th.gather(current_q_values_in, dim=1, index=replay_data.actions[:, 0].reshape(-1, 1).long())

            # Compute Huber loss (less sensitive to outliers)
            #loss_cd = F.smooth_l1_loss(current_q_values_cd, target_q_values_cd)

            # L2 loss
            loss_cd = F.mse_loss(current_q_values_cd, target_q_values_cd)
            losses_cd.append(loss_cd.item())

            #loss_in = F.smooth_l1_loss(current_q_values_in, target_q_values_in)
            loss_in = F.mse_loss(current_q_values_in, target_q_values_in)
            losses_in.append(loss_in.item())

            # Optimize the policy
            self.policy.optimizer_cd.zero_grad()
            loss_cd.backward()
            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.q_net_cd.parameters(), self.max_grad_norm)
            self.policy.optimizer_cd.step()

            self.policy.optimizer_in.zero_grad()
            loss_in.backward()
            th.nn.utils.clip_grad_norm_(self.policy.q_net_in.parameters(), self.max_grad_norm)
            self.policy.optimizer_in.step()

        # Increase update counter
        self._n_updates += gradient_steps

        # don't log every step for data efficiency
        if self._n_updates % 100 == 0:
            wandb.log({"train/n_updates": self._n_updates,
                       "train/loss_cd": np.mean(losses_cd).astype('float32'),
                       "train/loss_in": np.mean(losses_in).astype('float32')}, commit=False)

    def predict(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """
        Overrides the base_class predict function to include epsilon-greedy exploration.

        :param observation: the input observation
        :param state: The last states (can be None, used in recurrent policies)
        :param episode_start: The last masks (can be None, used in recurrent policies)
        :param deterministic: Whether or not to return deterministic actions.
        :return: the model's action and the next state
            (used in recurrent policies)
        """
        # predict q values
        qs_in, state = self.policy.q_net_in.predict(observation, state, episode_start, deterministic)
        qs_in = qs_in.reshape((self.n_envs, self.action_space.nvec[0]))
        qs_cd, state = self.policy.q_net_cd.predict(observation, state, episode_start, deterministic)
        qs_cd = qs_cd.reshape((self.n_envs, self.action_space.nvec[1]))

        # determine the mask for structure actions
        mask = self._get_action_mask(observation)
        # set masked entries to -inf
        qs_cd[mask[0], mask[1]] = -np.inf

        if not deterministic and np.random.rand() < self.exploration_rate:
            # compute inverse mask
            mask_ind = [[x, y] for x, y in zip(mask[0], mask[1])]
            inv_mask_ind = [[i for i in range(self.action_space.nvec[1])] for _ in range(self.n_envs)]
            [inv_mask_ind[x].remove(y) for x, y in mask_ind]

            if is_vectorized_observation(maybe_transpose(observation, self.observation_space), self.observation_space):
                if isinstance(observation, dict):
                    n_batch = observation[list(observation.keys())[0]].shape[0]
                else:
                    n_batch = observation.shape[0]
                action_cd = np.array([random.choice(inv_mask_ind[i]) for i in range(n_batch)])
                action_in = np.array([self.action_space.sample()[0] for i in range(n_batch)])
            else:
                raise NotImplementedError
                action = np.array(self.action_space.sample())
        else:
            # take argmax
            action_in = np.argmax(qs_in, axis=1)
            action_cd = np.argmax(qs_cd, axis=1)
        action = np.array([action_in, action_cd]).transpose()
        return action, state

    def _get_action_mask(self, observation):
        # Get graph adj for each environment by retrieving the last n_var**2 elements of the observation.
        curr_graph_adjs = observation[:, -(self.action_space.nvec[0] - 1) ** 2 - 1:-1]
        # delete diagonal elements
        curr_graph_adjs = np.delete(curr_graph_adjs, range(0, len(curr_graph_adjs[0]), self.action_space.nvec[0]),
                                    axis=1)
        # get indicies for illegal actions
        ill_del_actions = np.array(np.where(curr_graph_adjs == 0))
        ill_add_actions = np.array(np.where(curr_graph_adjs == 1))
        # adjust indicies for add actions
        ill_add_actions[1] = np.array(
            [a + ((self.action_space.nvec[0] - 1) ** 2 - self.action_space.nvec[0] + 1) for a in
             ill_add_actions[1]])
        # concatenate to create the mask
        mask = np.concatenate((ill_del_actions, ill_add_actions), axis=1)
        return mask

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "DQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        train_freq: TrainFreq,
        replay_buffer: ReplayBuffer,
        action_noise: Optional[ActionNoise] = None,
        learning_starts: int = 0,
        log_interval: Optional[int] = None,
    ) -> RolloutReturn:
        """
        Collect experiences and store them into a ``ReplayBuffer``.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param train_freq: How much experience to collect
            by doing rollouts of current policy.
            Either ``TrainFreq(<n>, TrainFrequencyUnit.STEP)``
            or ``TrainFreq(<n>, TrainFrequencyUnit.EPISODE)``
            with ``<n>`` being an integer greater than 0.
        :param action_noise: Action noise that will be used for exploration
            Required for deterministic policy (e.g. TD3). This can also be used
            in addition to the stochastic policy for SAC.
        :param learning_starts: Number of steps before learning for the warm-up phase.
        :param replay_buffer:
        :param log_interval: Log data every ``log_interval`` episodes
        :return:
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        num_collected_steps, num_collected_episodes = 0, 0

        assert isinstance(env, VecEnv), "You must pass a VecEnv"
        assert train_freq.frequency > 0, "Should at least collect one step or episode."

        if env.num_envs > 1:
            assert train_freq.unit == TrainFrequencyUnit.STEP, "You must use only one env when doing episodic training."

        # Vectorize action noise if needed
        if action_noise is not None and env.num_envs > 1 and not isinstance(action_noise, VectorizedActionNoise):
            action_noise = VectorizedActionNoise(action_noise, env.num_envs)

        if self.use_sde:
            self.actor.reset_noise(env.num_envs)

        callback.on_rollout_start()
        continue_training = True

        while should_collect_more_steps(train_freq, num_collected_steps, num_collected_episodes):
            if self.use_sde and self.sde_sample_freq > 0 and num_collected_steps % self.sde_sample_freq == 0:
                # Sample a new noise matrix
                self.actor.reset_noise(env.num_envs)

            # Select action randomly or according to policy
            actions, buffer_actions = self._sample_action(learning_starts, action_noise, env.num_envs)

            # Rescale and perform action
            new_obs, rewards, dones, infos = env.step(actions)
            [wandb.log({}, commit=True) for _ in range(len(new_obs))]  # to mactch the steps with number of workers

            self.num_timesteps += env.num_envs
            num_collected_steps += 1

            # Give access to local variables
            callback.update_locals(locals())
            # Only stop training if return value is False, not when it is None.
            if callback.on_step() is False:
                return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training=False)

            # Retrieve reward and episode length if using Monitor wrapper
            self._update_info_buffer(infos, dones)

            # Store data in replay buffer (normalized action and unnormalized observation)
            self._store_transition(replay_buffer, buffer_actions, new_obs, rewards, dones, infos)

            self._update_current_progress_remaining(self.num_timesteps, self._total_timesteps)

            # For DQN, check if the target network should be updated
            # and update the exploration schedule
            # For SAC/TD3, the update is dones as the same time as the gradient update
            # see https://github.com/hill-a/stable-baselines/issues/900
            self._on_step()

            for idx, done in enumerate(dones):
                if done:
                    wandb.log({'train/final_reward_avg_workers': sum(rewards)/len(rewards)}, commit=False)
                    # Update stats
                    num_collected_episodes += 1
                    self._episode_num += 1

                    if action_noise is not None:
                        kwargs = dict(indices=[idx]) if env.num_envs > 1 else {}
                        action_noise.reset(**kwargs)

                    # Log training infos
                    if log_interval is not None and self._episode_num % log_interval == 0:
                        self._dump_logs()
        callback.on_rollout_end()

        return RolloutReturn(num_collected_steps * env.num_envs, num_collected_episodes, continue_training)

    def _excluded_save_params(self) -> List[str]:
        return [*super()._excluded_save_params(), "q_net", "q_net_target"]

    def _get_torch_save_params(self) -> Tuple[List[str], List[str]]:
        state_dicts = ["policy", "policy.optimizer"]

        return state_dicts, []
