"""For defining eventual callback functions for the training loop"""
import torch
from stable_baselines3.common.callbacks import EventCallback, BaseCallback
from evaluation import evaluate_policy
import os
import numpy as np
from typing import Optional
import wandb


class EvalTrainTestCallback(EventCallback):
    """Slightly adapted version of stable_baselines.common.callbacks.EvalCallback"""
    def __init__(self, val_frequency: int, scms_val, n_vars: int, episode_length: int,
                 best_model_save_path: str,  verbose: int = 1,
                 callback_on_new_best: Optional[BaseCallback] = None, intervention_value: int = 20) -> None:
        super(EvalTrainTestCallback, self).__init__(callback_on_new_best, verbose=verbose)
        self.val_frequency = val_frequency
        self.scms_val = scms_val
        self.total_steps = 0
        self.best_model_save_path = best_model_save_path
        self.n_vars = n_vars
        self.episode_length = episode_length
        self.evaluations_timesteps = []
        self.evaluations_results_val = []
        self.best_mean_reward = np.inf
        self.intervention_value = intervention_value

    def _init_callback(self) -> None:
        # Create folders if needed
        if self.best_model_save_path is not None:
            os.makedirs(self.best_model_save_path, exist_ok=True)

    def _on_step(self) -> bool:
        if self.val_frequency > 0 and self.n_calls % self.val_frequency == 0:
            # run the policy on all testset envs and collect rewards
            episode_rewards_val = evaluate_policy(self.model, self.scms_val, 1, self.n_vars, self.episode_length,
                                                  intervention_value=self.intervention_value)[0]

            mean_reward_val = np.mean(episode_rewards_val)
            # Keep track of the last evaluation, useful for classes that derive from this callback
            self.last_mean_reward_val = mean_reward_val

            print("Eval val num_timesteps={}, avg_shd={:.2f}".format(self.num_timesteps, mean_reward_val))
            wandb.log({'eval/avg_shd': mean_reward_val}, commit=False)

            if mean_reward_val <= self.best_mean_reward:
                print("New best mean reward!")
                if self.best_model_save_path is not None:
                    self.model.q_net_cd.save(self.best_model_save_path + '/best_model_cd')
                    if self.verbose >= 2:
                        print(f"Saving model checkpoint to {self.best_model_save_path}/best_model_cd.zip")

                    self.model.q_net_in.save(self.best_model_save_path + '/best_model_in')
                    if self.verbose >= 2:
                        print(f"Saving model checkpoint to {self.best_model_save_path}/best_model_in.zip")
                self.best_mean_reward = mean_reward_val

                # Trigger callback if needed
                if self.callback is not None:
                    return self._on_event()
            print()

        return True


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, it only saves model checkpoints,
    you need to pass ``save_replay_buffer=True``,
    and ``save_vecnormalize=True`` to also save replay buffer checkpoints
    and normalization statistics checkpoints.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param name_prefix: Common prefix to the saved models
    :param save_replay_buffer: Save the model replay buffer
    :param save_vecnormalize: Save the ``VecNormalize`` statistics
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: str,
        name_prefix: str = "rl_model",
        save_replay_buffer: bool = False,
        save_vecnormalize: bool = False,
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.save_path = save_path
        self.name_prefix = name_prefix
        self.save_replay_buffer = save_replay_buffer
        self.save_vecnormalize = save_vecnormalize

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _checkpoint_path(self, checkpoint_type: str = "", extension: str = "", net_type: str = "") -> str:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :param checkpoint_type: empty for the model, "replay_buffer_"
            or "vecnormalize_" for the other checkpoints.
        :param extension: Checkpoint file extension (zip for model, pkl for others)
        :return: Path to the checkpoint
        """
        return os.path.join(self.save_path, f"{self.name_prefix}_{net_type}{checkpoint_type}.{extension}")

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            model_path = self._checkpoint_path(extension="zip", net_type='cd')
            self.model.q_net_cd.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            model_path = self._checkpoint_path(extension="zip", net_type='in')
            self.model.q_net_in.save(model_path)
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")

            if self.save_replay_buffer and hasattr(self.model, "replay_buffer") and self.model.replay_buffer is not None:
                # If model has a replay buffer, save it too
                replay_buffer_path = self._checkpoint_path("replay_buffer_", extension="pkl")
                self.model.save_replay_buffer(replay_buffer_path)
                if self.verbose > 1:
                    print(f"Saving model replay buffer checkpoint to {replay_buffer_path}")

            if self.save_vecnormalize and self.model.get_vec_normalize_env() is not None:
                # Save the VecNormalize statistics
                vec_normalize_path = self._checkpoint_path("vecnormalize_", extension="pkl")
                self.model.get_vec_normalize_env().save(vec_normalize_path)
                if self.verbose >= 2:
                    print(f"Saving model VecNormalize to {vec_normalize_path}")

        return True
