from __future__ import annotations

from typing import Any, Generator, Mapping
from eta_utility.eta_x import ETAx

import os
import pathlib
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecNormalize

from eta_utility import get_logger
from eta_utility.eta_x import ConfigOpt, ConfigOptRun
from eta_utility.eta_x.common import (
    CallbackEnvironment,
    initialize_model,
    is_env_closed,
    load_model,
    log_net_arch,
    log_run_info,
    log_to_file,
    merge_callbacks,
    vectorize_environment,
)

if TYPE_CHECKING:
    from typing import Any, Generator, Mapping

    from stable_baselines3.common.base_class import BaseAlgorithm
    from stable_baselines3.common.type_aliases import MaybeCallback
    from stable_baselines3.common.vec_env import VecEnv
    from stable_baselines3.common.vec_env.base_vec_env import VecEnvObs

log = get_logger("eta_x")

class ETAx2(ETAx):
    def __init__(
        self,
        root_path: str | os.PathLike,
        config_name: str,
        config_overwrite: Mapping[str, Any] | None = None,
        relpath_config: str | os.PathLike = "config/",
    ) -> None:
        super().__init__(root_path, config_name, config_overwrite, relpath_config)
        # Add any additional initialization steps specific to ETAx2 here if needed


    def play(self, series_name: str | None = None, run_name: str | None = None, run_description: str = "") -> None:
        """Play with previously learned agent model in environment.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        """
        with self.prepare_environments_models(series_name, run_name, run_description, reset=False, training=False):
            assert self.config_run is not None, "Run configuration could not be found. Call prepare_run first."
            assert (
                self.environments is not None
            ), "Initialized environments could not be found. Call prepare_environments first."
            assert self.model is not None, "Initialized model could not be found. Call prepare_model first."

            if self.config.settings.n_episodes_play is None:
                raise ValueError("Missing configuration value for playing: 'n_episodes_play' in section 'settings'")

            # Log some information about the model and configuration
            log_net_arch(self.model, self.config_run)
            log_run_info(self.config, self.config_run)

            n_episodes_stop = self.config.settings.n_episodes_play

            # Reset the environments before starting to play
            try:
                log.debug("Resetting environments before starting to play.")
                observations = self._reset_envs()
            except ValueError as e:
                raise ValueError(
                    "It is likely that returned observations do not conform to the specified state config."
                ) from e
            n_episodes = 0

            log.debug("Start playing process of agent in environment.")
            if self.config.settings.interact_with_env:
                log.info("Starting agent with environment/optimization interaction.")
            else:
                log.info("Starting without an additional interaction environment.")

            _round_actions = self.config.settings.round_actions
            _scale_actions = self.config.settings.scale_actions if self.config.settings.scale_actions is not None else 1

            episode_rewards = [] #added this one in custom play function

            while n_episodes < n_episodes_stop:
                try:
                    observations, dones, reward = self._play_step(_round_actions, _scale_actions, observations)
                    episode_rewards.append(reward[0])
                except BaseException as e:
                    log.error(
                        "Exception occurred during an environment step. Aborting and trying to reset environments."
                    )
                    try:
                        observations = self._reset_envs()
                    except BaseException as followup_exception:
                        raise e from followup_exception
                    log.debug("Environment reset successful - re-raising exception")
                    raise e

                n_episodes += sum(dones)
            
            return np.mean(episode_rewards)


    def _play_step(
        self, _round_actions: int | None, _scale_actions: float, observations: VecEnvObs
    ) -> tuple[VecEnvObs, np.ndarray]:
        assert self.environments is not None, "Initialized environments could not be found. Call prepare_run first."

        # action, _states = self.model.predict(observation=observations, deterministic=False)  # type: ignore
        action, _states = self.model.predict(observation=observations, deterministic=True)  # type: ignore

        # Type ignored because typing in stable_baselines appears to be incorrect
        # Round and scale actions if required
        if _round_actions is not None:
            action = np.round(action * _scale_actions, _round_actions)
        else:
            action *= _scale_actions
        # Some agents (i.e. MPC) can interact with an additional environment
        if self.config.settings.interact_with_env:
            assert (
                self.interaction_env is not None
            ), "Initialized interaction environments could not be found. Call prepare_run first."

            # Perform a step  with the interaction environment and update the normal environment with
            # its observations
            observations, rewards, dones, info = self.interaction_env.step(action)
            observations = np.array(self.environments.env_method("update", observations, indices=0))
            # Make sure to also reset the environment, if the interaction_env says it's done. For the interaction
            # env this is done inside the vectorizer.
            for idx in range(self.environments.num_envs):
                if dones[idx]:
                    info[idx]["terminal_observation"] = observations
                    observations[idx] = self._reset_env_interaction(observations)
        else:
            observations, rewards, dones, info = self.environments.step(action)
        return observations, dones, rewards