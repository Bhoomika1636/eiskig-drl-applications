from __future__ import annotations

import inspect
import os
import pathlib
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING

import numpy as np
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import VecMonitor, VecNormalize

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


class ETAx:
    """Initialize an optimization model and provide interfaces for optimization, learning and execution (play).

    :param root_path: Root path of the eta_x application (the configuration will be interpreted relative to this).
    :param config_name: Name of configuration .ini file in configuration directory (should be JSON format).
    :param config_overwrite: Dictionary to overwrite selected configurations.
    :param relpath_config: Relative path to configuration file, starting from root path.
    """

    def __init__(
        self,
        root_path: str | os.PathLike,
        config_name: str,
        config_overwrite: Mapping[str, Any] | None = None,
        relpath_config: str | os.PathLike = "config/",
    ) -> None:
        # Load configuration for the optimization
        _root_path = root_path if isinstance(root_path, pathlib.Path) else pathlib.Path(root_path)
        _relpath_config = relpath_config if isinstance(relpath_config, pathlib.Path) else pathlib.Path(relpath_config)
        #: Path to the configuration file.
        self.path_config = _root_path / _relpath_config / f"{config_name}.json"
        #: ConfigOpt object for the optimization run.
        self.config: ConfigOpt = ConfigOpt.from_json(self.path_config, root_path, config_overwrite)
        log.setLevel(int(self.config.settings.verbose * 10))

        #: Configuration for an optimization run.
        self.config_run: ConfigOptRun | None = None

        #: The vectorized environments.
        self.environments: VecEnv | VecNormalize | None = None
        #: Vectorized interaction environments.
        self.interaction_env: VecEnv | None = None
        #: The model or algorithm.
        self.model: BaseAlgorithm | None = None

    def prepare_run(self, series_name: str, run_name: str, run_description: str = "") -> None:
        """Prepare the learn and play methods by reading configuration, creating results folders and the model.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        :return: Boolean value indicating successful preparation.
        """
        self.config_run = ConfigOptRun(
            series=series_name,
            name=run_name,
            description=run_description,
            path_root=self.config.path_root,
            path_results=self.config.path_results,
            path_scenarios=self.config.path_scenarios,
        )
        self.config_run.create_results_folders()

        # Add file handler to parent logger to log the terminal output
        log_to_file(config=self.config, config_run=self.config_run)

        log.info("Run prepared successfully.")

    def prepare_model(self, reset: bool = False) -> None:
        """Check for existing model and load it or back it up and create a new model.

        :param reset: Flag to determine whether an existing model should be reset.
        """
        self._prepare_model(reset)

    def _prepare_model(self, reset: bool = False) -> None:
        """Check for existing model and load it or back it up and create a new model.

        :param reset: Flag to determine whether an existing model should be reset.
        """
        assert self.config_run is not None, (
            "Set the config_run attribute before trying to initialize the model "
            "(for example by calling prepare_run)."
        )
        assert self.environments is not None, (
            "Initialize the environments before trying to initialize the model" "(for example by calling prepare_run)."
        )

        path_model = self.config_run.path_run_model
        if path_model.is_file() and reset:
            log.info(f"Existing model detected: {path_model}")

            bak_name = path_model / f"_{datetime.fromtimestamp(path_model.stat().st_mtime).strftime('%Y%m%d_%H%M')}.bak"
            path_model.rename(bak_name)
            log.info(f"Reset is active. Existing model will be backed up. Backup file name: {bak_name}")
        elif path_model.is_file():
            log.info(f"Existing model detected: {path_model}. Loading model.")

            self.model = load_model(
                self.config.setup.agent_class,
                self.environments,
                self.config.settings.agent,
                self.config_run.path_run_model,
                tensorboard_log=self.config.setup.tensorboard_log,
                log_path=self.config_run.path_series_results,
            )
            return

        # Initialize the model if it wasn't loaded from a file
        self.model = initialize_model(
            self.config.setup.agent_class,
            self.config.setup.policy_class,
            self.environments,
            self.config.settings.agent,
            self.config.settings.seed,
            tensorboard_log=self.config.setup.tensorboard_log,
            log_path=self.config_run.path_series_results,
        )

    @contextmanager
    def prepare_environments(self, training: bool = True) -> Generator:
        """Context manager which prepares the environments and closes them after it exits.

        :param training: Should preparation be done for training (alternative: playing)?
        """
        try:
            self._prepare_environments(training)
            yield

        finally:
            # close all environments when done (kill processes)
            log.debug("Closing environments.")
            assert self.environments is not None, "Initialized environments could not be found."
            self.environments.close()
            if self.config.settings.interact_with_env:
                assert self.interaction_env is not None, "Initialized interaction environments could not be found."
                self.interaction_env.close()

    def _prepare_environments(self, training: bool = True) -> None:
        """Vectorize and prepare the environments and potentially the interaction environments.

        :param training: Should preparation be done for training (alternative: playing)?
        """
        assert self.config_run is not None, (
            "Set the config_run attribute before trying to initialize the environments "
            "(for example by calling prepare_run)."
        )

        env_class = self.config.setup.environment_class
        self.config_run.set_env_info(env_class)

        legacy_signature = {
            "env_id",
            "run_name",
            "general_settings",
            "path_settings",
            "env_settings",
            "verbose",
            "callback",
        }

        # Check whether the environment uses old style initialization and replace vectorize function accordingly
        env_params = inspect.signature(env_class.__init__).parameters
        if legacy_signature <= set(env_params):
            log.warning(
                f"Environment {env_class.__name__} uses a deprecated __init__ format. " f"Please consider updating."
            )
            self.environments = self._vectorize_legacy_environments("env", training)
        else:
            callback = CallbackEnvironment(self.config.settings.plot_interval)
            # Vectorize the environments
            self.environments = vectorize_environment(
                env_class,
                self.config_run,
                self.config.settings.environment,
                callback,
                self.config.settings.seed,
                self.config.settings.verbose,
                self.config.setup.vectorizer_class,
                self.config.settings.n_environments,
                training=training,
                monitor_wrapper=self.config.setup.monitor_wrapper,
                norm_wrapper_obs=self.config.setup.norm_wrapper_obs,
                norm_wrapper_reward=self.config.setup.norm_wrapper_reward,
            )

        if self.config.settings.interact_with_env:
            # Perform some checks to ensure the interaction environment is configured correctly.
            if self.config.setup.interaction_env_class is None:
                raise ValueError(
                    "If 'interact_with_env' is specified, an interaction env class must be specified as well."
                )
            elif self.config.settings.interaction_env is None:
                raise ValueError(
                    "If 'interact_with_env' is specified, the interaction_env settings must be specified as well."
                )
            interaction_env_class = self.config.setup.interaction_env_class
            self.config_run.set_interaction_env_info(interaction_env_class)

            # Check whether the interaction environment uses old style initialization and replace vectorize
            # function accordingly
            env_params = inspect.signature(interaction_env_class.__init__).parameters
            if legacy_signature == set(env_params):
                log.warning(
                    f"Environment {interaction_env_class.__class__.__name__} uses a deprecated __init__ format. "
                    f"Please consider updating."
                )
                self.interaction_env = self._vectorize_legacy_environments("interaction")
            else:
                callback = CallbackEnvironment(self.config.settings.plot_interval)
                # Vectorize the environment
                self.interaction_env = vectorize_environment(
                    interaction_env_class,
                    self.config_run,
                    self.config.settings.interaction_env,
                    callback,
                    self.config.settings.seed,
                    self.config.settings.verbose,
                    training=training,
                )

    def _vectorize_legacy_environments(self, typ: str = "env", training: bool = False) -> VecNormalize | VecEnv:
        """Vectorize the environment and automatically apply normalization wrappers if configured. If the
        environment is initialized as an interaction_env it will not have normalization wrappers and use the
        appropriate configuration automatically.

        .. deprecated:: v2.0.0
            Use the new style environment initialization instead, by explicitly specifying environment parameters in
            the init function of the environment.

        :param typ: Requested type of environment (normal: 'env' or interaction environment: 'interaction').
        :param training: Flag to identify whether the environment should be initialized for training or playing.
                         It true, it will be initialized for training.
        """
        assert self.config_run is not None, (
            "Set the config_run attribute before trying to initialize the environments "
            "(for example by calling prepare_run)."
        )
        # Create the vectorized environment
        log.debug(
            "Trying to vectorize the environment with the legacy initializer " "(consider updating to new style init)."
        )

        # Ensure n is 1 if the DummyVecEnv is used (it doesn't support more than one) or if typ is 'interaction'.
        if (
            self.config.setup.environment_class.__class__.__name__ == "DummyVecEnv"
            and self.config.settings.n_environments != 1
        ) or typ == "interaction":
            n = 1
            log.warning("Setting number of environments to 1 because DummyVecEnv (default) is used.")
        else:
            n = self.config.settings.n_environments

        # Prepare the requested type of environment (interaction or normal env).
        if typ == "env":
            env = self.config.setup.environment_class
            env_settings = self.config.settings.environment
        elif typ == "interaction":
            assert (
                self.config.setup.interaction_env_class is not None
            ), "If 'interact_with_env' is specified, an interaction env class must be specified as well."
            assert (
                self.config.settings.interaction_env is not None
            ), "If 'interact_with_env' is specified, an interaction env settings must be specified as well."
            env = self.config.setup.interaction_env_class
            env_settings = self.config.settings.interaction_env
        else:
            raise ValueError(f"The environment type must be either 'env' or 'interaction', '{typ}' given.")

        vectorizer = self.config.setup.vectorizer_class

        callback = CallbackEnvironment(self.config.settings.plot_interval)
        # Create the vectorized environments
        envs: VecEnv | VecNormalize
        envs = vectorizer(
            [
                lambda env_id=i + 1: env(  # type: ignore  # using legacy instantiation not recognized correctly.
                    env_id=env_id,
                    run_name=self.config_run.name,
                    general_settings=self.config.settings,
                    path_settings=self.config_run.paths,
                    env_settings=env_settings,
                    verbose=self.config.settings.verbose,
                    callback=callback,
                )
                for i in range(n)
            ]
        )

        if not typ == "interaction" and self.config.setup.monitor_wrapper:
            envs = VecMonitor(envs)

        # Automatically normalize the input features if type isn't an interaction env.
        if not typ == "interaction" and (self.config.setup.norm_wrapper_obs or self.config.setup.norm_wrapper_reward):
            # check if normalization data are available; then load
            if self.config_run.path_vec_normalize.is_file():
                log.info(
                    f"Normalization data detected. Loading running averages into normalization wrapper: \n"
                    f"\t {self.config_run.path_vec_normalize}"
                )
                envs = VecNormalize.load(str(self.config_run.path_vec_normalize), envs)
                envs.training = training
                envs.norm_obs = self.config.setup.norm_wrapper_obs
                envs.norm_reward = self.config.setup.norm_wrapper_reward
            else:
                log.info("No Normalization data detected.")
                envs = VecNormalize(
                    envs,
                    training=training,
                    norm_obs=self.config.setup.norm_wrapper_obs,
                    norm_reward=self.config.setup.norm_wrapper_reward,
                )

        return envs

    def learn(
        self,
        series_name: str | None = None,
        run_name: str | None = None,
        run_description: str = "",
        reset: bool = False,
        callbacks: MaybeCallback = None,
        use_eval_cb: bool = False,
        n_eval_envs: int = 2,
        eval_freq: int = 50000
    ) -> None:
        """Start the learning job for an agent with the specified environment.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        :param reset: Indication whether possibly existing models should be reset. Learning will be continued if
                           model exists and reset is false.
        :param callbacks: Provide additional callbacks to send to the model.learn() call.
        """
        if is_env_closed(self.environments) or self.model is None:
            _series_name = series_name if series_name is not None else ""
            _run_name = run_name if run_name is not None else ""
            self.prepare_run(_series_name, _run_name, run_description)

        assert self.config_run is not None, "Run configuration could not be found. Call prepare_run first."

        with self.prepare_environments(training=True):
            assert (
                self.environments is not None
            ), "Initialized environments could not be found. Call prepare_environments first."

            self.prepare_model(reset)
            assert self.model is not None, "Initialized model could not be found. Call prepare_model first."

            # Log some information about the model and configuration
            log_net_arch(self.model, self.config_run)
            log_run_info(self.config, self.config_run)

            # Genetic algorithm has a slightly different concept for saving since it does not stop between time steps
            if "n_generations" in self.config.settings.agent:
                save_freq = self.config.settings.save_model_every_x_episodes
                total_timesteps = self.config.settings.agent["n_generations"]
            else:
                # Check if all required config values are present
                if self.config.settings.episode_duration is None:
                    raise ValueError("Missing configuration values for learning: 'episode_duration'.")
                elif self.config.settings.sampling_time is None:
                    raise ValueError("Missing configuration values for learning: 'sampling_time'.")
                elif self.config.settings.n_episodes_learn is None:
                    raise ValueError("Missing configuration values for learning: 'n_episodes_learn'.")

                # define callback for periodically saving models
                save_freq = int(
                    self.config.settings.episode_duration
                    / self.config.settings.sampling_time
                    * self.config.settings.save_model_every_x_episodes
                )
                total_timesteps = int(
                    self.config.settings.n_episodes_learn
                    * self.config.settings.episode_duration
                    / self.config.settings.sampling_time
                )
            # custom code by David Askari -- BEGIN
            from stable_baselines3.common.callbacks import EvalCallback
            # own eval call that enables me to call the render methods on the evaluation environment after finishing episode
            class MyEvalCallback(EvalCallback):
                def __init__(self, eval_env, **kwargs):
                    super().__init__(eval_env, **kwargs)

                def _log_success_callback(self, locals_: Dict[str, Any], globals_: Dict[str, Any]) -> None:
                    """
                    Callback passed to the  ``evaluate_policy`` function
                    in order to log the success rate (when applicable),
                    for instance when using HER.

                    :param locals_:
                    :param globals_:
                    """
                    info = locals_["info"]

                    if locals_["done"]:
                        maybe_is_success = info.get("is_success")
                        if maybe_is_success is not None:
                            self._is_success_buffer.append(maybe_is_success)
                        self.eval_env.env_method("render", "human", name_suffix="Eval")#method_args={"name_suffix":"Eval"}
                        self.eval_env.env_method("render_episodes","Eval")
            if use_eval_cb:
                evalEnvs= vectorize_environment(
                    self.config.setup.environment_class,
                    self.config_run,
                    self.config.settings.environment,
                    None,
                    self.config.settings.seed+1,
                    self.config.settings.verbose,
                    self.config.setup.vectorizer_class,
                    n_eval_envs,
                    training=False,
                    monitor_wrapper=self.config.setup.monitor_wrapper,
                    norm_wrapper_obs=self.config.setup.norm_wrapper_obs,
                    norm_wrapper_reward=self.config.setup.norm_wrapper_reward,
                )
                eval_callback = MyEvalCallback(eval_env=evalEnvs, eval_freq=eval_freq, n_eval_episodes=1, deterministic=True, render=False, best_model_save_path=self.config.path_results / series_name /'best_model')
                callback_learn = merge_callbacks(
                    CheckpointCallback(
                        save_freq=save_freq,
                        save_path=str(self.config_run.path_series_results / "models"),
                        name_prefix=self.config_run.name,
                    ),
                    eval_callback,
                    callbacks,
                )
            else:
                callback_learn = merge_callbacks(
                    CheckpointCallback(
                        save_freq=save_freq,
                        save_path=str(self.config_run.path_series_results / "models"),
                        name_prefix=self.config_run.name,
                    ),
                    callbacks,
                )
            # custom code by David Askari -- END
            
            # Start learning
            log.info("Start learning process of agent in environment.")
            try:
                self.model.learn(
                    total_timesteps=total_timesteps,
                    callback=callback_learn,
                    tb_log_name=self.config_run.name,
                )
            except OSError:
                filename = str(self.config_run.path_series_results / f"{self.config_run.name}_model_before_error.pkl")
                log.info(f"Saving model to file: {filename}.")
                self.model.save(filename)
                raise

            # reset environment one more time to call environment callback one last time
            self.environments.reset()

            # save model
            log.debug(f"Saving model to file: {self.config_run.path_run_model}.")
            self.model.save(self.config_run.path_run_model)
            if isinstance(self.environments, VecNormalize):
                log.debug(f"Saving environment normalization data to file: {self.config_run.path_vec_normalize}.")
                self.environments.save(str(self.config_run.path_vec_normalize))

        log.info(f"Learning finished: {series_name} / {run_name}")

    def play(self, series_name: str | None = None, run_name: str | None = None, run_description: str = "") -> None:
        """Play with previously learned agent model in environment.

        :param series_name: Name for a series of runs.
        :param run_name: Name for a specific run.
        :param run_description: Description for a specific run.
        """
        if is_env_closed(self.environments) or self.model is None:
            _series_name = series_name if series_name is not None else ""
            _run_name = run_name if run_name is not None else ""
            self.prepare_run(_series_name, _run_name, run_description)

        assert self.config_run is not None, "Run configuration could not be found. Call prepare_run first."

        with self.prepare_environments(training=True):
            assert (
                self.environments is not None
            ), "Initialized environments could not be found. Call prepare_environments first."

            self.prepare_model(reset=False)
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

            while n_episodes < n_episodes_stop:
                try:
                    observations, dones = self._play_step(_round_actions, _scale_actions, observations)
                except BaseException as e:
                    log.error(
                        "Exception occurred during an environment step. Aborting and trying to reset environments."
                    )
                    observations = self._reset_envs()
                    log.debug("Environment reset successful - re-raising exception")
                    raise e

                n_episodes += sum(dones)

    def _play_step(
        self, _round_actions: int | None, _scale_actions: float, observations: VecEnvObs
    ) -> tuple[VecEnvObs, np.ndarray]:
        assert self.environments is not None, "Initialized environments could not be found. Call prepare_run first."

        action, _states = self.model.predict(observation=observations, deterministic=False)  # type: ignore
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
        return observations, dones

    def _reset_envs(self) -> VecEnvObs:
        """Reset the environments when interaction with another environment is taking place.

        :param observations: Observations from the interaction env.
        :return: observations after reset.
        """
        assert self.environments is not None, "Initialized environments could not be found. Call prepare_run first."
        log.debug("Resetting environments.")

        if self.config.settings.interact_with_env:
            assert (
                self.interaction_env is not None
            ), "Initialized interaction environments could not be found. Call prepare_run first."
            observations = self.interaction_env.reset()
            return self._reset_env_interaction(observations)
        else:
            return self.environments.reset()

    def _reset_env_interaction(self, observations: VecEnvObs) -> VecEnvObs:
        assert self.environments is not None, "Initialized environments could not be found. Call prepare_run first."
        log.debug("Resetting main environment during environment interaction.")

        try:
            observations = np.array(self.environments.env_method("first_update", observations, indices=0))
        except AttributeError as e:
            if "first_update" in str(e):
                observations = self.environments.reset()
            else:
                raise e

        return observations
