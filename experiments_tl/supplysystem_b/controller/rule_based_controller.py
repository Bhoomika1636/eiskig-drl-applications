from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from eta_utility.eta_x.agents import RuleBased

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv


class RuleBasedController(RuleBased):
    """
    Simple rule based controller for supplysystem_a.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class.
    """

    def __init__(self, policy: type[BasePolicy], env: VecEnv, verbose: int = 1, **kwargs: Any):
        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)

        # extract action and observation names from the environments state_config
        self.action_names = self.env.envs[0].state_config.actions
        self.observation_names = self.env.envs[0].state_config.observations

        # set initial state
        self.initial_state = np.zeros(self.action_space.shape)

    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """
        Controller of the model. This implements a simple PID controller

        :param observation: Observation from the environment.
        :returns: Resulting action from the PID controller.
        """

        actions = []
        # convert observation array to "human-readable" dictionary with keys
        observation = dict(zip(self.observation_names, observation))
        # initialize action dictionary
        action = dict.fromkeys(self.action_names, 0)

        # state variables
        temp_heat_hi = observation["s_temp_heat_storage_hi"] - 273.15
        temp_heat_lo = observation["s_temp_heat_storage_lo"] - 273.15
        temp_cold_hi = observation["s_temp_cold_storage_hi"] - 273.15
        temp_cold_lo = observation["s_temp_cold_storage_lo"] - 273.15

        # control rules
        # combinedheatpower
        if observation["s_u_combinedheatpower"] <= 0:  # off
            action["u_combinedheatpower"] = (temp_heat_hi < 72) * 1.0
        else:  # already on
            action["u_combinedheatpower"] = (temp_heat_lo < 69 and temp_heat_hi < 86) * 1.0
        # condensingboiler
        if observation["s_u_condensingboiler"] <= 0:  # off
            action["u_condensingboiler"] = (temp_heat_hi < 71) * 1.0
        else:  # already on
            action["u_condensingboiler"] = (temp_heat_lo < 68 and temp_heat_hi < 85) * 1.0
        # immersionheater
        if observation["s_u_immersionheater"] <= 0:  # off
            action["u_immersionheater"] = (temp_heat_hi < 70) * 1.0
        else:  # already on
            action["u_immersionheater"] = (temp_heat_lo < 67 and temp_heat_hi < 83) * 1.0
        # coolingtower
        if observation["s_u_coolingtower"] <= 0:  # off
            action["u_coolingtower"] = (temp_cold_lo > 15) * 1.0
        else:  # already on
            action["u_coolingtower"] = (temp_cold_hi > 15 and temp_cold_lo > 5) * 1.0
        # compressionchiller
        if observation["s_u_compressionchiller"] <= 0:  # off
            action["u_compressionchiller"] = (temp_cold_lo > 16) * 1.0
        else:  # already on
            action["u_compressionchiller"] = (temp_cold_hi > 16 and temp_cold_lo > 6) * 1.0
        # heatpump
        if observation["s_u_heatpump"] <= 0:  # off
            action["u_heatpump"] = (
                (temp_heat_hi < 70.5 or temp_cold_lo > 16.5) and (temp_heat_hi < 84 and temp_cold_lo > 7)
            ) * 1.0
        else:  # already on
            action["u_heatpump"] = (
                (temp_heat_lo < 70 or temp_cold_hi > 16.5) and (temp_heat_hi < 84 and temp_cold_lo > 7)
            ) * 1.0

        actions.append(list(action.values()))
        actions = actions[0]

        return np.array(actions)
