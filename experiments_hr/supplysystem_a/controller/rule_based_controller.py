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
        Controller of the model. This implements a rule based controller

        :param observation: Observation from the environment.
        :returns: Resulting action from the controller.
        """

        actions = []
        # convert observation array to "human-readable" dictionary with keys
        observations = dict(zip(self.observation_names, observation))
        # initialize action dictionary
        action = dict.fromkeys(self.action_names, 0)

        # state variables
        temp_heat_hi = observations["s_temp_heat_storage_hi"] - 273.15
        temp_heat_lo = observations["s_temp_heat_storage_lo"] - 273.15

        # control rules
        # combinedheatpower
        if observations["s_u_combinedheatpower"] <= 0:  # off
            action["u_combinedheatpower"] = (temp_heat_hi < 72) * 1.0
        else:  # already on
            action["u_combinedheatpower"] = (temp_heat_lo < 69 and temp_heat_hi < 86) * 1.0
        # condensingboiler
        if observations["s_u_condensingboiler"] <= 0:  # off
            action["u_condensingboiler"] = (temp_heat_hi < 71) * 1.0
        else:  # already on
            action["u_condensingboiler"] = (temp_heat_lo < 68 and temp_heat_hi < 85) * 1.0
        # immersionheater
        if observations["s_u_immersionheater"] <= 0:  # off
            action["u_immersionheater"] = (temp_heat_hi < 70) * 1.0
        else:  # already on
            action["u_immersionheater"] = (temp_heat_lo < 67 and temp_heat_hi < 83) * 1.0

        actions.append(list(action.values()))
        actions = actions[0]

        return np.array(actions)
