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

        
        # all actions are set to 0 as dummy variables - this rule base controller is not used since conventional strategy is in dymola
        action["u_AKT01"] = 0
        action["u_AKT02"] = 0
        action["u_AKT03"] = 0
        action["u_AKT04"] = 0
        action["u_valve_bypass"] = 0
        action["u_PKAB"] = 0
        action["u_PKW1"] = 0
        action["u_PKW2"] = 0
        action["u_PKW3"] = 0
        action["u_pump_PKAB"] = 0
        action["u_pump_PKW1"] = 0
        action["u_pump_PKW2"] = 0
        action["u_pump_PKW3"] = 0
        action["u_iceStorage"] = 0
        action["u_heatEx_PKWE"] = 0
        

        actions.append(list(action.values()))
        actions = actions[0]

        return np.array(actions)
