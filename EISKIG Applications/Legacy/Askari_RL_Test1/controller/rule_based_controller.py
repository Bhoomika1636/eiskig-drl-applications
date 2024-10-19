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

        # initialize action dictionary
        self.action = dict.fromkeys(self.action_names, 0)

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

        # state variables
        T_in_cooling = observations["T"]
        # Target temperatures
        T_target = 18.0
        # control rules
        if T_in_cooling>(T_target+1):
            self.action["u"] = 1
        elif T_in_cooling<(T_target-1):
            self.action["u"] = 0.01
        actions.append(list(self.action.values()))
        actions = actions[0]

        return np.array(actions)
