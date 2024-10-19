from __future__ import annotations

import random
from typing import TYPE_CHECKING

import numpy as np
from eta_utility.eta_x.agents import RuleBased

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv


class SpeedTestController(RuleBased):
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

        self.random_action_value = 0

        random.seed(111)

    # def control_rules(self, observation: np.ndarray) -> np.ndarray:
    def control_rules(self, observation) -> np.ndarray:
        """
        Controller of the model.

        :param observation: Observation from the environment.
        :returns: actions
        """

        # print("step performed")
        # print(type(observation))
        observation = dict(zip(self.observation_names, observation))

        action = dict.fromkeys(self.action_names, 0)  # this creates a dict with the action names and 0 as values

        # get observations

        action["bSetStatusOn_HeatExchanger1"] = random.randint(0, 1)
        action["bSetStatusOn_CHP1"] = random.randint(0, 1)
        action["bSetStatusOn_CHP2"] = random.randint(0, 1)
        action["bSetStatusOn_CondensingBoiler"] = random.randint(0, 1)
        action["bSetStatusOn_VSIStorage"] = random.randint(0, 1)
        action["bLoading_VSISystem"] = random.randint(0, 1)
        action["bSetStatusOn_OuterCapillaryTubeMats"] = random.randint(0, 1)
        action["bSetStatusOn_eChiller"] = random.randint(0, 1)
        action["bSetStatusOn_eChiller"] = 0
        action["bSetStatusOn_HVFASystem_CN"] = random.randint(0, 1)
        action["bLoading_HVFASystem_CN"] = random.randint(0, 1)
        action["bSetStatusOn_HVFASystem_HNLT"] = random.randint(0, 1)
        action["bLoading_HVFASystem_HNLT"] = random.randint(0, 1)
        action["bSetStatusOn_HeatPump"] = random.randint(0, 1)

        ####################################################################
        #                                                                  #
        #  The following actions are needed for env and FMU based on TSCL  #
        #                                                                  #
        ####################################################################

        # action["HeatExchanger1_bAlgorithmModeActivated"] = 1
        # action["CHP1_bAlgorithmModeActivated"] = 1
        # action["CHP2_bAlgorithmModeActivated"] = 1
        # action["CondensingBoiler_bAlgorithmModeActivated"] = 1
        # action["VSIStorage_bAlgorithmModeActivated"] = 1
        # action["OuterCapillaryTubeMats_bAlgorithmModeActivated"] = 1
        # action["eChiller_bAlgorithmModeActivated"] = 1
        # action["HVFASystem_CN_bAlgorithmModeActivated"] = 1
        # action["HVFASystem_HNLT_bAlgorithmModeActivated"] = 1
        # action["HeatPump_bAlgorithmModeActivated"] = 1

        actions = []
        actions.append(list(action.values()))
        # print(actions)
        actions = actions[0]

        return np.array(actions)
