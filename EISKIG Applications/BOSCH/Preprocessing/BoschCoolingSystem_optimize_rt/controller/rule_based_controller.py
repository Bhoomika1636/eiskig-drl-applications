from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from eta_utility.eta_x.agents import RuleBased

from common.controllerFunctions import ControlKT, ControlKKM

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

    def __init__(
        self, policy: type[BasePolicy], env: VecEnv, verbose: int = 1, **kwargs: Any
    ):
        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)

        # extract action and observation names from the environments state_config
        self.action_names = self.env.envs[0].state_config.actions
        self.observation_names = self.env.envs[0].state_config.observations

        # initialize action dictionary
        self.action = dict.fromkeys(self.action_names, 0)

        # set initial state
        self.initial_state = np.zeros(self.action_space.shape)
        self.prev_action = dict.fromkeys(self.action_names, 0)
        #time stuff
        self.sampling_time = 120 # TODO: automitization necessary!
        self.KT_Controller = ControlKT(self.sampling_time)
        self.KKM_controller = ControlKKM(self.sampling_time)
        self.error_wt_integrator = 0

    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """
        Controller of the model. This implements a rule based controller

        :param observation: Observation from the environment.
        :returns: Resulting action from the controller.
        """
        self.prev_action = self.action
        # convert observation array to "human-readable" dictionary with keys
        observations = dict(zip(self.observation_names, observation))

        # KTcontrols
        self.KT_Controller.update(observations['T_tank_cw'], observations['Tank_level_warm'])
        KTcontrols = self.KT_Controller.getControlls()
        
        self.action['u_P1'] = KTcontrols["gw_pump_führung"]
        self.action['u_KT_1'] = KTcontrols["gw_KT_führung_stufe1"]
        self.action['u_KT_2'] = KTcontrols["gw_KT_führung_stufe2"]

        self.action['u_P2'] = KTcontrols["gw_pump_folge1"]
        self.action['u_KT_3'] = KTcontrols["gw_KT_folge1_stufe1"]
        self.action['u_KT_4'] = KTcontrols["gw_KT_folge1_stufe2"]

        self.action['u_P3'] = KTcontrols["gw_pump_folge2"]
        self.action['u_KT_5'] = KTcontrols["gw_KT_folge2_stufe1"]
        self.action['u_KT_6'] = KTcontrols["gw_KT_folge2_stufe2"]

        # WT
        T_soll_kühl = 18
        k_WT = 1
        k_I = 1/200
        self.error_wt_integrator += (observations['T_in_coolingCircuit'] - T_soll_kühl) * self.sampling_time
        self.error_wt_integrator = np.clip(self.error_wt_integrator,0,360)
        control_signal_WT = k_WT * (observations['T_in_coolingCircuit'] - T_soll_kühl) + k_I *(self.error_wt_integrator)
        self.action['u_WT1'] = np.clip(control_signal_WT,a_min=0,a_max=1)
        self.action['u_WT2'] = np.clip(control_signal_WT,a_min=0,a_max=1)
        
        # KKM_controller
        self.action['u_KKM1_On'] = 1
        self.action['u_KKM2_On'] = 1
        self.action['u_KKM1_T_Target'] = 8
        self.action['u_KKM2_T_Target'] = 8
        # self.action['u_KKM1'], self.action['u_KKM2'] = self.KKM_controller.update(observation['Temperature_cwCircuit_in'], observation['Temperature_cwCircuit_out'])

        # reformat -> TODO: simplify
        actions = []
        actions.append(list(self.action.values()))
        return np.array(actions[0])