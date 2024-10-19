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
        # Define targets based on conditions:
        # KüWa: 12°C/14°C/28°C
        # KaWa: 6°C /8°C/ 11.5°C
        T_soll_kühl_prim = 14
        T_soll_kühl_sec = 17
        # T_wb = calculate_wet_bulb_temperature_stull(observations["d_weather_groundtemperature"],observations["u_weather_RelativeHumidity"])
        # if (T_wb>=14):
        #     T_soll_kühl = 26
        # else:
        #     T_soll_kühl = 14
        self.KT_Controller.set_T_soll(T_soll_kühl_prim)
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

        # # WT control better
        k_WT = 0.02 # noch höher, dafür K_I niedriger ?
        k_I = 10/(60*20) # -> max beitrag u = 0.4 
        self.error_wt_integrator += (observations['T_in_coolingCircuit'] - T_soll_kühl_sec) * self.sampling_time
        self.error_wt_integrator = np.clip(self.error_wt_integrator,-20*60,20*60) # 1 degree difference for 20 minutes
        control_signal_WT = k_WT * ((observations['T_in_coolingCircuit'] - T_soll_kühl_sec) + k_I *(self.error_wt_integrator))
        
        # WT control worse
        # k_WT = 1
        # k_I =  1/200 
        # self.error_wt_integrator += (observations['T_in_coolingCircuit'] - T_soll_kühl_sec) * self.sampling_time
        # self.error_wt_integrator = np.clip(self.error_wt_integrator,0,360)
        # control_signal_WT = k_WT * observations['T_in_coolingCircuit'] - T_soll_kühl_sec + k_I *(self.error_wt_integrator)

        self.action['u_WT1'] = np.clip(control_signal_WT,a_min=0,a_max=1)
        self.action['u_WT2'] = np.clip(control_signal_WT,a_min=0,a_max=1)
        
        # KKM_controller
        self.action['u_KKM1_On'] = 1
        self.action['u_KKM2_On'] = 1
        self.action['u_KKM1_On'], self.action['u_KKM2_On'] = self.KKM_controller.update(observations['Temperature_cwCircuit_in'], observations['Temperature_cwCircuit_out'])

        # reformat -> TODO: simplify
        actions = []
        actions.append(list(self.action.values()))
        return np.array(actions[0])
    
def calculate_wet_bulb_temperature_stull(T, RH):
    """
    Calculate the wet bulb temperature using the Stull formula.

    Parameters:
    - T: Air (dry bulb) temperature in degrees Celsius.
    - RH: Relative humidity as a percentage (0-100).

    Returns:
    - Wet bulb temperature in degrees Celsius.
    """
    T_wb = T * np.arctan(0.151977 * np.sqrt(RH + 8.313659)) \
           + np.arctan(T + RH) - np.arctan(RH - 1.676331) \
           + 0.00391838 * RH**(3/2) * np.arctan(0.023101 * RH) \
           - 4.686035
    return T_wb