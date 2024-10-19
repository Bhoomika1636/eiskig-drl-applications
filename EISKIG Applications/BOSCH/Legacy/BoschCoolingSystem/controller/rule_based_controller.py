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
        self.sampling_time = 30 # TODO: automitization necessary!
        self.KT_Controller = ControlKT(self.sampling_time)
        self.KKM_controller = ControlKKM(self.sampling_time)
        # get and define control parameters
        # self.KT_params = yaml.safe_load(open("experiments_hr\BoschCoolingSystem\controller\KTControlParams.yaml", 'r'))
        self.KKM_params = {
            "T_soll": 8,
            "XS_Spreizung_Folgemaschine": 3,
            "Verzögerung": 30*60,
        }

    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """
        Controller of the model. This implements a rule based controller

        :param observation: Observation from the environment.
        :returns: Resulting action from the controller.
        """
        self.prev_action = self.action
        # convert observation array to "human-readable" dictionary with keys
        observations = dict(zip(self.observation_names, observation))

        # Target temperatures
        T_soll_kalt = self.KKM_params['T_soll']

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
        control_signal_WT = k_WT * (observations['T_in_coolingCircuit'] - T_soll_kühl)
        self.action['u_WT1'] = np.clip(control_signal_WT,a_min=0,a_max=1)
        self.action['u_WT2'] = np.clip(control_signal_WT,a_min=0,a_max=1)
        
        # KKM_controller
        self.action['u_KKM1'] = 0
        self.action['u_KKM2'] = 0
        # self.action['u_KKM1'], self.action['u_KKM2'] = self.KKM_controller.update(observation['Temperature_cwCircuit_in'], observation['Temperature_cwCircuit_out'])
        
        # Pumps cooling water secondary
        self.action['u_P7'] = 1
        self.action['u_P8'] = 1
        self.action['u_P9'] = 1

        # Pumps cooling water primary
        self.action['u_P4'] = 1
        self.action['u_P5'] = 1
        self.action['u_P6'] = 1

        # reformat -> TODO: simplify
        actions = []
        actions.append(list(self.action.values()))
        return np.array(actions[0])


# # controller classes -> should be in seperate files
# class Hysteresis:
#     def __init__(self, th_low, th_high):
#         """
#         Initializes a Hysteresis instance with upper and lower thresholds.

#         Parameters:
#         - th_low: Lower threshold for the hysteresis.
#         - th_high: Upper threshold for the hysteresis.
#         """
#         self.th_low = th_low
#         self.th_high = th_high
#         self.state = False

#     def update(self, observation):
#         """
#         Updates the hysteresis state based on the given observation.

#         Parameters:
#         - observation: The input observation value.

#         Notes:
#         - If the observation is greater than or equal to the upper threshold, the state is set to True.
#         - If the observation is less than or equal to the lower threshold, the state is set to False.
#         - If the observation is between the lower and upper thresholds, the current state is maintained.
#         """
#         if observation >= self.th_high:
#             self.state = True
#         elif observation <= self.th_low:
#             self.state = False
#         # If observation is between th_low and th_high, maintain the current state

#     def get_state(self):
#         """
#         Returns the current state of the hysteresis.

#         Returns:
#         - state: Boolean value representing the current state of the hysteresis.
#         """
#         return self.state


# class DualHysteresis:
#     def __init__(self, th_low, th_high, tl_low, tl_high):
#         """
#         Initializes a DualHysteresis instance with upper and lower thresholds for two hysteresis instances.

#         Parameters:
#         - th_low: Lower threshold for the upper hysteresis.
#         - th_high: Upper threshold for the upper hysteresis.
#         - tl_low: Lower threshold for the lower hysteresis.
#         - tl_high: Upper threshold for the lower hysteresis.
#         """
#         self.upper_hysteresis = Hysteresis(th_low, th_high)
#         self.lower_hysteresis = Hysteresis(tl_low, tl_high)

#     def update(self, observation):
#         """
#         Updates both hysteresis instances based on the given observation and returns the AND connection of their states.

#         Parameters:
#         - observation: The input observation value.

#         Returns:
#         - output: Boolean value representing the AND connection of upper and lower hysteresis states.
#         """
#         self.upper_hysteresis.update(observation)
#         self.lower_hysteresis.update(observation)

#         # Calculate and return the AND connection of both hysteresis states
#         return self.upper_hysteresis.get_state() and self.lower_hysteresis.get_state()

# class ControlKT:
#     def __init__(self, sampling_time) -> None:
#         self.sampling_time = sampling_time

#         data = {
#             "Stufe": [
#                 "gw_pump_führung",
#                 "gw_KT_führung_stufe1",
#                 "gw_KT_führung_stufe2",
#                 "gw_pump_folge1",
#                 "gw_KT_folge1_stufe1",
#                 "gw_KT_folge1_stufe2",
#                 "gw_pump_folge2",
#                 "gw_KT_folge2_stufe1",
#                 "gw_KT_folge2_stufe2",
#             ],
#             "th_high": [15, 30, 40, 50, 60, 70, 80, 90, 95],
#             'time_thres': [120, 30, 120,120,120,120,120,120,120]
#         }
#         self.df = pd.DataFrame(data)
#         self.df['isOn'] = False
#         self.df['levelSafeIsOn'] = True
#         self.df['th_low'] = self.df['th_high'] - 13
#         self.df['timer'] = 0

#         self.T_cw_tank = 0
#         self.level_ww_tank = 0

#         self.Yh = 0
#         self.prevYh = 0

#         self.XS_steilheit_regler = 20
#         self.T_soll = 17
#         pass

#     def update(self, T_cw_tank, level_ww_tank):
#         self.prevYh = self.Yh

#         self.T_cw_tank = T_cw_tank
#         self.level_ww_tank = level_ww_tank
#         self.Yh = 50 + self.XS_steilheit_regler * (self.T_cw_tank - self.T_soll)

#         for index, row in self.df.iterrows():
#             stufe = row['Stufe']
#             th_low = row['th_low']
#             th_high = row['th_high']
#             time_thres = row['time_thres']
#             isOn = row['isOn']
            
#             if (not isOn) and self.Yh > th_high:
#                 self.df.at[index, 'timer'] += self.sampling_time  # Increment timer
#             elif isOn and self.Yh < th_low:
#                 self.df.at[index, 'timer'] += self.sampling_time  # Increment timer
#             else:
#                 # Y_h not over a threshold range
#                 self.df.at[index, 'timer'] = 0  # Reset timer

#             if self.df.at[index, 'timer'] > time_thres:
#                 # Timer exceeds time_thres, set isOn to True
#                 self.df.at[index, 'isOn'] = not self.df.at[index, 'isOn']
#                 self.df.at[index, 'timer'] = 0  # Reset timer
            
#         if level_ww_tank < 0.3: # TODO: what to do if level above max level?
#             self.df['levelSafeIsOn'] = False
#         else:
#             self.df['levelSafeIsOn'] = self.df['isOn']

#     def getControlls(self):
#         return self.df.set_index('Stufe')['levelSafeIsOn'].to_dict()


# class ControlKKM:
#     def __init__(self, sampling_time) -> None:
#         self.sampling_time = sampling_time
#         self.T_soll = 8
#         self.T_min = 6
#         self.T_max = 12
#         self.hysteresis = Hysteresis(self.T_min, self.T_max)
#         self.timer = 0
#         self.KKM1_on = False
#         self.KKM2_on = False
#         self.time_thres = 30*60  # 30 minutes
#         pass

#     def update(self, T_in_cold, T_out_cold):
#         T_spreizung = abs(T_in_cold - T_out_cold)
#         if T_spreizung>=3 and not self.KKM2_on:
#             self.timer += self.sampling_time
#         elif self.KKM2_on and T_spreizung<3:
#             self.timer += self.sampling_time
#         else:
#             self.timer = 0

#         if self.timer > self.time_thres:
#             self.KKM2_on = not self.KKM2_on

#         self.hysteresis.update(T_in_cold)
#         self.KKM1_on = self.hysteresis.get_state()
#         return self.KKM1_on, self.KKM2_on
