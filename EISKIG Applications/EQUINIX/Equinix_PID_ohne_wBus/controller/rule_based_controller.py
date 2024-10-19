from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from eta_utility.eta_x.agents import RuleBased

from eta_utility import get_logger, timeseries
from datetime import timedelta



if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv


class RuleBasedController(RuleBased):
    """
    Simple rule based controller for Equinix.

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
        self.t_an = 0.0
        self.t_aus = 0.0
        self.t_an1 = 0.0
        self.t_aus1 = 0.0
        
        
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
        T_nach_Tower = observation["s_T_nach_Tower"]
        T_air = observation["d_weather_drybulbtemperature"]+273.15
        T_vor_CoolingTower = observation["T_vor_CoolingTower"]
        time = observation["d_weather_time"]
    
        # T_Target
        action["T_target_Tower"] = 290.75
        action["T_target_Chiller1"] = 290.75
        action["T_target_Chiller2"] = 290.75
        action["u_CoolingTower_on_off"] = 1.0


        #Kreislaufpumpen
        action["u_Glykolpumpen"] = 0.9
        action["u_Wasserpumpen"] = 0.5

        #     #rulebased without time constraints
        # if T_air>T_vor_CoolingTower:
        #     action["u_CoolingTower_on_off"] = 0.0
        # elif T_air<T_vor_CoolingTower:
        #     action["u_CoolingTower_on_off"] = 1.0

        # if T_air>= 15.0:
        #    action["u_adiabatic"] =1.0
        # else:
        #     action["u_adiabatic"] = 0.0
        
        # if T_nach_Tower > 293.15:
            
        #         action["u_Chiller1_on_off"] = 1.0
        #         action["u_Chiller2_on_off"] = 1.0
        #         action["u_Pumpe_Chiller1"] = 0.9
        #         action["u_Pumpe_Chiller2"] = 0.9
        # else:
            
        #         action["u_Chiller1_on_off"] = 0.0
        #         action["u_Chiller2_on_off"] = 0.0
        #         action["u_Pumpe_Chiller1"] = 0.0
        #         action["u_Pumpe_Chiller2"] = 0.0


        # #rulebased with time constraints
         
        if T_air>T_vor_CoolingTower:
            action["u_CoolingTower_on_off"] = 0.0
        else:
            action["u_CoolingTower_on_off"] = 1.0
       

        if T_air>= 15.0:
           action["u_adiabatic"] =1.0
        else:
            action["u_adiabatic"] = 0.0
        
        #if observation["s_u_Chiller1_on_off"] and observation["s_u_Chiller2_on_off"]<=0.0:
        if observation["s_u_Chiller1_on_off"]==0:    
            if T_nach_Tower >= 293.15:
                    if self.t_an == 0.0: 
                        self.t_an = time
                    elif (time - self.t_an) >= 300:
                        action["u_Chiller1_on_off"] = 1.0
                        action["u_Chiller2_on_off"] = 1.0
                        action["u_Pumpe_Chiller1"] = 0.9
                        action["u_Pumpe_Chiller2"] = 0.9
                        self.t_an=0.0
            else:
                    self.t_an=0.0
                    action["u_Chiller1_on_off"] = 0.0
                    action["u_Chiller2_on_off"] = 0.0
                    action["u_Pumpe_Chiller1"] = 0.0
                    action["u_Pumpe_Chiller2"] = 0.0
        else:
            if T_nach_Tower<=293.15:
                if self.t_aus == 0.0:
                    self.t_aus = time
                elif  (time - self.t_aus) >= 600:
                    action["u_Chiller1_on_off"] = 0.0
                    action["u_Chiller2_on_off"] = 0.0
                    action["u_Pumpe_Chiller1"] = 0.0
                    action["u_Pumpe_Chiller2"] = 0.0
                    self.t_aus=0.0
            else:
                self.t_aus=0.0
                action["u_Chiller1_on_off"] = 1.0
                action["u_Chiller2_on_off"] = 1.0
                action["u_Pumpe_Chiller1"] = 0.9
                action["u_Pumpe_Chiller2"] = 0.9


        
        actions.append(list(action.values()))
        actions = actions[0]

        return np.array(actions)
