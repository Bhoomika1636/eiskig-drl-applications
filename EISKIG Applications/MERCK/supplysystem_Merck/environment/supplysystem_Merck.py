from __future__ import annotations

import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import numpy as np
import pandas as pd
from eta_utility import get_logger, timeseries
from eta_utility.eta_x import ConfigOptRun
from eta_utility.eta_x.envs import BaseEnvSim, StateConfig, StateVar

try:
    from plotter.plotter import ETA_Plotter, Heatmap, Linegraph

    plotter_available = True
except ImportError as e:
    plotter_available = False


from eta_utility.type_hints import StepResult, TimeStep
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy.special import expit

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import plotly.express as px

log = get_logger("eta_x.envs")


class SupplysystemMerck(BaseEnvSim):
    """
    SupplysystemB environment class from BaseEnvSim.

    :param env_id: Identification for the environment, useful when creating multiple environments
    :param config_run: Configuration of the optimization run
    :param seed: Random seed to use for generating random numbers in this environment
        (default: None / create random seed)
    :param verbose: Verbosity to use for logging (default: 2)
    :param callback: Callback which should be called after each episode
    :param sampling_time: Length of a timestep in seconds
    :param episode_duration: Duration of one episode in seconds
    """

    # set info
    version = "v0.17"
    description = "(c) Heiko Ranzau, Niklas Panten and Benedikt Grosch"
    fmu_name = "supplysystem_Merck"

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,  # config_run contains run_name, general_settings, path_settings, env_info
        seed: int | None = None,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        sampling_time: float,  # from settings section of config file
        episode_duration: TimeStep | str,  # from settings section of config file
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        scenario_files: Sequence[Mapping[str, Any]],
        random_sampling,
        random_params,
        SOC_min,
        SOC_max,
        variant,
        discretize_action_space,
        reward_shaping,

        use_conventional: bool, # used to switch between agent inputs and conventional strategy
        switch_cost_coolingTower,
        switch_cost_chiller,
        switch_cost_valve_bypass,
        switch_cost_pump_chiller,
        switch_cost_iceStorage,

        temperature_goal_cooling,
        temperature_cost_cooling_min,
        temperature_cost_cooling_max,
        temperature_cost_cooling,

        temperature_goal_cooled,
        temperature_cost_cooled_min,
        temperature_cost_cooled_max,
        temperature_cost_cooled,

        abort_costs,
        policyshaping_costs,

   
        


        **kwargs: Any,
    ):
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            #verbose=verbose,
            callback=callback,
            sampling_time=sampling_time,
            episode_duration=episode_duration,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            **kwargs,
        )

        # make variables readable class-wide
        self.random_sampling = random_sampling
        self.random_params = random_params
        self.SOC_min = SOC_min
        self.SOC_max = SOC_max
        self.discretize_action_space = discretize_action_space
        self.reward_shaping = reward_shaping

        self.use_conventional = use_conventional

        self.switch_cost_coolingTower = switch_cost_coolingTower
        self.switch_cost_chiller = switch_cost_chiller
        self.switch_cost_valve_bypass = switch_cost_valve_bypass
        self.switch_cost_pump_chiller = switch_cost_pump_chiller
        self.switch_cost_iceStorage = switch_cost_iceStorage

        self.temperature_goal_cooling = temperature_goal_cooling
        self.temperature_cost_cooling_min = temperature_cost_cooling_min
        self.temperature_cost_cooling_max = temperature_cost_cooling_max
        self.temperature_cost_cooling = temperature_cost_cooling

        self.temperature_goal_cooled = temperature_goal_cooled
        self.temperature_cost_cooled_min = temperature_cost_cooled_min
        self.temperature_cost_cooled_max = temperature_cost_cooled_max
        self.temperature_cost_cooled = temperature_cost_cooled

        self.abort_costs = abort_costs
        self.policyshaping_costs = policyshaping_costs
        

        # check for different possible state-sizes
        extended_state = False if ("reduced_state" in variant) else True
        mpc_state = True if ("mpc_state" in variant) else False
        self.extended_predictions = False
        if "extended_predictions" in variant:
            extended_state = True
            self.extended_predictions = True

        # initialize episode statistics
        self.episode_statistics = {}
        self.episode_archive = []
        self.episode_df = pd.DataFrame()

        # set integer prediction steps (15m,1h,6h)
        self.n_steps_15m = int(900 // self.sampling_time)
        self.n_steps_1h = int(3600 // self.sampling_time)
        self.n_steps_6h = int(21600 // self.sampling_time)

        # initialize integrators and longtime stats
        self.P_el_total_15min_buffer = []
        self.P_gs_total_15min_buffer = []
        self.n_steps_longtime = 0
        self.reward_longtime_average = 0

        self.initial_resets = 0

        # define state variables
        state_var_tuple = (
            StateVar(
                name="u_AKT01",
                ext_id="u.u_AKT01", # directly controlled through agent
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=0.9,
            ),
            StateVar(
                name="u_AKT02",
                ext_id="u.u_AKT02", # directly controlled through agent
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=0.9,
            ),
            StateVar(
                name="u_AKT03",
                ext_id="u.u_AKT03", # directly controlled through agent
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=0.9,
            ),
            StateVar(
                name="u_AKT04",
                ext_id="u.u_AKT04", # directly controlled through agent
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=0.9,
            ),
            StateVar(
                name="u_valve_bypass",
                ext_id="u.u_valve_bypass_AKT34", # directly controlled through agent
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_PKAB",
                ext_id="u.u_PKAB", # standardized for agent (0=off, 1=6°C, 2=5.5°C, 3=5°C)
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=3.0,
            ),
            StateVar(
                name="u_PKW1",
                ext_id="u.u_PKW1", # standardized for agent (0=off, 1=6°C, 2=5°C, 3=4°C)
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=3.0,
            ),
            StateVar(
                name="u_PKW2",
                ext_id="u.u_PKW2", # standardized for agent (0=off, 1=6°C, 2=5°C, 3=4°C)
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=3.0,
            ),
            StateVar(
                name="u_PKW3",
                ext_id="u.u_PKW3", # standardized for agent (0=off, 1=6°C, 2=5°C, 3=4°C)
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=3.0,
            ),
            StateVar(
                name="u_pump_PKAB",
                ext_id="u.u_pump_PKAB", # directly controlled through agent
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_pump_PKW1",
                ext_id="u.u_pump_PKW1", # directly controlled through agent
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_pump_PKW2",
                ext_id="u.u_pump_PKW2", # directly controlled through agent
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_pump_PKW3",
                ext_id="u.u_pump_PKW3", # directly controlled through agent
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_iceStorage",
                ext_id="u.u_iceStorage", # directly controlled through agent (same as conventional, 0=dormant, 1=charge, -1=discharge)
                is_ext_input=True,
                is_agent_action=True,
                low_value=-1.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_heatEx_PKWE",
                ext_id="u.u_heatEx_PKWE", # standardized for agent (0=off, 1=6°C, 2=5°C, 3=4°C)
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=3.0,
            ),

            
            # disturbances
            
            StateVar(
                name="d_air_temperature",
                ext_id="d.d_weaBus.TDryBul",
                scenario_id="air_temperature",  # [°C]
                from_scenario=True,
                is_agent_observation=True,
                is_ext_input=True,
                low_value=-20,
                high_value=45,
            ),
            StateVar(
                name="d_relative_air_humidity",
                ext_id="d.d_weaBus.relHum",
                scenario_id="relative_air_humidity",  # [%]
                from_scenario=True,
                is_agent_observation=True,
                is_ext_input=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="d_air_pressure",
                ext_id="d.d_weaBus.pAtm",
                scenario_id="air_pressure",  # [mbar]
                from_scenario=True,
                is_agent_observation=True,
                is_ext_input=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="d_weather_time",
                ext_id="u_weather_Time",
                is_ext_input=False,
                low_value=0,
                high_value=31968000,
            ),
            StateVar(
                name="d_Q_load",
                ext_id="d.d_Q_load",
                scenario_id="power_cold",
                from_scenario=True,
                is_ext_input=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=6500000.0,
            ),
            

            # states
            # observe the states of the actions (with internal condition check e.g. cooling tower is not on if disturbance)
            StateVar(
                name="s_u_AKT01",
                ext_id="s.s_AKT01.s_u", # same as conventional
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_AKT02",
                ext_id="s.s_AKT02.s_u", # same as conventional
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_AKT03",
                ext_id="s.s_AKT03.s_u", # same as conventional
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_AKT04",
                ext_id="s.s_AKT04.s_u", # same as conventional
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_valve_bypass",
                ext_id="s.s_u_valve_bypass_AKT34", # same as conventional
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_PKAB",
                ext_id="s.s_u_PKAB", # only for agent observation, standardized for agent (0=off, 1=6°C, 2=5.5°C, 3=5°C)
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=3.0,
            ),
            StateVar(
                name="s_u_PKW1",
                ext_id="s.s_u_PKW1", # only for agent observation, standardized for agent (0=off, 1=6°C, 2=5°C, 3=4°C)
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=3.0,
            ),
            StateVar(
                name="s_u_PKW2",
                ext_id="s.s_u_PKW2", # only for agent observation, standardized for agent (0=off, 1=6°C, 2=5°C, 3=4°C)
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=3.0,
            ),
            StateVar(
                name="s_u_PKW3",
                ext_id="s.s_u_PKW3", # only for agent observation, standardized for agent (0=off, 1=6°C, 2=5°C, 3=4°C)
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=3.0,
            ),
            StateVar(
                name="s_u_pump_PKAB",
                ext_id="s.s_PKAB.s_pump.s_u", # same as conventional
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_pump_PKW1",
                ext_id="s.s_PKW1.s_pump.s_u", # same as conventional
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_pump_PKW2",
                ext_id="s.s_PKW2.s_pump.s_u", # same as conventional
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_pump_PKW3",
                ext_id="s.s_PKW3.s_pump.s_u", # same as conventional
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_iceStorage",
                ext_id="s.s_iceSubsystem.s_u_mode", # same as conventional, 0=dormant, 1=charge, -1=discharge
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-1.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_u_heatEx_PKWE",
                ext_id="s.s_u_heatEx_PKWE", # only for agent observation, standardized for agent (0=off, 1=6°C, 2=5°C, 3=4°C)
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=3.0,
            ),

            # additional states for switching costs
            # needed because standardized agent states are not available in conventional strategy
            StateVar(
                name="s_PKAB",
                ext_id="s.s_PKAB.s_u",
                is_ext_output=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="s_PKW1",
                ext_id="s.s_PKW1.s_u",
                is_ext_output=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="s_PKW2",
                ext_id="s.s_PKW2.s_u",
                is_ext_output=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="s_PKW3",
                ext_id="s.s_PKW3.s_u",
                is_ext_output=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="s_heatEx_PKWE",
                ext_id="s.s_iceSubsystem.s_heatEx_PKWE", # 0 = off, 1 = on
                is_ext_output=True,
                low_value=0,
                high_value=1,
            ),

            # manual input
            StateVar(
                name="use_conv",
                ext_id="u.use_conv", # manual input
                is_ext_input=True,
                low_value=0.0,
                high_value=1.0,
            ),


            # from scenario
            StateVar(
                name="s_price_electricity",
                scenario_id="electrical_energy_price",
                from_scenario=True,
                is_agent_observation=True,
                low_value=-10,
                high_value=10,
            ),  # to obtain €/kWh
            
            # other states
            StateVar(
                name="s_P_el",
                ext_id="s.s_P_el",
                is_ext_output=True,
                #is_agent_observation=True,
                low_value=-100000,
                high_value=500000,
            ),
            StateVar(
                name="s_P_th_cooling",
                ext_id="s.s_P_th_cooling",
                is_ext_output=True,
                #is_agent_observation=True,
                low_value=-100000,
                high_value=500000,
            ),
            StateVar(
                name="s_T_cooling_flow",
                ext_id="s.s_pumpGroup_cooling.s_T_flow",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_P_cooling_return",
                ext_id="s.s_P_cooling_return",
                is_ext_output=True,
                #is_agent_observation=True,
                low_value=10000, # 0.1 bar
                high_value=1000000, # 10 bar
            ),
            StateVar(
                name="s_T_cooled_flow",
                ext_id="s.s_pumpGroup_cooled.s_T_flow",
                is_ext_output=True,
                is_agent_observation=True,
                #abort_condition_min=273.15 + 0 if self.use_conventional else 273.15 + 3,
                #abort_condition_max=273.15 + 100 if self.use_conventional else 273.15 + 10,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_AKT01_T_flow",
                ext_id="s.s_AKT01.s_T_flow",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_AKT02_T_flow",
                ext_id="s.s_AKT02.s_T_flow",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_AKT03_T_flow",
                ext_id="s.s_AKT03.s_T_flow",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_AKT04_T_flow",
                ext_id="s.s_AKT04.s_T_flow",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_AKT01_VFlow",
                ext_id="s.s_AKT01.VFlow",
                is_ext_output=True,
                #is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),
            StateVar(
                name="s_AKT02_VFlow",
                ext_id="s.s_AKT02.VFlow",
                is_ext_output=True,
                #is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),
            StateVar(
                name="s_AKT03_VFlow",
                ext_id="s.s_AKT03.VFlow",
                is_ext_output=True,
                #is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),
            StateVar(
                name="s_AKT04_VFlow",
                ext_id="s.s_AKT04.VFlow",
                is_ext_output=True,
                #is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),
            StateVar(
                name="s_PKAB_T_cooled_flow",
                ext_id="s.s_PKAB.s_T_cooled_flow",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_PKW1_T_cooled_flow",
                ext_id="s.s_PKW1.s_T_cooled_flow",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_PKW2_T_cooled_flow",
                ext_id="s.s_PKW2.s_T_cooled_flow",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_PKW3_T_cooled_flow",
                ext_id="s.s_PKW3.s_T_cooled_flow",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_PKAB_VFlow_cooled",
                ext_id="s.s_PKAB.VFlow_cooled",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),
            StateVar(
                name="s_PKW1_VFlow_cooled",
                ext_id="s.s_PKW1.VFlow_cooled",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),
            StateVar(
                name="s_PKW2_VFlow_cooled",
                ext_id="s.s_PKW2.VFlow_cooled",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),
            StateVar(
                name="s_PKW3_VFlow_cooled",
                ext_id="s.s_PKW3.VFlow_cooled",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),

            StateVar(
                name="s_SOC",
                ext_id="s.s_iceSubsystem.s_SOC",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="s_PKWE",
                ext_id="s.s_iceSubsystem.s_PKWE.s_u",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="s_discharge_T_cooled_flow",
                ext_id="s.s_iceSubsystem.s_T_cooled_flow_discharge",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_PKWE_T_cooled_flow",
                ext_id="s.s_iceSubsystem.s_T_cooled_flow_PKWE",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_discharge_Vflow_cooled",
                ext_id="s.s_iceSubsystem.VFlow_cooled_discharge",
                is_ext_output=True,
                #is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),
            StateVar(
                name="s_PKWE_VFlow_cooled",
                ext_id="s.s_iceSubsystem.VFlow_cooled_heatEx_PKWE",
                is_ext_output=True,
                #is_agent_observation=True,
                low_value=-1500,
                high_value=1500,
            ),

            # states for plots only

        )


        # build final state_config
        self.state_config = StateConfig(*state_var_tuple)

        # import all scenario files
        self.scenario_data = self.import_scenario(*scenario_files).fillna(method="ffill")

        # get action_space
        # TODO: implement this functionality into utility functions
        if self.discretize_action_space:
            # get number of actions agent has to give from state_config
            self.n_action_space = len(self.state_config.actions)
            # set 3 discrete actions (increase,decrease,equal) per control variable
            self.action_space = spaces.MultiDiscrete(np.full(self.n_action_space, 3))
            # customized for chp, condensingboiler, immersionheater, heatpump, coolingtower, compressionchiller
            self.action_disc_step = [
                [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], #AKT01 from 0 to 1: 3*180 sec = 9 minutes
                [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], #AKT02
                [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], #AKT03
                [0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9], #AKT04
                [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60],
                [0.0, 1.0, 2.0, 3.0], #PKAB from 0 to one 9 minutes
                [0.0, 1.0, 2.0, 3.0], #PKW1
                [0.0, 1.0, 2.0, 3.0], #PKW2
                [0.0, 1.0, 2.0, 3.0], #PKW3
                [0.0, 0.68, 0.77, 0.86, 0.95], #pump_PKAB
                [0.0, 0.59, 0.75, 0.9], #pump_PKW1
                [0.0, 0.59, 0.75, 0.9], #pump_PKW2
                [0.0, 0.7, 0.9], #pump_PKW3
                [-1.0, 0.0, 1.0], #iceStorage
                [0.0, 1.0, 2.0, 3.0] #heatEx_PKWE
            ]
            # initialize action
            self.action_disc_index = [0] * self.n_action_space
        else:
            self.action_space = self.state_config.continuous_action_space()

        # get observation_space (always continuous)
        self.observation_space = self.state_config.continuous_obs_space()

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.
        """

        # initialize additional_state and create state backup
        self.state_backup = self.state.copy()
        self.additional_state = {}

        # convert discrete actions into continious space if discrete action space is chosen
        if self.discretize_action_space:
            _action = self.convert_disc_action(action)
        else:
            _action = action
        
        # overwrite actions if out of boundaries (policy shaping), values are explicitly written for logging purposes
        T_cooling_flow = self.state["s_T_cooling_flow"] -273.15
        T_cooled_flow = self.state["s_T_cooled_flow"] -273.15
        
        
        self.policy_shaping_active = False
        self.state["fallback"] = 0.0

        if self.use_conventional:
            self.state["use_conv"] = 1.0
        else:
            self.state["use_conv"] = 0.0
            # policyshaping
            # check T_cooling_flow
            if T_cooling_flow > self.temperature_cost_cooling_max:
                _action[0] = 0.9
                _action[1] = 0.9
                _action[2] = 0.9
                _action[3] = 0.9
                _action[4] = 0.05
                #self.state["use_conv"] = 1.0
                self.policy_shaping_active = True
                self.state["fallback"] = 1.0
            elif T_cooling_flow < self.temperature_cost_cooling_min:
                _action[0] = 0.4 # activate one cooling tower (minimum) and set bypass valve to 100 percent
                _action[1] = 0.0
                _action[2] = 0.0
                _action[3] = 0.0
                _action[4] = 0.6
                #self.state["use_conv"] = 1.0
                self.policy_shaping_active = True
                self.state["fallback"] = 1.0

            # check T_cooled_flow
            if T_cooled_flow > self.temperature_cost_cooled_max:
                _action[5] = 1.0 # activate all chillers
                _action[6] = 1.0
                _action[7] = 1.0
                _action[8] = 1.0
                _action[9] = 0.68 #pump_PKAB minimize pump speed for decreased temperature
                _action[10] = 0.59 #pump_PKW1
                _action[11] = 0.59 #pump_PKW2
                _action[12] = 0.7  #pump_PKW3
                _action[13] = 0.0
                _action[14] = 0.0 #heatEx_PKWE off
                #self.state["use_conv"] = 1.0
                self.policy_shaping_active = True
                self.state["fallback"] = 1.0
            elif T_cooled_flow < self.temperature_cost_cooled_min:
                _action[5] = 1.0 # deactivate all chillers
                _action[6] = 1.0
                _action[7] = 0.0
                _action[8] = 0.0
                _action[9] = 0.68 #pump_PKAB deactivate all chiller pumps
                _action[10] = 0.59 #pump_PKW1
                _action[11] = 0.0 #pump_PKW2
                _action[12] = 0.0 #pump_PKW3
                _action[13] = 0.0
                _action[14] = 0.0 #heatEx_PKWE off
                #self.state["use_conv"] = 1.0
                self.policy_shaping_active = True
                self.state["fallback"] = 1.0

            # check activation of needed pumps
            if _action[5] > 0.0 and _action[9] <= 0.0: # if PKAB active and pump_PKAB not then activate pump_PKAB
                _action[9] = 0.68
                self.policy_shaping_active = True
            if _action[6] > 0.0 and _action[10] <= 0.0: # if PKW1 active and pump_PKW1 not then activate pump_PKW1
                _action[10] = 0.59
                self.policy_shaping_active = True
            if _action[7] > 0.0 and _action[11] <= 0.0: # if PKW2 active and pump_PKW2 not then activate pump_PKW2
                _action[11] = 0.59
                self.policy_shaping_active = True
            if _action[8] > 0.0 and _action[12] <= 0.0: # if PKW3 active and pump_PKW3 not then activate pump_PKW3
                _action[12] = 0.7
                self.policy_shaping_active = True

            # deactivation of pumps when not needed
            if _action[5] <= 0.0 and _action[9] > 0.0: # if PKAB inactive and pump_PKAB then deactivate pump_PKAB
                _action[9] = 0.0
                self.policy_shaping_active = True
            if _action[6] <= 0.0 and _action[10] > 0.0: # if PKW1 inactive and pump_PKW1 then deactivate pump_PKW1
                _action[10] = 0.0
                self.policy_shaping_active = True
            if _action[7] <= 0.0 and _action[11] > 0.0: # if PKW2 inactive and pump_PKW2 then deactivate pump_PKW2
                _action[11] = 0.0
                self.policy_shaping_active = True
            if _action[8] <= 0.0 and _action[12] > 0.0: # if PKW3 inactive and pump_PKW3 then deactivate pump_PKW3
                _action[12] = 0.0
                self.policy_shaping_active = True

            # check that charging iceStorage and usage of heatEx_PKWE ist not done simultanously
            if _action[13] >= 1.0 and _action[14] > 0.0: # deactivate heatEx_PKWE when charging iceStorage
                _action[14] = 0.0
                self.policy_shaping_active = True
        


        # check actions for vilidity, perform simulation step and load new external values for the next time step
        self._actions_valid(_action)
        self.state["step_success"], _ = self._update_state(_action)


        #print("Step success: " + str(self.state["step_success"]))
        print("Episode: ", self.n_episodes, "Progress: " + str(self.n_steps) + " of " + str(self.episode_duration/self.sampling_time), "Estimated remaining time: ", (datetime.now()-self.startTime)/self.n_steps*(self.episode_duration/self.sampling_time-self.n_steps))
        

        # check if state is in valid boundaries
        #self.state["step_abort"] = False if StateConfig.within_abort_conditions(self.state_config, self.state) else True
        # check if state is in valid boundaries
        try:
            self.state["step_abort"] = (
                False if StateConfig.within_abort_conditions(self.state_config, self.state) else True
            )
        except:
            self.state["step_abort"] = False

        # update predictions and virtual state for next time step
        self.state.update(self.update_predictions())
        self.state.update(self.update_virtual_state())
        # add time of the year in seconds
        starttime_of_year = pd.Timestamp(str(self.ts_current.index[self.n_steps].year) + "-01-01 00:00")
        self.state["d_weather_time"] = pd.Timedelta(
            self.ts_current.index[self.n_steps] - starttime_of_year
        ).total_seconds()

        # check if episode is over or not
        done = self._done() or not self.state["step_success"]
        done = done if not self.state["step_abort"] else True

        
        # calculate reward
        if self.state["step_success"]:
            # only if step successfull reward can be calculated normal
            reward = self.calc_reward()
            observations = self._observations()
        else:
            # otherwise we just give abort cost reward
            # since the step was not successfull, the observations will just be zeros.
            # reward = (-self.abort_costs * (1 - 0.5 * self.n_steps / self.n_episode_steps))
            self.state["reward_switching"] = 0
            self.state["reward_temperature_cooling"] = 0
            self.state["reward_temperature_cooled"] = 0
            self.state["reward_energy_electric"] = 0
            #self.state["reward_abort"] = 0
            self.state["reward_policyshaping"] = 0

            self.state["reward_total"] = (
                self.state["reward_switching"]
                + self.state["reward_temperature_cooling"]
                + self.state["reward_temperature_cooled"]
                + self.state["reward_energy_electric"]
                #+ self.state["reward_abort"]
                + self.state["reward_policyshaping"]
            )

            reward = self.state["reward_total"]

            self.state["reward_energy_electric"] = 0
            self.state["energy_electric_consumed"] = 0

            

            observations = np.zeros(len(self.state_config.observations))

        # update state_log
        self.state_log.append(self.state)

        return observations, reward, done, False, {}

    def update_predictions(self):

        prediction_dict = {}

        return prediction_dict

    def update_virtual_state(self):

        virtual_state = {}

        return virtual_state

    def calc_reward(self):
        """Calculates the step reward. Needs to be called from step() method after state update.

        :return: Normalized or non-normalized reward
        :rtype: Real
        """

        # switching costs helper function
        def switch_cost(u_old, u_new, penalty):
            if (u_old <= 0 and u_new > 0) or (u_old > 0 and u_new <= 0):  # if u_old != u_new :
                return penalty  # else 0.1*penalty*abs(u_new - u_old)
            else:
                return 0
        def switch_cost_every_step(u_old, u_new, penalty):
            if (u_old != u_new):  # if u_old != u_new :
                return penalty  # else 0.1*penalty*abs(u_new - u_old)
            else:
                return 0
        

        # switching costs
        self.state["reward_switching"] = (
            -switch_cost(
                self.state_backup["s_u_AKT01"], # it is changed every step in conventional strategy
                self.state["s_u_AKT01"],
                self.switch_cost_coolingTower,
            )
            -switch_cost(
                self.state_backup["s_u_AKT02"],
                self.state["s_u_AKT02"],
                self.switch_cost_coolingTower,
            )
            -switch_cost(
                self.state_backup["s_u_AKT03"],
                self.state["s_u_AKT03"],
                self.switch_cost_coolingTower,
            )
            -switch_cost(
                self.state_backup["s_u_AKT04"],
                self.state["s_u_AKT04"],
                self.switch_cost_coolingTower,
            )
            - switch_cost_every_step(
                self.state_backup["s_u_valve_bypass"], 
                self.state["s_u_valve_bypass"],
                self.switch_cost_valve_bypass,
            )
            - switch_cost(
                self.state_backup["s_PKAB"], # conventional strategy has static flow temperature = 5.5 always on -> zero switching costs
                self.state["s_PKAB"],
                self.switch_cost_chiller,
            )
            - switch_cost(
                self.state_backup["s_PKW1"], 
                self.state["s_PKW1"],
                self.switch_cost_chiller,
            )
            - switch_cost(
                self.state_backup["s_PKW2"], 
                self.state["s_PKW2"],
                self.switch_cost_chiller,
            )
            - switch_cost(
                self.state_backup["s_PKW3"], 
                self.state["s_PKW3"],
                self.switch_cost_chiller,
            )
            - switch_cost(
                self.state_backup["s_u_pump_PKAB"],
                self.state["s_u_pump_PKAB"],
                self.switch_cost_pump_chiller,
            )
            - switch_cost(
                self.state_backup["s_u_pump_PKW1"],
                self.state["s_u_pump_PKW1"],
                self.switch_cost_pump_chiller,
            )
            - switch_cost(
                self.state_backup["s_u_pump_PKW2"],
                self.state["s_u_pump_PKW2"],
                self.switch_cost_pump_chiller,
            )
            - switch_cost(
                self.state_backup["s_u_pump_PKW3"],
                self.state["s_u_pump_PKW3"],
                self.switch_cost_pump_chiller,
            )
            - switch_cost_every_step(
                self.state_backup["s_u_iceStorage"],
                self.state["s_u_iceStorage"],
                self.switch_cost_iceStorage,
            )
            - switch_cost(
                self.state_backup["s_heatEx_PKWE"],
                self.state["s_heatEx_PKWE"],
                self.switch_cost_chiller,
            )
        )

        self.state["switch_count"] = (
            -switch_cost(
                self.state_backup["s_u_AKT01"], # it is changed every step in conventional strategy
                self.state["s_u_AKT01"],
                1,
            )
            -switch_cost(
                self.state_backup["s_u_AKT02"],
                self.state["s_u_AKT02"],
                1,
            )
            -switch_cost(
                self.state_backup["s_u_AKT03"],
                self.state["s_u_AKT03"],
                1,
            )
            -switch_cost(
                self.state_backup["s_u_AKT04"],
                self.state["s_u_AKT04"],
                1,
            )
            - switch_cost_every_step(
                self.state_backup["s_u_valve_bypass"], 
                self.state["s_u_valve_bypass"],
                1,
            )
            - switch_cost(
                self.state_backup["s_PKAB"], # conventional strategy has static flow temperature = 5.5 always on -> zero switching costs
                self.state["s_PKAB"],
                1,
            )
            - switch_cost(
                self.state_backup["s_PKW1"], 
                self.state["s_PKW1"],
                1,
            )
            - switch_cost(
                self.state_backup["s_PKW2"], 
                self.state["s_PKW2"],
                1,
            )
            - switch_cost(
                self.state_backup["s_PKW3"], 
                self.state["s_PKW3"],
                1,
            )
            - switch_cost(
                self.state_backup["s_u_pump_PKAB"],
                self.state["s_u_pump_PKAB"],
                1,
            )
            - switch_cost(
                self.state_backup["s_u_pump_PKW1"],
                self.state["s_u_pump_PKW1"],
                1,
            )
            - switch_cost(
                self.state_backup["s_u_pump_PKW2"],
                self.state["s_u_pump_PKW2"],
                1,
            )
            - switch_cost(
                self.state_backup["s_u_pump_PKW3"],
                self.state["s_u_pump_PKW3"],
                1,
            )
            - switch_cost_every_step(
                self.state_backup["s_u_iceStorage"],
                self.state["s_u_iceStorage"],
                1,
            )
            - switch_cost(
                self.state_backup["s_heatEx_PKWE"],
                self.state["s_heatEx_PKWE"],
                1,
            )
        )

        # temperature costs (when availability of temperature levels are needed)


        # self.state["reward_temperature_cooling"] = reward_temp(
        #     self.state["s_T_cooling_flow"],
        #     self.temperature_goal_cooling+273.15,
        #     self.temperature_cost_cooling_min+273.15,
        #     self.temperature_cost_cooling_max+273.15,
        #     self.temperature_cost_cooling,
        #     k=3,
        # )
        
        # self.state["reward_temperature_cooled"] = reward_temp(
        #     self.state["s_T_cooled_flow"],
        #     self.temperature_goal_cooled+273.15,
        #     self.temperature_cost_cooled_min+273.15,
        #     self.temperature_cost_cooled_max+273.15,
        #     self.temperature_cost_cooled,
        #     k=3,
        # )
        boundary_distance = 1 # in degC (distance from fallback boundary)
        self.state["reward_temperature_cooling"] = reward_boundary(
            self.state["s_T_cooling_flow"],
            self.temperature_cost_cooling_min+273.15 +boundary_distance,
            self.temperature_cost_cooling_max+273.15 -boundary_distance,
            0,
            self.temperature_cost_cooling,
            smoothed=self.reward_shaping,
            k=6,
        )
        self.state["reward_temperature_cooled"] = reward_boundary(
            self.state["s_T_cooled_flow"],
            5 +273.15,
            7 +273.15,
            0,
            self.temperature_cost_cooled,
            smoothed=self.reward_shaping,
            k=6,
        )

        
       
        
       

        # energy costs
        
        self.state["reward_energy_electric"] = (
            -self.state["s_price_electricity"] * self.state["s_P_el"] * self.sampling_time / 3600 / 1000
        )
        

        # energy consumed
        self.state["energy_electric_consumed"] = self.state["s_P_el"] * self.sampling_time / 3600 / 1000

        # policyshaping costs
        self.state["reward_policyshaping"] = 0 # originally abort and policyshaping costs were summed
        if self.policy_shaping_active:
            self.state["reward_policyshaping"] -= self.policyshaping_costs # self.state["reward_other"] -= self.policyshaping_costs

        # total reward
        self.state["reward_total"] = (
            self.state["reward_switching"]
            + self.state["reward_temperature_cooling"]
            + self.state["reward_temperature_cooled"]
            + self.state["reward_energy_electric"]
            + self.state["reward_policyshaping"]
        )

        return self.state["reward_total"]

    def reset(
        self,
        *,
        seed: int | None = None,
    ):  # -> tuple[ObservationType, dict[str, Any]]:

        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """

        # Hack to work around the current issue of non deterministic seeding of first episode
        if self.initial_resets == 0 and seed is not None:
            self._np_random, _ = seeding.np_random(seed + 100)

        # get current time
        self.startTime = datetime.now()   


        # delete long time storage, since it takes up too much memory during training
        self.state_log_longtime = []

        # # save episode's stats
        if self.n_steps > 0:

            # create dataframe from state_log
            self.episode_df = pd.DataFrame(self.state_log)

            # derive certain episode statistics for logging and plotting
            self.episode_statistics["datetime_begin"] = self.ts_current.index[0].strftime("%Y-%m-%d %H:%M")
            self.episode_statistics["rewards_total"] = self.episode_df["reward_total"].sum()
            self.episode_statistics["rewards_switching"] = self.episode_df["reward_switching"].sum()
            self.episode_statistics["rewards_temperature_cooling"] = self.episode_df["reward_temperature_cooling"].sum()
            self.episode_statistics["rewards_temperature_cooled"] = self.episode_df["reward_temperature_cooled"].sum()
            self.episode_statistics["rewards_energy_electric"] = self.episode_df["reward_energy_electric"].sum()
            self.episode_statistics["rewards_policyshaping"] = self.episode_df["reward_policyshaping"].sum()
            self.episode_statistics["energy_electric_consumed"] = self.episode_df["energy_electric_consumed"].sum()
            self.episode_statistics["time"] = time.time() - self.episode_timer
            self.episode_statistics["n_steps"] = self.n_steps
            self.episode_statistics["s_T_cooling_flow_min"] = self.episode_df["s_T_cooling_flow"].min()
            self.episode_statistics["s_T_cooling_flow_max"] = self.episode_df["s_T_cooling_flow"].max()
            self.episode_statistics["s_T_cooled_flow_min"] = self.episode_df["s_T_cooled_flow"].min()
            self.episode_statistics["s_T_cooled_flow_max"] = self.episode_df["s_T_cooled_flow"].max()
            
            self.episode_statistics["switches_count"] = self.episode_df["switch_count"].sum() # Summe Schaltvorgänge
            self.episode_statistics["mean_switches_per_day"] = self.episode_statistics["switches_count"] / (self.n_steps * self.sampling_time * 3600 * 24) # Schaltvorgänge pro Tag
            #self.episode_statistics["mean_electric_power"] = self.episode_df["s_P_el"].mean() / 1000 # Mittelwert der el. Leistung in kW

            

            # # append episode store to episodes' archive
            self.episode_archive.append(list(self.episode_statistics.values()))

        # initialize additional state, since super().reset is used
        self.additional_state = {}

        # set FMU parameters for initialization
        self.model_parameters = {}

        # choose if conventional strategy or agent inputs are used
        #self.model_parameters["use_conventional"] = self.use_conventional

        # # randomly parametrize systems
        self.model_parameters["iceStorage_SOC_Start"] = self.np_random.uniform(self.SOC_min, self.SOC_max) if self.random_params else 0

        print(self.model_parameters)

        
        self.ts_current = timeseries.df_time_slice(
            self.scenario_data,
            self.scenario_time_begin,
            self.scenario_time_end,
            self.episode_duration + (self.n_steps_6h + 1) * self.sampling_time,
            random=self.np_random if self.random_sampling else False,
        )

        # read current date time
        self.episode_datetime_begin = self.ts_current.index[0]
        #self.additional_state["vs_time_daytime"] = self.episode_datetime_begin.hour

        # # reset virtual states and internal counters
        # self.P_el_total_15min_buffer = []
        # self.P_gs_total_15min_buffer = []
        # self.additional_state["vs_electric_power_total_15min"] = 0
        # self.additional_state["vs_gas_power_total_15min"] = 0

        # get scenario input for initialization (time step: 0)
        # self.additional_state.update(self.update_predictions())

        # add time of the year in seconds
        starttime_of_year = pd.Timestamp(str(self.ts_current.index[self.n_steps].year) + "-01-01 00:00")
        self.additional_state["d_weather_time"] = pd.Timedelta(
            self.ts_current.index[self.n_steps] - starttime_of_year
        ).total_seconds()

        # reset maximal peak electric power for penalty costs (necessary for peak shaving)
        # self.max_limit = self.power_cost_max

        # reset RNG, hack to work around the current issue of non deterministic seeding of first episode
        if self.initial_resets == 0:
            self._np_random = None
        self.initial_resets += 1

        # receive observations from simulation
        observations = super().reset(seed=seed)

        return observations

    def convert_disc_action(self, action_disc):
        """
        converts discrete actions from agent to continious FMU input space
        """
        float_action = []

        for idx, val in enumerate(action_disc):
            self.action_disc_index[idx] = np.clip(
                self.action_disc_index[idx] + (val - 1), 0, len(self.action_disc_step[idx]) - 1
            )
            float_action.append(self.action_disc_step[idx][self.action_disc_index[idx]])

        return np.array(float_action)

    def render_episodes(self):
        """
        output plot for all episodes
        see pandas visualization options on https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
        https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

        Parameters
        -----
        mode : (str)
        """
        # create dataframe
        episode_archive_df = pd.DataFrame(self.episode_archive, columns=list(self.episode_statistics.keys()))

        # write all data to csv after every episode
        episode_archive_df.to_csv(
            path_or_buf=os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(4)
                + "-"
                + str(self.env_id).zfill(2)
                + "_all-episodes.csv",
            ),
            sep=";",
            decimal=".",
        )

        # write another aggregated csv that contains all episodes (necessary for mpc and mpc_simple)
        csvpath = os.path.join(self.path_results, "all-episodes.csv")
        if os.path.exists(
            csvpath
        ):  # check if aggregated file already exists, which is the case when multiple runs are done with mpc and mpc_simple
            tocsvmode = "a"
            tocsvheader = False
        else:
            tocsvmode = "w"
            tocsvheader = True
        # write data to csv
        episode_archive_df.tail(1).to_csv(path_or_buf=csvpath, sep=";", decimal=".", mode=tocsvmode, header=tocsvheader)

        # plot settings
        # create figure and axes with custom layout/adjustments
        figure = plt.figure(figsize=(14, 14), dpi=200)
        axes = []
        axes.append(figure.add_subplot(2, 1, 1))
        axes.append(figure.add_subplot(2, 1, 2, sharex=axes[0]))

        # fig, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=200)
        x = np.arange(len(episode_archive_df.index))
        y = episode_archive_df

        # (1) Costs
        axes[0].plot(
            x, y["rewards_energy_electric"], label="Strom (netto)", color=(1.0, 0.75, 0.0), linewidth=1, alpha=0.9
        )
        axes[0].set_ylabel("kum. Kosten [€]")
        axes[0].set_xlabel("Episode")
        axes[0].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[0].margins(x=0.0, y=0.1)
        axes[0].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)

        # (2) Rewards
        cost_total = (
            y["rewards_energy_electric"]
        )
        axes[1].plot(x, cost_total, label="Kosten", color=(1.0, 0.75, 0.0), linewidth=1, alpha=0.9)
        axes[1].plot(
            x, y["rewards_temperature_cooling"], label="Kühlwasser", color=(0.75, 0, 0), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x, y["rewards_temperature_cooled"], label="Kaltwasser", color=(0.36, 0.61, 0.84), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x, y["rewards_switching"], label="Schaltvorgänge", color=(0.44, 0.19, 0.63), linewidth=1, alpha=0.9
        )
        axes[1].plot(x, y["rewards_policyshaping"], label="Policyshaping", color=(0.52, 0.75, 0.16), linewidth=1, alpha=0.9)
        axes[1].plot(x, y["rewards_total"], label="Gesamt", color=(0.1, 0.1, 0.1), linewidth=2)
        axes[1].set_ylabel("kum. + fikt. Kosten [€]")
        axes[1].set_xlabel("Episode")
        axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[1].margins(x=0.0, y=0.1)
        axes[1].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)

        plt.savefig(
            os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(4)
                + "-"
                + str(self.env_id).zfill(2)
                + "_all-episodes.png",
            )
        )
        plt.close(figure)


        # plotly
        # Beispiel-Daten, ersetze 'episode_archive_df' mit deinem DataFrame
        x = np.arange(len(episode_archive_df.index))
        y = episode_archive_df

        # Erstellen der Figure mit zwei vertikalen Subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.02)

        # (1) Kosten
        fig.add_trace(
            go.Scatter(x=x, y=y["rewards_energy_electric"], mode='lines', name="Strom (netto)", 
                    line=dict(color='rgb(255,191,0)', width=2), opacity=0.9),
            row=1, col=1
        )
        fig.update_yaxes(title_text="kum. Kosten [€]", row=1, col=1)
        fig.update_xaxes(title_text="Episode", row=1, col=1)

        # (2) Rewards
        fig.add_trace(
            go.Scatter(x=x, y=y["rewards_energy_electric"], mode='lines', name="Kosten", 
                    line=dict(color='rgb(255,191,0)', width=2), opacity=0.9),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y["rewards_temperature_cooling"], mode='lines', name="Kühlwasserversorgung", 
                    line=dict(color='rgb(191,0,0)', width=2), opacity=0.9),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y["rewards_temperature_cooled"], mode='lines', name="Kaltwasserversorgung", 
                    line=dict(color='rgb(92,156,215)', width=2), opacity=0.9),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y["rewards_switching"], mode='lines', name="Schaltvorgänge", 
                    line=dict(color='rgb(112,48,160)', width=2), opacity=0.9),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y["rewards_policyshaping"], mode='lines', name="Policyshaping", 
                    line=dict(color='rgb(133,192,41)', width=2), opacity=0.9),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=y["rewards_total"], mode='lines', name="Gesamt", 
                    line=dict(color='rgb(26,26,26)', width=2)),
            row=2, col=1
        )
        # fig.update_yaxes(title_text="kum. + fikt. Kosten [€]", row=2, col=1)
        # fig.update_xaxes(title_text="Episode", row=2, col=1)

        # Layout-Updates für die gesamte Figur
        fig.update_layout(height=800, width=1400, title_text="Visualisierung der Kosten und Rewards", 
                        showlegend=True, legend=dict(x=1, y=0.5, bgcolor='rgba(255,255,255,0.5)', bordercolor='rgba(0,0,0,0.5)'))

        # Zeige die Figur an
        #fig.show()

        fig.write_html(
            os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(4)
                + "-"
                + str(self.env_id).zfill(2)
                + "_all-episodes.html",
            )
        )


    def import_scenario(self, *scenario_paths: Mapping[str, Any], prefix_renamed: bool = True) -> pd.DataFrame:
        paths = []
        prefix = []
        int_methods = []
        scale_factors = []
        rename_cols = {}
        infer_datetime_from = []
        time_conversion_str = []

        for path in scenario_paths:
            paths.append(self.path_scenarios / path["path"])
            prefix.append(path.get("prefix", None))
            int_methods.append(path.get("interpolation_method", None))
            scale_factors.append(path.get("scale_factors", None))
            rename_cols.update(path.get("rename_cols", {})),
            infer_datetime_from.append(path.get("infer_datetime_cols", "string"))
            time_conversion_str.append(path.get("time_conversion_str", "%Y-%m-%d %H:%M"))

        return timeseries.scenario_from_csv(
            paths=paths,
            resample_time=self.sampling_time,
            start_time=self.scenario_time_begin,
            end_time=self.scenario_time_end,
            random=False,
            interpolation_method=int_methods,
            scaling_factors=scale_factors,
            rename_cols=rename_cols,
            prefix_renamed=prefix_renamed,
            infer_datetime_from=infer_datetime_from,
            time_conversion_str=time_conversion_str,
        )

    def render(self, mode="human", name_suffix=""):
        """
        output plots for last episode
        see pandas visualization options on https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
        https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

        Parameters
        -----
        mode : (str)
        """

        # save csv
        self.episode_df.to_csv(
            path_or_buf=os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(3)
                + "-"
                + str(self.env_id).zfill(2)
                + "_episode"
                + name_suffix
                + ".csv",
            ),
            sep=";",
            decimal=".",
        )

        # create figure and axes with custom layout/adjustments
        figure = plt.figure(figsize=(16, 22), dpi=200)
        axes = []
        axes.append(figure.add_subplot(5, 1, 1))
        axes.append(figure.add_subplot(5, 1, 2, sharex=axes[0]))
        axes.append(figure.add_subplot(10, 1, 5, sharex=axes[0]))
        axes.append(figure.add_subplot(10, 1, 6, sharex=axes[0]))
        axes.append(figure.add_subplot(5, 1, 4, sharex=axes[0]))
        axes.append(figure.add_subplot(10, 1, 9, sharex=axes[0]))
        axes.append(figure.add_subplot(10, 1, 10, sharex=axes[0]))
        plt.tight_layout()
        figure.subplots_adjust(left=0.125, bottom=0.05, right=0.9, top=0.95, wspace=0.2, hspace=0.05)

        # set x/y axe and datetime begin
        x = self.episode_df.index
        y = self.episode_df
        dt_begin = self.episode_datetime_begin
        sampling_time = self.sampling_time

        # (1) - Plot actions as heatmap
        axes[0].set_yticks(np.arange(15)) # axes[0].set_yticks(np.arange(len(self.state_config.actions)))
        axes[0].set_yticklabels(
            ["AKT01",
            "AKT02",
            "AKT03",
            "AKT04",
            "Bypass-Ventil",
            "PKAB",
            "Pumpe PKAB",
            "PKW1",
            "Pumpe PKW1",
            "PKW2",
            "Pumpe PKW2",
            "PKW3",
            "Pumpe PKW3",
            "PKWE", # vorher "PKWE Direktbetrieb"
            "Eisspeicher"]
        )
        # Definiere die zwei Colormaps
        cmap_reds = plt.cm.Reds
        cmap_blues_r = plt.cm.Blues_r

        # Kombiniere die Colormaps
        colors1 = cmap_blues_r(np.linspace(0, 1, 128))
        colors2 = cmap_reds(np.linspace(0, 1, 128))
        colors = np.vstack((colors1, colors2))
        my_cmap = LinearSegmentedColormap.from_list('my_colormap', colors)


        im = axes[0].imshow(
            y[["s_u_AKT01",
              "s_u_AKT02",
              "s_u_AKT03",
              "s_u_AKT04",
              "s_u_valve_bypass",
              "s_PKAB",
              "s_u_pump_PKAB",
              "s_PKW1",
              "s_u_pump_PKW1",
              "s_PKW2",
              "s_u_pump_PKW2",
              "s_PKW3",
              "s_u_pump_PKW3",
              "s_PKWE", # vorher "s_heatEx_PKWE" -> jetzt sieht man nur die PKWE-Leistung und kann über den Eisspeicher Modus sehen, ob Direktmodus oder Ladebetrieb
              "s_u_iceStorage"]].transpose(), cmap=my_cmap, vmin=-1, vmax=1, aspect="auto", interpolation="nearest"
        )
        # add colorbar
        ax_pos = axes[0].get_position().get_points().flatten()
        ax_colorbar = figure.add_axes(
            [0.93, ax_pos[1] + 0.05, 0.01, ax_pos[3] - ax_pos[1] - 0.1]
        )  ## the parameters are the specified position you set
        figure.colorbar(im, ax=axes[0], shrink=0.9, cax=ax_colorbar)

        timeRange = np.arange(
            (1 - dt_begin.minute / 60) * 60 * 60 / sampling_time,
            self.episode_duration / sampling_time,
            1 * 60 * 60 / sampling_time,
        )
        dt_begin = dt_begin.replace(microsecond=0, second=0, minute=0)
        ticknames = []
        tickpos = []
        for i in timeRange:
            tickdate = dt_begin + timedelta(seconds=i * sampling_time)
            if tickdate.hour in [6, 12, 18]:
                tickpos.append(i)
                ticknames.append(tickdate.strftime("%H"))
            elif tickdate.hour == 0:
                tickpos.append(i)
                ticknames.append(tickdate.strftime("%d.%m.%y"))
        # Let the horizontal axes labeling appear on top
        axes[0].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, rotation=45)
        axes[0].set_xlabel("Zeit")
        axes[0].xaxis.set_label_position("top")
        # ax.set_xticks(np.arange(df1.shape[1]+1)-.5, minor=True)
        axes[0].set_yticks(np.arange(15 + 1) - 0.5, minor=True)
        axes[0].tick_params(which="minor", bottom=False, left=False)
        # grid settings
        axes[0].grid(which="minor", color="w", linestyle="-", linewidth=3)
        axes[0].xaxis.grid(color=(1, 1, 1, 0.1), linestyle="-", linewidth=1)
        # add ticks and tick labels
        axes[0].set_xticks(tickpos)
        axes[0].set_xticklabels(ticknames)
        # Rotate the tick labels and set their alignment.
        plt.setp(axes[0].get_yticklabels(), rotation=30, ha="right", va="center", rotation_mode="anchor")

        # (2) - Plot Temperatures
        axes[1].plot(
            x,
            [self.temperature_cost_cooling_min] * len(x),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle="--",
            #label="T Kühlwasser minimal",
            label="Temperaturgrenzen",
        )
        axes[1].plot(
            x,
            [self.temperature_cost_cooling_max] * len(x),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle="--",
            #label="T Kühlwasser maximal",
        )

        axes[1].plot(
            x,
            y["s_T_cooling_flow"] - 273.15,
            color=(192 / 255, 0, 0),
            label="Kühlwasser Vorlauf"
        )
        #axes[1].plot(x, y["s_temp_heat_storage_lo"] - 273.15, color=(192 / 255, 0, 0), linestyle="--", label="Wärmespeicher (un)")
        axes[1].plot(
            x,
            [self.temperature_cost_cooled_min] * len(x),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle="--",
            #label="T Kaltwasser minimal",
        )
        axes[1].plot(
            x,
            [self.temperature_cost_cooled_max] * len(x),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle="--",
            #label="T Kaltwasser maximal",
        )
        axes[1].plot(
            x,
            y["s_T_cooled_flow"] - 273.15,
            color=(91 / 255, 155 / 255, 213 / 255),
            #linestyle="--",
            label="Kaltwasser Vorlauf",
        )
        axes[1].step(
            x,
            y["fallback"], #y["use_conv"],
            color=(0.1, 0.1, 0.1),
            #linestyle="--",
            where = "post",
            label="Fallback aktiv {0,1}",
        )
        
        



        # weather
        axes[1].fill_between(
            x, y["d_air_temperature"], color=(0.44, 0.68, 0.28), linewidth=0.1, alpha=0.3, label="Außenluft"
        )
        # settings
        axes[1].set_ylabel("Temperatur [°C]")
        axes[1].margins(x=0.0, y=0.1)
        axes[1].set_axisbelow(True)
        axes[1].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[1].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # (2) - Battery and relative Humidity (%)
        # prodState = y["d_time_till_availability"].copy()
        # prodState[prodState > 0] = 0
        # prodState[prodState < 0] = 10
        # axes[2].fill_between(x, prodState, color=(0.1, 0.1, 0.1), linewidth=0.1, alpha=0.3, label="Prod. Modus")
        axes[2].plot(
            x,
            y["s_SOC"],
            color=(0.1, 0.1, 0.1),
            #linestyle="--",
            label="Eisspeicher SOC",
        )
        axes[2].fill_between(
            x,
            y["d_relative_air_humidity"],
            color=(0.44, 0.68, 0.28),
            linewidth=0.1,
            alpha=0.3,
            label="Außenluftfeuchte",
        )
        

        # settings
        axes[2].set_ylabel("Zustand [%]")
        axes[2].margins(x=0.0, y=0.1)
        axes[2].set_axisbelow(True)
        axes[2].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        axes[2].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[2].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # (3) - Prices
        axes[3].plot(x, y["s_price_electricity"], color=(1.0, 0.75, 0.0), linewidth=1.5, label="Strompreis")
        #axes[3].plot(x, y["s_price_gas"], color=(0.65, 0.65, 0.65), label="Erdgas")
        axes[3].set_ylabel("Energiepreis (netto) [€/kWh]")
        axes[3].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[3].margins(x=0.0, y=0.1)
        axes[3].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        axes[3].set_axisbelow(True)
        axes[3].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)

        # (4) - Plot power      
        axes[4].plot(
            x,
            y["d_Q_load"] * 1e-3,
            color=(0.36, 0.61, 0.84),
            linestyle="--",
            linewidth=1.5,
            #alpha=0.9,
            label="Kältelast Prod.",
        )
        axes[4].plot(
            x, y["s_P_el"] * 1e-3, color=(1.0, 0.75, 0.0), linewidth=1.5, label="Strom Netzleistung"
        )

        axes[4].set_ylabel("Leistung [kW]")
        axes[4].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[4].margins(x=0.0, y=0.1)
        axes[4].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        axes[4].set_axisbelow(True)
        axes[4].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)

        # (5) - Costs
        axes[5].plot(
            x,
            -y["reward_energy_electric"].cumsum(),
            label="Stromkosten",
            color=(1.0, 0.75, 0.0),
            linewidth=1.5,
            #alpha=0.9,
        )
        # axes[5].plot(
        #     x, y["reward_energy_gas"].cumsum(), label="Erdgas (netto)", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9
        # )
        # axes[5].plot(
        #     x,
        #     y["reward_energy_taxes"].cumsum(),
        #     label="Steuern & Umlagen",
        #     color=(0.184, 0.333, 0.592),
        #     linewidth=1,
        #     alpha=0.9,
        # )
        # axes[5].plot(
        #     x,
        #     y["reward_power_electric"].cumsum(),
        #     label="el. Lastspitzen",
        #     color=(0.929, 0.49, 0.192),
        #     linewidth=1,
        #     alpha=0.9,
        # )
        axes[5].set_ylabel("kum. Kosten [€]")
        axes[5].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[5].margins(x=0.0, y=0.1)
        axes[5].set_axisbelow(True)
        axes[5].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        axes[5].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[5].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # (6) Rewards
        cost_total = (
            y["reward_energy_electric"].cumsum()
            # + y["reward_energy_gas"].cumsum()
            # + y["reward_energy_taxes"].cumsum()
            # + y["reward_power_electric"].cumsum()
        )
        axes[6].plot(x, cost_total, label="Stromkosten", color=(1.0, 0.75, 0.0), linewidth=1.5)
        axes[6].plot(
            x,
            y["reward_temperature_cooling"].cumsum(),
            label="Kühlwasser",
            color=(0.75, 0, 0),
            linewidth=1.5,
            #alpha=0.9,
        )
        axes[6].plot(
            x,
            y["reward_temperature_cooled"].cumsum(),
            label="Kaltwasser",
            color=(0.36, 0.61, 0.84),
            linewidth=1.5,
            #alpha=0.9,
        )
        axes[6].plot(
            x, y["reward_switching"].cumsum(), label="Schaltvorgänge", color=(0.44, 0.19, 0.63), linewidth=1.5, alpha=0.9
        )
        #axes[6].plot(x, y["reward_abort"].cumsum(), label="Abbruch", color=(1, 0.2, 1), linewidth=1, alpha=0.9)
        axes[6].plot(x, y["reward_policyshaping"].cumsum(), label="Policyshaping", color=(0.52, 0.75, 0.16), linewidth=1.5)
        axes[6].plot(x, y["reward_total"].cumsum(), label="Gesamt", color=(0.1, 0.1, 0.1), linewidth=2)
        axes[6].set_ylabel("kum. Belohnung [€-äquiv.]")
        axes[6].set_xlabel("Zeit")
        axes[6].set_axisbelow(True)
        axes[6].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[6].margins(x=0.0, y=0.1)
        axes[6].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        # add ticks and tick labels
        axes[6].set_xticks(tickpos)
        axes[6].set_xticklabels(ticknames, rotation=45)

        # save and close figure
        plt.savefig(
            os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(3)
                + "-"
                + str(self.env_id).zfill(2)
                + "_episode"
                + name_suffix
                + ".png",
            )
        )
        plt.savefig(
            os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(3)
                + "-"
                + str(self.env_id).zfill(2)
                + "_episode"
                + name_suffix
                + ".pdf",
            ),
            backend="pgf"
        )
        plt.close(figure)

        # plotly
        date_format = "%d.%m.%Y %H:%M:%S"
        temperature_cost_cooling_min = 15
        temperature_cost_cooling_max = 27.5
        temperature_cost_cooled_min = 5
        temperature_cost_cooled_max = 7.5
        
        

        
        episode_df = self.episode_df.shift(1).dropna()
        episode_df.reset_index(drop=True, inplace=True)
        x = pd.to_datetime(episode_df.d_weather_time, unit='s', origin=datetime(year=2017, month=1, day=1, hour=0, minute=0, second=0))
        datetime_start = x[0]
        datetime_end = x.tail(1).iloc[0]
            

        actions = episode_df[[
            "s_u_AKT01", "s_u_AKT02", "s_u_AKT03", "s_u_AKT04",
            "s_u_valve_bypass", "s_PKAB", "s_u_pump_PKAB", "s_PKW1",
            "s_u_pump_PKW1", "s_PKW2", "s_u_pump_PKW2", "s_PKW3",
            "s_u_pump_PKW3", "s_PKWE", "s_u_iceStorage"
        ]].transpose()

        actions = actions.iloc[::-1]  # Kehrt die Reihenfolge der Zeilen um
        # Aktionen Labels
        action_labels = [
            "Eisspeicher", "PKWE", "Pumpe PKW3", "PKW3", "Pumpe PKW2", "PKW2", "Pumpe PKW1", "PKW1",
            "Pumpe PKAB", "PKAB", "Bypass-Ventil", "AKT04", "AKT03", "AKT02", "AKT01"
        ]

        fig = make_subplots(
            rows=7, cols=1, shared_xaxes=True,
            vertical_spacing=0.02,
            subplot_titles=("Aktionen", "Temperaturen [°C]", "Eisspeicher and Relative Luftfeuchte [%]", "Strompreis (netto) [€/kWh]", "Leistung [kW]", "kum. Kosten [€]", "Rewards [€-äquiv.]")
        )

        # (1) - Action Heatmap
        # Benutzerdefinierte Farbskala aus vorhandenen Farbskalen erstellen
        blues = px.colors.sequential.Blues_r[:-1]  # Blau zu Weiß
        reds = px.colors.sequential.Reds[1:]  # Weiß zu Rot
        custom_colorscale = blues + reds

        fig.add_trace(
            go.Heatmap(
                z=actions.values,
                zmin=-1,
                zmax=1,
                x = x,  # or appropriate timestamp if needed
                y=action_labels,
                colorscale=custom_colorscale,
                colorbar=dict(len=0.12, y=0.94),
                showscale=True,
                ygap=1    # Abstand zwischen den Zellen in y-Richtung
            ),
            row=1, col=1
        )

        # (2) - Temperature Plot
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['d_air_temperature'], name='Außenlufttemperatur',line=dict(color='lightgreen'), fill='tozeroy', legend='legend1'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['s_T_cooling_flow'] - 273.15, name='Kühlwasser Vorlauf', legend='legend1'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['s_T_cooled_flow'] - 273.15, name='Kaltwasser Vorlauf', line=dict(color='blue'), legend='legend1'),
            row=2, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['fallback'], name='Fallback aktiv {0,1}', legend='legend1', line=dict(color='Black'),),
            row=2, col=1
        )
        

        fig.add_trace(
            go.Scatter(
                x=x, 
                y=[temperature_cost_cooling_min] * len(episode_df.index),
                name="Minimale Kühlwassertemperatur",
                mode='lines',
                line=dict(color='grey', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=x, 
                y=[temperature_cost_cooling_max] * len(episode_df.index),
                name="Maximale Kühlwassertemperatur",
                mode='lines',
                line=dict(color='grey', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=x, 
                y=[temperature_cost_cooled_min] * len(episode_df.index),
                name="Minimale Kaltwassertemperatur",
                mode='lines',
                line=dict(color='grey', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        fig.add_trace(
            go.Scatter(
                x=x, 
                y=[temperature_cost_cooled_max] * len(episode_df.index),
                name="Maximale Kaltwassertemperatur",
                mode='lines',
                line=dict(color='grey', dash='dash'),
                showlegend=False
            ),
            row=2, col=1
        )

        # (3) - Battery and Humidity
        
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['d_relative_air_humidity'], name='Außenluftfeuchte',line=dict(color='lightgreen'), fill='tozeroy', legend='legend2'),
            row=3, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['s_SOC'], name='Eisspeicher SOC', legend='legend2', line=dict(color='Black'),),
            row=3, col=1
        )
        

        # (4) - Energy Prices
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['s_price_electricity'], name='Strompreis', legend='legend3', line=dict(color='Orange')),
            row=4, col=1
        )

        # (5) - Power
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['s_P_th_cooling'] * 1e-3, name='Kühlleistung AKT', legend='legend4', line=dict(color='Cyan')),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['d_Q_load'] * 1e-3, name='Kälteleistung Prod.', legend='legend4', line=dict(color='Blue')),
            row=5, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['s_P_el'] * 1e-3, name='Strom Netzleistung', legend='legend4', line=dict(color='Orange')),
            row=5, col=1
        )
        

        # (6) - Costs
        fig.add_trace(
            go.Scatter(x=x, y=-episode_df['reward_energy_electric'].cumsum(), name='Stromkosten', legend='legend5', line=dict(color='Orange')),
            row=6, col=1
        )

        # (7) - Rewards
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['reward_energy_electric'].cumsum(), name='Stromkosten', line=dict(color='Orange'), legend='legend6'),
            row=7, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['reward_temperature_cooling'].cumsum(), name='Kühlwasser', line=dict(color='Red'), legend='legend6'),
            row=7, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['reward_temperature_cooled'].cumsum(), name='Kaltwasser', line=dict(color='Blue'), legend='legend6'),
            row=7, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['reward_switching'].cumsum(), name='Schaltvorgänge', line=dict(color='Purple'), legend='legend6'),
            row=7, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['reward_policyshaping'].cumsum(), name='Policyshaping', line=dict(color='Green'), legend='legend6'),
            row=7, col=1
        )
        fig.add_trace(
            go.Scatter(x=x, y=episode_df['reward_total'].cumsum(), name='Gesamt', line=dict(color='Black'), legend='legend6'),
            row=7, col=1
        )

        fig.update_layout(height=2200, width=1200, title_text="Kälteversorgungssystem")
        fig.update_layout(showlegend=True)

        fig.update_layout(
            
            title_text="Supply System Analysis",
            legend1=dict(
                x=1,  # x-Position der Legende, außerhalb des Plots
                y=0.85,    # y-Position der Legende, oben
                xanchor='left',  # Verankerung der Legende
                yanchor='top'
            ),
            legend2=dict(
                x=1,  # x-Position der Legende, außerhalb des Plots
                y=0.65,    # y-Position der Legende, oben
                xanchor='left',  # Verankerung der Legende
                yanchor='top'
            ),
            legend3=dict(
                x=1,  # x-Position der Legende, außerhalb des Plots
                y=0.5,    # y-Position der Legende, oben
                xanchor='left',  # Verankerung der Legende
                yanchor='top'
            ),
            legend4=dict(
                x=1,  # x-Position der Legende, außerhalb des Plots
                y=0.35,    # y-Position der Legende, oben
                xanchor='left',  # Verankerung der Legende
                yanchor='top'
            ),
            legend5=dict(
                x=1,  # x-Position der Legende, außerhalb des Plots
                y=0.2,    # y-Position der Legende, oben
                xanchor='left',  # Verankerung der Legende
                yanchor='top'
            ),
            legend6=dict(
                x=1,  # x-Position der Legende, außerhalb des Plots
                y=0.05,    # y-Position der Legende, oben
                xanchor='left',  # Verankerung der Legende
                yanchor='top'
            )
        )
        
        fig.update_xaxes(range=[datetime_start, datetime_end], row=1, col=1, showticklabels=True, tickfont=dict(size=10))
        fig.update_xaxes(range=[datetime_start, datetime_end], row=2, col=1, showticklabels=True, tickfont=dict(size=10))
        fig.update_xaxes(range=[datetime_start, datetime_end], row=3, col=1, showticklabels=True, tickfont=dict(size=10))
        fig.update_xaxes(range=[datetime_start, datetime_end], row=4, col=1, showticklabels=True, tickfont=dict(size=10))
        fig.update_xaxes(range=[datetime_start, datetime_end], row=5, col=1, showticklabels=True, tickfont=dict(size=10))
        fig.update_xaxes(range=[datetime_start, datetime_end], row=6, col=1, showticklabels=True, tickfont=dict(size=10))
        fig.update_xaxes(range=[datetime_start, datetime_end], row=7, col=1, showticklabels=True, tickfont=dict(size=10))

        # Speichern der Figur als HTML
        fig.write_html(
            os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(3)
                + "-"
                + str(self.env_id).zfill(2)
                + "_episode"
                + name_suffix
                + ".html")
        )
            







        return


def reward_boundary(state, state_min, state_max, reward, penalty, smoothed=True, k=1):
    """
    reward function for boundaries enabling hard reward/penalties or smoothed by sigmoid function
    Parameters
    -----
        state : (float)
            the state value to be checked
        min : (float) or None
        max : (float) or None
        reward : (float)
            reward given if within min/max boundary
        penalty : (float)
            penalty given if outside of min/max boundary
        smoothed : (bool)
            should reward be smoothed by use if sigmoid function ?
        k : (float)
            modify width of sigmoid smoothing 1/(1+exp(-k*x)) - higher is steeper
    """
    # catch cases when min/max are not defined
    if state_min == None:
        state_min = -1e10
    if state_max == None:
        state_max = 1e10

    if smoothed:
        # return reward - (reward+penalty)*(expit(k*(state-state_max))+expit(k*(state_min-state)))
        return (
            reward
            - (reward + penalty) * (expit(k * (state - state_max)) + expit(k * (state_min - state)))
            - k * max(state - state_max, 0)
            - k * max(state_min - state, 0)
        )
    else:
        return reward if (state > state_min and state < state_max) else -penalty



def reward_potential(state, state_min, state_max, penalty, k=2, base_reward=0):
    mid_point = (state_min + state_max) / 2
    distance = state - mid_point

    c = (penalty-base_reward) / ((state_max-mid_point)**k)

    reward = 0 - base_reward - abs(c * (distance**k))
    return reward

def reward_temp(state, goal, state_min, state_max, penalty, k=2, base_reward=0):
    
    distance = state - goal

    c = (penalty-base_reward) / ((state_max-goal)**k)
    d = (penalty-base_reward) / ((goal-state_min)**k)
    def f_pos (x): return abs(c * (x**k))
    def f_neg (x): return abs(d * (x**k))

    reward = 0 - base_reward - (f_pos(distance) if distance>0 else f_neg(distance))
    return reward