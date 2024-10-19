from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

import controller.rule_based_controller as rule_based_controller
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eta_utility import get_logger, timeseries
from eta_utility.eta_x import ConfigOptRun
from eta_utility.eta_x.envs import BaseEnvSim, StateConfig, StateVar
from eta_utility.type_hints import StepResult, TimeStep
from gymnasium import spaces
from gymnasium.utils import seeding
from scipy.special import expit

log = get_logger("eta_x.envs")


class SupplysystemETA(BaseEnvSim):
    """
    Supplysystem for ETA Application environment class from BaseEnvSim.

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
    # fmu_name = "supplysystem_ETA_variance"
    fmu_name = "supplysystem_ETA"  # simply change to supplysystem_ETA if variance is not needed

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,  # config_run contains run_name, general_settings, path_settings, env_info
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        sampling_time: float,  # from settings section of config file
        episode_duration: TimeStep | str,  # from settings section of config file
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        scenario_files: Sequence[Mapping[str, Any]],
        random_sampling,
        variance_min: float,
        variance_max: float,
        variance_parameters: list,
        extended_observations,  # different variants for observations can be selected
        # continuous_action_space,
        with_storages,
        use_complex_AFA_model,
        allow_policy_shaping,
        allow_limiting_CHP_switches,
        reward_shaping,
        temperature_HNHT_Buffer_init_min,
        temperature_HNHT_Buffer_init_max,
        temperature_HNHT_VSI_init_min,
        temperature_HNHT_VSI_init_max,
        temperature_HNLT_Buffer_init_min,
        temperature_HNLT_Buffer_init_max,
        temperature_HNLT_HVFA_init_min,
        temperature_HNLT_HVFA_init_max,
        temperature_CN_Buffer_init_min,
        temperature_CN_Buffer_init_max,
        temperature_CN_HVFA_init_min,
        temperature_CN_HVFA_init_max,
        temperature_AFA_init_fixed,
        # temperature_heat_init_min,
        # temperature_heat_init_max,
        # temperature_cool_init_min,
        # temperature_cool_init_max,
        poweron_cost_HeatExchanger1,  # cost for powering on
        poweron_cost_CHP1,
        poweron_cost_CHP2,
        poweron_cost_CondensingBoiler,
        poweron_cost_VSIStorage,
        switching_cost_bLoading_VSISystem,
        poweron_cost_HVFASystem_HNLT,
        switching_cost_bLoading_HVFASystem_HNLT,
        poweron_cost_eChiller,
        poweron_cost_HVFASystem_CN,
        switching_cost_bLoading_HVFASystem_CN,
        poweron_cost_OuterCapillaryTubeMats,
        poweron_cost_HeatPump,
        runtime_violation_cost_CHP1,
        runtime_violation_cost_CHP2,
        runtime_violation_cost_HeatPump,
        runtime_violation_cost_eChiller,
        runtime_violation_cost_OuterCapillaryTubeMats,
        abort_costs,  # penalty for abort simulation
        abort_cost_for_unsuccessfull_step,
        policyshaping_costs,  # penalty if policy shaping is necessary
        temperature_cost_HNHT,  # penalty for exceeding temperature in heating network
        temperature_reward_HNHT,  # reward when temperature in heating network is not exceeded
        temperature_cost_HNLT,
        temperature_reward_HNLT,
        temperature_cost_CN,
        temperature_reward_CN,
        HNHT_temperature_reward_min_T,
        HNHT_temperature_reward_max_T,
        HNLT_temperature_reward_min_T,
        HNLT_temperature_reward_max_T,
        CN_temperature_reward_min_T,
        CN_temperature_reward_max_T,
        offset_policy_shaping_threshold,
        ### add other networks
        # temperature_cost_heat, #penalty for exceeding temperature in heating network
        # temperature_reward_heat, #reward when temperature in heating network is not exceeded
        # temperature_cost_heat_min, #not used anywhere
        # temperature_cost_heat_max, #not used anywhere
        # temperature_cost_prod_heat_min, # min value when calculating reward for heating network
        # temperature_cost_prod_heat_max, # max value when calculating reward for heating network
        # temperature_cost_cool, #penalty for exceeding temperature in cooling network
        # temperature_reward_cool, #reward when temperature in cooling network is not exceeded
        # temperature_cost_cool_min, #not used anywhere
        # temperature_cost_cool_max, #not used anywhere
        # temperature_cost_prod_cool_min, # min value when calculating reward for cooling network
        # temperature_cost_prod_cool_max, # max value when calculating reward for cooling network
        power_cost_max,
        tax_el_per_kwh,
        tax_el_produced_per_kwh,
        tax_gas_per_kwh,
        peak_cost_per_kw,
        **kwargs: Any,
    ):
        super().__init__(
            env_id=env_id,
            config_run=config_run,
            verbose=verbose,
            callback=callback,
            sampling_time=sampling_time,
            episode_duration=episode_duration,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            **kwargs,
        )

        # make variables readable class-wide
        self.random_sampling = random_sampling
        self.variance_min = variance_min
        self.variance_max = variance_max
        self.variance_parameters = variance_parameters
        # self.continuous_action_space = continuous_action_space
        self.reward_shaping = reward_shaping
        self.allow_policy_shaping = allow_policy_shaping
        self.allow_limiting_CHP_switches = allow_limiting_CHP_switches
        self.chp1_switch_counter = 0
        self.chp2_switch_counter = 0
        self.temperature_HNHT_Buffer_init_min = temperature_HNHT_Buffer_init_min
        self.temperature_HNHT_Buffer_init_max = temperature_HNHT_Buffer_init_max
        self.temperature_HNHT_VSI_init_min = temperature_HNHT_VSI_init_min
        self.temperature_HNHT_VSI_init_max = temperature_HNHT_VSI_init_max

        self.temperature_HNLT_Buffer_init_min = temperature_HNLT_Buffer_init_min
        self.temperature_HNLT_Buffer_init_max = temperature_HNLT_Buffer_init_max
        self.temperature_HNLT_HVFA_init_min = temperature_HNLT_HVFA_init_min
        self.temperature_HNLT_HVFA_init_max = temperature_HNLT_HVFA_init_max

        self.temperature_CN_Buffer_init_min = temperature_CN_Buffer_init_min
        self.temperature_CN_Buffer_init_max = temperature_CN_Buffer_init_max
        self.temperature_CN_HVFA_init_min = temperature_CN_HVFA_init_min
        self.temperature_CN_HVFA_init_max = temperature_CN_HVFA_init_max

        self.temperature_AFA_init_fixed = temperature_AFA_init_fixed

        self.poweron_cost_HeatExchanger1 = poweron_cost_HeatExchanger1  # cost for powering on
        self.poweron_cost_CHP1 = poweron_cost_CHP1
        self.poweron_cost_CHP2 = poweron_cost_CHP2
        self.poweron_cost_CondensingBoiler = poweron_cost_CondensingBoiler
        self.poweron_cost_VSIStorage = poweron_cost_VSIStorage
        self.switching_cost_bLoading_VSISystem = switching_cost_bLoading_VSISystem
        self.poweron_cost_HVFASystem_HNLT = poweron_cost_HVFASystem_HNLT
        self.switching_cost_bLoading_HVFASystem_HNLT = switching_cost_bLoading_HVFASystem_HNLT
        self.poweron_cost_eChiller = poweron_cost_eChiller
        self.poweron_cost_HVFASystem_CN = poweron_cost_HVFASystem_CN
        self.switching_cost_bLoading_HVFASystem_CN = switching_cost_bLoading_HVFASystem_CN
        self.poweron_cost_OuterCapillaryTubeMats = poweron_cost_OuterCapillaryTubeMats
        self.poweron_cost_HeatPump = poweron_cost_HeatPump

        self.runtime_violation_cost_CHP1 = runtime_violation_cost_CHP1
        self.runtime_violation_cost_CHP2 = runtime_violation_cost_CHP2
        self.runtime_violation_cost_HeatPump = runtime_violation_cost_HeatPump
        self.runtime_violation_cost_eChiller = runtime_violation_cost_eChiller
        self.runtime_violation_cost_OuterCapillaryTubeMats = runtime_violation_cost_OuterCapillaryTubeMats

        self.abort_costs = abort_costs
        self.abort_cost_for_unsuccessfull_step = abort_cost_for_unsuccessfull_step
        self.policyshaping_costs = policyshaping_costs

        self.temperature_cost_HNHT = temperature_cost_HNHT  # penalty for exceeding temperature in heating network
        self.temperature_reward_HNHT = (
            temperature_reward_HNHT  # reward when temperature in heating network is not exceeded
        )
        self.temperature_cost_HNLT = temperature_cost_HNLT
        self.temperature_reward_HNLT = temperature_reward_HNLT
        self.temperature_cost_CN = temperature_cost_CN
        self.temperature_reward_CN = temperature_reward_CN
        self.HNHT_temperature_reward_min_T = HNHT_temperature_reward_min_T
        self.HNHT_temperature_reward_max_T = HNHT_temperature_reward_max_T
        self.HNLT_temperature_reward_min_T = HNLT_temperature_reward_min_T
        self.HNLT_temperature_reward_max_T = HNLT_temperature_reward_max_T
        self.CN_temperature_reward_min_T = CN_temperature_reward_min_T
        self.CN_temperature_reward_max_T = CN_temperature_reward_max_T

        self.offset_policy_shaping_threshold = offset_policy_shaping_threshold

        self.power_cost_max = power_cost_max
        self.tax_el_per_kwh = tax_el_per_kwh
        self.tax_el_produced_per_kwh = tax_el_produced_per_kwh
        self.tax_gas_per_kwh = tax_gas_per_kwh
        self.peak_cost_per_kw = peak_cost_per_kw

        # initialize episode statistics
        self.episode_statistics = {}
        self.episode_archive = []
        self.episode_df = pd.DataFrame()

        # set integer prediction steps (15m,1h,6h)

        self.n_steps_15m = int(900 // self.sampling_time)
        self.n_steps_1h = int(3600 // self.sampling_time)
        # self.n_steps_2h = int(3600*2 // self.sampling_time)
        self.n_steps_3h = int(3600 * 3 // self.sampling_time)
        # self.n_steps_4h = int(3600*4 // self.sampling_time)
        # self.n_steps_5h = int(3600*5 // self.sampling_time)
        self.n_steps_6h = int(21600 // self.sampling_time)

        # initialize integrators and longtime stats
        self.P_el_total_15min_buffer = []
        self.P_gs_total_15min_buffer = []
        self.n_steps_longtime = 0
        self.reward_longtime_average = 0
        self.initial_resets = 0

        self.startTime = datetime.now()

        # define state variables
        state_var_tuple = (
            #################### agent actions ####################
            StateVar(
                name="bSetStatusOn_HeatExchanger1",
                ext_id="bSetStatusOn_HeatExchanger1",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bSetStatusOn_CHP1",
                ext_id="bSetStatusOn_CHP1",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bSetStatusOn_CHP2",
                ext_id="bSetStatusOn_CHP2",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bSetStatusOn_CondensingBoiler",
                ext_id="bSetStatusOn_CondensingBoiler",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bSetStatusOn_VSIStorage",
                ext_id="bSetStatusOn_VSIStorage",
                is_ext_input=True,
                is_agent_action=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bLoading_VSISystem",
                ext_id="bLoading_VSISystem",
                is_ext_input=True,
                is_agent_action=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bSetStatusOn_HVFASystem_HNLT",
                ext_id="bSetStatusOn_HVFASystem_HNLT",
                is_ext_input=True,
                is_agent_action=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bLoading_HVFASystem_HNLT",
                ext_id="bLoading_HVFASystem_HNLT",
                is_ext_input=True,
                is_agent_action=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bSetStatusOn_eChiller",
                ext_id="bSetStatusOn_eChiller",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bSetStatusOn_HVFASystem_CN",
                ext_id="bSetStatusOn_HVFASystem_CN",
                is_ext_input=True,
                is_agent_action=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bLoading_HVFASystem_CN",
                ext_id="bLoading_HVFASystem_CN",
                is_ext_input=True,
                is_agent_action=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bSetStatusOn_OuterCapillaryTubeMats",
                ext_id="bSetStatusOn_OuterCapillaryTubeMats",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="bSetStatusOn_HeatPump",
                ext_id="bSetStatusOn_HeatPump",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            #################### current states of systems ####################
            StateVar(
                name="Out_bSetStatusOn_HeatExchanger1",
                ext_id="Out_bSetStatusOn_HeatExchanger1",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bSetStatusOn_CHP1",
                ext_id="Out_bSetStatusOn_CHP1",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bSetStatusOn_CHP2",
                ext_id="Out_bSetStatusOn_CHP2",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bSetStatusOn_CondensingBoiler",
                ext_id="Out_bSetStatusOn_CondensingBoiler",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bSetStatusOn_VSIStorage",
                ext_id="Out_bSetStatusOn_VSIStorage",
                is_ext_output=True,
                is_agent_observation=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bLoading_VSISystem",
                ext_id="Out_bLoading_VSISystem",
                is_ext_output=True,
                is_agent_observation=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bSetStatusOn_HVFASystem_HNLT",
                ext_id="Out_bSetStatusOn_HVFASystem_HNLT",
                is_ext_output=True,
                is_agent_observation=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bLoading_HVFASystem_HNLT",
                ext_id="Out_bLoading_HVFASystem_HNLT",
                is_ext_output=True,
                is_agent_observation=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bSetStatusOn_eChiller",
                ext_id="Out_bSetStatusOn_eChiller",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bSetStatusOn_HVFASystem_CN",
                ext_id="Out_bSetStatusOn_HVFASystem_CN",
                is_ext_output=True,
                is_agent_observation=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bLoading_HVFASystem_CN",
                ext_id="Out_bLoading_HVFASystem_CN",
                is_ext_output=True,
                is_agent_observation=with_storages,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bSetStatusOn_OuterCapillaryTubeMats",
                ext_id="Out_bSetStatusOn_OuterCapillaryTubeMats",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="Out_bSetStatusOn_HeatPump",
                ext_id="Out_bSetStatusOn_HeatPump",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            # add others here too
            #################### disturbances ####################
            StateVar(
                name="d_HNHT_prod_heat_demand_consumer",
                ext_id="HNHT_prod_heat_demand_consumer",
                scenario_id="hnht_consumer",  # name of column in csv
                from_scenario=True,
                is_ext_input=True,
                # is_agent_observation=True,
                low_value=0.0,
                high_value=250000.0,
            ),
            # StateVar(
            #     name="d_HNHT_prod_heat_demand_producer",
            #     ext_id="HNHT_prod_heat_demand_producer",
            #     scenario_id="hnht_producer", #name of column in csv
            #     from_scenario=True,
            #     is_ext_input=True,
            #     #is_agent_observation=True,
            #     low_value=-250000.0,
            #     high_value=0.0,
            # ),
            StateVar(
                name="d_HNLT_prod_heat_demand_consumer",
                ext_id="HNLT_prod_heat_demand_consumer",
                scenario_id="hnlt_consumer",  # name of column in csv
                from_scenario=True,
                is_ext_input=True,
                # is_agent_observation=True,
                low_value=0.0,
                high_value=250000.0,
            ),
            StateVar(
                name="d_HNLT_prod_heat_demand_producer",
                ext_id="HNLT_prod_heat_demand_producer",
                scenario_id="hnlt_producer",  # name of column in csv
                from_scenario=True,
                is_ext_input=True,
                # is_agent_observation=True,
                low_value=-250000.0,
                high_value=0,
            ),
            StateVar(
                name="d_CN_prod_heat_demand_consumer",
                ext_id="CN_prod_heat_demand_consumer",
                scenario_id="cn_consumer",  # name of column in csv
                from_scenario=True,
                is_ext_input=True,
                # is_agent_observation=True,
                low_value=-250000.0,
                high_value=0,
            ),
            # The following is imported from Factory_2017/2018 - but is it relevant for my application?
            # StateVar(
            #     name="d_production_electric_power",
            #     ext_id="u_electric_power_demand_production",
            #     scenario_id="power_electricity",
            #     from_scenario=True,
            #     is_ext_input=True,
            #     is_agent_observation=extended_state,
            #     low_value=0.0,
            #     high_value=300000.0,
            # ),
            # StateVar(
            #     name="d_production_gas_power",
            #     ext_id="u_gas_power_demand_production",
            #     scenario_id="power_gas",
            #     from_scenario=True,
            #     is_ext_input=True,
            #     low_value=0.0,
            #     high_value=300000.0,
            # ),
            #################### weather ####################
            # StateVar(
            #     name="weather_RelativeHumidity",
            #     ext_id="weather_RelativeHumidity",
            #     scenario_id="relative_air_humidity",  # [%]
            #     from_scenario=True,
            #     is_agent_observation=True,
            #     is_ext_input=True,
            #     low_value=0,
            #     high_value=100,
            # ),
            StateVar(
                name="weather_T_amb",
                ext_id="weather_T_amb",
                scenario_id="air_temperature",  # [°C]
                from_scenario=True,
                is_agent_observation=True,
                is_ext_input=True,
                low_value=-20,
                high_value=45,
            ),
            StateVar(
                name="weather_T_amb_Mean",
                ext_id="weather_T_amb_Mean",
                scenario_id="air_temperature_mean_6h",  # [°C]
                from_scenario=True,
                is_agent_observation=extended_observations,  # ToDo: change to extended
                low_value=-20,
                high_value=45,
            ),
            StateVar(
                name="weather_T_Ground_1m",
                ext_id="weather_T_Ground_1m",
                scenario_id="ts100",  # [°C]
                from_scenario=True,
                is_ext_input=True,
                low_value=-20,
                high_value=45,
            ),
            # StateVar(       #time of year in seconds
            #     name="d_weather_time",
            #     ext_id="u_weather_Time",
            #     is_ext_input=False, #ToDo: Check for this again
            #     low_value=0,
            #     high_value=31968000,
            # ),
            #################### time ####################
            StateVar(
                name="time_daytime",
                ext_id="time_daytime",
                is_agent_observation=True,
                is_ext_input=use_complex_AFA_model,
                low_value=0,
                high_value=24,
            ),
            StateVar(
                name="time_month",
                ext_id="time_month",
                scenario_id="current_month",
                from_scenario=True,
                is_agent_observation=True,
                is_ext_input=use_complex_AFA_model,
                low_value=0,
                high_value=12,
            ),
            #################### virtual states ####################
            StateVar(
                name="vs_electric_power_total_15min",
                # is_agent_observation=True,
                low_value=-100000,
                high_value=500000,
            ),
            StateVar(
                name="vs_gas_power_total_15min",
                # is_agent_observation=True, #before it was = extended_state
                low_value=-100000,
                high_value=500000,
            ),
            #################### prices ####################
            # prices
            # StateVar(
            #     name="s_price_electricity",
            #     scenario_id="electrical_energy_price",
            #     from_scenario=True,
            #     is_agent_observation=True,
            #     low_value=-1000,
            #     high_value=1000,
            # ),  # to obtain €/kWh
            StateVar(
                name="s_price_electricity_00h",
                scenario_id="electrical_energy_price",
                from_scenario=True,
                is_agent_observation=True,
                low_value=-1000,
                high_value=1000,
            ),
            StateVar(name="s_price_electricity_01h", is_agent_observation=True, low_value=-1000, high_value=1000),
            StateVar(name="s_price_electricity_03h", is_agent_observation=True, low_value=-1000, high_value=1000),
            StateVar(name="s_price_electricity_06h", is_agent_observation=True, low_value=-1000, high_value=1000),
            StateVar(
                name="s_price_gas",
                scenario_id="gas_price",
                from_scenario=True,
                is_agent_observation=True,
                low_value=-1000,
                high_value=1000,
            ),  # to obtain €/kWh
            #################### energy consumption in production ####################
            StateVar(
                name="gas_power_consumption",
                ext_id="gas_power_consumption",
                is_ext_output=True,
                low_value=0,
                high_value=500000,
            ),
            # used electricity in production
            StateVar(
                name="electric_power_consumption",
                ext_id="electric_power_consumption",
                is_ext_output=True,
                low_value=-100000,
                high_value=500000,
            ),
            #################### Temperature observations ####################
            StateVar(
                name="HNHT_Buffer_fUpperTemperature",
                ext_id="HNHT_Buffer_fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=extended_observations,
                abort_condition_min=15,  # how should these be set?
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT_Buffer_fMidTemperature",
                ext_id="HNHT_Buffer_fMidTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                abort_condition_min=15,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT_Buffer_fLowerTemperature",
                ext_id="HNHT_Buffer_fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=extended_observations,
                abort_condition_min=15,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT_Buffer_fUpperTemperature",
                ext_id="HNLT_Buffer_fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=extended_observations,
                abort_condition_min=3,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT_Buffer_fMidTemperature",
                ext_id="HNLT_Buffer_fMidTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                abort_condition_min=3,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT_Buffer_fLowerTemperature",
                ext_id="HNLT_Buffer_fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=extended_observations,
                abort_condition_min=3,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN_Buffer_fUpperTemperature",
                ext_id="CN_Buffer_fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=extended_observations,
                abort_condition_min=3,
                abort_condition_max=50,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN_Buffer_fMidTemperature",
                ext_id="CN_Buffer_fMidTemperature",
                is_ext_output=True,
                is_agent_observation=True,
                abort_condition_min=3,
                abort_condition_max=50,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN_Buffer_fLowerTemperature",
                ext_id="CN_Buffer_fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=extended_observations,
                abort_condition_min=3,
                abort_condition_max=50,
                low_value=0,
                high_value=100,
            ),
            # HVFA and VSI Storage
            StateVar(
                name="HNLT_HVFA_fUpperTemperature",
                ext_id="HNLT_HVFA_fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=extended_observations,
                abort_condition_min=1,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNLT_HVFA_fLowerTemperature",
                ext_id="HNLT_HVFA_fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=with_storages,
                abort_condition_min=1,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN_HVFA_fUpperTemperature",
                ext_id="CN_HVFA_fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=extended_observations,
                abort_condition_min=1,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="CN_HVFA_fLowerTemperature",
                ext_id="CN_HVFA_fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=with_storages,
                abort_condition_min=1,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT_VSI_fUpperTemperature",
                ext_id="HNHT_VSI_fUpperTemperature",
                is_ext_output=True,
                is_agent_observation=with_storages,
                abort_condition_min=15,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT_VSI_fMidTemperature",
                ext_id="HNHT_VSI_fMidTemperature",
                is_ext_output=True,
                is_agent_observation=extended_observations,
                abort_condition_min=15,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="HNHT_VSI_fLowerTemperature",
                ext_id="HNHT_VSI_fLowerTemperature",
                is_ext_output=True,
                is_agent_observation=with_storages,
                abort_condition_min=15,
                abort_condition_max=97,
                low_value=0,
                high_value=100,
            ),
        )

        # build final state_config
        self.state_config = StateConfig(*state_var_tuple)
        print("The selected observations are: ", self.state_config.observations)

        # import all scenario files
        self.scenario_data = self.import_scenario(*scenario_files).fillna(
            method="ffill"
        )  # add another ffillna cause only values which are missing beceause of resampling are interpolated in eta_utility

        # get action_space
        self.n_action_space = len(
            self.state_config.actions
        )  # get number of actions agent has to give from state_config
        self.action_space = spaces.MultiDiscrete(np.full(self.n_action_space, 2))  # set 2 discrete actions (On, Off)
        self.action_disc_index = [0] * self.n_action_space  # initialize action

        # get observation_space (always continuous)
        self.observation_space = self.state_config.continuous_obs_space()

        """
        # TODO: implement this functionality into utility functions
        if self.discretize_action_space:

            # get number of actions agent has to give from state_config
            self.n_action_space = len(self.state_config.actions)
            # set 3 discrete actions (increase,decrease,equal) per control variable
            self.action_space = spaces.MultiDiscrete(np.full(self.n_action_space, 3))
            # customized for chp, condensingboiler, immersionheater, heatpump, coolingtower, compressionchiller
            self.action_disc_step = [
                [0, 0.5, 0.75, 1],
                [0, 0.25, 0.5, 0.75, 1],
                [0, 0.25, 0.5, 0.75, 1],
                [0, 0.5, 0.75, 1],
                [0, 0.5, 0.75, 1],
                [0, 0.5, 0.75, 1],
            ]
            # initialize action
            self.action_disc_index = [0] * self.n_action_space
        else:
            self.action_space = self.state_config.continuous_action_space()
        """

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.
        """
        # initialize additional_state and create state backup
        self.state_backup = self.state.copy()
        self.additional_state = {}

        # create the following when its the first step ever
        try:
            self.action_backup
            # self.policy_shaping_active
        except:
            self.action_backup = action
            self.policy_shaping_active = False
            self.runtime_counter = {"CHP1": 0, "CHP2": 0, "HeatPump": 0, "eChiller": 0, "OuterCapillaryTubeMats": 0}

        _action = action

        # now we do policy shaping
        if self.allow_policy_shaping:

            HNHT_Buffer_lower_threshold = self.HNHT_temperature_reward_min_T - self.offset_policy_shaping_threshold
            HNHT_Buffer_upper_threshold = self.HNHT_temperature_reward_max_T + self.offset_policy_shaping_threshold

            HNLT_Buffer_lower_threshold = self.HNLT_temperature_reward_min_T - self.offset_policy_shaping_threshold
            HNLT_Buffer_upper_threshold = self.HNLT_temperature_reward_max_T + self.offset_policy_shaping_threshold

            CN_Buffer_lower_threshold = self.CN_temperature_reward_min_T - self.offset_policy_shaping_threshold
            CN_Buffer_upper_threshold = self.CN_temperature_reward_max_T + self.offset_policy_shaping_threshold

            # check if reward shaping is necessary in this step
            if (
                self.state["HNHT_Buffer_fMidTemperature"] < HNHT_Buffer_lower_threshold
                or self.state["HNHT_Buffer_fMidTemperature"] > HNHT_Buffer_upper_threshold
            ):
                policy_shaping_necessary = True
            elif (
                self.state["HNLT_Buffer_fMidTemperature"] < HNLT_Buffer_lower_threshold
                or self.state["HNLT_Buffer_fMidTemperature"] > HNLT_Buffer_upper_threshold
            ):
                policy_shaping_necessary = True
            elif (
                self.state["CN_Buffer_fMidTemperature"] < CN_Buffer_lower_threshold
                or self.state["CN_Buffer_fMidTemperature"] > CN_Buffer_upper_threshold
            ):
                policy_shaping_necessary = True
            else:
                policy_shaping_necessary = False

            # if there was no policy shaping in the last step, but now there is, the Hysteresis Controller should be initalized
            if self.policy_shaping_active == False and policy_shaping_necessary == True:
                self.init_Hysterese_Controllers()

            if policy_shaping_necessary:
                # set policy_shaping for next iteration and reward function
                self.policy_shaping_active == True
                print("Policy Shaping active")

                # ask the rule based controller for actions, but be careful! The rules are defined in this environment, so they can differ from the controller.rule_based_controller.py
                _action = self.control_rules(observation=self.state)

        #######################################

        if self.allow_limiting_CHP_switches:
            """
            this if-statement checks how many steps were performed since last chp switch and allows to change the state
            the node_in is adjusted accordingly
            """
            if self.n_steps > 1:  # only check, if minimum one step was performed
                action_chp1 = _action[1]  # get current action
                action_chp2 = _action[2]  # get current action

                if action_chp1 == self.action_chp1_backup:
                    self.chp1_switch_counter = self.chp1_switch_counter + 1
                    self.state["overwrite_CHP1_action"] = False
                else:
                    if self.chp1_switch_counter * self.sampling_time / 60 <= 20:  # switching is not ok
                        # node_in.update({str(self.state_config.map_ext_ids["bSetStatusOn_CHP1"]): self.action_chp1_backup})
                        _action[1] = self.action_chp1_backup
                        self.chp1_switch_counter = self.chp1_switch_counter + 1
                        self.state["overwrite_CHP1_action"] = True
                    else:  # switch is ok
                        self.chp1_switch_counter = 0
                        self.state["overwrite_CHP1_action"] = False

                if action_chp2 == self.action_chp2_backup:
                    self.chp2_switch_counter = self.chp2_switch_counter + 1
                    self.state["overwrite_CHP2_action"] = False
                else:
                    if self.chp2_switch_counter * self.sampling_time / 60 <= 20:  # switching is not ok
                        # node_in.update({str(self.state_config.map_ext_ids["bSetStatusOn_CHP2"]): self.action_chp2_backup})
                        _action[2] = self.action_chp2_backup
                        self.chp2_switch_counter = self.chp2_switch_counter + 1
                        self.state["overwrite_CHP2_action"] = True
                    else:
                        self.chp2_switch_counter = 0
                        self.state["overwrite_CHP2_action"] = False
            else:
                self.state["overwrite_CHP1_action"] = False
                self.state["overwrite_CHP2_action"] = False

            # log infos for user
            if self.state["overwrite_CHP1_action"] == True:
                log.info(f"The CHP1 state was overwritten from {action_chp1} to {self.action_chp1_backup}")
            if self.state["overwrite_CHP2_action"] == True:
                log.info(f"The CHP2 state was overwritten from {action_chp2} to {self.action_chp2_backup}")

        if self.allow_limiting_CHP_switches and self.n_steps > 0:
            if self.state["overwrite_CHP1_action"] == False:
                self.action_chp1_backup = _action[1]
            if self.state["overwrite_CHP2_action"] == False:
                self.action_chp2_backup = _action[2]

        ########################################################

        # check actions for vilidity, perform simulation step and load new external values for the next time step

        self._actions_valid(_action)
        _action = _action.astype(bool)  # convert actions to bool

        self.state["step_success"], _ = self._update_state(_action)  # update state witch actions

        # check if state is in valid boundaries
        try:
            self.state["step_abort"] = (
                False if StateConfig.within_abort_conditions(self.state_config, self.state) else True
            )
        except:
            self.state["step_abort"] = False

        # update predictions and virtual state for next time step
        self.state.update(self.update_predictions())
        if self.state["step_success"]:
            self.state.update(self.update_virtual_state())

        # add time of the year in seconds
        # starttime_of_year = pd.Timestamp(str(self.ts_current.index[self.n_steps].year) + "-01-01 00:00")
        # self.state["d_weather_time"] = pd.Timedelta(
        #     self.ts_current.index[self.n_steps] - starttime_of_year
        # ).total_seconds()

        # check if episode is over or not
        done = self._done() or not self.state["step_success"]
        done = done if not self.state["step_abort"] else True

        print("Episode: ", self.n_episodes, "Progress: " + str(self.n_steps) + " of " + str(self.episode_duration/self.sampling_time), "time steps - Estimated remaining time: ", (datetime.now()-self.startTime)/self.n_steps*(self.episode_duration/self.sampling_time-self.n_steps))

        # calculate reward
        if self.state["step_success"]:
            # only if step successfull reward can be calculated normal
            reward = self.calc_reward(action)
            observations = self._observations()
        else:
            # otherwise we just give abort cost reward
            # since the step was not successfull, the observations will just be zeros.
            # reward = (-self.abort_costs * (1 - 0.5 * self.n_steps / self.n_episode_steps))
            self.state["reward_switching"] = 0
            self.state["reward_temperature_HNHT"] = 0
            self.state["reward_temperature_HNLT"] = 0
            self.state["reward_temperature_CN"] = 0
            self.state["reward_energy_electric"] = 0
            self.state["reward_energy_gas"] = 0
            self.state["reward_energy_taxes"] = 0
            self.state["reward_power_electric"] = 0
            if self.abort_cost_for_unsuccessfull_step == True:
                self.state["reward_other"] = -self.abort_costs * (1 - 0.5 * self.n_steps / self.n_episode_steps)
            else:
                self.state["reward_other"] = 0

            self.state["reward_total"] = (
                self.state["reward_switching"]
                + self.state["reward_temperature_HNHT"]
                + self.state["reward_temperature_HNLT"]
                + self.state["reward_temperature_CN"]
                + self.state["reward_energy_electric"]
                + self.state["reward_energy_gas"]
                + self.state["reward_energy_taxes"]
                + self.state["reward_power_electric"]
                + self.state["reward_other"]
            )

            reward = self.state["reward_total"]

            observations = np.zeros(len(self.state_config.observations))

        # update state_log
        self.state_log.append(self.state)

        return observations, reward, done, False, {}

    def update_predictions(self):

        prediction_dict = {}

        # electricity price [€/kWh]
        prediction_dict["s_price_electricity_01h"] = self.ts_current["electrical_energy_price"].iloc[
            self.n_steps + self.n_steps_1h
        ]
        prediction_dict["s_price_electricity_03h"] = self.ts_current["electrical_energy_price"].iloc[
            self.n_steps + self.n_steps_3h
        ]
        prediction_dict["s_price_electricity_06h"] = self.ts_current["electrical_energy_price"].iloc[
            self.n_steps + self.n_steps_6h
        ]

        return prediction_dict

    def update_virtual_state(self):

        virtual_state = {}

        # daytime
        virtual_state["time_daytime"] = (
            self.ts_current.index[self.n_steps].hour + self.ts_current.index[self.n_steps].minute / 60
        )

        # running 15min average electric power  # TODO: replace by using state_log!
        self.P_el_total_15min_buffer.append(self.state["electric_power_consumption"])
        if len(self.P_el_total_15min_buffer) > self.n_steps_15m:
            self.P_el_total_15min_buffer.pop(0)
        virtual_state["vs_electric_power_total_15min"] = sum(self.P_el_total_15min_buffer) / len(
            self.P_el_total_15min_buffer
        )

        # running 15min average gas power  # TODO: replace by using state_log!
        self.P_gs_total_15min_buffer.append(self.state["gas_power_consumption"])
        if len(self.P_gs_total_15min_buffer) > self.n_steps_15m:
            self.P_gs_total_15min_buffer.pop(0)
        virtual_state["vs_gas_power_total_15min"] = sum(self.P_gs_total_15min_buffer) / len(
            self.P_gs_total_15min_buffer
        )

        return virtual_state

    def calc_reward(self, action):
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

        def runtime_counter(u_old, u_new, old_runtime):
            if u_old == 0 and u_new == 0:  # off
                new_runtime = 0
            elif u_old <= 0 and u_new > 0:  # turned on
                new_runtime = 0
            elif u_old > 0 and u_new > 0:  # on
                new_runtime = old_runtime + self.sampling_time
            elif u_old > 0 and u_new <= 0:  # turned off
                new_runtime = old_runtime + self.sampling_time
            else:
                raise NotImplemented
            return new_runtime

        def complex_switch_cost(u_old, u_new, runtime, min_runtime, penalty):
            if u_old > 0 and u_new <= 0 and runtime < min_runtime:  # system turned off
                adjusted_penalty = ((min_runtime - runtime) / min_runtime) * penalty

                return adjusted_penalty  # return penalty when min runtime is not achieved

            else:
                return 0

        # switching costs
        total_switch_costs = (
            -switch_cost(
                # bSetStatusOn_HeatExchanger1
                self.action_backup[0],
                action[0],
                self.poweron_cost_HeatExchanger1,
            )
            - switch_cost(
                # bSetStatusOn_CHP1
                self.action_backup[1],
                action[1],
                self.poweron_cost_CHP1,
            )
            - switch_cost(
                # bSetStatusOn_CHP2
                self.action_backup[2],
                action[2],
                self.poweron_cost_CHP2,
            )
            - switch_cost(
                # bSetStatusOn_CondensingBoiler
                self.action_backup[3],
                action[3],
                self.poweron_cost_CondensingBoiler,
            )
            - switch_cost(
                # bSetStatusOn_VSIStorage
                self.action_backup[4],
                action[4],
                self.poweron_cost_VSIStorage,
            )
            - switch_cost(
                # bLoading_VSISystem
                self.action_backup[5],
                action[5],
                self.switching_cost_bLoading_VSISystem,
            )
            - switch_cost(
                # bSetStatusOn_HVFASystem_HNLT
                self.action_backup[6],
                action[6],
                self.poweron_cost_HVFASystem_HNLT,
            )
            - switch_cost(
                # bLoading_HVFASystem_HNLT
                self.action_backup[7],
                action[7],
                self.switching_cost_bLoading_HVFASystem_HNLT,
            )
            - switch_cost(
                # bSetStatusOn_eChiller
                self.action_backup[8],
                action[8],
                self.poweron_cost_eChiller,
            )
            - switch_cost(
                # bSetStatusOn_HVFASystem_CN
                self.action_backup[9],
                action[9],
                self.poweron_cost_HVFASystem_CN,
            )
            - switch_cost(
                # bLoading_HVFASystem_CN
                self.action_backup[10],
                action[10],
                self.switching_cost_bLoading_HVFASystem_CN,
            )
            - switch_cost(
                # bSetStatusOn_OuterCapillaryTubeMats
                self.action_backup[11],
                action[11],
                self.poweron_cost_OuterCapillaryTubeMats,
            )
            - switch_cost(
                # bSetStatusOn_HeatPump
                self.action_backup[12],
                action[12],
                self.poweron_cost_HeatPump,
            )
        )

        CHP1_runtime = runtime_counter(
            u_old=self.action_backup[1], u_new=action[1], old_runtime=self.runtime_counter["CHP1"]
        )
        CHP2_runtime = runtime_counter(
            u_old=self.action_backup[2], u_new=action[2], old_runtime=self.runtime_counter["CHP2"]
        )
        HeatPump_runtime = runtime_counter(
            u_old=self.action_backup[12], u_new=action[12], old_runtime=self.runtime_counter["HeatPump"]
        )
        eChiller_runtime = runtime_counter(
            u_old=self.action_backup[8], u_new=action[8], old_runtime=self.runtime_counter["eChiller"]
        )
        AFA_runtime = runtime_counter(
            u_old=self.action_backup[11], u_new=action[11], old_runtime=self.runtime_counter["OuterCapillaryTubeMats"]
        )

        self.runtime_counter.update({"CHP1": CHP1_runtime})
        self.runtime_counter.update({"CHP2": CHP2_runtime})
        self.runtime_counter.update({"HeatPump": HeatPump_runtime})
        self.runtime_counter.update({"eChiller": eChiller_runtime})
        self.runtime_counter.update({"OuterCapillaryTubeMats": AFA_runtime})

        # print(self.runtime_counter)

        runtime_violation_costs = (
            -complex_switch_cost(
                u_old=self.action_backup[1],
                u_new=action[1],
                runtime=self.runtime_counter["CHP1"],
                min_runtime=3 * 60 * 60 * 100,  # 3 hours minimum runtime on average
                penalty=self.runtime_violation_cost_CHP1,
            )
            - complex_switch_cost(
                u_old=self.action_backup[2],
                u_new=action[2],
                runtime=self.runtime_counter["CHP2"],
                min_runtime=3 * 60 * 60 * 100,  # 3 hours minimum runtime on average
                penalty=self.runtime_violation_cost_CHP2,
            )
            - complex_switch_cost(
                u_old=self.action_backup[12],
                u_new=action[12],
                runtime=self.runtime_counter["HeatPump"],
                min_runtime=20 * 60 * 100,  # 20 min minimum runtime on average
                penalty=self.runtime_violation_cost_HeatPump,
            )
            - complex_switch_cost(
                u_old=self.action_backup[8],
                u_new=action[8],
                runtime=self.runtime_counter["eChiller"],
                min_runtime=20 * 60 * 100,  # 20 min minimum runtime on average
                penalty=self.runtime_violation_cost_eChiller,
            )
            - complex_switch_cost(
                u_old=self.action_backup[11],
                u_new=action[11],
                runtime=self.runtime_counter["OuterCapillaryTubeMats"],
                min_runtime=10 * 60 * 100,  # 10 min minimum runtime on average
                penalty=self.runtime_violation_cost_OuterCapillaryTubeMats,
            )
        )

        self.state["reward_switching"] = total_switch_costs + runtime_violation_costs

        # after switching costs are calculated, action backup can be overwritten
        self.action_backup = action

        # temperature costs (when availability of temperature levels are needed)
        self.state["reward_temperature_HNHT"] = reward_boundary(
            self.state[
                "HNHT_Buffer_fMidTemperature"
            ],  # ToDO: only use upper Temperature or rather use Mid Temperature?
            self.HNHT_temperature_reward_min_T,
            self.HNHT_temperature_reward_max_T,
            self.temperature_reward_HNHT,
            self.temperature_cost_HNHT,
            smoothed=self.reward_shaping,
            k=6,
        )
        self.state["reward_temperature_HNLT"] = reward_boundary(
            self.state[
                "HNLT_Buffer_fMidTemperature"
            ],  # ToDO: only use upper Temperature or rather use Mid Temperature?
            self.HNLT_temperature_reward_min_T,
            self.HNLT_temperature_reward_max_T,
            self.temperature_reward_HNLT,
            self.temperature_cost_HNLT,
            smoothed=self.reward_shaping,
            k=6,
        )
        self.state["reward_temperature_CN"] = reward_boundary(
            self.state["CN_Buffer_fMidTemperature"],  # ToDO: only use upper Temperature or rather use Mid Temperature?
            self.CN_temperature_reward_min_T,
            self.CN_temperature_reward_max_T,
            self.temperature_reward_CN,
            self.temperature_cost_CN,
            smoothed=self.reward_shaping,
            k=6,
        )

        # here the power consumption is calculated by substracting the consumption of production systems
        # base_power_electric = (
        #     self.state["s_electric_power_total"] - self.state["d_production_electric_power"]
        # )  # total consumption of supply systems O
        # base_power_gas = self.state["s_gas_power_total"] - self.state["d_production_gas_power"]

        self.state["reward_energy_electric"] = (
            -self.state["s_price_electricity_00h"]
            * self.state["electric_power_consumption"]
            * self.sampling_time
            / 3600
            / 1000
        )
        self.state["reward_energy_gas"] = (
            -self.state["s_price_gas"] * self.state["gas_power_consumption"] * self.sampling_time / 3600 / 1000
        )

        # energy consumed for plots
        self.state["energy_electric_consumed"] = (
            self.state["electric_power_consumption"] * self.sampling_time / 3600 / 1000
        )
        self.state["energy_gas_consumed"] = self.state["gas_power_consumption"] * self.sampling_time / 3600 / 1000

        """
        # energy taxes costs
        tax_el_per_kwh = self.tax_el_per_kwh  # [€/kWh]
        tax_el_produced_per_kwh = self.tax_el_produced_per_kwh  # [€/kWh]
        tax_gas_per_kwh = self.tax_gas_per_kwh  # [€/kWh]

        # taxes on electricity consumption
        if self.state["s_P_el_combinedheatpower"] < 0:
            if self.state["s_electric_power_total"] > 0:
                tax_el = (
                    (self.state["electric_power_consumption"] - self.state["s_P_el_combinedheatpower"])
                    * tax_el_per_kwh
                    * self.sampling_time
                    / 3600
                    / 1000
                )
                tax_el -= (
                    abs(self.state["s_P_el_combinedheatpower"])
                    * (tax_el_per_kwh - tax_el_produced_per_kwh)
                    * self.sampling_time
                    / 3600
                    / 1000
                )
            else:
                tax_el = (
                    -(abs(self.state["s_electric_power_total"]) - abs(self.state["electric_power_consumption"]))
                    * (tax_el_per_kwh - tax_el_produced_per_kwh)
                    * self.sampling_time
                    / 3600
                    / 1000
                )
        else:
            tax_el = self.state["electric_power_consumption"] * tax_el_per_kwh * self.sampling_time / 3600 / 1000

        # taxes on gas consumption
        tax_gs = tax_gas_per_kwh * self.state["gas_power_consumption"] * self.sampling_time / 3600 / 1000  # [€]

        # total energy taxes
        self.state["reward_energy_taxes"] = -tax_el - tax_gs  # [€]

        """

        self.state["reward_energy_taxes"] = 0  # ToDO: check if it is possible to lock at P_el from CHPs

        # power costs for peak load pricing
        # update peak electric load, when the average load of the past 15 mintes is greater then the last max_limit
        if self.state["vs_electric_power_total_15min"] > self.max_limit:
            peak_difference = (self.state["vs_electric_power_total_15min"] - self.max_limit) / 1000  # [kW]
            self.state["reward_power_electric"] = -1 * (peak_difference * self.peak_cost_per_kw)  # [€]
            self.max_limit = self.state["vs_electric_power_total_15min"]  # update value
        else:
            self.state["reward_power_electric"] = 0

        # other costs
        if self.abort_cost_for_unsuccessfull_step == True:
            self.state["reward_other"] = (
                -self.abort_costs * (1 - 0.5 * self.n_steps / self.n_episode_steps)
                if (self.state["step_abort"] or not self.state["step_success"])
                else 0
            )
        else:
            self.state["reward_other"] = 0

        # policyshaping costs
        if self.policy_shaping_active:
            self.state["reward_other"] -= self.policyshaping_costs

        # total reward
        self.state["reward_total"] = (
            self.state["reward_switching"]
            + self.state["reward_temperature_HNHT"]
            + self.state["reward_temperature_HNLT"]
            + self.state["reward_temperature_CN"]
            + self.state["reward_energy_electric"]
            + self.state["reward_energy_gas"]
            + self.state["reward_energy_taxes"]
            + self.state["reward_power_electric"]
            + self.state["reward_other"]
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

        # delete long time storage, since it takes up too much memory during training
        self.state_log_longtime = []

        # get current time
        self.startTime = datetime.now()  

        # # save episode's stats
        if self.n_steps > 0:

            # create dataframe from state_log
            self.episode_df = pd.DataFrame(self.state_log)

            # The problem is, that for the actions, there is no value for the first time step, so we bfill the generated NaNs
            self.episode_df.bfill(inplace=True)

            # derive certain episode statistics for logging and plotting
            self.episode_statistics["datetime_begin"] = self.ts_current.index[0].strftime("%Y-%m-%d %H:%M")
            self.episode_statistics["rewards_total"] = self.episode_df["reward_total"].sum()
            self.episode_statistics["rewards_switching"] = self.episode_df["reward_switching"].sum()
            self.episode_statistics["reward_temperature_HNHT"] = self.episode_df["reward_temperature_HNHT"].sum()
            self.episode_statistics["reward_temperature_HNLT"] = self.episode_df["reward_temperature_HNLT"].sum()
            self.episode_statistics["reward_temperature_CN"] = self.episode_df["reward_temperature_CN"].sum()
            self.episode_statistics["rewards_energy_electric"] = self.episode_df["reward_energy_electric"].sum()
            self.episode_statistics["rewards_energy_gas"] = self.episode_df["reward_energy_gas"].sum()
            self.episode_statistics["rewards_energy_taxes"] = self.episode_df["reward_energy_taxes"].sum()
            self.episode_statistics["rewards_power_electric"] = self.episode_df["reward_power_electric"].sum()
            self.episode_statistics["rewards_other"] = self.episode_df["reward_other"].sum()
            self.episode_statistics["energy_electric_consumed"] = self.episode_df["energy_electric_consumed"].sum()
            self.episode_statistics["energy_gas_consumed"] = self.episode_df["energy_gas_consumed"].sum()
            self.episode_statistics["time"] = time.time() - self.episode_timer
            self.episode_statistics["n_steps"] = self.n_steps
            self.episode_statistics["power_electric_max"] = self.episode_df["vs_electric_power_total_15min"].max()
            # self.episode_statistics["temp_heat_min"] = self.episode_df["s_temp_heat_storage_hi"].min()
            # self.episode_statistics["temp_heat_max"] = self.episode_df["s_temp_heat_storage_hi"].max()

            # # append episode store to episodes' archive
            self.episode_archive.append(list(self.episode_statistics.values()))

        # initialize additional state, since super().reset is used
        self.additional_state = {}

        # set FMU parameters for initialization
        self.model_parameters = {}

        # randomly parametrize systems
        if "none" not in self.variance_parameters:
            # CN
            self.model_parameters["HydraulischeSwitch_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["SV146_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV146_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["PU_HeatPump_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, 1.0
            )  # max 1!
            self.model_parameters["HeatPump_variance_P"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["HeatPump_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)

            # HNLT
            self.model_parameters["HydraulicSwitch_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV605_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV605_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV235_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV235_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV246_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV246_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV_XX_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV_XX_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV600_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV600_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["PWT6_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["PU600_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["PU235_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["PU_HeatPump_HNLT_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, 1.0
            )  # max 1!
            self.model_parameters["PU215_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["aFA_simple_2_1_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)

            # Container
            self.model_parameters["HVFA_CN_795_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["HVFA_CN_796_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["HVFA_HNLT_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV138_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV138_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV_CN_HVFA_pressuredrop_variance_dp"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["SV_CN_HVFA_pressuredrop_variance_riseTime"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["SV_HNLT_HVFA_AFA_Correction_variance_dp"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["SV_HNLT_HVFA_AFA_Correction_variance_riseTime"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["SV_HNLT_HVFA_pressuredrop_variance_dp"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["SV_HNLT_HVFA_pressuredrop_variance_riseTime"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["RV105_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV105_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV105_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV105_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["Consumer_Producer_Switch_variance_dp"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["Consumer_Producer_Switch_variance_riseTime"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["Consumer_Producer_Switch1_variance_dp"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["Consumer_Producer_Switch1_variance_riseTime"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["SV106_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV106_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV205_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV205_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV205_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV205_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV206_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV206_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["PWT4_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["PWT5_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["PU138_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, 1.0
            )  # max 1!
            self.model_parameters["PU105_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["PU205_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["eChiller_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)

            # HNHT
            self.model_parameters["VSI_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["HydraulicSwitch_HNHT_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["SV305_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV305_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV315_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV315_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV331_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV331_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV322_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV322_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV321_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV321_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV307_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV307_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV306_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["SV306_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV215_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV215_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV322_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV322_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV321_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["RV321_variance_riseTime"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["PWT1_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["PU315_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["PU307_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["PU306_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["PU331_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, 1.0
            )  # max 1!
            self.model_parameters["PU322_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, 1.0
            )  # max 1!
            self.model_parameters["PU321_OperatingStrategy_variance"] = self.np_random.uniform(
                self.variance_min, 1.0
            )  # max 1!
            self.model_parameters["CondensingBoiler_variance_P"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["CondensingBoiler_variance_dp"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["CHP1_variance_P"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["CHP1_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["CHP2_variance_P"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["CHP2_variance_dp"] = self.np_random.uniform(self.variance_min, self.variance_max)

            # Check if parameters have to be changed or should stay 1.0
            if "all" not in self.variance_parameters:
                self.model_parameters = {
                    key: (1.0 if key not in self.variance_parameters else value)
                    for key, value in self.model_parameters.items()
                }

            # Load fixed parameterset, if available
            parameter_set = os.path.join(self.path_results, "fixed_parameterset.json")
            try:
                with open(parameter_set) as json_file:
                    # Load the dictionary from the file
                    self.model_parameters = json.load(json_file)
                log.info("Fixed parameter set found and loaded.")
            except FileNotFoundError:
                log.info("Fixed parameter set not found, taking random parameters.")
            log.info("Current model parameters:\n\t" + str(self.model_parameters))

            # save current parameterset
            current_parameterset_path = os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes + int(self.initial_resets > 0)).zfill(4)
                + "-"
                + str(self.env_id).zfill(2)
                + "_parameters.json",
            )
            with open(current_parameterset_path, "w") as json_file:
                # Save the dictionary to the file
                json.dump(self.model_parameters, json_file)

        # set the start temperatures for the different networks
        self.model_parameters["T_start_HNHT"] = self.np_random.uniform(
            self.temperature_HNHT_Buffer_init_min, self.temperature_HNHT_Buffer_init_max
        )
        self.model_parameters["T_start_VSI"] = self.np_random.uniform(
            self.temperature_HNHT_VSI_init_min, self.temperature_HNHT_VSI_init_max
        )

        self.model_parameters["T_start_HNLT"] = self.np_random.uniform(
            self.temperature_HNLT_Buffer_init_min, self.temperature_HNLT_Buffer_init_max
        )

        # AFA Glycol side is init later

        self.model_parameters["T_start_HVFA_HNLT"] = self.np_random.uniform(
            self.temperature_HNLT_HVFA_init_min, self.temperature_HNLT_HVFA_init_max
        )

        self.model_parameters["T_start_CN"] = self.np_random.uniform(
            self.temperature_CN_Buffer_init_min, self.temperature_CN_Buffer_init_max
        )
        self.model_parameters["T_start_HVFA_CN"] = self.np_random.uniform(
            self.temperature_CN_HVFA_init_min, self.temperature_CN_HVFA_init_max
        )

        # get current slice of timeseries dataframe, extended by maximum prediction horizon (6h)
        # and one additional step because step 0 = init conditions
        self.ts_current = timeseries.df_time_slice(
            self.scenario_data,
            self.scenario_time_begin,
            self.scenario_time_end,
            self.episode_duration + (self.n_steps_6h + 1) * self.sampling_time,
            random=self.np_random if self.random_sampling else False,
        )

        # Now, when the current time is known, init the Glycol side of the AFA with the current outside temperature

        if self.temperature_AFA_init_fixed == True:  # for playing use fixed AFA temperature to allow fair comparison
            self.model_parameters["T_start_AFA_Glycol"] = self.np_random.uniform(
                self.ts_current["air_temperature"].iloc[0] + 273.15, self.ts_current["air_temperature"].iloc[0] + 273.15
            )
        else:
            temperature_AFA_Glycol_init_max = max(
                self.ts_current["air_temperature"].iloc[0] + 273.15, self.temperature_HNLT_Buffer_init_max
            )
            self.model_parameters["T_start_AFA_Glycol"] = self.np_random.uniform(
                self.ts_current["air_temperature"].iloc[0] + 273.15, temperature_AFA_Glycol_init_max
            )  # AFA can be between outside temp and HNLT Buffer temp

        # read current date time
        self.episode_datetime_begin = self.ts_current.index[0]
        self.additional_state["time_daytime"] = self.episode_datetime_begin.hour

        # reset virtual states and internal counters
        self.P_el_total_15min_buffer = []
        self.P_gs_total_15min_buffer = []
        self.additional_state["vs_electric_power_total_15min"] = 0
        self.additional_state["vs_gas_power_total_15min"] = 0

        # get scenario input for initialization (time step: 0)
        self.additional_state.update(self.update_predictions())

        # add time of the year in seconds
        # starttime_of_year = pd.Timestamp(str(self.ts_current.index[self.n_steps].year) + "-01-01 00:00")
        # self.additional_state["d_weather_time"] = pd.Timedelta(
        #     self.ts_current.index[self.n_steps] - starttime_of_year
        # ).total_seconds()

        # reset maximal peak electric power for penalty costs (necessary for peak shaving)
        self.max_limit = self.power_cost_max

        # reset RNG, hack to work around the current issue of non deterministic seeding of first episode
        if self.initial_resets == 0:
            self._np_random = None
        self.initial_resets += 1

        # receive observations from simulation
        observations = super().reset(seed=seed)

        return observations

    def convert_disc_action(self, action_disc):
        """
        !! This function is not needed here, since discrete inputs are used in FMU !!

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
            decimal=",",
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
        episode_archive_df.tail(1).to_csv(path_or_buf=csvpath, sep=";", decimal=",", mode=tocsvmode, header=tocsvheader)

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
            x,
            y["rewards_energy_electric"].to_numpy(),
            label="Strom (netto)",
            color=(1.0, 0.75, 0.0),
            linewidth=1,
            alpha=0.9,
        )
        axes[0].plot(
            x,
            y["rewards_energy_gas"].to_numpy(),
            label="Erdgas (netto)",
            color=(0.65, 0.65, 0.65),
            linewidth=1,
            alpha=0.9,
        )
        axes[0].plot(
            x,
            y["rewards_energy_taxes"].to_numpy(),
            label="Steuern & Umlagen",
            color=(0.184, 0.333, 0.592),
            linewidth=1,
            alpha=0.9,
        )
        axes[0].plot(
            x,
            y["rewards_power_electric"].to_numpy(),
            label="el. Lastspitzen",
            color=(0.929, 0.49, 0.192),
            linewidth=1,
            alpha=0.9,
        )
        axes[0].set_ylabel("kum. Kosten [€]")
        axes[0].set_xlabel("Episode")
        axes[0].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[0].margins(x=0.0, y=0.1)
        axes[0].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)

        # (2) Rewards
        cost_total = (
            y["rewards_energy_electric"]
            + y["rewards_energy_gas"]
            + y["rewards_energy_taxes"]
            + y["rewards_power_electric"]
        )
        axes[1].plot(x, cost_total.to_numpy(), label="Kosten", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9)
        axes[1].plot(
            x, y["reward_temperature_HNHT"].to_numpy(), label="HNHT", color=(0.75, 0, 0), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x, y["reward_temperature_HNLT"].to_numpy(), label="HNLT", color=(0, 0.75, 0.25), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x, y["reward_temperature_CN"].to_numpy(), label="CN", color=(0.36, 0.61, 0.84), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x,
            y["rewards_switching"].to_numpy(),
            label="Schaltvorgänge",
            color=(0.44, 0.19, 0.63),
            linewidth=1,
            alpha=0.9,
        )
        axes[1].plot(x, y["rewards_other"].to_numpy(), label="Sonstige", color=(0.1, 0.1, 0.1), linewidth=1, alpha=0.9)
        axes[1].plot(x, y["rewards_total"].to_numpy(), label="Gesamt", color=(0.1, 0.1, 0.1), linewidth=2)
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
            decimal=",",
        )

        # create figure and axes with custom layout/adjustments
        figure = plt.figure(figsize=(14, 22), dpi=200)
        axes = []
        axes.append(figure.add_subplot(5, 1, 1))
        axes.append(figure.add_subplot(5, 1, 2, sharex=axes[0]))
        axes.append(figure.add_subplot(10, 1, 5, sharex=axes[0]))
        axes.append(figure.add_subplot(10, 1, 6, sharex=axes[0]))
        axes.append(figure.add_subplot(5, 1, 4, sharex=axes[0]))
        axes.append(figure.add_subplot(10, 1, 9, sharex=axes[0]))
        axes.append(figure.add_subplot(10, 1, 10, sharex=axes[0]))
        # axes.append(figure.add_subplot(10, 1, 3, sharex=axes[0]))
        # axes.append(figure.add_subplot(10, 1, 7, sharex=axes[0]))
        plt.tight_layout()
        figure.subplots_adjust(left=0.125, bottom=0.05, right=0.9, top=0.95, wspace=0.2, hspace=0.05)

        # set x/y axe and datetime begin
        x = self.episode_df.index.to_numpy()
        y = self.episode_df
        dt_begin = self.episode_datetime_begin
        sampling_time = self.sampling_time

        # (1) - Plot actions as heatmap
        axes[0].set_yticks(np.arange(len(self.state_config.actions)))
        axes[0].set_yticklabels(
            [
                "PWT 1",
                "CHP1",
                "CHP2",
                "Gas Boiler",
                "VSI On/Off",
                "VSI Loading",
                "HVFA HNLT On/Off",
                "HVFA HNLT Loading",
                "eChiller",
                "HVFA CN On/Off",
                "HVFA CN Loading",
                "AFA",
                "Wärmepumpe",
            ]
        )
        # the follwing actions are changed from bool to int
        im = axes[0].imshow(
            y[self.state_config.actions].astype(int).transpose(),
            cmap="Reds",
            vmin=0,
            vmax=1,
            aspect="auto",
            interpolation="none",
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
                ticknames.append(tickdate.strftime("%d.%m.'%y"))
        # Let the horizontal axes labeling appear on top
        axes[0].tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, rotation=45)
        axes[0].set_xlabel("Zeit (UTC)")
        axes[0].xaxis.set_label_position("top")
        # ax.set_xticks(np.arange(df1.shape[1]+1)-.5, minor=True)
        axes[0].set_yticks(np.arange(len(self.state_config.actions) + 1) - 0.5, minor=True)
        axes[0].tick_params(which="minor", bottom=False, left=False)
        # grid settings
        axes[0].grid(which="minor", color="w", linestyle="-", linewidth=3)
        axes[0].xaxis.grid(color=(1, 1, 1, 0.1), linestyle="-", linewidth=1)
        # add ticks and tick labels
        axes[0].set_xticks(tickpos)
        axes[0].set_xticklabels(np.array(ticknames))
        # Rotate the tick labels and set their alignment.
        plt.setp(axes[0].get_yticklabels(), rotation=30, ha="right", va="center", rotation_mode="anchor")

        # (2) - Plot Storages

        ##### HNHT #####
        axes[1].plot(
            x,
            np.array([self.HNHT_temperature_reward_min_T] * len(x)),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle=":",
            label="HNHT T_min",
        )
        axes[1].plot(
            x,
            np.array([self.HNHT_temperature_reward_max_T] * len(x)),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle=":",
            label="HNHT T_max",
        )

        axes[1].plot(
            x,
            y["HNHT_Buffer_fUpperTemperature"].to_numpy(),
            color=(192 / 255, 0, 0),
            linestyle="--",
            label="HNHT Buffer (oben)",
        )
        axes[1].plot(
            x, y["HNHT_Buffer_fMidTemperature"].to_numpy(), color=(192 / 255, 0, 0), label="HNHT Buffer (mitte)"
        )
        axes[1].plot(
            x,
            y["HNHT_Buffer_fLowerTemperature"].to_numpy(),
            color=(192 / 255, 0, 0),
            linestyle="--",
            label="HNHT Buffer (unten)",
        )

        axes[1].plot(
            x,
            y["HNHT_VSI_fUpperTemperature"].to_numpy(),
            color=(230 / 255, 125 / 255, 60 / 255),
            linestyle="--",
            label="VSI Speicher (oben)",
        )
        axes[1].plot(
            x,
            y["HNHT_VSI_fMidTemperature"].to_numpy(),
            color=(230 / 255, 125 / 255, 60 / 255),
            label="VSI Speicher (mitte)",
        )
        axes[1].plot(
            x,
            y["HNHT_VSI_fLowerTemperature"].to_numpy(),
            color=(230 / 255, 125 / 255, 60 / 255),
            linestyle="--",
            label="VSI Speicher (unten)",
        )

        # settings
        axes[1].set_ylabel("Temperatur [°C]")
        axes[1].margins(x=0.0, y=0.1)
        axes[1].set_axisbelow(True)
        axes[1].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[1].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        ##### HNLT #####
        axes[2].plot(
            x,
            np.array([self.HNLT_temperature_reward_min_T] * len(x)),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle=":",
            label="HNLT T_min",
        )
        axes[2].plot(
            x,
            np.array([self.HNLT_temperature_reward_max_T] * len(x)),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle=":",
            label="HNLT T_max",
        )

        axes[2].plot(
            x,
            y["HNLT_Buffer_fUpperTemperature"].to_numpy(),
            color=(70 / 255, 195 / 255, 60 / 255),
            linestyle="--",
            label="HNLT Buffer (oben)",
        )
        axes[2].plot(
            x,
            y["HNLT_Buffer_fMidTemperature"].to_numpy(),
            color=(70 / 255, 195 / 255, 60 / 255),
            label="HNLT Buffer (mitte)",
        )
        axes[2].plot(
            x,
            y["HNLT_Buffer_fLowerTemperature"].to_numpy(),
            color=(70 / 255, 195 / 255, 60 / 255),
            linestyle="--",
            label="HNLT Buffer (unten)",
        )

        axes[2].plot(
            x,
            y["HNLT_HVFA_fUpperTemperature"].to_numpy(),
            color=(180 / 255, 230 / 255, 80 / 255),
            label="HNLT HVFA (oben)",
        )

        axes[2].plot(
            x,
            y["HNLT_HVFA_fLowerTemperature"].to_numpy(),
            color=(180 / 255, 230 / 255, 80 / 255),
            linestyle="--",
            label="HNLT HVFA (unten)",
        )

        # settings
        axes[2].set_ylabel("Temperatur [°C]")
        axes[2].margins(x=0.0, y=0.1)
        axes[2].set_axisbelow(True)
        axes[2].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        axes[2].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[2].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        ##### CN #####
        axes[3].plot(
            x,
            np.array([self.CN_temperature_reward_min_T] * len(x)),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle=":",
            label="CN T_min",
        )
        axes[3].plot(
            x,
            np.array([self.CN_temperature_reward_max_T] * len(x)),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle=":",
            label="CN T_max",
        )

        axes[3].plot(
            x,
            y["CN_Buffer_fUpperTemperature"].to_numpy(),
            color=(91 / 255, 155 / 255, 213 / 255),
            linestyle="--",
            label="CN Buffer (oben)",
        )
        axes[3].plot(
            x,
            y["CN_Buffer_fMidTemperature"].to_numpy(),
            color=(91 / 255, 155 / 255, 213 / 255),
            label="CN Buffer (mitte)",
        )
        axes[3].plot(
            x,
            y["CN_Buffer_fLowerTemperature"].to_numpy(),
            color=(91 / 255, 155 / 255, 213 / 255),
            linestyle="--",
            label="CN Buffer (unten)",
        )

        axes[3].plot(
            x,
            y["CN_HVFA_fUpperTemperature"].to_numpy(),
            color=(0 / 255, 200 / 255, 200 / 255),
            label="CN HVFA (oben)",
        )

        axes[3].plot(
            x,
            y["CN_HVFA_fLowerTemperature"].to_numpy(),
            color=(0 / 255, 200 / 255, 200 / 255),
            linestyle="--",
            label="CN HVFA (unten)",
        )

        # settings
        axes[3].set_ylabel("Temperatur [°C]")
        axes[3].margins(x=0.0, y=0.1)
        axes[3].set_axisbelow(True)
        axes[3].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        axes[3].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[3].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # heat demand production
        axes[4].plot(
            x,
            y["d_HNHT_prod_heat_demand_consumer"].to_numpy() / 1000,
            color=(192 / 255, 0, 0),
            label="HNHT Verbraucher",
        )
        # axes[4].plot(x,y["d_HNHT_prod_heat_demand_producer"]/1000,
        #     color=(192 / 255, 0, 0),linestyle="--",label="HNHT Erzeuger",
        # )

        axes[4].plot(
            x,
            y["d_HNLT_prod_heat_demand_consumer"].to_numpy() / 1000,
            color=(70 / 255, 195 / 255, 60 / 255),
            label="HNLT Verbraucher",
        )
        axes[4].plot(
            x,
            y["d_HNLT_prod_heat_demand_producer"].to_numpy() / 1000,
            color=(70 / 255, 195 / 255, 60 / 255),
            linestyle="--",
            label="HNLT Erzeuger",
        )
        axes[4].plot(
            x,
            y["d_CN_prod_heat_demand_consumer"].to_numpy() / 1000,
            color=(91 / 255, 155 / 255, 213 / 255),
            label="CN Verbraucher",
        )

        axes[4].set_ylabel("Wärmebedarf [kW]")
        axes[4].margins(x=0.0, y=0.1)
        axes[4].set_axisbelow(True)
        axes[4].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        axes[4].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[4].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # # (5) - T_amb and T_ground

        axes[5].plot(
            x,
            y["weather_T_amb"].to_numpy(),
            color=(255 / 255, 128 / 255, 0 / 255),
            label="Außentemperatur",
        )
        axes[5].plot(
            x,
            y["weather_T_amb_Mean"].to_numpy(),
            color=(100 / 255, 50 / 255, 0 / 255),
            linestyle="--",
            label="Außentemperatur Mean 6h",
        )
        axes[5].plot(
            x,
            y["weather_T_Ground_1m"].to_numpy(),
            color=(204 / 255, 0 / 255, 204 / 255),
            label="Bodentemperatur (1m)",
        )

        axes[5].set_ylabel("Temperatur [°C]")
        axes[5].margins(x=0.0, y=0.1)
        axes[5].set_axisbelow(True)
        axes[5].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        axes[5].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[5].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # # (2) - Battery and relative Humidity (%)
        # prodState = y["d_time_till_availability"].copy()
        # prodState[prodState > 0] = 0
        # prodState[prodState < 0] = 10
        # axes[2].fill_between(x, prodState, color=(0.1, 0.1, 0.1), linewidth=0.1, alpha=0.3, label="Prod. Modus")
        # axes[2].fill_between(
        #     x,
        #     y["d_weather_relativehumidity"],
        #     color=(0.44, 0.68, 0.28),
        #     linewidth=0.1,
        #     alpha=0.3,
        #     label="Außenluftfeuchte",
        # )

        # # settings
        # axes[2].set_ylabel("Zustand [%]")
        # axes[2].margins(x=0.0, y=0.1)
        # axes[2].set_axisbelow(True)
        # axes[2].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        # axes[2].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        # axes[2].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # (3) - Prices
        # axes[6].plot(x, y["s_price_electricity"], color=(1.0, 0.75, 0.0), label="Strom")
        # axes[6].plot(x, y["s_price_gas"], color=(0.65, 0.65, 0.65), label="Erdgas")
        # axes[6].set_ylabel("Energiepreis (netto) [€/kWh]")
        # axes[6].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        # axes[6].margins(x=0.0, y=0.1)
        # axes[6].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        # axes[6].set_axisbelow(True)
        # axes[6].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)

        # (7) - Costs
        # axes[6].plot(
        #     x,
        #     y["reward_energy_electric"].cumsum(),
        #     label="Strom (netto)",
        #     color=(1.0, 0.75, 0.0),
        #     linewidth=1,
        #     alpha=0.9,
        # )
        # axes[6].plot(
        #     x, y["reward_energy_gas"].cumsum(), label="Erdgas (netto)", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9
        # )
        # axes[6].plot(
        #     x,
        #     y["reward_energy_taxes"].cumsum(),
        #     label="Steuern & Umlagen",
        #     color=(0.184, 0.333, 0.592),
        #     linewidth=1,
        #     alpha=0.9,
        # )
        # axes[6].plot(
        #     x,
        #     y["reward_power_electric"].cumsum(),
        #     label="el. Lastspitzen",
        #     color=(0.929, 0.49, 0.192),
        #     linewidth=1,
        #     alpha=0.9,
        # )
        # axes[6].set_ylabel("kum. Kosten [€]")
        # axes[6].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        # axes[6].margins(x=0.0, y=0.1)
        # axes[6].set_axisbelow(True)
        # axes[6].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        # axes[6].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        # axes[6].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # (8) Rewards
        cost_total = (
            y["reward_energy_electric"].cumsum()
            + y["reward_energy_gas"].cumsum()
            + y["reward_energy_taxes"].cumsum()
            + y["reward_power_electric"].cumsum()
        ).to_numpy()
        axes[6].plot(x, cost_total, label="Kosten", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9)
        axes[6].plot(
            x,
            y["reward_temperature_HNHT"].cumsum().to_numpy(),
            label="HNHT",
            color=(0.75, 0, 0),
            linewidth=1,
            alpha=0.9,
        )
        axes[6].plot(
            x,
            y["reward_temperature_HNLT"].cumsum().to_numpy(),
            label="HNLT",
            color=(50 / 255, 50 / 255, 50 / 255),
            linewidth=1,
            alpha=0.9,
        )
        axes[6].plot(
            x,
            y["reward_temperature_CN"].cumsum().to_numpy(),
            label="CN",
            color=(0.36, 0.61, 0.84),
            linewidth=1,
            alpha=0.9,
        )
        axes[6].plot(
            x,
            y["reward_switching"].cumsum().to_numpy(),
            label="Schaltvorgänge",
            color=(0.44, 0.19, 0.63),
            linewidth=1,
            alpha=0.9,
        )
        axes[6].plot(
            x, y["reward_other"].cumsum().to_numpy(), label="Sonstige", color=(0.1, 0.1, 0.1), linewidth=1, alpha=0.9
        )
        axes[6].plot(x, y["reward_total"].cumsum().to_numpy(), label="Gesamt", color=(0.1, 0.1, 0.1), linewidth=2)
        axes[6].set_ylabel("Rewards [€-äquiv.]")
        axes[6].set_xlabel("Zeit (UTC)")
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
        plt.close(figure)

        """
        if plotter_available:
            # HTML PLotter

            xaxis_title = "Zeit (UTC)"
            x2 = self.episode_df.index

            actions = Heatmap(x2, xaxis_title=xaxis_title, height=750, width=1900)
            actions.line(y["u_immersionheater"], name="Tauchsieder")
            actions.line(y["u_condensingboiler"], name="Gasbrennwertgerät")
            actions.line(y["u_combinedheatpower"], name="Blockheizkraftwerk")
            actions.line(y["u_heatpump"], name="Wärmepumpe")
            actions.line(y["u_coolingtower"], name="Kühlturm")
            actions.line(y["u_compressionchiller"], name="Kältemaschine")

            storages = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="Temperatur [°C]", height=750, width=1900)
            storages.line(
                [self.temperature_cost_prod_heat_min - 273.15] * len(x),
                "T heiß minimal",
                color="rgb(50,50,50)",
                dash="dash",
            )
            storages.line(
                [self.temperature_cost_prod_heat_max - 273.15] * len(x),
                "T heiß maximal",
                color="rgb(50,50,50)",
                dash="dash",
            )
            storages.line(y["s_temp_heat_storage_hi"] - 273.15, "Wärmespeicher (ob)", color="rgb(192,0,0)")
            storages.line(y["s_temp_heat_storage_lo"] - 273.15, "Wärmespeicher (un)", color="rgb(192,0,0)", dash="dash")
            storages.line(
                [self.temperature_cost_prod_cool_min - 273.15] * len(x),
                "T kalt minimal",
                color="rgb(50,50,50)",
                dash="dash",
            )
            storages.line(
                [self.temperature_cost_prod_cool_max - 273.15] * len(x),
                "T kalt maximal",
                color="rgb(50,50,50)",
                dash="dash",
            )
            storages.line(
                y["s_temp_cold_storage_hi"] - 273.15, "Wärmespeicher (ob)", color="rgb(91,155,213)", dash="dash"
            )
            storages.line(y["s_temp_cold_storage_lo"] - 273.15, "Wärmespeicher (un)", color="rgb(91,155,213)")
            storages.line(y["d_weather_drybulbtemperature"], "Außenluft", color="rgb(100,100,100)")

            humidity = Linegraph(
                x2, xaxis_title=xaxis_title, yaxis_title="Luftfeuchtigkeit [%]", height=350, width=1900
            )
            humidity.line(y["d_weather_relativehumidity"], "rel. Außenluftfeuchte", color="rgb(100,100,100)")

            prices = Linegraph(
                x2, xaxis_title=xaxis_title, yaxis_title="Energiepreis (netto) [€/kWh]", height=750, width=1900
            )
            prices.line(y["s_price_electricity"], "Strom", color="rgb(255,191,0)")
            prices.line(y["s_price_gas"], "Erdgas", color="rgb(165,165,165)")

            power = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="Leistung [kW]", height=750, width=1900)
            power.line(y["s_electric_power_total"] * 1e-3, "Strom Netz", color="rgb(255,191,0)")
            power.line(y["vs_electric_power_total_15min"] * 1e-3, "Strom Netz (Ø15m)", color="rgb(255,191,0)")
            power.line(y["vs_gas_power_total_15min"] * 1e-3, "Erdgas Netz (Ø15m)", color="rgb(165,165,165)")
            power.line(
                y["d_production_heat_power"] * 1e-3, "Wärmelast Prod.", width=1, dash="dash", color="rgb(191,0,0)"
            )
            power.line(
                y["d_production_cool_power"] * 1e-3, "Kältelast Prod.", width=1, dash="dash", color="rgb(92,156,214)"
            )
            power.line(
                y["d_production_electric_power"] * 1e-3, "Strom Prod.", width=1, dash="dash", color="rgb(255,191,0)"
            )
            power.line(
                y["d_production_gas_power"] * 1e-3, "Erdgas Prod.", width=1, dash="dash", color="rgb(165,165,165)"
            )
            power.line(
                (y["s_electric_power_total"] - y["d_production_electric_power"]) * 1e-3,
                "Strom TGA",
                width=0,
                dash="dash",
                color="rgb(255,191,0)",
                fill="tozeroy",
            )
            power.line(
                (y["s_gas_power_total"] - y["d_production_gas_power"]) * 1e-3,
                "Erdgas TGA",
                width=0,
                dash="dash",
                color="rgb(165,165,165)",
                fill="tozeroy",
            )

            costs = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="kum. Kosten [€]", height=500, width=1900)
            costs.line(y["reward_energy_electric"].cumsum(), "Strom (netto)", width=1, color="rgb(255,191,0)")
            costs.line(y["reward_energy_gas"].cumsum(), "Erdgas (netto)", width=1, color="rgb(165,165,165)")
            costs.line(y["reward_energy_taxes"].cumsum(), "Steuern & Umlagen", width=1, color="rgb(47,85,151)")
            costs.line(y["reward_power_electric"].cumsum(), "el. Lastspitzen", width=1, color="rgb(237,125,49)")

            rewards = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="Rewards [€-äquiv.]", height=500, width=1900)
            rewards.line(cost_total, "Kosten", width=1, color="rgb(165,165,165)")
            rewards.line(y["reward_temperature_heat"].cumsum(), "Wärmeversorgung", width=1, color="rgb(191,0,0)")
            rewards.line(y["reward_temperature_cool"].cumsum(), "Kälteversorgung", width=1, color="rgb(92,156,214)")
            rewards.line(y["reward_switching"].cumsum(), "Schaltvorgänge", width=1, color="rgb(112,48,160)")
            rewards.line(y["reward_other"].cumsum(), "Sonstige", width=1, color="rgb(25,25,25)")
            rewards.line(y["reward_total"].cumsum(), "Gesamt", color="rgb(25,25,25)")

            plot = ETA_Plotter(actions, storages, humidity, prices, power, costs, rewards)
            plot.plot_html(
                os.path.join(
                    self.path_results,
                    self.config_run.name
                    + "_"
                    + str(self.n_episodes).zfill(3)
                    + "-"
                    + str(self.env_id).zfill(2)
                    + "_episode"
                    + name_suffix
                    + ".html",
                )
            )
        """

        return

    def init_Hysterese_Controllers(self):
        fTargetTemperature_HNHT = 70
        fTargetTemperature_HNLT_Cooling = 40
        fTargetTemperature_HNLT_Heating = 35
        fTargetTemperature_CN = 18

        self.Controller_CHP_Prio = HysteresisController(hysteresis_range=4, target=15)
        self.Controller_CHP1 = HysteresisController(hysteresis_range=0, target=0)  # this is set later
        self.Controller_CHP2 = HysteresisController(hysteresis_range=0, target=0)  # this is set later
        self.Controller_CondensingBoiler = HysteresisController(hysteresis_range=10, target=fTargetTemperature_HNHT + 1)

        self.Controller_VSI_Unloading = HysteresisController(hysteresis_range=4, target=fTargetTemperature_HNHT - 0)

        self.Controller_VSI_Unloading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNHT, inverted=True
        )
        self.Controller_VSI_Loading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNHT
        )

        self.Controller_OuterCapillaryTubeMats = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Cooling + 2, inverted=True  # 42
        )  # cooling application
        self.Controller_OuterCapillaryTubeMats_Permission = HysteresisController(
            hysteresis_range=6, target=0
        )  # target is set later

        self.Controller_HeatExchanger1 = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Heating + 4
        )  # 39

        self.Controller_eChiller = HysteresisController(hysteresis_range=6, target=fTargetTemperature_CN, inverted=True)

        self.Controller_HeatPump = HysteresisController(hysteresis_range=5, target=fTargetTemperature_HNLT_Heating + 0)
        self.Controller_HeatPump_Permission = HysteresisController(
            hysteresis_range=3, target=fTargetTemperature_CN - 3, inverted=True
        )

        self.Controller_HVFA_CN_Loading_Permission = HysteresisController(
            hysteresis_range=2, target=fTargetTemperature_CN, inverted=True
        )
        self.Controller_HVFA_CN_Unloading_Permission = HysteresisController(
            hysteresis_range=2, target=fTargetTemperature_CN
        )
        self.Controller_Buffer_HVFA_CN_Loading = HysteresisController(hysteresis_range=2, target=fTargetTemperature_CN)
        self.Controller_Buffer_HVFA_CN_Unloading = HysteresisController(
            hysteresis_range=2, target=fTargetTemperature_CN, inverted=True
        )

        self.Controller_Controller_HVFA_HNLT_Loading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Cooling
        )
        self.Controller_Controller_HVFA_HNLT_Unloading_Permission = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Heating, inverted=True
        )
        self.Controller_Buffer_HVFA_HNLT_Loading = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Cooling, inverted=True
        )
        self.Controller_Buffer_HVFA_HNLT_Unloading = HysteresisController(
            hysteresis_range=4, target=fTargetTemperature_HNLT_Heating + 2
        )

    # def control_rules(self, observation: np.ndarray) -> np.ndarray:
    def control_rules(self, observation) -> np.ndarray:
        """
        Controller of the model.

        :param observation: Observation from the environment.
        :returns: actions
        """

        assert isinstance(observation, dict), "observation is not from type dict"

        # action = dict.fromkeys(self.action_names, 0) #this creates a dict with the action names and 0 as values
        action = {
            "bSetStatusOn_HeatExchanger1": 0,
            "bSetStatusOn_CHP1": 0,
            "bSetStatusOn_CHP2": 0,
            "bSetStatusOn_CondensingBoiler": 0,
            "bSetStatusOn_VSIStorage": 0,
            "bLoading_VSISystem": 0,
            "bSetStatusOn_HVFASystem_HNLT": 0,
            "bLoading_HVFASystem_HNLT": 0,
            "bSetStatusOn_eChiller": 0,
            "bSetStatusOn_HVFASystem_CN": 0,
            "bLoading_HVFASystem_CN": 0,
            "bSetStatusOn_OuterCapillaryTubeMats": 0,
            "bSetStatusOn_HeatPump": 0,
        }

        # get observations
        HNHT_Buffer_fMidTemperature = observation["HNHT_Buffer_fMidTemperature"]
        HNLT_Buffer_fMidTemperature = observation["HNLT_Buffer_fMidTemperature"]
        CN_Buffer_fMidTemperature = observation["CN_Buffer_fMidTemperature"]

        VSI_fUpperTemperature = observation["HNHT_VSI_fUpperTemperature"]
        VSI_fLowerTemperature = observation["HNHT_VSI_fLowerTemperature"]

        # CN_HVFA_fUpperTemperature = observation["CN_HVFA_fUpperTemperature"]
        CN_HVFA_fLowerTemperature = observation["CN_HVFA_fLowerTemperature"]
        HNLT_HVFA_fLowerTemperature = observation["HNLT_HVFA_fLowerTemperature"]

        T_Mean = observation["weather_T_amb_Mean"]
        T_amb = observation["weather_T_amb"]

        # define parameters
        fTargetTemperature_HNHT = 70

        # ask for controller outputs for actions which are more complex
        if self.Controller_CHP_Prio.update(actual_value=T_Mean) == 1:
            CHP1_hysteresis_range = 14
            CHP2_hysteresis_range = 12
            fOffset_TargetTemperature_CHP1 = 1
            fOffset_TargetTemperature_CHP2 = 0

            # change hysteresis settings
            self.Controller_CHP1.change_hysteresis_range(CHP1_hysteresis_range)
            self.Controller_CHP1.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP1)

            self.Controller_CHP2.change_hysteresis_range(CHP2_hysteresis_range)
            self.Controller_CHP2.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP2)
        else:
            CHP1_hysteresis_range = 12
            CHP2_hysteresis_range = 14
            fOffset_TargetTemperature_CHP1 = 0
            fOffset_TargetTemperature_CHP2 = 1
            self.Controller_CHP1.change_hysteresis_range(CHP1_hysteresis_range)
            self.Controller_CHP1.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP1)

            self.Controller_CHP2.change_hysteresis_range(CHP2_hysteresis_range)
            self.Controller_CHP2.change_target(fTargetTemperature_HNHT - fOffset_TargetTemperature_CHP2)

        # set actions
        action["bSetStatusOn_CHP1"] = self.Controller_CHP1.update(actual_value=HNHT_Buffer_fMidTemperature)
        action["bSetStatusOn_CHP2"] = self.Controller_CHP2.update(actual_value=HNHT_Buffer_fMidTemperature)

        action["bSetStatusOn_CondensingBoiler"] = self.Controller_CondensingBoiler.update(
            actual_value=HNHT_Buffer_fMidTemperature
        )

        # VSI
        Controller_Output_VSI_Unloading = self.Controller_VSI_Unloading.update(actual_value=HNHT_Buffer_fMidTemperature)
        Controller_Output_VSI_Unloading_Permission = self.Controller_VSI_Unloading_Permission.update(
            actual_value=VSI_fUpperTemperature
        )
        Controller_Output_VSI_Loading_Permission = self.Controller_VSI_Loading_Permission.update(
            actual_value=VSI_fLowerTemperature
        )

        if Controller_Output_VSI_Unloading and Controller_Output_VSI_Unloading_Permission:
            action["bSetStatusOn_VSIStorage"] = 1
        elif action["bSetStatusOn_CHP1"] == 1 or action["bSetStatusOn_CHP2"] == 1:
            if Controller_Output_VSI_Unloading == 0 and Controller_Output_VSI_Loading_Permission == 1:
                action["bSetStatusOn_VSIStorage"] = 1
            else:
                action["bSetStatusOn_VSIStorage"] = 0
        else:
            action["bSetStatusOn_VSIStorage"] = 0

        if Controller_Output_VSI_Unloading == 1 and Controller_Output_VSI_Unloading_Permission == 1:
            action["bLoading_VSISystem"] = 0
        else:
            action["bLoading_VSISystem"] = 1

        # HNHT-HNLT Linkage
        action["bSetStatusOn_HeatExchanger1"] = self.Controller_HeatExchanger1.update(
            actual_value=HNLT_Buffer_fMidTemperature
        )

        # HNLT
        Controller_Output_OuterCapillaryTubeMats = self.Controller_OuterCapillaryTubeMats.update(
            actual_value=HNLT_Buffer_fMidTemperature
        )
        self.Controller_OuterCapillaryTubeMats_Permission.change_target(HNLT_Buffer_fMidTemperature - 6)
        Controller_Output_OuterCapillaryTubeMats_Permission = self.Controller_OuterCapillaryTubeMats_Permission.update(
            actual_value=T_amb
        )
        if Controller_Output_OuterCapillaryTubeMats and Controller_Output_OuterCapillaryTubeMats_Permission:
            action["bSetStatusOn_OuterCapillaryTubeMats"] = 1
        else:
            action["bSetStatusOn_OuterCapillaryTubeMats"] = 0

        # HNLT HVFA
        Controller_Controller_HVFA_HNLT_Loading_Permission_Output = (
            self.Controller_Controller_HVFA_HNLT_Loading_Permission.update(HNLT_HVFA_fLowerTemperature)
        )
        Controller_Controller_HVFA_HNLT_Unloading_Permission_Output = (
            self.Controller_Controller_HVFA_HNLT_Unloading_Permission.update(HNLT_HVFA_fLowerTemperature)
        )
        Controller_Buffer_HVFA_HNLT_Loading_Output = self.Controller_Buffer_HVFA_HNLT_Loading.update(
            HNLT_Buffer_fMidTemperature
        )
        Controller_Buffer_HVFA_HNLT_Unloading_Output = self.Controller_Buffer_HVFA_HNLT_Unloading.update(
            HNLT_Buffer_fMidTemperature
        )

        if Controller_Controller_HVFA_HNLT_Loading_Permission_Output and Controller_Buffer_HVFA_HNLT_Loading_Output:
            action["bSetStatusOn_HVFASystem_HNLT"] = 1
        elif (
            Controller_Controller_HVFA_HNLT_Unloading_Permission_Output and Controller_Buffer_HVFA_HNLT_Unloading_Output
        ):
            action["bSetStatusOn_HVFASystem_HNLT"] = 1
        else:
            action["bSetStatusOn_HVFASystem_HNLT"] = 0

        if Controller_Controller_HVFA_HNLT_Loading_Permission_Output and Controller_Buffer_HVFA_HNLT_Loading_Output:
            action["bLoading_HVFASystem_HNLT"] = 1
        else:
            action["bLoading_HVFASystem_HNLT"] = 0

        # CN
        action["bSetStatusOn_eChiller"] = self.Controller_eChiller.update(CN_Buffer_fMidTemperature)

        # CN HVFA
        Controller_HVFA_CN_Loading_Permission_Output = self.Controller_HVFA_CN_Loading_Permission.update(
            CN_HVFA_fLowerTemperature
        )
        Controller_HVFA_CN_Unloading_Permission_Output = self.Controller_HVFA_CN_Unloading_Permission.update(
            CN_HVFA_fLowerTemperature
        )

        Controller_Buffer_HVFA_CN_HVFA_Loading_Output = self.Controller_Buffer_HVFA_CN_Loading.update(
            CN_Buffer_fMidTemperature
        )
        Controller_Buffer_HVFA_CN_HVFA_Unoading_Output = self.Controller_Buffer_HVFA_CN_Unloading.update(
            CN_Buffer_fMidTemperature
        )

        if Controller_HVFA_CN_Loading_Permission_Output and Controller_Buffer_HVFA_CN_HVFA_Loading_Output:
            action["bSetStatusOn_HVFASystem_CN"] = 1
        elif Controller_HVFA_CN_Unloading_Permission_Output and Controller_Buffer_HVFA_CN_HVFA_Unoading_Output:
            action["bSetStatusOn_HVFASystem_CN"] = 1
        else:
            action["bSetStatusOn_HVFASystem_CN"] = 0

        if Controller_HVFA_CN_Loading_Permission_Output and Controller_Buffer_HVFA_CN_HVFA_Loading_Output:
            action["bLoading_HVFASystem_CN"] = 1
        else:
            action["bLoading_HVFASystem_CN"] = 0

        # HNLT-CN Linkage
        Controller_Output_HeatPump = self.Controller_HeatPump.update(HNLT_Buffer_fMidTemperature)
        Controller_Output_HeatPump_Permission = self.Controller_HeatPump_Permission.update(CN_Buffer_fMidTemperature)
        if Controller_Output_HeatPump and Controller_Output_HeatPump_Permission:
            action["bSetStatusOn_HeatPump"] = 1
        else:
            action["bSetStatusOn_HeatPump"] = 0

        # print(action)
        actions = []
        actions.append(list(action.values()))
        # print(actions)
        actions = actions[0]

        return np.array(actions)


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


class HysteresisController:
    def __init__(self, hysteresis_range, target, inverted=False):
        self.hysteresis_range = hysteresis_range  # This is the hysteresis range
        self.target = target  # This is the target temperature
        self.inverted = inverted  # This should be True e.g. for eChiller, which should be 1 when T is too high
        self.output = 0  # The output is always init with 0

    def update(self, actual_value):
        if self.inverted == False:
            if self.output == 0:  # controller is off
                # controller output still 0 if input value below threshhold
                self.output = 1 if actual_value <= self.target - self.hysteresis_range / 2 else 0
            else:  # controller is on
                self.output = 0 if actual_value >= self.target + self.hysteresis_range / 2 else 1

        else:
            if self.output == 0:  # controller is off
                # controller output still 0 if input value below threshhold
                self.output = 1 if actual_value >= self.target + self.hysteresis_range / 2 else 0
            else:  # controller is on
                self.output = 0 if actual_value <= self.target - self.hysteresis_range / 2 else 1

        return self.output

    def change_hysteresis_range(self, hysteresis_range):
        self.hysteresis_range = hysteresis_range

    def change_target(self, target):
        self.target = target
