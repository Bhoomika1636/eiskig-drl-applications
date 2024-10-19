from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from eta_utility import get_logger, timeseries
from eta_utility.eta_x import ConfigOptRun
from eta_utility.eta_x.envs import BaseEnvSim, StateConfig, StateVar
from gymnasium.utils import seeding

try:
    from plotter.plotter import ETA_Plotter, Heatmap, Linegraph

    plotter_available = True
except ImportError as e:
    plotter_available = False


from eta_utility.type_hints import StepResult, TimeStep
from gymnasium import spaces
from scipy.special import expit

log = get_logger("eta_x.envs")


class SupplysystemB(BaseEnvSim):
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
    # fmu_name = "supplysystem_b_variance"
    fmu_name = "supplysystem_b"

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
        variant,
        discretize_action_space,
        reward_shaping,
        temperature_heat_init_min,
        temperature_heat_init_max,
        temperature_cool_init_min,
        temperature_cool_init_max,
        temperature_difference_top_bottom,
        poweron_cost_combinedheatpower,
        poweron_cost_condensingboiler,
        poweron_cost_immersionheater,
        poweron_cost_heatpump,
        poweron_cost_compressionchiller,
        poweron_cost_coolingtower,
        abort_costs,
        extended_obs_4_entropy,
        policyshaping_costs,
        temperature_cost_heat,
        temperature_reward_heat,
        temperature_cost_heat_min,
        temperature_cost_heat_max,
        temperature_cost_prod_heat_min,
        temperature_cost_prod_heat_max,
        temperature_cost_cool,
        temperature_reward_cool,
        temperature_cost_cool_min,
        temperature_cost_cool_max,
        temperature_cost_prod_cool_min,
        temperature_cost_prod_cool_max,
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
        self.discretize_action_space = discretize_action_space
        self.reward_shaping = reward_shaping
        self.temperature_heat_init_min = temperature_heat_init_min
        self.temperature_heat_init_max = temperature_heat_init_max
        self.temperature_cool_init_min = temperature_cool_init_min
        self.temperature_cool_init_max = temperature_cool_init_max
        self.temperature_difference_top_bottom = temperature_difference_top_bottom
        self.poweron_cost_combinedheatpower = poweron_cost_combinedheatpower
        self.poweron_cost_condensingboiler = poweron_cost_condensingboiler
        self.poweron_cost_immersionheater = poweron_cost_immersionheater
        self.poweron_cost_heatpump = poweron_cost_heatpump
        self.poweron_cost_compressionchiller = poweron_cost_compressionchiller
        self.poweron_cost_coolingtower = poweron_cost_coolingtower
        self.abort_costs = abort_costs
        self.policyshaping_costs = policyshaping_costs
        self.temperature_cost_heat = temperature_cost_heat
        self.temperature_reward_heat = temperature_reward_heat
        self.temperature_cost_heat_min = temperature_cost_heat_min
        self.temperature_cost_heat_max = temperature_cost_heat_max
        self.temperature_cost_prod_heat_min = temperature_cost_prod_heat_min
        self.temperature_cost_prod_heat_max = temperature_cost_prod_heat_max
        self.temperature_cost_cool = temperature_cost_cool
        self.temperature_reward_cool = temperature_reward_cool
        self.temperature_cost_cool_min = temperature_cost_cool_min
        self.temperature_cost_cool_max = temperature_cost_cool_max
        self.temperature_cost_prod_cool_min = temperature_cost_prod_cool_min
        self.temperature_cost_prod_cool_max = temperature_cost_prod_cool_max
        self.power_cost_max = power_cost_max
        self.tax_el_per_kwh = tax_el_per_kwh
        self.tax_el_produced_per_kwh = tax_el_produced_per_kwh
        self.tax_gas_per_kwh = tax_gas_per_kwh
        self.peak_cost_per_kw = peak_cost_per_kw

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
                name="u_combinedheatpower",
                ext_id="u_cHP",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_condensingboiler",
                ext_id="u_CondensingBoiler",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_immersionheater",
                ext_id="u_ImmersionHeater",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_heatpump",
                ext_id="u_CompressionChiller",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_coolingtower",
                ext_id="u_CoolingTower",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="u_compressionchiller",
                ext_id="u_CompressionChiller_CT",
                is_ext_input=True,
                is_agent_action=True,
                low_value=0.0,
                high_value=1.0,
            ),
            # disturbances
            StateVar(
                name="d_production_heat_power",
                ext_id="u_heat_power_demand_production",
                scenario_id="power_heat",
                from_scenario=True,
                is_ext_input=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=250000.0,
            ),
            StateVar(
                name="d_production_cool_power",
                ext_id="u_heat_power_from_production",
                scenario_id="power_cold",
                from_scenario=True,
                is_ext_input=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=150000.0,
            ),
            StateVar(
                name="d_production_electric_power",
                ext_id="u_electric_power_demand_production",
                scenario_id="power_electricity",
                from_scenario=True,
                is_ext_input=True,
                is_agent_observation=extended_obs_4_entropy,
                low_value=0.0,
                high_value=300000.0,
            ),
            StateVar(
                name="d_production_gas_power",
                ext_id="u_gas_power_demand_production",
                scenario_id="power_gas",
                from_scenario=True,
                is_ext_input=True,
                is_agent_observation=extended_obs_4_entropy,
                low_value=0.0,
                high_value=300000.0,
            ),
            StateVar(
                name="d_time_till_availability",
                scenario_id="time_availability",
                from_scenario=True,
                is_agent_observation=extended_state,
                low_value=-18,
                high_value=54,
            ),
            StateVar(
                name="d_weather_drybulbtemperature",
                ext_id="u_weather_DryBulbTemperature",
                scenario_id="air_temperature",  # [°C]
                from_scenario=True,
                is_agent_observation=True,
                is_ext_input=True,
                low_value=-20,
                high_value=45,
            ),
            StateVar(
                name="d_weather_relativehumidity",
                ext_id="u_weather_RelativeHumidity",
                scenario_id="relative_air_humidity",  # [%]
                from_scenario=True,
                is_agent_observation=True,
                is_ext_input=True,
                low_value=0,
                high_value=100,
            ),
            StateVar(
                name="d_weather_time",
                ext_id="u_weather_Time",
                is_ext_input=True,
                low_value=0,
                high_value=31968000,
            ),
            StateVar(
                name="d_weather_globalhorizontalradiation",
                ext_id="u_weather_GlobalHorizontalRadiation",
                scenario_id="global_radiation",  # [W/m2]
                from_scenario=True,
                is_ext_input=True,
                low_value=0,
                high_value=1000,
            ),
            StateVar(
                name="d_weather_opaqueskycover",
                ext_id="u_weather_OpaqueSkyCover",
                scenario_id="clouds",  # [*100 %]
                from_scenario=True,
                is_ext_input=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="d_weather_directnormalradiation",
                ext_id="u_weather_DirectNormalRadiation",
                scenario_id="direct_radiation",  # [W/m2]
                from_scenario=True,
                is_ext_input=True,
                low_value=0,
                high_value=1000,
            ),
            StateVar(
                name="d_weather_diffusehorizontalradiation",
                ext_id="u_weather_DiffuseHorizontalRadiation",
                scenario_id="diffuse_radiation",  # [W/m2]
                from_scenario=True,
                is_ext_input=True,
                low_value=0,
                high_value=1000,
            ),
            StateVar(
                name="d_weather_windspeed",
                ext_id="u_weather_WindSpeed",
                scenario_id="wind_speed",  # [m/s]
                from_scenario=True,
                is_ext_input=True,
                low_value=0,
                high_value=30,
            ),
            StateVar(
                name="d_weather_precip_depth",
                ext_id="u_weather_Precip_Depth",
                scenario_id="rainfall",  # [mm]
                from_scenario=True,
                is_ext_input=True,
                low_value=-1000,
                high_value=1000,
            ),
            StateVar(
                name="d_weather_winddirection",
                ext_id="u_weather_WindDirection",
                scenario_id="global_radiation",  # [deg]
                from_scenario=True,
                is_ext_input=True,
                low_value=0,
                high_value=360,
            ),
            StateVar(
                name="d_weather_precipitation",
                ext_id="u_weather_Precipitation",
                scenario_id="rain_indicator",  # [1 = rain, 0 = no rain]
                from_scenario=True,
                is_ext_input=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="d_weather_groundtemperature",
                ext_id="u_weather_GroundTemperature",
                scenario_id="air_temperature",  # [°C]
                from_scenario=True,
                is_ext_input=True,
                low_value=-20,
                high_value=45,
            ),
            # predictions
            StateVar(
                name="p_production_delta_heat_power_1h",
                is_agent_observation=True,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_heat_power_6h",
                is_agent_observation=extended_state,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_cool_power_1h",
                is_agent_observation=True,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_cool_power_6h",
                is_agent_observation=extended_state,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_electric_power_15m",
                is_agent_observation=True,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_electric_power_1h",
                is_agent_observation=extended_state,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_production_delta_electric_power_6h",
                is_agent_observation=extended_state,
                low_value=-2e6,
                high_value=2e6,
            ),
            StateVar(
                name="p_delta_price_electricity_15m",
                is_agent_observation=extended_state,
                low_value=-1e2,
                high_value=1e2,
            ),
            StateVar(name="p_delta_price_electricity_1h", is_agent_observation=True, low_value=-1e2, high_value=1e2),
            StateVar(name="p_delta_price_electricity_6h", is_agent_observation=True, low_value=-1e2, high_value=1e2),
            # virtual states
            StateVar(
                name="vs_electric_power_total_15min", is_agent_observation=True, low_value=-100000, high_value=500000
            ),
            StateVar(
                name="vs_gas_power_total_15min",
                is_agent_observation=extended_state,
                low_value=-100000,
                high_value=500000,
            ),
            StateVar(name="vs_time_daytime", is_agent_observation=True, low_value=0, high_value=24),
            # states
            StateVar(
                name="s_price_electricity",
                scenario_id="electrical_energy_price",
                from_scenario=True,
                is_agent_observation=True,
                low_value=-10,
                high_value=10,
            ),  # to obtain €/kWh
            StateVar(
                name="s_price_gas",
                scenario_id="gas_price",
                from_scenario=True,
                # is_agent_observation=extended_state,
                is_agent_observation=extended_obs_4_entropy,
                low_value=-10,
                high_value=10,
            ),  # to obtain €/kWh
            StateVar(
                name="s_gas_power_total",
                ext_id="gas_power_consumption",
                is_ext_output=True,
                is_agent_observation=extended_obs_4_entropy,
                low_value=0,
                high_value=500000,
            ),
            StateVar(
                name="s_electric_power_total",
                ext_id="electric_power_consumption",
                is_ext_output=True,
                is_agent_observation=extended_obs_4_entropy,
                low_value=-100000,
                high_value=500000,
            ),
            StateVar(
                name="s_temp_heatsupply",
                ext_id="Temp_warmwater_before_production.T",
                is_ext_output=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_temp_coldsupply",
                ext_id="Temp_coldwater_before_production.T",
                is_ext_output=True,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_temp_heat_storage_hi",
                ext_id="TempSen_storage_heat_upper.T",
                is_ext_output=True,
                is_agent_observation=True,
                abort_condition_min=273.15 + 15,
                abort_condition_max=273.15 + 97,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_temp_heat_storage_lo",
                ext_id="TempSen_storage_heat_lower.T",
                is_ext_output=True,
                is_agent_observation=True,
                abort_condition_min=273.15 + 15,
                abort_condition_max=273.15 + 97,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_temp_cold_storage_hi",
                ext_id="TempSen_storage_cold_upper.T",
                is_ext_output=True,
                is_agent_observation=True,
                abort_condition_min=273.15 + 1,
                abort_condition_max=273.15 + 60,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_temp_cold_storage_lo",
                ext_id="TempSen_storage_cold_lower.T",
                is_ext_output=True,
                is_agent_observation=True,
                abort_condition_min=273.15 + 1,
                abort_condition_max=273.15 + 60,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_temp_before_CT",
                ext_id="Temp_before_CoolingTower.T",
                is_ext_output=True,
                is_agent_observation=mpc_state,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_temp_after_CT",
                ext_id="Temp_after_CoolingTower.T",
                is_ext_output=True,
                is_agent_observation=mpc_state,
                low_value=273,
                high_value=373,
            ),
            StateVar(
                name="s_u_combinedheatpower",
                ext_id="cHP.s_u",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_P_el_combinedheatpower",
                ext_id="cHP.P_el",
                is_ext_output=True,
                is_agent_observation=mpc_state,
                low_value=-60000,
                high_value=60000,
            ),
            StateVar(
                name="s_P_gs_combinedheatpower",
                ext_id="cHP.P_gs",
                is_ext_output=True,
                is_agent_observation=mpc_state,
                low_value=0,
                high_value=200000,
            ),
            StateVar(
                name="s_u_condensingboiler",
                ext_id="condensingBoiler.s_u",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_P_gs_condensingboiler",
                ext_id="condensingBoiler.P_gs",
                is_ext_output=True,
                is_agent_observation=mpc_state,
                low_value=0,
                high_value=8000000,
            ),
            StateVar(
                name="s_u_immersionheater",
                ext_id="immersionHeater.s_u",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0.0,
                high_value=1.0,
            ),
            StateVar(
                name="s_P_el_immersionheater",
                ext_id="immersionHeater.P_el",
                is_ext_output=True,
                is_agent_observation=mpc_state,
                low_value=0,
                high_value=150000,
            ),
            StateVar(
                name="s_u_heatpump",
                ext_id="compressionChiller_Tanks.s_u",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="P_el_heatpump",
                ext_id="compressionChiller_Tanks.P_el",
                is_ext_output=True,
                is_agent_observation=mpc_state,
                low_value=0,
                high_value=1000000,
            ),
            StateVar(
                name="s_u_compressionchiller",
                ext_id="compressionChiller_CT.s_u",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="P_el_compressionchiller",
                ext_id="compressionChiller_CT.P_el",
                is_ext_output=True,
                is_agent_observation=mpc_state,
                low_value=0,
                high_value=1000000,
            ),
            StateVar(
                name="s_u_coolingtower",
                ext_id="coolingTower_open.s_u",
                is_ext_output=True,
                is_agent_observation=True,
                low_value=0,
                high_value=1,
            ),
            StateVar(
                name="P_el_coolingtower",
                ext_id="coolingTower_open.P_el",
                is_ext_output=True,
                is_agent_observation=mpc_state,
                low_value=0,
                high_value=1000000,
            ),
        )

        # add predictions when extended predictions are required
        if self.extended_predictions:
            predictions_state_var_tuple = ()
            for i in range(self.n_steps_6h):
                predictions_state_var_tuple = (
                    *predictions_state_var_tuple,
                    StateVar(
                        name="p_production_delta_heat_power_t_plus_" + str(i + 1),
                        is_agent_observation=extended_state,
                        low_value=-2e6,
                        high_value=2e6,
                    ),
                )
            for i in range(self.n_steps_6h):
                predictions_state_var_tuple = (
                    *predictions_state_var_tuple,
                    StateVar(
                        name="p_production_delta_cool_power_t_plus_" + str(i + 1),
                        is_agent_observation=extended_state,
                        low_value=-2e6,
                        high_value=2e6,
                    ),
                )
            for i in range(self.n_steps_6h):
                predictions_state_var_tuple = (
                    *predictions_state_var_tuple,
                    StateVar(
                        name="p_production_delta_electric_power_t_plus_" + str(i + 1),
                        is_agent_observation=extended_state,
                        low_value=-2e6,
                        high_value=2e6,
                    ),
                )
            for i in range(self.n_steps_6h):
                predictions_state_var_tuple = (
                    *predictions_state_var_tuple,
                    StateVar(
                        name="p_delta_price_electricity_t_plus_" + str(i + 1),
                        is_agent_observation=extended_state,
                        low_value=-1e2,
                        high_value=1e2,
                    ),
                )
            state_var_tuple = (*state_var_tuple, *predictions_state_var_tuple)

        # build final state_config
        self.state_config = StateConfig(*state_var_tuple)
        print("The selected observations are: ", self.state_config.observations)

        # import all scenario files
        self.scenario_data = self.import_scenario(*scenario_files).fillna(
            method="ffill"
        )  # add another ffillna cause only values which are missing beceause of resampling are interpolated in eta_utility

        # get action_space
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
        T_heat_max = max(self.state["s_temp_heat_storage_hi"], self.state["s_temp_heatsupply"]) - 273.15
        T_heat_min = min(self.state["s_temp_heat_storage_hi"], self.state["s_temp_heatsupply"]) - 273.15
        T_cold_max = max(self.state["s_temp_cold_storage_lo"], self.state["s_temp_coldsupply"]) - 273.15
        T_cold_min = min(self.state["s_temp_cold_storage_lo"], self.state["s_temp_coldsupply"]) - 273.15
        self.policy_shaping_active = False
        # warm net check
        if T_heat_max > 95:
            self.additional_state["u_combinedheatpower"] = 0
            self.additional_state["u_condensingboiler"] = 0
            self.additional_state["u_immersionheater"] = 0
            self.additional_state["u_heatpump"] = 0
            self.additional_state["u_coolingtower"] = _action[4]
            self.additional_state["u_compressionchiller"] = _action[5]
            _action = np.array(
                [
                    self.additional_state["u_combinedheatpower"],
                    self.additional_state["u_condensingboiler"],
                    self.additional_state["u_immersionheater"],
                    self.additional_state["u_heatpump"],
                    self.additional_state["u_coolingtower"],
                    self.additional_state["u_compressionchiller"],
                ]
            )
            self.policy_shaping_active = True
        elif T_heat_min < 45:
            self.additional_state["u_combinedheatpower"] = 1
            self.additional_state["u_condensingboiler"] = 1
            self.additional_state["u_immersionheater"] = 1
            self.additional_state["u_heatpump"] = 1
            self.additional_state["u_coolingtower"] = _action[4]
            self.additional_state["u_compressionchiller"] = _action[5]
            _action = np.array(
                [
                    self.additional_state["u_combinedheatpower"],
                    self.additional_state["u_condensingboiler"],
                    self.additional_state["u_immersionheater"],
                    self.additional_state["u_heatpump"],
                    self.additional_state["u_coolingtower"],
                    self.additional_state["u_compressionchiller"],
                ]
            )
            self.policy_shaping_active = True
        # cold net check
        if T_cold_max > 31:
            self.additional_state["u_combinedheatpower"] = _action[0]
            self.additional_state["u_condensingboiler"] = _action[1]
            self.additional_state["u_immersionheater"] = _action[2]
            self.additional_state["u_coolingtower"] = 1
            self.additional_state["u_compressionchiller"] = 1
            self.additional_state["u_heatpump"] = 0
            _action = np.array(
                [
                    self.additional_state["u_combinedheatpower"],
                    self.additional_state["u_condensingboiler"],
                    self.additional_state["u_immersionheater"],
                    self.additional_state["u_heatpump"],
                    self.additional_state["u_coolingtower"],
                    self.additional_state["u_compressionchiller"],
                ]
            )
            self.policy_shaping_active = True
        elif T_cold_min < 4:
            self.additional_state["u_combinedheatpower"] = _action[0]
            self.additional_state["u_condensingboiler"] = _action[1]
            self.additional_state["u_immersionheater"] = _action[2]
            self.additional_state["u_coolingtower"] = 0
            self.additional_state["u_compressionchiller"] = 0
            self.additional_state["u_heatpump"] = 0
            _action = np.array(
                [
                    self.additional_state["u_combinedheatpower"],
                    self.additional_state["u_condensingboiler"],
                    self.additional_state["u_immersionheater"],
                    self.additional_state["u_heatpump"],
                    self.additional_state["u_coolingtower"],
                    self.additional_state["u_compressionchiller"],
                ]
            )
            self.policy_shaping_active = True

        # check actions for vilidity, perform simulation step and load new external values for the next time step
        self._actions_valid(_action)
        self.state["step_success"], _ = self._update_state(_action)

        # check if state is in valid boundaries
        self.state["step_abort"] = False if StateConfig.within_abort_conditions(self.state_config, self.state) else True

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
        reward = self.calc_reward()

        # update state_log
        self.state_log.append(self.state)

        observations = self._observations()

        return observations, reward, done, False, {}

    def update_predictions(self):

        prediction_dict = {}

        # [W] heat power demand of production
        prediction_dict["p_production_delta_heat_power_1h"] = (
            self.ts_current["power_heat"].iloc[self.n_steps : self.n_steps + self.n_steps_1h].mean()
            - self.ts_current["power_heat"].iloc[self.n_steps]
        )
        prediction_dict["p_production_delta_heat_power_6h"] = (
            self.ts_current["power_heat"].iloc[self.n_steps : self.n_steps + self.n_steps_6h].mean()
            - self.ts_current["power_heat"].iloc[self.n_steps]
        )

        # [W] cooling power demand of production
        prediction_dict["p_production_delta_cool_power_1h"] = (
            self.ts_current["power_cold"].iloc[self.n_steps : self.n_steps + self.n_steps_1h].mean()
            - self.ts_current["power_cold"].iloc[self.n_steps]
        )
        prediction_dict["p_production_delta_cool_power_6h"] = (
            self.ts_current["power_cold"].iloc[self.n_steps : self.n_steps + self.n_steps_6h].mean()
            - self.ts_current["power_cold"].iloc[self.n_steps]
        )

        # [W] electric power demand of production
        prediction_dict["p_production_delta_electric_power_15m"] = (
            self.ts_current["power_electricity"].iloc[self.n_steps : self.n_steps + self.n_steps_15m].mean()
            - self.ts_current["power_electricity"].iloc[self.n_steps]
        )
        prediction_dict["p_production_delta_electric_power_1h"] = (
            self.ts_current["power_electricity"].iloc[self.n_steps : self.n_steps + self.n_steps_1h].mean()
            - self.ts_current["power_electricity"].iloc[self.n_steps]
        )
        prediction_dict["p_production_delta_electric_power_6h"] = (
            self.ts_current["power_electricity"].iloc[self.n_steps : self.n_steps + self.n_steps_6h].mean()
            - self.ts_current["power_electricity"].iloc[self.n_steps]
        )

        # electricity price [€/kWh]
        prediction_dict["p_delta_price_electricity_15m"] = (
            self.ts_current["electrical_energy_price"].iloc[self.n_steps : self.n_steps + self.n_steps_15m].mean()
            - self.ts_current["electrical_energy_price"].iloc[self.n_steps]
        )
        prediction_dict["p_delta_price_electricity_1h"] = (
            self.ts_current["electrical_energy_price"].iloc[self.n_steps : self.n_steps + self.n_steps_1h].mean()
            - self.ts_current["electrical_energy_price"].iloc[self.n_steps]
        )
        prediction_dict["p_delta_price_electricity_6h"] = (
            self.ts_current["electrical_energy_price"].iloc[self.n_steps : self.n_steps + self.n_steps_6h].mean()
            - self.ts_current["electrical_energy_price"].iloc[self.n_steps]
        )

        # add  predictions when extended predictions are required
        if self.extended_predictions:
            for i in range(self.n_steps_6h):
                prediction_dict["p_production_delta_heat_power_t_plus_" + str(i + 1)] = (
                    self.ts_current["power_heat"].iloc[self.n_steps + i + 1]
                    - self.ts_current["power_heat"].iloc[self.n_steps]
                )
            for i in range(self.n_steps_6h):
                prediction_dict["p_production_delta_cool_power_t_plus_" + str(i + 1)] = (
                    self.ts_current["power_cold"].iloc[self.n_steps + i + 1]
                    - self.ts_current["power_cold"].iloc[self.n_steps]
                )
            for i in range(self.n_steps_6h):
                prediction_dict["p_production_delta_electric_power_t_plus_" + str(i + 1)] = (
                    self.ts_current["power_electricity"].iloc[self.n_steps + i + 1]
                    - self.ts_current["power_electricity"].iloc[self.n_steps]
                )
            for i in range(self.n_steps_6h):
                prediction_dict["p_delta_price_electricity_t_plus_" + str(i + 1)] = (
                    self.ts_current["electrical_energy_price"].iloc[self.n_steps + i + 1]
                    - self.ts_current["electrical_energy_price"].iloc[self.n_steps]
                )

        return prediction_dict

    def update_virtual_state(self):

        virtual_state = {}

        # daytime
        virtual_state["vs_time_daytime"] = (
            self.ts_current.index[self.n_steps].hour + self.ts_current.index[self.n_steps].minute / 60
        )

        # running 15min average electric power  # TODO: replace by using state_log!
        self.P_el_total_15min_buffer.append(self.state["s_electric_power_total"])
        if len(self.P_el_total_15min_buffer) > self.n_steps_15m:
            self.P_el_total_15min_buffer.pop(0)
        virtual_state["vs_electric_power_total_15min"] = sum(self.P_el_total_15min_buffer) / len(
            self.P_el_total_15min_buffer
        )

        # running 15min average gas power  # TODO: replace by using state_log!
        self.P_gs_total_15min_buffer.append(self.state["s_gas_power_total"])
        if len(self.P_gs_total_15min_buffer) > self.n_steps_15m:
            self.P_gs_total_15min_buffer.pop(0)
        virtual_state["vs_gas_power_total_15min"] = sum(self.P_gs_total_15min_buffer) / len(
            self.P_gs_total_15min_buffer
        )

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

        # switching costs
        self.state["reward_switching"] = (
            -switch_cost(
                self.state_backup["s_u_combinedheatpower"],
                self.state["s_u_combinedheatpower"],
                self.poweron_cost_combinedheatpower,
            )
            - switch_cost(
                self.state_backup["s_u_condensingboiler"],
                self.state["s_u_condensingboiler"],
                self.poweron_cost_condensingboiler,
            )
            - switch_cost(
                self.state_backup["s_u_immersionheater"],
                self.state["s_u_immersionheater"],
                self.poweron_cost_immersionheater,
            )
            - switch_cost(
                self.state_backup["s_u_heatpump"],
                self.state["s_u_heatpump"],
                self.poweron_cost_heatpump,
            )
            - switch_cost(
                self.state_backup["s_u_compressionchiller"],
                self.state["s_u_compressionchiller"],
                self.poweron_cost_compressionchiller,
            )
            - switch_cost(
                self.state_backup["s_u_coolingtower"],
                self.state["s_u_coolingtower"],
                self.poweron_cost_coolingtower,
            )
        )

        # temperature costs (when availability of temperature levels are needed)
        self.state["reward_temperature_heat"] = reward_boundary(
            self.state["s_temp_heat_storage_hi"],
            self.temperature_cost_prod_heat_min,
            self.temperature_cost_prod_heat_max,
            self.temperature_reward_heat,
            self.temperature_cost_heat,
            smoothed=self.reward_shaping,
            k=6,
        )
        self.state["reward_temperature_cool"] = reward_boundary(
            self.state["s_temp_cold_storage_lo"],
            self.temperature_cost_prod_cool_min,
            self.temperature_cost_prod_cool_max,
            self.temperature_reward_cool,
            self.temperature_cost_cool,
            smoothed=self.reward_shaping,
            k=6,
        )

        # energy costs
        base_power_electric = (
            self.state["s_electric_power_total"] - self.state["d_production_electric_power"]
        )  # total consumption of supply systems O
        base_power_gas = self.state["s_gas_power_total"] - self.state["d_production_gas_power"]
        self.state["reward_energy_electric"] = (
            -self.state["s_price_electricity"] * base_power_electric * self.sampling_time / 3600 / 1000
        )
        self.state["reward_energy_gas"] = -self.state["s_price_gas"] * base_power_gas * self.sampling_time / 3600 / 1000

        # energy consumed
        self.state["energy_electric_consumed"] = base_power_electric * self.sampling_time / 3600 / 1000
        self.state["energy_gas_consumed"] = base_power_gas * self.sampling_time / 3600 / 1000

        # energy taxes costs
        tax_el_per_kwh = self.tax_el_per_kwh  # [€/kWh]
        tax_el_produced_per_kwh = self.tax_el_produced_per_kwh  # [€/kWh]
        tax_gas_per_kwh = self.tax_gas_per_kwh  # [€/kWh]

        # taxes on electricity consumption
        if self.state["s_P_el_combinedheatpower"] < 0: #if P_el of CHp is > 0
            if self.state["s_electric_power_total"] > 0: #total power consumption of supply system and production system
                tax_el = (
                    (base_power_electric - self.state["s_P_el_combinedheatpower"])
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
                    -(abs(self.state["s_electric_power_total"]) - abs(base_power_electric))
                    * (tax_el_per_kwh - tax_el_produced_per_kwh)
                    * self.sampling_time
                    / 3600
                    / 1000
                )
        else:
            tax_el = base_power_electric * tax_el_per_kwh * self.sampling_time / 3600 / 1000

        # taxes on gas consumption
        tax_gs = tax_gas_per_kwh * base_power_gas * self.sampling_time / 3600 / 1000  # [€]

        # total energy taxes
        self.state["reward_energy_taxes"] = -tax_el - tax_gs  # [€]

        # power costs for peak load pricing
        # update peak electric load, when the average load of the past 15 mintes is greater then the last max_limit
        if self.state["vs_electric_power_total_15min"] > self.max_limit:
            peak_difference = (self.state["vs_electric_power_total_15min"] - self.max_limit) / 1000  # [kW]
            self.state["reward_power_electric"] = -1 * (peak_difference * self.peak_cost_per_kw)  # [€]
            self.max_limit = self.state["vs_electric_power_total_15min"]  # update value
        else:
            self.state["reward_power_electric"] = 0

        # other costs
        self.state["reward_other"] = (
            -self.abort_costs * (1 - 0.5 * self.n_steps / self.n_episode_steps)
            if (self.state["step_abort"] or not self.state["step_success"])
            else 0
        )

        # policyshaping costs
        if self.policy_shaping_active:
            self.state["reward_other"] -= self.policyshaping_costs

        # total reward
        self.state["reward_total"] = (
            self.state["reward_switching"]
            + self.state["reward_temperature_heat"]
            + self.state["reward_temperature_cool"]
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

        # # save episode's stats
        if self.n_steps > 0:

            # create dataframe from state_log
            self.episode_df = pd.DataFrame(self.state_log)

            # derive certain episode statistics for logging and plotting
            self.episode_statistics["datetime_begin"] = self.ts_current.index[0].strftime("%Y-%m-%d %H:%M")
            self.episode_statistics["rewards_total"] = self.episode_df["reward_total"].sum()
            self.episode_statistics["rewards_switching"] = self.episode_df["reward_switching"].sum()
            self.episode_statistics["rewards_temperature_heat"] = self.episode_df["reward_temperature_heat"].sum()
            self.episode_statistics["rewards_temperature_cool"] = self.episode_df["reward_temperature_cool"].sum()
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
            self.episode_statistics["temp_heat_min"] = self.episode_df["s_temp_heat_storage_hi"].min()
            self.episode_statistics["temp_heat_max"] = self.episode_df["s_temp_heat_storage_hi"].max()

            # # append episode store to episodes' archive
            self.episode_archive.append(list(self.episode_statistics.values()))

        # initialize additional state, since super().reset is used
        self.additional_state = {}

        # set FMU parameters for initialization
        self.model_parameters = {}

        if "none" not in self.variance_parameters:
            # randomly parametrize systems
            self.model_parameters["pipe_simple_chp_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["pipe_simple_boiler_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["pipe_simple_CompressionChiller_warm_side_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["pipe_simple_CompressionChiller_cold_side_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["pipe_simple_production1_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["pipe_simple_production2_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["pipe_simple_CompressionChiller2_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["pipe_simple_CoolingTower_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["watertank_warm_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["watertank_cold_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["val_after_HeatPump_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["val_before_CoolingTower_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["heatExchanger_counterflow_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["idealPump2_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["idealPump3_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["idealPump4_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["idealPump5_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["idealPump1_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["idealPump_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["coolingTower_open_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["chp_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["condensingBoiler_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["immersionHeater_variance"] = self.np_random.uniform(self.variance_min, self.variance_max)
            self.model_parameters["compressionChiller_Tanks_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )
            self.model_parameters["compressionChiller_CT_variance"] = self.np_random.uniform(
                self.variance_min, self.variance_max
            )

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

        # randomly set start temperatures
        self.model_parameters["T_start_warmwater"] = self.np_random.uniform(
            self.temperature_heat_init_min, self.temperature_heat_init_max
        )
        self.model_parameters["T_start_coldwater"] = self.np_random.uniform(
            self.temperature_cool_init_min, self.temperature_cool_init_max
        )

        self.additional_state["s_temp_heat_storage_hi"] = self.model_parameters["T_start_warmwater"]
        self.additional_state["s_temp_heat_storage_lo"] = (
            self.model_parameters["T_start_warmwater"] - self.temperature_difference_top_bottom
        )

        self.additional_state["s_temp_cold_storage_lo"] = self.model_parameters["T_start_coldwater"]
        self.additional_state["s_temp_cold_storage_hi"] = (
            self.model_parameters["T_start_coldwater"] + self.temperature_difference_top_bottom
        )

        self.additional_state["s_temp_heatsupply"] = self.model_parameters["T_start_warmwater"]
        self.additional_state["s_temp_coldsupply"] = self.model_parameters["T_start_coldwater"]

        # get current slice of timeseries dataframe, extended by maximum prediction horizon (6h)
        # and one additional step because step 0 = init conditions
        self.ts_current = timeseries.df_time_slice(
            self.scenario_data,
            self.scenario_time_begin,
            self.scenario_time_end,
            self.episode_duration + (self.n_steps_6h + 1) * self.sampling_time,
            random=self.np_random if self.random_sampling else False,
        )

        # read current date time
        self.episode_datetime_begin = self.ts_current.index[0]
        print("Start Time: ",self.episode_datetime_begin)
        self.additional_state["vs_time_daytime"] = self.episode_datetime_begin.hour

        # reset virtual states and internal counters
        self.P_el_total_15min_buffer = []
        self.P_gs_total_15min_buffer = []
        self.additional_state["vs_electric_power_total_15min"] = 0
        self.additional_state["vs_gas_power_total_15min"] = 0

        # get scenario input for initialization (time step: 0)
        self.additional_state.update(self.update_predictions())

        # add time of the year in seconds
        starttime_of_year = pd.Timestamp(str(self.ts_current.index[self.n_steps].year) + "-01-01 00:00")
        self.additional_state["d_weather_time"] = pd.Timedelta(
            self.ts_current.index[self.n_steps] - starttime_of_year
        ).total_seconds()

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
            x, y["rewards_energy_electric"], label="Strom (netto)", color=(1.0, 0.75, 0.0), linewidth=1, alpha=0.9
        )
        axes[0].plot(
            x, y["rewards_energy_gas"], label="Erdgas (netto)", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9
        )
        axes[0].plot(
            x, y["rewards_energy_taxes"], label="Steuern & Umlagen", color=(0.184, 0.333, 0.592), linewidth=1, alpha=0.9
        )
        axes[0].plot(
            x, y["rewards_power_electric"], label="el. Lastspitzen", color=(0.929, 0.49, 0.192), linewidth=1, alpha=0.9
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
        axes[1].plot(x, cost_total, label="Kosten", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9)
        axes[1].plot(
            x, y["rewards_temperature_heat"], label="Wärmeversorgung", color=(0.75, 0, 0), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x, y["rewards_temperature_cool"], label="Kälteversorgung", color=(0.36, 0.61, 0.84), linewidth=1, alpha=0.9
        )
        axes[1].plot(
            x, y["rewards_switching"], label="Schaltvorgänge", color=(0.44, 0.19, 0.63), linewidth=1, alpha=0.9
        )
        axes[1].plot(x, y["rewards_other"], label="Sonstige", color=(0.1, 0.1, 0.1), linewidth=1, alpha=0.9)
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
        plt.tight_layout()
        figure.subplots_adjust(left=0.125, bottom=0.05, right=0.9, top=0.95, wspace=0.2, hspace=0.05)

        # set x/y axe and datetime begin
        x = self.episode_df.index
        y = self.episode_df
        dt_begin = self.episode_datetime_begin
        sampling_time = self.sampling_time

        # (1) - Plot actions as heatmap
        axes[0].set_yticks(np.arange(len(self.state_config.actions)))
        axes[0].set_yticklabels(
            ["Blockheizkraftwerk", "Gasbrennwertgerät", "Tauchsieder", "Wärmepumpe", "Kühlturm", "Kältemaschine"]
        )
        im = axes[0].imshow(
            y[self.state_config.actions].transpose(), cmap="Reds", vmin=0, vmax=1, aspect="auto", interpolation="none"
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
        axes[0].set_xticklabels(ticknames)
        # Rotate the tick labels and set their alignment.
        plt.setp(axes[0].get_yticklabels(), rotation=30, ha="right", va="center", rotation_mode="anchor")

        # (2) - Plot Storages
        axes[1].plot(
            x,
            [self.temperature_cost_prod_heat_min - 273.15] * len(x),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle="--",
            label="T minimal",
        )
        axes[1].plot(
            x,
            [self.temperature_cost_prod_heat_max - 273.15] * len(x),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle="--",
            label="T maximal",
        )

        axes[1].plot(x, y["s_temp_heat_storage_hi"] - 273.15, color=(192 / 255, 0, 0), label="Wärmespeicher (ob)")
        axes[1].plot(
            x, y["s_temp_heat_storage_lo"] - 273.15, color=(192 / 255, 0, 0), linestyle="--", label="Wärmespeicher (un)"
        )
        axes[1].plot(
            x,
            [self.temperature_cost_prod_cool_min - 273.15] * len(x),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle="--",
            label="T minimal",
        )
        axes[1].plot(
            x,
            [self.temperature_cost_prod_cool_max - 273.15] * len(x),
            color=(50 / 255, 50 / 255, 50 / 255),
            linestyle="--",
            label="T maximal",
        )

        axes[1].plot(
            x,
            y["s_temp_cold_storage_hi"] - 273.15,
            color=(91 / 255, 155 / 255, 213 / 255),
            linestyle="--",
            label="Kältespeicher (ob)",
        )
        axes[1].plot(
            x, y["s_temp_cold_storage_lo"] - 273.15, color=(91 / 255, 155 / 255, 213 / 255), label="Kältespeicher (un)"
        )
        # weather
        axes[1].fill_between(
            x, y["d_weather_drybulbtemperature"], color=(0.44, 0.68, 0.28), linewidth=0.1, alpha=0.3, label="Außenluft"
        )
        # settings
        axes[1].set_ylabel("Temperatur [°C]")
        axes[1].margins(x=0.0, y=0.1)
        axes[1].set_axisbelow(True)
        axes[1].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)
        axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[1].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)

        # (2) - Battery and relative Humidity (%)
        prodState = y["d_time_till_availability"].copy()
        prodState[prodState > 0] = 0
        prodState[prodState < 0] = 10
        axes[2].fill_between(x, prodState, color=(0.1, 0.1, 0.1), linewidth=0.1, alpha=0.3, label="Prod. Modus")
        axes[2].fill_between(
            x,
            y["d_weather_relativehumidity"],
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
        axes[3].plot(x, y["s_price_electricity"], color=(1.0, 0.75, 0.0), label="Strom")
        axes[3].plot(x, y["s_price_gas"], color=(0.65, 0.65, 0.65), label="Erdgas")
        axes[3].set_ylabel("Energiepreis (netto) [€/kWh]")
        axes[3].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        axes[3].margins(x=0.0, y=0.1)
        axes[3].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        axes[3].set_axisbelow(True)
        axes[3].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="-", linewidth=1)

        # (4) - Plot power
        axes[4].plot(
            x, y["s_electric_power_total"] * 1e-3, color=(1.0, 0.75, 0.0), linewidth=2, alpha=0.5, label="Strom Netz"
        )
        axes[4].plot(
            x,
            y["vs_electric_power_total_15min"] * 1e-3,
            color=(1.0, 0.75, 0.0),
            linewidth=2,
            alpha=0.9,
            label="Strom Netz (Ø15m)",
        )
        axes[4].plot(
            x,
            y["vs_gas_power_total_15min"] * 1e-3,
            color=(0.65, 0.65, 0.65),
            linewidth=2,
            alpha=0.9,
            label="Erdgas Netz (Ø15m)",
        )
        axes[4].plot(
            x,
            y["d_production_heat_power"] * 1e-3,
            color=(0.75, 0, 0),
            linestyle="--",
            linewidth=1,
            alpha=0.9,
            label="Wärmelast Prod.",
        )
        axes[4].plot(
            x,
            y["d_production_cool_power"] * 1e-3,
            color=(0.36, 0.61, 0.84),
            linestyle="--",
            linewidth=1,
            alpha=0.9,
            label="Kältelast Prod.",
        )
        axes[4].plot(
            x,
            y["d_production_electric_power"] * 1e-3,
            color=(1.0, 0.75, 0.0),
            linestyle="--",
            linewidth=1,
            alpha=0.9,
            label="Strom Prod.",
        )
        axes[4].plot(
            x,
            y["d_production_gas_power"] * 1e-3,
            color=(0.65, 0.65, 0.65),
            linestyle="--",
            linewidth=1,
            alpha=0.9,
            label="Erdgas Prod.",
        )
        axes[4].fill_between(
            x,
            (y["s_electric_power_total"] - y["d_production_electric_power"]) * 1e-3,
            label="Strom TGA",
            color=(1.0, 0.75, 0.0),
            linewidth=0.1,
            alpha=0.4,
        )
        axes[4].fill_between(
            x,
            (y["s_gas_power_total"] - y["d_production_gas_power"]) * 1e-3,
            label="Erdgas TGA",
            color=(0.65, 0.65, 0.65),
            linewidth=0.1,
            alpha=0.4,
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
            y["reward_energy_electric"].cumsum(),
            label="Strom (netto)",
            color=(1.0, 0.75, 0.0),
            linewidth=1,
            alpha=0.9,
        )
        axes[5].plot(
            x, y["reward_energy_gas"].cumsum(), label="Erdgas (netto)", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9
        )
        axes[5].plot(
            x,
            y["reward_energy_taxes"].cumsum(),
            label="Steuern & Umlagen",
            color=(0.184, 0.333, 0.592),
            linewidth=1,
            alpha=0.9,
        )
        axes[5].plot(
            x,
            y["reward_power_electric"].cumsum(),
            label="el. Lastspitzen",
            color=(0.929, 0.49, 0.192),
            linewidth=1,
            alpha=0.9,
        )
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
            + y["reward_energy_gas"].cumsum()
            + y["reward_energy_taxes"].cumsum()
            + y["reward_power_electric"].cumsum()
        )
        axes[6].plot(x, cost_total, label="Kosten", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9)
        axes[6].plot(
            x,
            y["reward_temperature_heat"].cumsum(),
            label="Wärmeversorgung",
            color=(0.75, 0, 0),
            linewidth=1,
            alpha=0.9,
        )
        axes[6].plot(
            x,
            y["reward_temperature_cool"].cumsum(),
            label="Kälteversorgung",
            color=(0.36, 0.61, 0.84),
            linewidth=1,
            alpha=0.9,
        )
        axes[6].plot(
            x, y["reward_switching"].cumsum(), label="Schaltvorgänge", color=(0.44, 0.19, 0.63), linewidth=1, alpha=0.9
        )
        axes[6].plot(x, y["reward_other"].cumsum(), label="Sonstige", color=(0.1, 0.1, 0.1), linewidth=1, alpha=0.9)
        axes[6].plot(x, y["reward_total"].cumsum(), label="Gesamt", color=(0.1, 0.1, 0.1), linewidth=2)
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
