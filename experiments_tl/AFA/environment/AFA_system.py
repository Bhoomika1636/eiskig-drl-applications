from __future__ import annotations

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


class AFA_System(BaseEnvSim):
    """
    SupplysystemA environment class from BaseEnvSim.

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
    description = "Parameter Identification Outer Capillary Tube Mats"
    fmu_name = "AFA_erweitert"

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
        temperature_start_east,
        temperature_start_west,
        #temperature_start_basement,
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
        self.temperature_start_west = temperature_start_west
        self.temperature_start_east = temperature_start_east
        #self.temperature_start_basement = temperature_start_basement

        # initialize episode statistics
        self.episode_statistics = {}
        self.episode_archive = []
        self.episode_df = pd.DataFrame()

        # set integer prediction steps (15m,1h,6h)
        self.n_steps_15m = int(900 // self.sampling_time)
        self.n_steps_1h = int(3600 // self.sampling_time)
        self.n_steps_6h = int(21600 // self.sampling_time)

        # initialize integrators and longtime stats
        self.initial_resets = 0

        # define state variables
        state_var_tuple = (
            #define fmu inputs
            # StateVar(
            #     name="u_Pump",
            #     ext_id="u_PU600",
            #     scenario_id="u_PU600",
            #     is_ext_input=True,
            #     from_scenario=True,
            #     #is_agent_action=True, #this is an agent action now, but maybe it would be better to have the value in a scenario file and use this instead
            #     low_value=int(0),
            #     high_value=int(1),
            # ),
            StateVar(
                name="Pump_control",
                ext_id="PU600_onoff",
                scenario_id="controlpump",
                is_ext_input=True,
                from_scenario=True,
                #is_agent_action=True,
                low_value=int(0),
                high_value=int(1),
            ),
            StateVar(
                name="RV640_control",
                ext_id="RV640_onoff",
                scenario_id="u_rv640",
                is_ext_input=True,
                from_scenario=True,
                #is_agent_action=True, #this is an agent action now, but maybe it would be better to have the value in a scenario file and use this instead
                low_value=int(0),
                high_value=int(1),
            ),
            StateVar(
                name="RV660_control",
                ext_id="RV660_onoff",
                scenario_id="u_rv660",
                is_ext_input=True,
                from_scenario=True,
                #is_agent_action=True, #this is an agent action now, but maybe it would be better to have the value in a scenario file and use this instead
                low_value=int(0),
                high_value=int(1),
            ),
            StateVar(
                name="RV600_control",
                ext_id="RV600_onoff",
                scenario_id="u_rv600",
                is_ext_input=True,
                from_scenario=True,
                #is_agent_action=True,
                low_value=int(0),
                high_value=int(1),
            ),
            # ambient temperature
            StateVar(
                name="T_ambient",
                ext_id="T_amb",
                scenario_id="air_temperature",
                from_scenario=True,
                is_ext_input=True,
                is_agent_observation=True,
                low_value=int(-100),
                high_value=int(500),
            ),
            # sun intensity
            StateVar(
                name="global_radiation",
                ext_id="Solar_irradiation",
                scenario_id="global_radiation",
                from_scenario=True,
                is_ext_input=True,
                is_agent_observation=True,
                low_value=int(-100),
                high_value=int(100),
            ), 
            StateVar(
                name="Wind_dir",
                ext_id="Wind_direction",
                scenario_id="wind_direction",
                from_scenario=True,
                is_ext_input=True,
                is_agent_observation=True,
                low_value=int(0),
                high_value=int(360.0),
            ),
            StateVar(
                name="Wind_speed",
                ext_id="Wind_speed",
                scenario_id="wind_speed",
                from_scenario=True,
                is_ext_input=True,
                #is_agent_observation=True,
                low_value=int(-100),
                high_value=int(100),
            ),
            StateVar(
                name="T_int_PU_real",
                ext_id="T_int_PU_real",
                scenario_id="t_int_pu600",
                from_scenario=True,
                #is_ext_input=True,
                #is_agent_observation=True,
                low_value=int(-100),
                high_value=int(100),
            ),
            StateVar(
                name="T_ext_PU_real",
                ext_id="T_ext_PU_real",
                scenario_id="t_ext_pu600",
                from_scenario=True,
                #is_ext_input=True,
                #is_agent_observation=True,
                low_value=int(-100),
                high_value=int(100),
            ),
            StateVar(
                name="T_TS660_real",
                ext_id="T_TS660_real",
                scenario_id="t_ts660",
                from_scenario=True,
                #is_ext_input=True,
                #is_agent_observation=True,
                low_value=int(-100),
                high_value=int(100),
            ),
            StateVar(
                name="T_TS640_real",
                ext_id="T_TS640_real",
                scenario_id="t_ts640",
                from_scenario=True,
                #is_ext_input=True,
                #is_agent_observation=True,
                low_value=int(-100),
                high_value=int(100),
            ),      
            # internal variables
            # StateVar(
            #     name="conductivity",
            #     ext_id="conductivity_outside_layer",
            #     scenario_id="conductivity_outside_layer",
            #     from_scenario=True,
            #     is_ext_input=True,
            #     is_agent_observation=True,
            #     low_value=1.0,
            #     high_value=5.0,
            # ),
            # StateVar(
            #     name="heat_capacity",
            #     ext_id="heat_capacity_outside_layer",
            #     scenario_id="heat_capacity_outside_layer",
            #     from_scenario=True,
            #     is_ext_input=True,
            #     is_agent_observation=True,
            #     low_value=900.0,
            #     high_value=1100.0,
            # ),
            # StateVar(
            #     name="density",
            #     ext_id="density_outside_layer",
            #     scenario_id="density_outside_layer",
            #     from_scenario=True,
            #     is_ext_input=True,
            #     is_agent_observation=True,
            #     low_value=2.0,
            #     high_value=3.0,
            # ),
            # StateVar(
            #     name="emission_coefficient",
            #     ext_id="ebsilon_concrete",
            #     scenario_id="ebsilon_concrete",
            #     from_scenario=True,
            #     is_ext_input=True,
            #     is_agent_observation=True,
            #     low_value=0.92,
            #     high_value=0.97,
            # ),
            # StateVar(
            #     name="Volume_basement",
            #     ext_id="V_glycol_basement",
            #     scenario_id="V_glycol_basement",
            #     from_scenario=True,
            #     is_ext_input=True,
            #     is_agent_observation=True,
            #     low_value=120.0,
            #     high_value=150.0,
            # ),
            # Sensor outputs         
            StateVar(
                name="T_Pumpe_intern",
                ext_id="T_in_PU600.T",
                is_ext_output=True,
                low_value=int(-100),
                high_value=int(100),
            ),
            StateVar(
                name="T_Pumpe_external",
                ext_id="T_PU_Ext.T",
                is_ext_output=True,
                low_value=int(-100),
                high_value=int(100),
            ),
            StateVar(
                name="T_PWT6_return",
                ext_id="T_AFA_return.T",
                is_ext_output=True,
                low_value=int(-100),
                high_value=int(100),
            ),

            StateVar(
                name="T_TS660",
                ext_id="TS_660.T",
                is_ext_output=True,
                low_value=int(-100),
                high_value=int(100),
            ),
             StateVar(
                name="T_TS640",
                ext_id="TS_640.T",
                is_ext_output=True,
                low_value=int(-100),
                high_value=int(100),
            ),
            StateVar(
                name="V_PU600",
                ext_id="vFlow_PU600.V_flow",
                is_ext_output=True,
               low_value=int(0),
                high_value=int(100),
            ),
            StateVar(
                name="V_RV640",
                ext_id="vFlow_RV640.V_flow",
                is_ext_output=True,
                low_value=int(0),
                high_value=int(100),
            ),
            StateVar(
                name="V_RV660",
                ext_id="vFlow_RV660.V_flow",
                is_ext_output=True,
                low_value=int(0),
                high_value=int(100),
            ),
        )

        # build final state_config
        self.state_config = StateConfig(*state_var_tuple)

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

    def step(self, action: np.ndarray) -> StepResult:
        """Perform one time step and return its results.

        :param action: Actions to perform in the environment.
        :return: The return value represents the state of the environment after the step was performed.
        """

        # initialize additional_state and create state backup
        self.state_backup = self.state.copy()
        self.additional_state = {}

        #write action to _action
        _action = action


        # check actions for vilidity, perform simulation step and load new external values for the next time step
        self._actions_valid(_action)
        self.state["step_success"], _ = self._update_state(_action)

        # check if state is in valid boundaries
        self.state["step_abort"] = False if StateConfig.within_abort_conditions(self.state_config, self.state) else True

        # check if episode is over or not
        done = self._done() or not self.state["step_success"]
        done = done if not self.state["step_abort"] else True

        # calculate reward
        reward = self.calc_reward()

        # update state_log
        self.state_log.append(self.state)

        observations = self._observations()

        return observations, reward, done, False, {}



    def calc_reward(self):
        """Calculates the step reward. Needs to be called from step() method after state update.

        :return: Normalized or non-normalized reward
        :rtype: Real
        """

        # for this application no reward calculation is needed
        reward = 0
        

        return reward

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
            self._np_random, _ = seeding.np_random(seed - 1)

        # delete long time storage, since it takes up too much memory during training
        self.state_log_longtime = []

        # # save episode's stats
        if self.n_steps > 0:

            # create dataframe from state_log
            self.episode_df = pd.DataFrame(self.state_log)

            # derive certain episode statistics for logging and plotting
            self.episode_statistics["datetime_begin"] = self.ts_current.index[0].strftime("%Y-%m-%d %H:%M")
            self.episode_statistics["time"] = time.time() - self.episode_timer
            self.episode_statistics["n_steps"] = self.n_steps

            # # append episode store to episodes' archive
            self.episode_archive.append(list(self.episode_statistics.values()))

        # initialize additional state, since super().reset is used
        self.additional_state = {}

        # set FMU parameters for initialization
        self.model_parameters = {}
        self.model_parameters["T_start_west"] = self.temperature_start_west
        self.model_parameters["T_start_east"] = self.temperature_start_east
        #self.model_parameters["T_start_basement"] = self.temperature_start_basement

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

        # reset RNG, hack to work around the current issue of non deterministic seeding of first episode
        if self.initial_resets == 0:
            self._np_random = None
        self.initial_resets += 1

        # receive observations from simulation
        observations = super().reset(seed=seed)

        return observations


    def render_episodes(self):
        """
        output plot for all episodes
        see pandas visualization options on https://pandas.pydata.org/pandas-docs/stable/user_guide/visualization.html
        https://matplotlib.org/gallery/images_contours_and_fields/image_annotated_heatmap.html

        Parameters
        -----
        mode : (str)
        """
        # # create dataframe
        # episode_archive_df = pd.DataFrame(self.episode_archive, columns=list(self.episode_statistics.keys()))

        # # write all data to csv after every episode
        # episode_archive_df.to_csv(
        #     path_or_buf=os.path.join(
        #         self.path_results,
        #         self.config_run.name
        #         + "_"
        #         + str(self.n_episodes).zfill(4)
        #         + "-"
        #         + str(self.env_id).zfill(2)
        #         + "_all-episodes.csv",
        #     ),
        #     sep=";",
        #     decimal=".",
        # )

        # # write another aggregated csv that contains all episodes (necessary for mpc and mpc_simple)
        # csvpath = os.path.join(self.path_results, "all-episodes.csv")
        # if os.path.exists(
        #     csvpath
        # ):  # check if aggregated file already exists, which is the case when multiple runs are done with mpc and mpc_simple
        #     tocsvmode = "a"
        #     tocsvheader = False
        # else:
        #     tocsvmode = "w"
        #     tocsvheader = True
        # # write data to csv
        # episode_archive_df.tail(1).to_csv(path_or_buf=csvpath, sep=";", decimal=".", mode=tocsvmode, header=tocsvheader)

        # # plot settings
        # # create figure and axes with custom layout/adjustments
        # figure = plt.figure(figsize=(14, 14), dpi=200)
        # axes = []
        # axes.append(figure.add_subplot(2, 1, 1))
        # axes.append(figure.add_subplot(2, 1, 2, sharex=axes[0]))

        # # fig, axes = plt.subplots(2, 1, figsize=(10, 4), dpi=200)
        # x = np.arange(len(episode_archive_df.index))
        # y = episode_archive_df

        # # (1) Costs
        # axes[0].plot(
        #     x, y["rewards_energy_electric"], label="Strom (netto)", color=(1.0, 0.75, 0.0), linewidth=1, alpha=0.9
        # )
        # axes[0].plot(
        #     x, y["rewards_energy_gas"], label="Erdgas (netto)", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9
        # )
        # axes[0].plot(
        #     x, y["rewards_energy_taxes"], label="Steuern & Umlagen", color=(0.184, 0.333, 0.592), linewidth=1, alpha=0.9
        # )
        # axes[0].plot(
        #     x, y["rewards_power_electric"], label="el. Lastspitzen", color=(0.929, 0.49, 0.192), linewidth=1, alpha=0.9
        # )
        # axes[0].set_ylabel("kum. Kosten [€]")
        # axes[0].set_xlabel("Episode")
        # axes[0].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        # axes[0].margins(x=0.0, y=0.1)
        # axes[0].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="solid", linewidth=1)

        # # (2) Rewards
        # cost_total = (
        #     y["rewards_energy_electric"]
        #     + y["rewards_energy_gas"]
        #     + y["rewards_energy_taxes"]
        #     + y["rewards_power_electric"]
        # )
        # axes[1].plot(x, cost_total, label="Kosten", color=(0.65, 0.65, 0.65), linewidth=1, alpha=0.9)
        # axes[1].plot(
        #     x, y["rewards_temperature_heat"], label="Wärmeversorgung", color=(0.75, 0, 0), linewidth=1, alpha=0.9
        # )
        # axes[1].plot(
        #     x, y["rewards_switching"], label="Schaltvorgänge", color=(0.44, 0.19, 0.63), linewidth=1, alpha=0.9
        # )
        # axes[1].plot(x, y["rewards_other"], label="Sonstige", color=(0.1, 0.1, 0.1), linewidth=1, alpha=0.9)
        # axes[1].plot(x, y["rewards_total"], label="Gesamt", color=(0.1, 0.1, 0.1), linewidth=2)
        # axes[1].set_ylabel("kum. + fikt. Kosten [€]")
        # axes[1].set_xlabel("Episode")
        # axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc="center left", ncol=1, fontsize="x-small")
        # axes[1].margins(x=0.0, y=0.1)
        # axes[1].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="solid", linewidth=1)

        # plt.savefig(
        #     os.path.join(
        #         self.path_results,
        #         self.config_run.name
        #         + "_"
        #         + str(self.n_episodes).zfill(4)
        #         + "-"
        #         + str(self.env_id).zfill(2)
        #         + "_all-episodes.png",
        #     )
        # )
        # plt.close(figure)

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
        figure = plt.figure(figsize=(14, 22), dpi=200)
        axes = []
        axes.append(figure.add_subplot(5, 1, 1))
        axes.append(figure.add_subplot(5, 1, 2, sharex=axes[0]))
        axes.append(figure.add_subplot(5, 1, 3, sharex=axes[0]))
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
        axes[0].set_yticklabels([])
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
        axes[0].grid(which="minor", color="w", linestyle="solid", linewidth=3)
        axes[0].xaxis.grid(color=(1, 1, 1, 0.1), linestyle="solid", linewidth=1)
        # add ticks and tick labels
        axes[0].set_xticks(tickpos)
        axes[0].set_xticklabels(ticknames)
        # Rotate the tick labels and set their alignment.
        plt.setp(axes[0].get_yticklabels(), rotation=30, ha="right", va="center", rotation_mode="anchor")

        # (2) - Plot Scenario Data

        axes[1].plot(x, y["T_ambient"], color=(192 / 255, 0, 0), label="Außentemperatur")
        #axes[1].plot(x, y["global_radiation"] , color=(1, 0, 0), label="global_radiation")
        ax2 = axes[1].twinx()
        ax2.plot(x, y["global_radiation"], color="blue", label="Globalstrahlung (10 Min)")

        # settings
        axes[1].set_ylabel("Temperatur [°C]")
        axes[1].margins(x=0.0, y=0.1)
        axes[1].set_axisbelow(True)
        axes[1].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="solid", linewidth=1)
        #axes[1].legend(bbox_to_anchor=(1.0, 0.5), loc="center right", ncol=1, fontsize="x-small")
        axes[1].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        ax2.set_ylabel("Radiation [J/cm²]")

        handles1, labels1 = axes[1].get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        axes[1].legend(handles1 + handles2, labels1 + labels2, loc="upper right", ncol=1, fontsize="x-small")

        # (3) - Simulated Temperaturs vs. Real Temperatures
        axes[2].plot(x, y["T_Pumpe_intern"]-273.15, color="blue", linestyle="dotted", label="T_PU_int_calc")
        axes[2].plot(x, y["T_int_PU_real"], color="green", linestyle="solid", label="T_PU_int_real")
        axes[2].plot(x, y["T_TS660"]-273.15, color=(1.0, 0.75, 0.0), label="T_TS660_calc")
        axes[2].plot(x, y["T_TS660_real"], color="red", linestyle="-", label="T_TS660_real")
        
        #axes[2].plot(x, y["T_TS640"], color=(0.65, 0.65, 0.65), label="T_TS640")

        #settings
        axes[2].set_ylabel("Temperatur [°C]")
        axes[2].legend(loc="upper right", ncol=1, fontsize="x-small")
        axes[2].margins(x=0.0, y=0.1)
        axes[2].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        axes[2].set_axisbelow(True)
        axes[2].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="solid", linewidth=1)

        # (4) - Plot other stuff here
        axes[3].plot(x, y["T_Pumpe_intern"]-273.15, color=(1.0, 0.75, 0.0), label="T_PU_int_calc")
        axes[3].plot(x, y["T_Pumpe_external"]-273.15, color="red", linestyle="dotted", label="T_PU_ext_calc")
        axes[3].plot(x, y["T_int_PU_real"], color=(0.65, 0.65, 0.65), label="T_PU_int_real")
        axes[3].plot(x, y["T_ext_PU_real"], color="blue", label="T_PU_ext_real")
        #axes[2].plot(x, y["T_TS640"], color=(0.65, 0.65, 0.65), label="T_TS640")

        #settings
        axes[3].set_ylabel("Temperatur [°C]")
        axes[3].legend(loc="upper right", ncol=1, fontsize="x-small")
        axes[3].margins(x=0.0, y=0.1)
        axes[3].tick_params(axis="x", which="both", bottom=False, top=False, labelbottom=False)
        axes[3].set_axisbelow(True)
        axes[3].grid(color=(0.9, 0.9, 0.9, 0.1), linestyle="solid", linewidth=1)

        # (5) - Plot other stuff here
        axes[4].plot(x, y["Wind_speed"], color=(1.0, 0.75, 0.0), label="Wind_speed")

        axes[4].set_ylabel("Windgeschwindigkeit [m/s]")
    

        # (6) - Plot other stuff here
        

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


        plotter_available = False # disable plotter, this can be deleted, if plotter is implemented properly
        if plotter_available:
            # HTML PLotter

            xaxis_title = "Zeit (UTC)"
            x2 = self.episode_df.index

            actions = Heatmap(x2, xaxis_title=xaxis_title, height=750, width=1900)
            actions.line(y["u_immersionheater"], name="Tauchsieder")
            actions.line(y["u_condensingboiler"], name="Gasbrennwertgerät")
            actions.line(y["u_combinedheatpower"], name="Blockheizkraftwerk")

            storages = Linegraph(x2, xaxis_title=xaxis_title, yaxis_title="Temperatur [°C]", height=750, width=1900)
            storages.line(
                [self.temperature_cost_prod_heat_min - 273.15] * len(x), "T minimal", color="rgb(50,50,50)", dash="dash"
            )
            storages.line(
                [self.temperature_cost_prod_heat_max - 273.15] * len(x), "T maximal", color="rgb(50,50,50)", dash="dash"
            )
            storages.line(y["s_temp_heat_storage_hi"] - 273.15, "Wärmespeicher (ob)", color="rgb(192,0,0)")
            storages.line(y["s_temp_heat_storage_lo"] - 273.15, "Wärmespeicher (un)", color="rgb(192,0,0)", dash="dash")

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
            rewards.line(y["reward_switching"].cumsum(), "Schaltvorgänge", width=1, color="rgb(112,48,160)")
            rewards.line(y["reward_other"].cumsum(), "Sonstige", width=1, color="rgb(25,25,25)")
            rewards.line(y["reward_total"].cumsum(), "Gesamt", color="rgb(25,25,25)")

            plot = ETA_Plotter(actions, storages, prices, power, costs, rewards)
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