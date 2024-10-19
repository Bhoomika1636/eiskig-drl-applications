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

try:
    from plotter.plotter import ETA_Plotter, Heatmap, Linegraph

    plotter_available = True
except ImportError as e:
    plotter_available = False


from eta_utility.type_hints import StepResult, TimeStep
from gym import spaces
from scipy.special import expit

import plotly.express as px
import warnings
import common.helpers as helpers
import pprint

log = get_logger("eta_x.envs")


class BoschSystem(BaseEnvSim):
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
    description = "(c) Heiko Ranzau, Niklas Panten and Benedikt Grosch"
    fmu_name = "BoschCoolingSystem_weather_4_2"

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,  # config_run contains run_name, general_settings, path_settings, env_info
        seed: int | None = None,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        random_sampling,
        sampling_time: float,  # from settings section of config file
        episode_duration: TimeStep | str,  # from settings section of config file
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        scenario_files: Sequence[Mapping[str, Any]],
        discretize_action_space,
        up_down_discrete,
        reward_shaping,
        use_policy_shaping=False,
        sim_parameters: dict | None,
        **kwargs: Any,
    ):
        super().__init__(
            env_id,
            config_run,
            seed,
            verbose,
            callback,
            sampling_time=sampling_time,
            episode_duration=episode_duration,
            scenario_time_begin=scenario_time_begin,
            scenario_time_end=scenario_time_end,
            **kwargs,
        )
        # custom class variables
        self.discretize_action_space = discretize_action_space
        self.up_down_discrete = up_down_discrete
        self.reward_shaping = reward_shaping
        self.random_sampling = random_sampling
        self.policy_shaping_active = 0
        self.use_policy_shaping = use_policy_shaping
        #
        self.sim_parameters = sim_parameters
        # set same random seed for np ---> deactivate for real model training and testing !!!! only for DEBUG
        np.random.seed(seed)

        # initialize episode statistics
        self.episode_statistics = {}
        self.episode_archive = []
        self.episode_df = pd.DataFrame()

        state_var_tuple = helpers.getStateVars('experiments_hr\BoschCoolingSystem2\environment\BoschStateVars.csv', includeWeather=True, includeEnergyPrice=True)
        # build final state_config
        self.state_config = StateConfig(*state_var_tuple)
        self.lastcwDemand = 0
        self.lastkueDemand = 0
        # import all scenario files
        self.scenario_data = self.import_scenario(*scenario_files)

        # get action_space
        # TODO: implement this functionality into utility functions
        if self.discretize_action_space:
            # get number of actions agent has to give from state_config
            self.n_action_space = len(self.state_config.actions)
            if up_down_discrete:
                # set 3 discrete actions (increase,decrease,equal) per control variable
                self.action_space = spaces.MultiDiscrete(np.full(self.n_action_space, 3))
            else:
                self.action_space = spaces.MultiDiscrete([2,2,2,2,2,5,5,3,3,3,3,3,3,5,5])
            # customized for P1,...,KKM,...KT,...,WT
            self.action_disc_step = [[0, 1], [0,1], [0, 1], [0, 1], [0, 1], [6,7,8,9,10], [6,7,8,9,10], [0, 0.5, 1],[0, 0.5, 1],[0, 0.5, 1],[0, 0.5, 1],[0, 0.5, 1],[0, 0.5, 1],[0, 0.25, 0.5, 0.75, 1],[0, 0.25, 0.5, 0.75, 1]]            # initialize action
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

        # set random necesarry coolingdemand -- only if not taken from scenario !!!
        # self.state['CoolingDemandCoolingWater'] = 0.0
        # self.state['CoolingDemandColdWater'] = 0.0
        # if (self.n_steps % int(3600*1.5/self.sampling_time)) == 0: # update demand every 1.5 hours 
        #     self.lastcwDemand=np.random.uniform(0, 2000000)
        #     self.lastkueDemand = np.random.uniform(0, 2000000)
        #     self.state['CoolingDemandCoolingWater'] = self.lastkueDemand
        #     self.state['CoolingDemandColdWater'] = self.lastcwDemand
        # else:
        #     self.state['CoolingDemandCoolingWater'] = self.lastkueDemand
        #     self.state['CoolingDemandColdWater'] = self.lastcwDemand
        
        # MODIFIED_ Random Action choosen 
        action = np.random.random_integers(low=0, high=2, size=action.shape)
        # convert discrete actions into continious space if discrete action space is chosen
        if self.discretize_action_space:
            _action = self.convert_disc_action(action)
        else:
            _action = action
        # save agent action to allow one state to carry the non overwritten action 
        agent_action = _action
        # use action overwriting if enabled
        self.policy_shaping_active = 0
        if self.use_policy_shaping:
            # warm net check
            _action = self.policy_shape_cold(self.state['Temperature_cwCircuit_in'], _action, punishment=2)
            _action = self.policy_shape_cooling(self.state['T_in_coolingCircuit'], _action, punishment=2)
            _action = self.policy_shape_pumps(_action, punishment=2)
        
        #print(f"Start: nSteps:{self.n_steps}, EnvID:{self.env_id}")
        #print(_action)
        # check actions for vilidity, perform simulation step and load new external values for the next time step
        self._actions_valid(_action) # why every step - at the beginning is sufficient ???
        self.state["step_success"], _ = self._update_state(_action) # updates state incomplete if it fails-> issues with withinAbortConditions
        if not self.state["step_success"]:
            raise RuntimeError
        # check if state is in valid boundaries
        try:
            #print(self.state)
            self.state["step_abort"] = False if StateConfig.within_abort_conditions(self.state_config, self.state) else True
        except:
            print(self.state)
            print('-------')
            print(self.state_backup)
            self.state = self.state_backup
            self.state["step_abort"] = False
            self.state["step_success"] = False
        
        # if policy shaping is active, I dont want it to stop. And I want to set the action state (u not s_u) to the original agent action, not to the overwritten one
        if self.use_policy_shaping is True:
            self.state["step_abort"] = False
            self.reset_agent_action(agent_action) 
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

        return observations, reward, done, {}

    def calc_reward(self):
        """Calculates the step reward. Needs to be called from step() method after state update.

        :return: Normalized or non-normalized reward
        :rtype: Real
        """
        # switching costs helper function
        def switch_cost(u_old, u_new, penalty):
            if (u_old <= 0 and u_new > 0) or (u_old > 0 and u_new <= 0):  # if u_old != u_new :
                return penalty  # else 0.1*penalty*abs(u_new - u_old)....in example penalty= 1 to 4
            else:
                return 0

        # switching costs
        #P1,2,3,KKMon1,2,KKM_T1,2,KT1,2,3,4,5,6,WT1,2
        switching_reward = -1
        if self.discretize_action_space:
            switchingArray = np.array([switch_cost(self.state[x], self.state_backup[x], switching_reward) for x in self.state.keys() if "s_u" in x])
            switchingArray = switchingArray*np.array([1,1,1, 1,1,1,1, 1,1,1,1,1,1 ,1,1])
            self.state["reward_switching"] = switchingArray.sum()
        else:
            switchingArray = np.array([-abs((self.state[x]-self.state_backup[x])/self.sampling_time) for x in self.state.keys() if "s_u" in x])
            # only for eval begin 
            switchingArray = np.array([abs((self.state[x]-self.state_backup[x])) for x in self.state.keys() if "s_u" in x])
            switchingArray[switchingArray<0.25] = 0
            switchingArray[switchingArray>=0.25] = -1
            # only for eval end 
            switchingArray = switchingArray*np.array([1,1,1, 1,1,1,1,1,1,1,1,1,1,1,1])
            self.state["reward_switching"] = switchingArray.sum() 
        
        # temperature costs (when availability of temperature levels are needed)
        self.state["reward_temperature_heat"] = reward_boundary(
            self.state["T_in_coolingCircuit"],
            17,
            19,
            0,
            5, # = 1 in example
            smoothed=True,
            k=7,
            k_lin=0.1
        ) + reward_boundary(
            self.state["Temperature_cwCircuit_in"],
            7,
            9,
            0,
            5,
            smoothed=True,
            k=7,
            k_lin = 0.1
        )

        # enery costs 
        self.state["energy_cost_cooling_eur/h"] = self.state["s_price_electricity"] * self.state['P_el_ges_kw']
        self.state["reward_energy_electric"] = (
            -self.state["energy_cost_cooling_eur/h"] * 0.4
        )
        # useful evaluation metrics
        self.state["energy_cost_cooling_eur"] =  self.state["energy_cost_cooling_eur/h"]* self.sampling_time/3600 # cost accumulation for one step ---> sum for accumulation 
        self.state["cooling_system_coeff"] = 1000*self.state['P_el_ges_kw']/(self.state["CoolingDemandColdWater"]+self.state["CoolingDemandCoolingWater"]) #--> mean for accumulation 
        self.state["switches_per_hour"] = (self.state["reward_switching"]/switching_reward)*3600/self.sampling_time # switchings per hour --> take mean for accumulation 
        # cost for crossing abort conditions -> doesnt make sense to adapt to the time it runs, because it's always as bad to cross boundries 
        if (self.state["step_abort"]): # or not self.state["step_success"]
            self.state["reward_abort"] = (-50000)
        else:
            self.state["reward_abort"]=0

        # policyshaping costs
        self.state["reward_policy_overwrite"] = - 3*self.policy_shaping_active

        # total reward
        self.state["reward_total"] = (
            self.state["reward_switching"]
            + self.state["reward_temperature_heat"]
            + self.state["reward_abort"]
            + self.state["reward_energy_electric"]
            + self.state["reward_policy_overwrite"]
        )
        # return the reward
        return self.state["reward_total"]

    def reset(self) -> np.ndarray:

        """Resets the state of the environment and returns an initial observation.

        Returns: observation (object): the initial observation of the
            space.
        """

        # delete long time storage, since it takes up too much memory during training
        self.state_log_longtime = []

        # # save episode's stats
        if self.n_steps > 0:

            # create dataframe from state_log
            self.episode_df = pd.DataFrame(self.state_log)

            # derive certain episode statistics for logging and plotting
            #rewards
            self.episode_statistics["rewards_total"] = self.episode_df["reward_total"].sum()
            self.episode_statistics["rewards_switching"] = self.episode_df["reward_switching"].sum()
            self.episode_statistics["rewards_temperature_heat"] = self.episode_df["reward_temperature_heat"].sum()
            self.episode_statistics["reward_abort"] = self.episode_df["reward_abort"].sum()
            self.episode_statistics["reward_policy_overwrite"] = self.episode_df["reward_policy_overwrite"].sum()
            self.episode_statistics["reward_energy_electric"] = self.episode_df["reward_energy_electric"].sum()
            # additional helpful stats
            self.episode_statistics["energy_cost_cooling_eur"] = self.episode_df["energy_cost_cooling_eur"].sum()
            self.episode_statistics["cooling_system_coeff"] = self.episode_df["cooling_system_coeff"].mean()
            self.episode_statistics["switches_per_hour"] = self.episode_df["switches_per_hour"].mean()
            # time refs 
            self.episode_statistics["time"] = time.time() - self.episode_timer
            self.episode_statistics["n_steps"] = self.n_steps
            self.episode_statistics["datetime_begin"] = self.ts_current.index[0].strftime("%Y-%m-%d %H:%M")

            # # append episode store to episodes' archive
            self.episode_archive.append(list(self.episode_statistics.values()))

        # initialize additional state, since super().reset is used
        self.additional_state = {}

        # set FMU parameters for initialization
        self.model_parameters = {}
        self.model_parameters["coolingTowerCircuitFull.tank_coolingWater.T_start"] = 273.15 + self.np_random.uniform(12, 28)
        self.model_parameters['coolingTowerCircuitFull.tank_warmWater.T_start'] = 273.15 + self.np_random.uniform(20, 40)
        # for runtime optimization
        if self.sim_parameters is not None:
            for key in self.sim_parameters:
                self.model_parameters[key] = self.sim_parameters[key]
        # self.model_parameters["system.T_ambient"] = 273.15 + self.np_random.uniform(6, 20)

        # self.model_parameters['coolingWaterCircuitFull.pipe_simple.vol.T_start'] = 273.15 + 18.0  # self.np_random.uniform(12, 28)
        # self.model_parameters['compressionChillerCircuit.coldWaterCircuit.pipe_simple.vol.T_start'] = 273.15 + 8.0 # self.np_random.uniform(12, 28)
        
        # get current slice of timeseries dataframe, extended by maximum prediction horizon (6h)
        # and one additional step because step 0 = init conditions
        self.ts_current = timeseries.df_time_slice(
            self.scenario_data,
            self.scenario_time_begin,
            self.scenario_time_end,
            self.episode_duration +  1 * self.sampling_time,
            random=self.np_random if self.random_sampling else False,
        )

        # read current date time
        self.episode_datetime_begin = self.ts_current.index[0]
        self.additional_state["vs_time_daytime"] = self.episode_datetime_begin.hour
        
        # add time of the year in seconds
        starttime_of_year = pd.Timestamp(str(self.ts_current.index[self.n_steps].year) + "-01-01 00:00")
        self.additional_state["d_weather_time"] = pd.Timedelta(
            self.ts_current.index[self.n_steps] - starttime_of_year
        ).total_seconds()

        # receive observations from simulation
        observations = super().reset()

        return observations

    def convert_disc_action(self, action_disc):
        """
        converts discrete actions from agent to continious FMU input space
        """
        float_action = []

        for idx, val in enumerate(action_disc):
            if self.up_down_discrete:
                self.action_disc_index[idx] = np.clip(self.action_disc_index[idx] + (val - 1), 0, len(self.action_disc_step[idx]) - 1)
            else:
                self.action_disc_index[idx] = val
            float_action.append(self.action_disc_step[idx][self.action_disc_index[idx]])

        return np.array(float_action)

    def render_episodes(self, name_suffix=""):
        """

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
                + "_all-episodes"+ name_suffix+".csv",
            ),
            sep=";",
            decimal=".",
        )

        # plot settings
        # create figure and axes with custom layout/adjustments
        for i in episode_archive_df.columns:
            try:
                episode_archive_df[i] = episode_archive_df[i].astype(float)
            except ValueError:
                episode_archive_df.drop(columns=i,inplace=True)
        try:
            fig = px.line(episode_archive_df, x=episode_archive_df.index, y=episode_archive_df.columns)
            fig.write_html(os.path.join(
                    self.path_results,
                    self.config_run.name
                    + "_"
                    + str(self.n_episodes).zfill(4)
                    + "-"
                    + str(self.env_id).zfill(2)
                    + "_all-episodes.html",
                ))
        except:
            print(episode_archive_df)
            warnings.warn("There was an issue with plotting the all_episode stats render!(-probably empty for some reason)")

    def render(self, mode="human", name_suffix=""):
        """
        output plots for last episode

        Parameters
        -----
        mode : (str)
        """
        try:
            # set x/y axe and datetime begin
            x = self.episode_df.index
            y = self.episode_df
            # build time index for episode timecut
            dt_begin = self.episode_datetime_begin
            sampling_time = self.sampling_time
            
            timeRange = np.arange(
                0,
                self.n_steps*self.sampling_time+1,
                sampling_time,
            )
            dt_begin = dt_begin.replace(microsecond=0, second=0, minute=0)
            df_time = pd.DataFrame()
            df_time['dateTime'] = [dt_begin + timedelta(seconds=i) for i in timeRange]
            # print(self.episode_df['u_KT_1'].sum()+self.episode_df['u_KT_2'].sum()+self.episode_df['u_KT_3'].sum()+self.episode_df['u_KT_4'].sum())
            
            # add datetime to df of states
            try:
                self.episode_df.index = df_time["dateTime"]
            except: # is relevant for Eval callback 
                warnings.warn("Fitting DateTime not found - probably there was a reset before the final render call.")
                # print(f"Env id:{self.env_id}, df:{self.episode_df}")
            
            for i in self.episode_df.columns:
                try:
                    self.episode_df[i] = self.episode_df[i].astype(float)
                except:
                    self.episode_df.drop(columns=i,inplace=True)
            
            #print(f"{name_suffix}_Env id:{self.env_id},\n df.columns: {self.episode_df.columns}\n\n")
            #print(f"{name_suffix}_Env id:{self.env_id},\n df.columns: {self.episode_df.index}\n\n")
            
            fig = px.line(self.episode_df, x=self.episode_df.index, y=self.episode_df.columns)
            fig.write_html(os.path.join(
                self.path_results,
                self.config_run.name
                + "_"
                + str(self.n_episodes).zfill(3)
                + "-"
                + str(self.env_id).zfill(2)
                + "_episode"
                + name_suffix
                + ".html",
            ))
        except:
            warnings.warn("Render of episode did not work! Some issue needs to be resolved!")
        # Create a Matplotlib plot
        # x_data = np.array(self.episode_df.reset_index().index)
        # y_data1 = self.episode_df["T_in_coolingCircuit"].values
        # y_data2 = self.episode_df['Temperature_cwCircuit_in'].values

        # # Create a new Matplotlib figure and axis
        # fig, ax = plt.subplots()

        # # Plot all three columns
        # ax.plot(x_data, y_data1, label='Column 2')
        # ax.plot(x_data, y_data2, label='Column 3')

        # # Set labels and title
        # ax.set_xlabel('X Axis')
        # ax.set_ylabel('Y Axis')
        # ax.set_title('Plotting Columns from DataFrame')

        # # Add legend
        # ax.legend()

        # # Get the current figure
        # fig = plt.gcf()

        # # Convert the Matplotlib plot to an image in ndarray format
        # fig.canvas.draw()
        # image_array = np.array(fig.canvas.renderer._renderer)
        #fig_img = px.line(self.episode_df, x=self.episode_df.index, y=["T_in_coolingCircuit",'Temperature_cwCircuit_in','reward_switching'])
        return #image_array

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

    def policy_shape_cooling(self, T_in_cooling, _action, punishment = 1):
        if T_in_cooling> 28:
            self.state["s_u_P1"] = 1#_action[0]
            self.state["s_u_P2"] = 1#_action[1]
            self.state["s_u_P3"] = 1#_action[2]
            self.state["s_u_KKM1_On"] = _action[3]
            self.state["s_u_KKM2_On"] = _action[4]
            self.state["s_u_KKM1_T_Target"] = _action[5]
            self.state["s_u_KKM2_T_Target"] = _action[6]
            self.state["s_u_KT_1"] = 1
            self.state["s_u_KT_2"] = 1
            self.state["s_u_KT_3"] = 1
            self.state["s_u_KT_4"] = 1
            self.state["s_u_KT_5"] = 1
            self.state["s_u_KT_6"] = 1
            self.state["s_u_WT1"] = 1
            self.state["s_u_WT2"] = 1
            _action = np.array(
                [
                    self.state["s_u_P1"],
                    self.state["s_u_P2"],
                    self.state["s_u_P3"],
                    self.state["s_u_KKM1_On"],
                    self.state["s_u_KKM2_On"],
                    self.state["s_u_KKM1_T_Target"],
                    self.state["s_u_KKM2_T_Target"],
                    self.state["s_u_KT_1"],
                    self.state["s_u_KT_2"],
                    self.state["s_u_KT_3"],
                    self.state["s_u_KT_4"],
                    self.state["s_u_KT_5"],
                    self.state["s_u_KT_6"],
                    self.state["s_u_WT1"],
                    self.state["s_u_WT2"],
                ]
            )
            self.policy_shaping_active += punishment
        if T_in_cooling< 12:
            self.state["s_u_P1"] = 1#_action[0]
            self.state["s_u_P2"] = 1#_action[1]
            self.state["s_u_P3"] = 1#_action[2]
            self.state["s_u_KKM1_On"] = _action[3]
            self.state["s_u_KKM2_On"] = _action[4]
            self.state["s_u_KKM1_T_Target"] = _action[5]
            self.state["s_u_KKM2_T_Target"] = _action[6]
            self.state["s_u_KT_1"] = 0
            self.state["s_u_KT_2"] = 0
            self.state["s_u_KT_3"] = 0
            self.state["s_u_KT_4"] = 0
            self.state["s_u_KT_5"] = 0
            self.state["s_u_KT_6"] = 0
            self.state["s_u_WT1"] = 0
            self.state["s_u_WT2"] = 0
            _action = np.array(
                [
                    self.state["s_u_P1"],
                    self.state["s_u_P2"],
                    self.state["s_u_P3"],
                    self.state["s_u_KKM1_On"],
                    self.state["s_u_KKM2_On"],
                    self.state["s_u_KKM1_T_Target"],
                    self.state["s_u_KKM2_T_Target"],
                    self.state["s_u_KT_1"],
                    self.state["s_u_KT_2"],
                    self.state["s_u_KT_3"],
                    self.state["s_u_KT_4"],
                    self.state["s_u_KT_5"],
                    self.state["s_u_KT_6"],
                    self.state["s_u_WT1"],
                    self.state["s_u_WT2"],
                ]
            )
            self.policy_shaping_active += punishment
        return _action

    def policy_shape_cold(self, T_in_cold,_action, punishment=1):
        if T_in_cold> 15:
            self.state["s_u_P1"] = _action[0]
            self.state["s_u_P2"] = _action[1]
            self.state["s_u_P3"] = _action[2]
            self.state["s_u_KKM1_On"] = 1
            self.state["s_u_KKM2_On"] = 1
            self.state["s_u_KKM1_T_Target"] = 8
            self.state["s_u_KKM2_T_Target"] = 8
            self.state["s_u_KT_1"] = _action[7]
            self.state["s_u_KT_2"] = _action[8]
            self.state["s_u_KT_3"] = _action[9]
            self.state["s_u_KT_4"] = _action[10]
            self.state["s_u_KT_5"] = _action[11]
            self.state["s_u_KT_6"] = _action[12]
            self.state["s_u_WT1"] = _action[13]
            self.state["s_u_WT2"] = _action[14]
            _action = np.array(
                [
                    self.state["s_u_P1"],
                    self.state["s_u_P2"],
                    self.state["s_u_P3"],
                    self.state["s_u_KKM1_On"],
                    self.state["s_u_KKM2_On"],
                    self.state["s_u_KKM1_T_Target"],
                    self.state["s_u_KKM2_T_Target"],
                    self.state["s_u_KT_1"],
                    self.state["s_u_KT_2"],
                    self.state["s_u_KT_3"],
                    self.state["s_u_KT_4"],
                    self.state["s_u_KT_5"],
                    self.state["s_u_KT_6"],
                    self.state["s_u_WT1"],
                    self.state["s_u_WT2"],
                ]
            )
            self.policy_shaping_active += punishment
        if T_in_cold< 6:
            self.state["s_u_P1"] = _action[0]
            self.state["s_u_P2"] = _action[1]
            self.state["s_u_P3"] = _action[2]
            self.state["s_u_KKM1_On"] = 0
            self.state["s_u_KKM2_On"] = 0
            self.state["s_u_KKM1_T_Target"] = 8
            self.state["s_u_KKM2_T_Target"] = 8
            self.state["s_u_KT_1"] = _action[7]
            self.state["s_u_KT_2"] = _action[8]
            self.state["s_u_KT_3"] = _action[9]
            self.state["s_u_KT_4"] = _action[10]
            self.state["s_u_KT_5"] = _action[11]
            self.state["s_u_KT_6"] = _action[12]
            self.state["s_u_WT1"] = _action[13]
            self.state["s_u_WT2"] = _action[14]
            _action = np.array(
                [
                    self.state["s_u_P1"],
                    self.state["s_u_P2"],
                    self.state["s_u_P3"],
                    self.state["s_u_KKM1_On"],
                    self.state["s_u_KKM2_On"],
                    self.state["s_u_KKM1_T_Target"],
                    self.state["s_u_KKM2_T_Target"],
                    self.state["s_u_KT_1"],
                    self.state["s_u_KT_2"],
                    self.state["s_u_KT_3"],
                    self.state["s_u_KT_4"],
                    self.state["s_u_KT_5"],
                    self.state["s_u_KT_6"],
                    self.state["s_u_WT1"],
                    self.state["s_u_WT2"],
                ]
            )
            self.policy_shaping_active += punishment
        return _action

    def policy_shape_pumps(self, _action, punishment=1):
        if _action[0]<=0.1 and (_action[7]>0.1 or _action[8]>0.1):
            _action[0] = 1
            self.state["s_u_P1"] = _action[0]
            self.policy_shaping_active += punishment
        if _action[1]<=0.1 and (_action[9]>0.1 or _action[10]>0.1):
            _action[1] = 1
            self.state["s_u_P2"] = _action[1]
            self.policy_shaping_active += punishment
        if _action[2]<=0.1 and (_action[11]>0.1 or _action[12]>0.1):
            _action[2] = 1
            self.state["s_u_P3"] = _action[2]
            self.policy_shaping_active += punishment
        return _action

    def reset_agent_action(self, _action):
        self.state["u_P1"] = _action[0]
        self.state["u_P2"] = _action[1]
        self.state["u_P3"] = _action[2]
        self.state["u_KKM1_On"] = _action[3]
        self.state["u_KKM2_On"] = _action[4]
        self.state["u_KKM1_T_Target"] = _action[5]
        self.state["u_KKM2_T_Target"] = _action[6]
        self.state["u_KT_1"] = _action[7]
        self.state["u_KT_2"] = _action[8]
        self.state["u_KT_3"] = _action[9]
        self.state["u_KT_4"] = _action[10]
        self.state["u_KT_5"] = _action[11]
        self.state["u_KT_6"] = _action[12]
        self.state["u_WT1"] = _action[13]
        self.state["u_WT2"] = _action[14]
        return None

def reward_boundary(state, state_min, state_max, reward, penalty, smoothed=True, k=1, k_lin=0):
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
            should reward be smoothed by use of sigmoid function ?
        k : (float)
            modify width of sigmoid smoothing 1/(1+exp(-k*x)) - higher is steeper
    """
    # catch cases when min/max are not defined
    if state_min is None:
        state_min = -1e10
    if state_max is None:
        state_max = 1e10

    if smoothed:
        # return reward - (reward+penalty)*(expit(k*(state-state_max))+expit(k*(state_min-state)))
        # I dont know in which context this is useful -> its NOT a sigmoid below
        return (
            reward
            - (reward + penalty) * (expit(k * (state - state_max)) + expit(k * (state_min - state)))
            - k_lin * max(state - state_max, 0)
            - k_lin * max(state_min - state, 0)
        )
    else:
        return reward if (state > state_min and state < state_max) else -penalty