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

import common.helpers as helpers

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
    fmu_name = "BoschCoolingSystem"

    def __init__(
        self,
        env_id: int,
        config_run: ConfigOptRun,  # config_run contains run_name, general_settings, path_settings, env_info
        seed: int | None = None,
        verbose: int = 2,
        callback: Callable | None = None,
        *,
        random_sampling,
        abort_costs,
        sampling_time: float,  # from settings section of config file
        episode_duration: TimeStep | str,  # from settings section of config file
        scenario_time_begin: datetime | str,
        scenario_time_end: datetime | str,
        discretize_action_space,
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
        self.abort_costs = abort_costs
        self.discretize_action_space = discretize_action_space
        self.random_sampling = random_sampling

        # initialize episode statistics
        self.episode_statistics = {}
        self.episode_archive = []
        self.episode_df = pd.DataFrame()

        # define state variables
        # # helperfunction to convert CSV to state_var_tuple
        # def getStateVars(pathToCSV, decimal=',', sep=';'):
        #     df = pd.read_csv(pathToCSV, sep=sep,decimal=decimal, true_values=["True"], false_values=["False"])
        #     stateVarList = []

        #     for i, row in df.iterrows():
        #         stateVar = StateVar.from_dict(row.dropna())
        #         stateVarList.append(stateVar)
            
        #     state_var_tuple = tuple(stateVarList)
            
        #     return state_var_tuple

        state_var_tuple = helpers.getStateVars('experiments_hr\BoschCoolingSystem\environment\BoschStateVars.csv')
        # build final state_config
        self.state_config = StateConfig(*state_var_tuple)

        # # import all scenario files
        # self.scenario_data = self.import_scenario(*scenario_files)

        # get action_space
        # TODO: implement this functionality into utility functions
        if self.discretize_action_space:
            # get number of actions agent has to give from state_config
            self.n_action_space = len(self.state_config.actions)
            # set 3 discrete actions (increase,decrease,equal) per control variable
            self.action_space = spaces.MultiDiscrete(np.full(self.n_action_space, 3))
            # customized for chp, condensingboiler, immersionheater
            self.action_disc_step = [[0, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1], [0, 0.25, 0.5, 0.75, 1]]
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

        # set random necesarry coolingdemand
        self.state['CoolingDemandCoolingWater'] = 0 # np.random.uniform(0, 300000)
        self.state['CoolingDemandColdWater'] = 0

        # convert discrete actions into continious space if discrete action space is chosen
        if self.discretize_action_space:
            _action = self.convert_disc_action(action)
        else:
            _action = action

        # check actions for vilidity, perform simulation step and load new external values for the next time step
        self._actions_valid(_action) # why every step - at the beginning is sufficient ???
        self.state["step_success"], _ = self._update_state(_action) # updates state incomplete if it fails-> issues with withinAbortConditions

        # check if state is in valid boundaries
        self.state["step_abort"] = False if StateConfig.within_abort_conditions(self.state_config, self.state) else True
        # update predictions and virtual state for next time step
        # deleted! if relevant check in supplysystem_a file

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
                return penalty  # else 0.1*penalty*abs(u_new - u_old)
            else:
                return 0

        # switching costs -- TODO
        self.state["reward_switching"] = (
            # -switch_cost(
            #     self.state_backup["s_u"],
            #     self.state["s_u"],
            #     0.01,
            # )
            -1*abs(self.state["s_u_P1"] - self.state_backup["s_u_P1"])/self.sampling_time
        ) 
        [-abs(self.state[x]-self.state_backup[x]) for x in self.state.keys() if "s_u" in x]  # noqa: E999
        
        # temperature costs (when availability of temperature levels are needed)
        self.state["reward_temperature_heat"] = reward_boundary(
            self.state["T_in_coolingCircuit"],
            17,
            19,
            0,
            0.5,
            smoothed=True,
            k=6,
        ) + reward_boundary(
            self.state["Temperature_cwCircuit_in"],
            6,
            12,
            0,
            0.5,
            smoothed=True,
            k=6,
        )

        # other costs -> relevant
        self.state["reward_other"] = (
            -self.abort_costs * (1 - 0.5 * self.n_steps / self.n_episode_steps)
            if (self.state["step_abort"] or not self.state["step_success"])
            else 0
        )

        # total reward
        self.state["reward_total"] = (
            self.state["reward_switching"]
            + self.state["reward_temperature_heat"]
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
            # self.episode_statistics["datetime_begin"] = self.ts_current.index[0].strftime("%Y-%m-%d %H:%M")
            self.episode_statistics["rewards_total"] = self.episode_df["reward_total"].sum()
            self.episode_statistics["rewards_switching"] = self.episode_df["reward_switching"].sum()
            self.episode_statistics["rewards_temperature_heat"] = self.episode_df["reward_temperature_heat"].sum()
            self.episode_statistics["rewards_other"] = self.episode_df["reward_other"].sum()
            self.episode_statistics["time"] = time.time() - self.episode_timer
            self.episode_statistics["n_steps"] = self.n_steps

            # # append episode store to episodes' archive
            self.episode_archive.append(list(self.episode_statistics.values()))

        # initialize additional state, since super().reset is used
        self.additional_state = {}

        # set FMU parameters for initialization
        self.model_parameters = {}
        # self.model_parameters["fullHeatExchangerCircuit.port_b4.m_flow"] = -0.01
        
        # receive observations from simulation
        observations = super().reset()

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

        # plot settings
        # create figure and axes with custom layout/adjustments

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

    def render(self, mode="human", name_suffix=""):
        """
        output plots for last episode

        Parameters
        -----
        mode : (str)
        """

        # save csv -- why?
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
        # set x/y axe and datetime begin
        x = self.episode_df.index
        y = self.episode_df
        
        # dt_begin = self.episode_datetime_begin
        sampling_time = self.sampling_time

        # timeRange = np.arange(
        #     (1 - dt_begin.minute / 60) * 60 * 60 / sampling_time,
        #     self.episode_duration / sampling_time,
        #     1 * 60 * 60 / sampling_time,
        # )
        fig = px.line(self.episode_df, x=self.episode_df.index, y=["T_tank_cw","Tank_level_warm","u_P1"])
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
            should reward be smoothed by use of sigmoid function ?
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
