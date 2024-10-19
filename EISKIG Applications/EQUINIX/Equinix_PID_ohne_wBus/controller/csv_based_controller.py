from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from eta_utility.eta_x.agents import RuleBased

if TYPE_CHECKING:
    from typing import Any

    from stable_baselines3.common.base_class import BasePolicy
    from stable_baselines3.common.vec_env import VecEnv


class CSVBasedController(RuleBased):
    """
    Simple rule based controller for supplysystem_a.

    :param policy: Agent policy. Parameter is not used in this agent and can be set to NoPolicy.
    :param env: Environment to be controlled.
    :param verbose: Logging verbosity.
    :param kwargs: Additional arguments as specified in stable_baselins3.commom.base_class.
    """

    def __init__(
        self,
        policy: type[BasePolicy],
        env: VecEnv,
        verbose: int = 1,
        csv_path="/",
        steps_per_episode=480,
        **kwargs: Any,
    ):
        super().__init__(policy=policy, env=env, verbose=verbose, **kwargs)

        # extract action and observation names from the environments state_config
        self.action_names = self.env.envs[0].state_config.actions
        self.observation_names = self.env.envs[0].state_config.observations
        self.csv_path = csv_path
        self.steps_per_episode = steps_per_episode
        self.counter = 1  # because the first step is the initialization

        # Function to check if a column name starts with a given prefix
        def column_starts_with(column_name, prefix):
            return column_name.startswith(prefix)

        # Read only the first row to get the column names
        df_header = pd.read_csv(self.csv_path, sep=";", nrows=0)
        # Filter the column names that start with 's_u_'
        filtered_columns = [col for col in df_header.columns if column_starts_with(col, "s_u_")]
        # Now read only the filtered columns from the CSV
        self.control_signals_df = pd.read_csv(self.csv_path, sep=";", usecols=filtered_columns)
        # set initial state
        self.initial_state = np.zeros(self.action_space.shape)

    def control_rules(self, observation: np.ndarray) -> np.ndarray:
        """
        Controller of the model. This implements a simple, rule-based controller
        """

        # Initialize action dictionary with zeros
        action = dict.fromkeys(self.action_names, 0)
        # Use dictionary comprehension to construct the action2 dictionary
        action = {key: self.control_signals_df["s_" + key].iloc[self.counter] for key in self.action_names}
        # Convert the dictionary values directly to a list
        actions = list(action.values())
        # Update counter
        self.counter = (self.counter % self.steps_per_episode) + 1

        return np.array(actions)
