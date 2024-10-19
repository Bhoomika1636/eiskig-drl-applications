from __future__ import annotations

import pathlib
import time

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


"""
Notes 

This script uses a different env, which uses the QSolar scenario with the new AFA model.
In addition, the P_el of all pumps is included in the total P_el of the system.

"""

def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    # config_experiment_1 = {
    #     "settings": {
    #         "episode_duration": 60 * 60 * 24 * 30, #30 days
    #         "sampling_time": 30, #communication time with simulation, for faster simulation, can be set to 60 seconds too
    #         "n_episodes_play": 1,
    #         "plot_interval": 1,
    #     },
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2017-12-25 00:00",
    #         "random_sampling": False,
    #     },
    # }

    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_ETA_rule_based_QSolar",
    #     config_overwrite=config_experiment_1,
    #     relpath_config="config/",
    # )

    # experiment_1.play("rule_based_January", "experiment_1")

    config_experiment_1 = {
        "settings": {
            "episode_duration": 60 * 60 * 24 * 30, #30 days
            "sampling_time": 30, #communication time with simulation, for faster simulation, can be set to 60 seconds too
            "n_episodes_play": 1,
            "plot_interval": 1,
        },
        "environment_specific": {
            "scenario_time_begin": "2017-06-01 00:00",
            "scenario_time_end": "2017-12-25 00:00",
            "random_sampling": False,
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_rule_based_QSolar",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("rule_based_June", "experiment_1")


if __name__ == "__main__":
    main()
