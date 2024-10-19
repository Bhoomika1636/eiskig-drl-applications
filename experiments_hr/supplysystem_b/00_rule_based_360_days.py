from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    config_experiment_1 = {
        "settings": {"episode_duration": 86400 * 1, #duration in seconds: 1 day
                     "n_episodes_play": 1, # number of episodes to play
                     "plot_interval": 1 # plot every 1 episode
                     },
        "environment_specific": {
            "temperature_cool_init_max": 288, 
            "temperature_cool_init_min": 288,
            "temperature_heat_init_max": 342.5,
            "temperature_heat_init_min": 342.5,
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-27 00:00",
            "random_sampling": False, #no random epsiode sampling but instead start at scenario_time_begin
            
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_rule_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("rule_based_controller_1_day", "experiment_1")


if __name__ == "__main__":
    main()
