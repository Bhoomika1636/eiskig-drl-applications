from __future__ import annotations

import pathlib
import time

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    config_experiment_1 = {
        "settings": {
            "episode_duration": 60 * 60 * 1,
            "sampling_time": 60, 
            "n_episodes_play": 1,
            "plot_interval": 1,
        },
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-25 00:00",
            "random_sampling": False, #to start the episode at scenario time begin
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_rule_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("rule_based_test", "experiment_1")


if __name__ == "__main__":
    main()
