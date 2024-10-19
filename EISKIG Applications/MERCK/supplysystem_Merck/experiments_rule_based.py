from __future__ import annotations

import pathlib
from datetime import datetime

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    config_experiment_1 = {
        "settings": {"episode_duration": 86400, "n_episodes_play": 1, "plot_interval": 1, "sampling_time": 180}, # 86400 * 3
        "environment_specific": {
            "scenario_time_begin": "2017-06-01 00:00",
            "scenario_time_end": "2018-01-01 00:00", # "2017-12-27 00:00" if random sampling = true!!!
            "discretize_action_space": True,
            "random_params": False,
            "random_sampling": False,
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_Merck_rule_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )


    startTime = datetime.now()   
    print("startTime: ", startTime.strftime('%Y-%m-%d %H:%M:%S'))
    #experiment_1.play("ppo_test_agent_5a", "agent_5a")
    experiment_1.play("Tests_TL_1", "summer_new1")
    endTime = datetime.now()
    print("endTime: ", endTime.strftime('%Y-%m-%d %H:%M:%S'))
    runTime = endTime - startTime
    print("runTime: ", runTime)


if __name__ == "__main__":
    main()
