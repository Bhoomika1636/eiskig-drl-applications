from __future__ import annotations

import pathlib

from eta_utility import LOG_INFO, get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule
import time

def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    ##################
    #                #
    #    TRAINING    #
    #                #
    ##################

    config_experiment_1_learn = {
        "settings": {"sampling_time": 50,"n_environments": 4, "n_episodes_learn": 100, "episode_duration": 1000, "plot_interval": 50},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "n_epochs": 4,
            # "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cpu",  # "cuda" on systems with cuda installed
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )
    ts = time.time()
    experiment_1.learn("ppo_test", "experiment_1")
    ts2 =time.time()

    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    config_experiment_1_play = {
        "settings": {"episode_duration": 3000,"sampling_time": 30, "n_episodes_play": 1, "plot_interval": 1},
        "environment_specific": {
            "scenario_time_begin": "2018-03-17 00:00",
            "scenario_time_end": "2018-05-30 00:00",
            "random_sampling": False,
        }
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_ppo",
        config_overwrite=config_experiment_1_play,
        relpath_config="config/",
    )

    experiment_1.play("ppo_test", "experiment_1")
    print(ts2-ts)


if __name__ == "__main__":
    main()
