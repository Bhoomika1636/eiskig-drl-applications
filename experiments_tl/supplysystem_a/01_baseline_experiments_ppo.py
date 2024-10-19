from __future__ import annotations

import pathlib

from eta_utility import LOG_INFO, get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    ##################
    #                #
    #    TRAINING    #
    #                #
    ##################

    config_experiment_1_learn = {
        "settings": {"n_environments": 16, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )

    experiment_1.learn("ppo_test", "experiment_1")

    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################


    # experiment_1.play("ppo_test", "experiment_1")


if __name__ == "__main__":
    main()
