from __future__ import annotations

import pathlib

from eta_utility import get_logger
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
        "settings": {"n_environments": 1, "n_episodes_learn": 1, "episode_duration": 259200},
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-25 00:00",
            "extended_observations": False,
        },
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }
    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )
    experiment_1.learn("ppo_speed_test_env1", "experiment_1")

    ########################################################################

    config_experiment_1_learn = {
        "settings": {"n_environments": 4, "n_episodes_learn": 4, "episode_duration": 259200},
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-25 00:00",
            "extended_observations": False,
        },
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }
    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )
    experiment_1.learn("ppo_speed_test_env4", "experiment_1")

    ########################################################################

    config_experiment_1_learn = {
        "settings": {"n_environments": 8, "n_episodes_learn": 8, "episode_duration": 259200},
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-25 00:00",
            "extended_observations": False,
        },
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }
    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )
    experiment_1.learn("ppo_speed_test_env8", "experiment_1")

    ########################################################################

    config_experiment_1_learn = {
        "settings": {"n_environments": 12, "n_episodes_learn": 12, "episode_duration": 259200},
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-25 00:00",
            "extended_observations": False,
        },
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }
    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )
    experiment_1.learn("ppo_speed_test_env12", "experiment_1")

    ########################################################################

    config_experiment_1_learn = {
        "settings": {"n_environments": 16, "n_episodes_learn": 16, "episode_duration": 259200},
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-25 00:00",
            "extended_observations": False,
        },
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }
    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )
    experiment_1.learn("ppo_speed_test_env16", "experiment_1")

    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    # config_experiment_1_play = {
    #     "settings": {"n_environments": 1, "n_episodes_play": 10, "episode_duration": 86400 * 3, "plot_interval": 1},
    #     "agent_specific": {"policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]}},

    # }

    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_ETA_ppo",
    #     config_overwrite=config_experiment_1_play,
    #     relpath_config="config/",
    # )

    # experiment_1.play("ppo_test", "experiment_1")


if __name__ == "__main__":
    main()
