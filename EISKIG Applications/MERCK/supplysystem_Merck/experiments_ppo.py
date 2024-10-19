from __future__ import annotations

import pathlib
from datetime import datetime

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

    """
    Notes:
    - the original netarch was [500, dict(pi=[400, 300], vf=[400, 300])] , but since a newer eta-utility version is used, it needs to be changed
    
    """

    config_experiment_1_learn = {
        "settings": {"n_environments": 8, "sampling_time": 180, "n_episodes_learn": 1600, "episode_duration": 86400*3, "plot_interval": 5, "save_model_every_x_episodes": 5},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0008, 0.00002).value,
            "n_steps": 128,
            "batch_size": 128,
            # "policy_kwargs": {"net_arch": [500, 400, 300]},
            "device": "cpu",  # "cuda" on systems with cuda installed
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_Merck_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )

    experiment_1.learn("ppo_agent_x", "agent_x")

    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    # config_experiment_1_play = {
    #     "settings": {"n_environments": 1, "episode_duration": 86400*3, "n_episodes_play": 1, "plot_interval": 1, "sampling_time": 180},
    #     "agent_specific": {"policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]}},
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-06-01 00:00",
    #         "scenario_time_end": "2017-12-27 00:00", # "2017-12-27 00:00" if random sampling = true!!!
    #         # "scenario_time_begin": "2017-01-01 00:00",
    #         # "scenario_time_end": "2018-01-01 00:00",
    #         "discretize_action_space": True,
    #         "random_params": False, # select a random start SOC for the iceStorage
    #         "random_sampling": False,
    #         "use_conventional": False,
    #     },
    # }

    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_Merck_ppo",
    #     config_overwrite=config_experiment_1_play,
    #     relpath_config="config/",
    # )

    # # startTime = datetime.now()   
    # # print("startTime: ", startTime.strftime('%Y-%m-%d %H:%M:%S'))

    # experiment_1.play("ppo_test_agent_7", "agent_7")
    # #experiment_1.play("rule_based_vergleich", "random_episodes")
    
    # # endTime = datetime.now()
    # # print("endTime: ", endTime.strftime('%Y-%m-%d %H:%M:%S'))
    # # runTime = endTime - startTime
    # # print("runTime: ", runTime)




if __name__ == "__main__":
    main()
