from __future__ import annotations

import pathlib
import time

from eta_utility import get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    # config_experiment_1 = {
    #     "settings": {"episode_duration": 259200,
    #                 "sampling_time": 30,
    #                 "n_episodes_play": 10,
    #                 "plot_interval": 1},
    #     "environment_specific": {
    #                     "scenario_time_begin": "2017-01-01 00:00",
    #                     "scenario_time_end": "2018-12-27 00:00",
    #                     "allow_policy_shaping":False,
    #                     "allow_limiting_CHP_switches":False,
    #                     "extended_observations": True,
    #                     "use_complex_AFA_model":False
    #                      }
    # }

    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_ETA_rule_based",
    #     config_overwrite=config_experiment_1,
    #     relpath_config="config/")

    # experiment_1.play("FinalPerformance_RuleBased_Phase_1", "experiment_1")

    #####################################################################################
    # DRL -Agent a2
    #####################################################################################

    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 1,
    #                  "n_episodes_play":10,
    #                  "episode_duration": 259200},
    #     "environment_specific":{"scenario_time_begin": "2017-01-01 00:00",
    #                             "scenario_time_end": "2018-12-27 00:00",
    #                             "allow_policy_shaping":False,
    #                             "extended_observations": False,
    #                             "abort_cost_for_unsuccessfull_step": False,
    #                             "with_storages": True,
    #                             "use_complex_AFA_model":False},
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0001, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_ETA_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/",
    # )

    # experiment_1.play("ppo_agent_baseline_continued", "experiment_final_performance")

    #####################################################################################
    # DRL -Agent c
    #####################################################################################

    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 1,
    #                  "n_episodes_play":10,
    #                  "episode_duration": 259200},
    #     "environment_specific":{"scenario_time_begin": "2017-01-01 00:00",
    #                             "scenario_time_end": "2018-12-27 00:00",
    #                             "allow_policy_shaping":False,
    #                             "extended_observations": False,
    #                             "abort_cost_for_unsuccessfull_step": False,
    #                             "with_storages": True,
    #                             "use_complex_AFA_model":False},
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0001, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_ETA_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/",
    # )

    # experiment_1.play("ppo_agent_without_abort_cost", "experiment_final_performance")

    #####################################################################################
    # DRL -Agent b2
    #####################################################################################

    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 1,
    #                  "n_episodes_play":10,
    #                  "episode_duration": 259200},
    #     "environment_specific":{"scenario_time_begin": "2017-01-01 00:00",
    #                             "scenario_time_end": "2018-12-27 00:00",
    #                             "allow_policy_shaping":False,
    #                             "extended_observations": True,
    #                             "abort_cost_for_unsuccessfull_step": False,
    #                             "with_storages": True,
    #                             "use_complex_AFA_model":False},
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0001, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_ETA_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/",
    # )

    # experiment_1.play("ppo_agent_extended_obs_continued", "experiment_final_performance")
    #
    #####################################################################################
    # DRL -Agent e
    #####################################################################################

    config_experiment_1_learn = {
        "settings": {"n_environments": 1, "n_episodes_play": 10, "episode_duration": 259200},
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2018-12-27 00:00",
            "allow_policy_shaping": False,
            "extended_observations": False,
            "abort_cost_for_unsuccessfull_step": False,
            "with_storages": True,
            "use_complex_AFA_model": True,
        },
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0001, 0.00002).value,
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

    experiment_1.play("ppo_agent_final_2", "experiment_final_performance")

    #####################################################################################
    # RuleBased FMU 2.0
    #####################################################################################

    config_experiment_1 = {
        "settings": {"episode_duration": 259200, "sampling_time": 30, "n_episodes_play": 10, "plot_interval": 1},
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2018-12-27 00:00",
            "allow_policy_shaping": False,
            "allow_limiting_CHP_switches": False,
            "extended_observations": True,
            "use_complex_AFA_model": True,
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_rule_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("FinalPerformance_RuleBased_Phase_2", "experiment_1")


if __name__ == "__main__":
    main()
