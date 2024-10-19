from __future__ import annotations

import pathlib

from eta_utility import LOG_INFO, get_logger
from experiments_hr.supplysystem_a.common.Custom_ETAx import ETAx2
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

    ###########################

    # Baseline Experiment

    ###########################

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific":{
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2017-12-27 00:00",
    #         "poweron_cost_combinedheatpower": 0.4,
    #         "poweron_cost_condensingboiler": 0.2,
    #         "poweron_cost_immersionheater": 0.1},
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx2(
    #     root_path=root_path,
    #     config_name="supplysystem_a_ppo_HyperparameterOpt",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/",
    # )
    # experiment_1.learn("ppo_baseline_datasetsplit", "experiment_1")


    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy",
    #               "norm_wrapper_reward": False},
    #     "settings": {"n_environments": 1, "n_episodes_play": 10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific":{
    #         "scenario_time_begin": "2018-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "poweron_cost_combinedheatpower": 0.4,
    #         "poweron_cost_condensingboiler": 0.2,
    #         "poweron_cost_immersionheater": 0.1},
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx2(
    #     root_path=root_path,
    #     config_name="supplysystem_a_ppo_HyperparameterOpt",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/",
    # )

    # mean_reward = experiment_1.play("ppo_baseline_datasetsplit", "experiment_1_play_deterministic")
    # print("mean_reward", mean_reward) #-0.2856

    ###########################

    # Baseline Experiment extended observations

    ###########################

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific":{
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2017-12-27 00:00",
    #         "poweron_cost_combinedheatpower": 0.4,
    #         "poweron_cost_condensingboiler": 0.2,
    #         "poweron_cost_immersionheater": 0.1,
    #         "variant":"extended_observations"},
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx2(
    #     root_path=root_path,
    #     config_name="supplysystem_a_ppo_HyperparameterOpt",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/",
    # )
    # experiment_1.learn("ppo_baseline_datasetsplit_extended_obs", "experiment_1")


    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy",
    #               "norm_wrapper_reward": False},
    #     "settings": {"n_environments": 1, "n_episodes_play": 10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific":{
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2017-12-27 00:00",
    #         "poweron_cost_combinedheatpower": 0.4,
    #         "poweron_cost_condensingboiler": 0.2,
    #         "poweron_cost_immersionheater": 0.1,
    #         "variant":"extended_observations"},
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx2(
    #     root_path=root_path,
    #     config_name="supplysystem_a_ppo_HyperparameterOpt",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/",
    # )

    # # mean_reward = experiment_1.play("ppo_baseline_datasetsplit_extended_obs", "experiment_1_play")
    # mean_reward = experiment_1.play("ppo_baseline_datasetsplit_extended_obs", "experiment_1_play_2017")
    # print("mean_reward", mean_reward) 
    
    ### Results ####
    # deterministic: -0.147482
    # not deterministic: -0.14462

    ###########################

    # Hyper Search Test 5 Run 1

    ###########################

    # config_experiment_1_learn = {
    #     "setup": {"norm_wrapper_reward": False},
    #     "settings": {"n_environments": 1, "n_episodes_play": 10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific":{
    #         "scenario_time_begin": "2018-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "poweron_cost_combinedheatpower": 0.4,
    #         "poweron_cost_condensingboiler": 0.2,
    #         "poweron_cost_immersionheater": 0.1,
    #         "variant":"extended_observations"},
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0001, 0.00002).value,
    #         "policy_kwargs": {"net_arch": [64, 32, 32]},
    #         "n_steps": 64,
    #         "batch_size":64,
    #         "ent_coef":0.002249,
    #         "clip_range":0.1,
    #         "n_epochs":1,
    #         "gae_lambda":0.98
    #     },
    # }
    # experiment_1 = ETAx2(
    #     root_path=root_path,
    #     config_name="supplysystem_a_ppo_HyperparameterOpt",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/",
    # )

    # mean_reward = experiment_1.play("Hyperparameter_Search_PPO_Test_5_Trail_1", "experiment_1_play_deterministic")
    # print("mean_reward", mean_reward) #-0.1537

    ###########################

    # Hyper Search Test 6 Run 14

    ###########################


    config_experiment_1_learn = {
        "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy",
                  "norm_wrapper_reward": False},
        "settings": {"n_environments": 1, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
        "environment_specific":{
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-27 00:00",
            "poweron_cost_combinedheatpower": 0.4,
            "poweron_cost_condensingboiler": 0.2,
            "poweron_cost_immersionheater": 0.1,
            "variant":"extended_observations"},
        "agent_specific": {
        # "gamma": 0.99,
        "n_steps": 2048,
        "ent_coef": 2.6300995627563196e-07,
        "learning_rate": LinearSchedule(0.0003, 0.00002).value,
        "vf_coef": 0.1145,
        "max_grad_norm": 0.5,
        "gae_lambda": 0.95,
        "batch_size": 64,
        "n_epochs": 10,
        "clip_range": 0.1,
        "verbose": 1,
        "policy_kwargs": {
            "net_arch": [128, 64, 64]
        },
        "device": "cuda"
    }
    }

    experiment_1 = ETAx2(
        root_path=root_path,
        config_name="supplysystem_a_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )

    experiment_1.play("Hyperparameter_Search_PPO_Test_6_Trail_14", "experiment_1_play_2017")

    # config_experiment_1_learn = {
    #     # "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     # "settings": {"n_environments": 4, "n_episodes_learn": 3000, "n_episodes_play":1, "episode_duration": 259200, "plot_interval": 1},
    #     "settings": {"n_environments": 1, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     # "agent_specific": {
    #     #     "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #     #     "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #     #     "device": "cuda",  # "cuda" on systems with cuda installed
    #     # },
    #     "agent_specific": {
    #     "gamma": 0.99,
    #     "n_steps": 64,
    #     "ent_coef": 1.39e-07,
    #     "learning_rate": LinearSchedule(0.0001522, 0.00002).value,
    #     "vf_coef": 0.1145,
    #     "max_grad_norm": 0.5,
    #     "gae_lambda": 0.9,
    #     "batch_size": 64,
    #     "n_epochs": 10,
    #     "clip_range": 0.4,
    #     "verbose": 1,
    #     "policy_kwargs": {
    #         "net_arch": [64, 32, 32]
    #     },
    #     "device": "cuda"
    # }
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_a_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/",
    # )
    # experiment_1.play("Hyperparameter_Search_Test_Trail_24", "experiment_1_play")



if __name__ == "__main__":
    main()
