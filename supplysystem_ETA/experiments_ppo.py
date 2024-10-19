from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    ####################################
    #                                  #
    #        ppo agent testing         #
    #                                  #
    ####################################
    config_experiment_1_learn = {
        "setup": {
            # "environment_import":"environment.supplysystem_ETA_variance.SupplysystemETA",
            "policy_import": "common.CustomPolicies.CustomActorCriticPolicy"
        },
        "settings": {"n_environments": 1, "n_episodes_learn": 1, "n_episodes_play": 1, "episode_duration": 259200},
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2018-12-27 00:00",
            "extended_observations": False,
            "abort_cost_for_unsuccessfull_step": False,
            "with_storages": True,
            "use_complex_AFA_model": True,
            "temperature_HNHT_Buffer_init_min": 343.15,
            "temperature_HNHT_Buffer_init_max": 343.15,
            "temperature_HNHT_VSI_init_min": 343.15,
            "temperature_HNHT_VSI_init_max": 343.15,
            "temperature_HNLT_Buffer_init_min": 310.65,
            "temperature_HNLT_Buffer_init_max": 310.65,
            "temperature_HNLT_HVFA_init_min": 295.15,
            "temperature_HNLT_HVFA_init_max": 295.15,
            "temperature_CN_Buffer_init_min": 291.15,
            "temperature_CN_Buffer_init_max": 291.15,
            "temperature_CN_HVFA_init_min": 291.15,
            "temperature_CN_HVFA_init_max": 291.15,
            "temperature_AFA_init_fixed": True,
            "variance_min": 1.0,
            "variance_max": 1.0,
            "distribution": "uniform",
            "variance_parameters": ["all"],
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
    experiment_1.learn("ppo_agent_test1", "experiment_1")
    # experiment_1.play("ppo_agent_baseline", "experiment_1_play")

    ####################################
    #                                  #
    #        ppo agent baseline        #
    #                                  #
    ####################################
    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 1, "n_episodes_learn": 8 * 50, "n_episodes_play": 10, "episode_duration": 259200},
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "extended_observations": False,
    #         "abort_cost_for_unsuccessfull_step": False,
    #         "with_storages": True,
    #         "use_complex_AFA_model": True,
    #         "temperature_HNHT_Buffer_init_min": 343.15,
    #         "temperature_HNHT_Buffer_init_max":343.15,
    #         "temperature_HNHT_VSI_init_min":343.15,
    #         "temperature_HNHT_VSI_init_max":343.15,
    #         "temperature_HNLT_Buffer_init_min":310.65,
    #         "temperature_HNLT_Buffer_init_max":310.65,
    #         "temperature_HNLT_HVFA_init_min":295.15,
    #         "temperature_HNLT_HVFA_init_max":295.15,
    #         "temperature_CN_Buffer_init_min":291.15,
    #         "temperature_CN_Buffer_init_max":291.15,
    #         "temperature_CN_HVFA_init_min":291.15,
    #         "temperature_CN_HVFA_init_max":291.15,
    #         "temperature_AFA_init_fixed": True,
    # "variance_min": 1.0,
    # "variance_max": 1.0,
    # "distribution": "uniform",
    # "variance_parameters": ["all"],
    #         "poweron_cost_HeatExchanger1":0.5,
    #         "poweron_cost_CHP1":4,
    #         "poweron_cost_CHP2":4,
    #         "poweron_cost_CondensingBoiler":1,
    #         "poweron_cost_VSIStorage":0.5,
    #         "switching_cost_bLoading_VSISystem":0.5,
    #         "poweron_cost_HVFASystem_HNLT":0.5,
    #         "switching_cost_bLoading_HVFASystem_HNLT":0.5,
    #         "poweron_cost_eChiller":1,
    #         "poweron_cost_HVFASystem_CN":0.5,
    #         "switching_cost_bLoading_HVFASystem_CN":0.5,
    #         "poweron_cost_OuterCapillaryTubeMats":0.5,
    #         "poweron_cost_HeatPump":1,
    #     },
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
    # # experiment_1.learn("ppo_agent_final_continued", "experiment_2")
    # experiment_1.play("ppo_agent_baseline", "experiment_1_play")

    ####################################
    #                                  #
    #  ppo agent no switching costs    #
    #                                  #
    ####################################

    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 8, "n_episodes_learn": 8 * 20, "episode_duration": 259200},
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "extended_observations": False,
    #         "abort_cost_for_unsuccessfull_step": False,
    #         "with_storages": True,
    #         "use_complex_AFA_model": True,
    # "variance_min": 1.0,
    # "variance_max": 1.0,
    # "distribution": "uniform",
    # "variance_parameters": ["all"],
    #         "poweron_cost_HeatExchanger1":0,
    #         "poweron_cost_CHP1":0,
    #         "poweron_cost_CHP2":0,
    #         "poweron_cost_CondensingBoiler":0,
    #         "poweron_cost_VSIStorage":0,
    #         "switching_cost_bLoading_VSISystem":0,
    #         "poweron_cost_HVFASystem_HNLT":0,
    #         "switching_cost_bLoading_HVFASystem_HNLT":0,
    #         "poweron_cost_eChiller":0,
    #         "poweron_cost_HVFASystem_CN":0,
    #         "switching_cost_bLoading_HVFASystem_CN":0,
    #         "poweron_cost_OuterCapillaryTubeMats":0,
    #         "poweron_cost_HeatPump":0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.00004, 0.00002).value,
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
    # experiment_1.learn("ppo_agent_no_switching_costs", "experiment_1_continued")

    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 1, "n_episodes_learn": 8 * 50, "n_episodes_play": 10, "episode_duration": 259200},
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "extended_observations": False,
    #         "abort_cost_for_unsuccessfull_step": False,
    #         "with_storages": True,
    #         "use_complex_AFA_model": True,
    #         "temperature_HNHT_Buffer_init_min": 343.15,
    #         "temperature_HNHT_Buffer_init_max":343.15,
    #         "temperature_HNHT_VSI_init_min":343.15,
    #         "temperature_HNHT_VSI_init_max":343.15,
    #         "temperature_HNLT_Buffer_init_min":310.65,
    #         "temperature_HNLT_Buffer_init_max":310.65,
    #         "temperature_HNLT_HVFA_init_min":295.15,
    #         "temperature_HNLT_HVFA_init_max":295.15,
    #         "temperature_CN_Buffer_init_min":291.15,
    #         "temperature_CN_Buffer_init_max":291.15,
    #         "temperature_CN_HVFA_init_min":291.15,
    #         "temperature_CN_HVFA_init_max":291.15,
    #         "temperature_AFA_init_fixed": True,
    # "variance_min": 1.0,
    # "variance_max": 1.0,
    # "distribution": "uniform",
    # "variance_parameters": ["all"],
    #         "poweron_cost_HeatExchanger1":0.5,
    #         "poweron_cost_CHP1":4,
    #         "poweron_cost_CHP2":4,
    #         "poweron_cost_CondensingBoiler":1,
    #         "poweron_cost_VSIStorage":0.5,
    #         "switching_cost_bLoading_VSISystem":0.5,
    #         "poweron_cost_HVFASystem_HNLT":0.5,
    #         "switching_cost_bLoading_HVFASystem_HNLT":0.5,
    #         "poweron_cost_eChiller":1,
    #         "poweron_cost_HVFASystem_CN":0.5,
    #         "switching_cost_bLoading_HVFASystem_CN":0.5,
    #         "poweron_cost_OuterCapillaryTubeMats":0.5,
    #         "poweron_cost_HeatPump":1,
    #     },
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
    # experiment_1.play("ppo_agent_no_switching_costs", "experiment_1_play2")

    ######################################
    #                                    #
    #  ppo agent switching costs var 1  #
    #                                    #
    ######################################

    #

    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 1, "n_episodes_learn": 8 * 50, "n_episodes_play": 10, "episode_duration": 259200},
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "extended_observations": False,
    #         "abort_cost_for_unsuccessfull_step": False,
    #         "with_storages": True,
    #         "use_complex_AFA_model": True,
    #         "temperature_HNHT_Buffer_init_min": 343.15,
    #         "temperature_HNHT_Buffer_init_max":343.15,
    #         "temperature_HNHT_VSI_init_min":343.15,
    #         "temperature_HNHT_VSI_init_max":343.15,
    #         "temperature_HNLT_Buffer_init_min":310.65,
    #         "temperature_HNLT_Buffer_init_max":310.65,
    #         "temperature_HNLT_HVFA_init_min":295.15,
    #         "temperature_HNLT_HVFA_init_max":295.15,
    #         "temperature_CN_Buffer_init_min":291.15,
    #         "temperature_CN_Buffer_init_max":291.15,
    #         "temperature_CN_HVFA_init_min":291.15,
    #         "temperature_CN_HVFA_init_max":291.15,
    #         "temperature_AFA_init_fixed": True,
    # "variance_min": 1.0,
    # "variance_max": 1.0,
    # "distribution": "uniform",
    # "variance_parameters": ["all"],
    #         "poweron_cost_HeatExchanger1":0.5,
    #         "poweron_cost_CHP1":4,
    #         "poweron_cost_CHP2":4,
    #         "poweron_cost_CondensingBoiler":1,
    #         "poweron_cost_VSIStorage":0.5,
    #         "switching_cost_bLoading_VSISystem":0.5,
    #         "poweron_cost_HVFASystem_HNLT":0.5,
    #         "switching_cost_bLoading_HVFASystem_HNLT":0.5,
    #         "poweron_cost_eChiller":1,
    #         "poweron_cost_HVFASystem_CN":0.5,
    #         "switching_cost_bLoading_HVFASystem_CN":0.5,
    #         "poweron_cost_OuterCapillaryTubeMats":0.5,
    #         "poweron_cost_HeatPump":1,
    #     },
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
    # experiment_1.play("ppo_agent_switching_costs_var1", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 8, "n_episodes_learn": 8 * 50, "episode_duration": 259200,"seed":123},
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "extended_observations": False,
    #         "abort_cost_for_unsuccessfull_step": False,
    #         "with_storages": True,
    #         "use_complex_AFA_model": True,
    # "variance_min": 1.0,
    # "variance_max": 1.0,
    # "distribution": "uniform",
    # "variance_parameters": ["all"],
    #         "poweron_cost_HeatExchanger1":0.05,
    #         "poweron_cost_CHP1":0.4,
    #         "poweron_cost_CHP2":0.4,
    #         "poweron_cost_CondensingBoiler":0.1,
    #         "poweron_cost_VSIStorage":0.05,
    #         "switching_cost_bLoading_VSISystem":0.05,
    #         "poweron_cost_HVFASystem_HNLT":0.05,
    #         "switching_cost_bLoading_HVFASystem_HNLT":0.05,
    #         "poweron_cost_eChiller":0.1,
    #         "poweron_cost_HVFASystem_CN":0.05,
    #         "switching_cost_bLoading_HVFASystem_CN":0.05,
    #         "poweron_cost_OuterCapillaryTubeMats":0.05,
    #         "poweron_cost_HeatPump":0.1
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
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
    # experiment_1.learn("ppo_agent_switching_costs_var1_3", "experiment_1")

    ######################################
    #                                    #
    #  ppo agent switching costs  var 2  #
    #                                    #
    ######################################

    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 8, "n_episodes_learn": 8 * 50, "episode_duration": 259200,"seed":123},
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "extended_observations": False,
    #         "abort_cost_for_unsuccessfull_step": False,
    #         "with_storages": True,
    #         "use_complex_AFA_model": True,
    # "variance_min": 1.0,
    # "variance_max": 1.0,
    # "distribution": "uniform",
    # "variance_parameters": ["all"],
    #         "poweron_cost_HeatExchanger1":0,
    #         "poweron_cost_CHP1":0,
    #         "poweron_cost_CHP2":0,
    #         "poweron_cost_CondensingBoiler":0,
    #         "poweron_cost_VSIStorage":0,
    #         "switching_cost_bLoading_VSISystem":0,
    #         "poweron_cost_HVFASystem_HNLT":0,
    #         "switching_cost_bLoading_HVFASystem_HNLT":0,
    #         "poweron_cost_eChiller":0,
    #         "poweron_cost_HVFASystem_CN":0,
    #         "switching_cost_bLoading_HVFASystem_CN":0,
    #         "poweron_cost_OuterCapillaryTubeMats":0,
    #         "poweron_cost_HeatPump":0,
    #         "runtime_violation_cost_CHP1":1000,
    #         "runtime_violation_cost_CHP2":1000,
    #         "runtime_violation_cost_HeatPump":100,
    #         "runtime_violation_cost_eChiller":100,
    #         "runtime_violation_cost_OuterCapillaryTubeMats":100,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
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
    # experiment_1.learn("ppo_agent_switching_costs_var2", "experiment_1")

    # config_experiment_1_learn = {
    #     "settings": {"n_environments": 1, "n_episodes_learn": 8 * 50, "n_episodes_play":10, "episode_duration": 259200,"seed":123},
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "extended_observations": False,
    #         "abort_cost_for_unsuccessfull_step": False,
    #         "with_storages": True,
    #         "use_complex_AFA_model": True,
    # "variance_min": 1.0,
    # "variance_max": 1.0,
    # "distribution": "uniform",
    # "variance_parameters": ["all"],
    #         "poweron_cost_HeatExchanger1":0,
    #         "poweron_cost_CHP1":0,
    #         "poweron_cost_CHP2":0,
    #         "poweron_cost_CondensingBoiler":0,
    #         "poweron_cost_VSIStorage":0,
    #         "switching_cost_bLoading_VSISystem":0,
    #         "poweron_cost_HVFASystem_HNLT":0,
    #         "switching_cost_bLoading_HVFASystem_HNLT":0,
    #         "poweron_cost_eChiller":0,
    #         "poweron_cost_HVFASystem_CN":0,
    #         "switching_cost_bLoading_HVFASystem_CN":0,
    #         "poweron_cost_OuterCapillaryTubeMats":0,
    #         "poweron_cost_HeatPump":0,
    #         "runtime_violation_cost_CHP1":1000,
    #         "runtime_violation_cost_CHP2":1000,
    #         "runtime_violation_cost_HeatPump":100,
    #         "runtime_violation_cost_eChiller":100,
    #         "runtime_violation_cost_OuterCapillaryTubeMats":100,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
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
    # experiment_1.play("ppo_agent_switching_costs_var2", "experiment_1_play")


if __name__ == "__main__":
    main()
