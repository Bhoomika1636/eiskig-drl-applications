from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule

from sb3_contrib.ppo_mask.ppo_mask import MaskablePPO
from sb3_contrib.ppo_mask.policies import MlpPolicy


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent


    #########################
    #                       #
    #        Tests          #
    #                       #
    #########################

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 1, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Test", "experiment_1")


    # config_experiment_1_learn = {
    #     "setup": {
    #         "environment_import": "environment.supplysystem_b_Maskable.SupplysystemB",
    #         "policy_import": "sb3_contrib.ppo_mask.policies.MlpPolicy",
    #         "agent_import": "sb3_contrib.ppo_mask.ppo_mask.MaskablePPO"
    #         },
    #     "settings": {"n_environments": 1, "n_episodes_learn": 1, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Mask_Test1", "experiment_1")

    #########################
    #                       #
    #      Test series 1    #
    #                       #
    #########################

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_1_Baseline", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "extended_obs_4_entropy":True
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_1_Baseline_extended_obs_1", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "poweron_cost_combinedheatpower": 0,
    #         "poweron_cost_condensingboiler": 0,
    #         "poweron_cost_immersionheater": 0,
    #         "poweron_cost_heatpump": 0,
    #         "poweron_cost_compressionchiller": 0,
    #         "poweron_cost_coolingtower": 0,
    #         "extended_obs_4_entropy":True
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_1_NoSwitching_Costs_extended_obs_1", "experiment_1")


    # config_experiment_1_learn = {
    #     # "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_1_SplitNet", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "poweron_cost_combinedheatpower": 0,
    #         "poweron_cost_condensingboiler": 0,
    #         "poweron_cost_immersionheater": 0,
    #         "poweron_cost_heatpump": 0,
    #         "poweron_cost_compressionchiller": 0,
    #         "poweron_cost_coolingtower": 0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_1_NoSwitching_Costs", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "poweron_cost_combinedheatpower": 0.4,
    #         "poweron_cost_condensingboiler": 0.2,
    #         "poweron_cost_immersionheater": 0.1,
    #         "poweron_cost_heatpump": 0.1,
    #         "poweron_cost_compressionchiller": 0.3,
    #         "poweron_cost_coolingtower": 0.2,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_1_LowSwitching_Costs", "experiment_1")

    #########################
    #                       #
    #      Test series 2    #
    #                       #
    #########################

    # config_experiment_1_learn = {
    #     "setup": {"agent_import": "sb3_contrib.ppo_recurrent.ppo_recurrent.RecurrentPPO",
    #         "policy_import": "sb3_contrib.ppo_recurrent.policies.MlpLstmPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 1500, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_Lstm", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {"agent_import": "sb3_contrib.ppo_recurrent.ppo_recurrent.RecurrentPPO",
    #         "policy_import": "sb3_contrib.ppo_recurrent.policies.MlpLstmPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 1500, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "poweron_cost_combinedheatpower": 0,
    #         "poweron_cost_condensingboiler": 0,
    #         "poweron_cost_immersionheater": 0,
    #         "poweron_cost_heatpump": 0,
    #         "poweron_cost_compressionchiller": 0,
    #         "poweron_cost_coolingtower": 0,
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_Lstm_NoSwitchingCosts", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "poweron_cost_combinedheatpower": 0,
    #         "poweron_cost_condensingboiler": 0,
    #         "poweron_cost_immersionheater": 0,
    #         "poweron_cost_heatpump": 0,
    #         "poweron_cost_compressionchiller": 0,
    #         "poweron_cost_coolingtower": 0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "ent_coef": 0.05,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_EntCoef_05", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "poweron_cost_combinedheatpower": 0,
    #         "poweron_cost_condensingboiler": 0,
    #         "poweron_cost_immersionheater": 0,
    #         "poweron_cost_heatpump": 0,
    #         "poweron_cost_compressionchiller": 0,
    #         "poweron_cost_coolingtower": 0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "ent_coef": 0.005,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_EntCoef_005", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "poweron_cost_combinedheatpower": 0,
    #         "poweron_cost_condensingboiler": 0,
    #         "poweron_cost_immersionheater": 0,
    #         "poweron_cost_heatpump": 0,
    #         "poweron_cost_compressionchiller": 0,
    #         "poweron_cost_coolingtower": 0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "ent_coef": 0,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_EntCoef_000", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         # "poweron_cost_combinedheatpower": 0,
    #         # "poweron_cost_condensingboiler": 0,
    #         # "poweron_cost_immersionheater": 0,
    #         # "poweron_cost_heatpump": 0,
    #         # "poweron_cost_compressionchiller": 0,
    #         # "poweron_cost_coolingtower": 0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "ent_coef": 0.005,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_EntCoef_005_WithSwitching", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {
    #         "environment_import": "environment.supplysystem_b_Maskable.SupplysystemB",
    #         "policy_import": "sb3_contrib.ppo_mask.policies.MlpPolicy",
    #         "agent_import": "sb3_contrib.ppo_mask.ppo_mask.MaskablePPO"
    #         },
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         # "poweron_cost_combinedheatpower": 0.4,
    #         # "poweron_cost_condensingboiler": 0.2,
    #         # "poweron_cost_immersionheater": 0.1,
    #         # "poweron_cost_heatpump": 0.1,
    #         # "poweron_cost_compressionchiller": 0.3,
    #         # "poweron_cost_coolingtower": 0.2,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_MaskablePPO_WithMaskAsTempLimitPolicyShaping", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {
    #         "environment_import": "environment.supplysystem_b_Maskable.SupplysystemB",
    #         "policy_import": "sb3_contrib.ppo_mask.policies.MlpPolicy",
    #         "agent_import": "sb3_contrib.ppo_mask.ppo_mask.MaskablePPO"
    #         },
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "abort_costs":5,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_MaskablePPO_WithMaskAsTempLimitPolicyShaping_2", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {
    #         "environment_import": "environment.supplysystem_b_Maskable.SupplysystemB",
    #         "policy_import": "sb3_contrib.ppo_mask.policies.MlpPolicy",
    #         "agent_import": "sb3_contrib.ppo_mask.ppo_mask.MaskablePPO"
    #         },
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "abort_costs":5,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_MaskablePPO_WithMaskAsTempLimitPolicyShaping_3", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {
    #         "environment_import": "environment.supplysystem_b_Maskable.SupplysystemB",
    #         "policy_import": "sb3_contrib.ppo_mask.policies.MlpPolicy",
    #         "agent_import": "sb3_contrib.ppo_mask.ppo_mask.MaskablePPO"
    #         },
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "abort_costs":5,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_MaskablePPO_WithMaskAsTempLimitPolicyShaping_4", "experiment_1")

    # config_experiment_1_learn = {
    #     "setup": {
    #         "environment_import": "environment.supplysystem_b_Maskable.SupplysystemB",
    #         "policy_import": "sb3_contrib.ppo_mask.policies.MlpPolicy",
    #         "agent_import": "sb3_contrib.ppo_mask.ppo_mask.MaskablePPO"
    #         },
    #     "settings": {"n_environments": 4, "n_episodes_learn": 3000, "episode_duration": 259200, "plot_interval": 100},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "abort_costs":5,
    #         "poweron_cost_combinedheatpower": 0.4,
    #         "poweron_cost_condensingboiler": 0.2,
    #         "poweron_cost_immersionheater": 0.1,
    #         "poweron_cost_heatpump": 0.1,
    #         "poweron_cost_compressionchiller": 0.3,
    #         "poweron_cost_coolingtower": 0.2,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "ent_coef": 0.05,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.learn("Entropie_Test_2_MaskablePPO_WithMaskAsTempLimitPolicyShaping_5", "experiment_1")


    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 3000, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_1_Baseline", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 3000, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         # "poweron_cost_combinedheatpower": 0,
    #         # "poweron_cost_condensingboiler": 0,
    #         # "poweron_cost_immersionheater": 0,
    #         # "poweron_cost_heatpump": 0,
    #         # "poweron_cost_compressionchiller": 0,
    #         # "poweron_cost_coolingtower": 0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_1_NoSwitching_Costs", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 3000, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         # "poweron_cost_combinedheatpower": 0,
    #         # "poweron_cost_condensingboiler": 0,
    #         # "poweron_cost_immersionheater": 0,
    #         # "poweron_cost_heatpump": 0,
    #         # "poweron_cost_compressionchiller": 0,
    #         # "poweron_cost_coolingtower": 0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "ent_coef": 0.05,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_2_EntCoef_05", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 3000, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         # "poweron_cost_combinedheatpower": 0,
    #         # "poweron_cost_condensingboiler": 0,
    #         # "poweron_cost_immersionheater": 0,
    #         # "poweron_cost_heatpump": 0,
    #         # "poweron_cost_compressionchiller": 0,
    #         # "poweron_cost_coolingtower": 0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "ent_coef": 0.005,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_2_EntCoef_005", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 3000, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         # "poweron_cost_combinedheatpower": 0,
    #         # "poweron_cost_condensingboiler": 0,
    #         # "poweron_cost_immersionheater": 0,
    #         # "poweron_cost_heatpump": 0,
    #         # "poweron_cost_compressionchiller": 0,
    #         # "poweron_cost_coolingtower": 0,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "ent_coef": 0,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_2_EntCoef_000", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "setup": {"agent_import": "sb3_contrib.ppo_recurrent.ppo_recurrent.RecurrentPPO",
    #         "policy_import": "sb3_contrib.ppo_recurrent.policies.MlpLstmPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 1500, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_2_Lstm", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "setup": {"agent_import": "sb3_contrib.ppo_recurrent.ppo_recurrent.RecurrentPPO",
    #         "policy_import": "sb3_contrib.ppo_recurrent.policies.MlpLstmPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 1500, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         # "poweron_cost_combinedheatpower": 0,
    #         # "poweron_cost_condensingboiler": 0,
    #         # "poweron_cost_immersionheater": 0,
    #         # "poweron_cost_heatpump": 0,
    #         # "poweron_cost_compressionchiller": 0,
    #         # "poweron_cost_coolingtower": 0,
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_2_Lstm_NoSwitchingCosts", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "setup": {
    #         "environment_import": "environment.supplysystem_b_Maskable.SupplysystemB",
    #         "policy_import": "sb3_contrib.ppo_mask.policies.MlpPolicy",
    #         "agent_import": "sb3_contrib.ppo_mask.ppo_mask.MaskablePPO"
    #         },
    #     "settings": {"n_environments": 1, "n_episodes_learn": 3000, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific": {
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "abort_costs":5,
    #     },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, 400, 300]},
    #         "device": "cpu",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_2_MaskablePPO_WithMaskAsTempLimitPolicyShaping_4", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 3000, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "extended_obs_4_entropy":True
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_1_Baseline_extended_obs_1", "experiment_1_play")

    # config_experiment_1_learn = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_learn": 3000, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
    #     "environment_specific":{
    #         "variance_min": 1,
    #         "variance_max": 1,
    #         "poweron_cost_combinedheatpower": 0,
    #         "poweron_cost_condensingboiler": 0,
    #         "poweron_cost_immersionheater": 0,
    #         "poweron_cost_heatpump": 0,
    #         "poweron_cost_compressionchiller": 0,
    #         "poweron_cost_coolingtower": 0,
    #         "extended_obs_4_entropy":True
    #         },
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_b_ppo",
    #     config_overwrite=config_experiment_1_learn,
    #     relpath_config="config/")
    # experiment_1.play("Entropie_Test_1_NoSwitching_Costs_extended_obs_1", "experiment_1_play")

    config_experiment_1_learn = {
        "setup": {
            "environment_import": "environment.supplysystem_b_Maskable.SupplysystemB",
            "policy_import": "sb3_contrib.ppo_mask.policies.MlpPolicy",
            "agent_import": "sb3_contrib.ppo_mask.ppo_mask.MaskablePPO"
            },
        "settings": {"n_environments": 1, "n_episodes_learn": 3000, "n_episodes_play":10, "episode_duration": 259200, "plot_interval": 1},
        "environment_specific": {
            "variance_min": 1,
            "variance_max": 1,
            "abort_costs":5,
            # "poweron_cost_combinedheatpower": 0.4,
            # "poweron_cost_condensingboiler": 0.2,
            # "poweron_cost_immersionheater": 0.1,
            # "poweron_cost_heatpump": 0.1,
            # "poweron_cost_compressionchiller": 0.3,
            # "poweron_cost_coolingtower": 0.2,
        },
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "ent_coef": 0.05,
            "policy_kwargs": {"net_arch": [500, 400, 300]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }
    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/")
    experiment_1.play("Entropie_Test_2_MaskablePPO_WithMaskAsTempLimitPolicyShaping_5", "experiment_1_play")

if __name__ == "__main__":
    main()
