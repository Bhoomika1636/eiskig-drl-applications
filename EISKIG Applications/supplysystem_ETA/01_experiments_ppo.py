from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    """
    Deterministic Actions:

    When determinstic actions are used (to be changed in eta-utility) the temperature reward boundaries should be changed as commented below.

    Switching Costs:
    In addition, lower switching costs should be used, as commented below, to have a more balanced reward function. Experiments have not been carried out yet.
    
    """

    ####################################
    #                                  #
    #        ppo agent baseline        #
    #                                  #
    ####################################
    config_experiment_1_learn = {
        "setup": {
            # "environment_import":"environment.supplysystem_ETA_variance.SupplysystemETA",
            "policy_import": "common.CustomPolicies.CustomActorCriticPolicy"
        },
        "settings": {"n_environments": 16, 
                     "n_episodes_learn": 400, 
                     "n_episodes_play": 1, 
                     "episode_duration": 86400 * 6},# 6 days
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2018-12-27 00:00",
#             "poweron_cost_HeatExchanger1":0.2,
#             "poweron_cost_CHP1":0.8,
#             "poweron_cost_CHP2":0.8,
#             "poweron_cost_CondensingBoiler":0.2,
#             "poweron_cost_VSIStorage":0.2,
#             "switching_cost_bLoading_VSISystem":0.2,
#             "poweron_cost_HVFASystem_HNLT":0.2,
#             "switching_cost_bLoading_HVFASystem_HNLT":0.2,
#             "poweron_cost_eChiller":0.8,
#             "poweron_cost_HVFASystem_CN":0.2,
#             "switching_cost_bLoading_HVFASystem_CN":0.2,
#             "poweron_cost_OuterCapillaryTubeMats":0.2,
#             "poweron_cost_HeatPump":0.8,
#             "HNHT_temperature_reward_min_T":50, #+5
#             "HNHT_temperature_reward_max_T":80, #-5
#             "HNLT_temperature_reward_min_T":15, #+5
#             "HNLT_temperature_reward_max_T":45, #-5
#             "CN_temperature_reward_min_T":12, #+2
#             "CN_temperature_reward_max_T":23, #-2
#             "offset_policy_shaping_threshold":5,
#             "temperature_cost_HNHT":2, #1
#             "temperature_cost_HNLT":2, #1
#             "temperature_cost_CN":2, #1
        },
        "agent_specific": {
            "learning_rate": LinearSchedule(0.00035, 0.000015).value,
            "policy_kwargs": {"net_arch": [500, 400, 300]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }
    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )
    experiment_1.learn("ppo_agent_baseline", "experiment_1")
    # experiment_1.play("ppo_agent_baseline", "experiment_1_play") # also change scenario time begin and end for test scenario

    


if __name__ == "__main__":
    main()
