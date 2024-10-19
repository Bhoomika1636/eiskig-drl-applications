from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    """
    CAUTION! This script is used for the live application of optmization algorithms. 

    As soon as this script is started, real systems are switched in the ETA factory, 
    which represents a corresponding risk if the application is not prepared accordingly. 
    This script should therefore not be started without prior consultation with an ETA Wimi!

    """

    # config_experiment_1_play = {
    #     "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
    #     "settings": {"n_environments": 1, "n_episodes_play": 1, "episode_duration": 60 * 60 * 8},
    #     "environment_specific": {"allow_limiting_CHP_switches": True},
    #     "agent_specific": {
    #         "learning_rate": LinearSchedule(0.0002, 0.00002).value,
    #         "batch_size": 256,
    #         "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
    #         "device": "cuda",  # "cuda" on systems with cuda installed
    #     },
    # }
    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_ETA_ppo_live",
    #     config_overwrite=config_experiment_1_play,
    #     relpath_config="config/",
    # )

    # experiment_1.play("ppo_agent_baseline_continued_LiveApplication", "experiment_1206")


if __name__ == "__main__":
    main()
