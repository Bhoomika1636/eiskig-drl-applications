from __future__ import annotations

import pathlib
import time

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    config_experiment_1 = {
        "setup": {"agent_import": "controller.speed_test_controller.SpeedTestController"},
        "settings": {"episode_duration": 60 * 60 * 6, "sampling_time": 30, "n_episodes_play": 1, "plot_interval": 1},
        "environment_specific": {
            "scenario_time_begin": "2017-04-30 00:00",
            "scenario_time_end": "2017-04-30 18:00",
            "allow_policy_shaping": False,
            "allow_limiting_CHP_switches": False,
            "extended_observations": True,
            # winter. 20.01.2017
            # summer 07.07.2017
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_rule_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    sim_time_start = time.time()
    experiment_1.play("SpeedTest_Baseline", "experiment_1")

    sim_time_elapsed = time.time() - sim_time_start

    print("sim_time_elapsed", sim_time_elapsed)

    ####################################################################
    #                                                                  #
    #           The following is the speed test for TSCL env           #
    #                                                                  #
    ####################################################################

    # config_experiment_1 = {
    #     "setup": {
    #         "environment_import": "environment.supplysystem_ETA_TSCL.SupplysystemETA",
    #         "agent_import": "controller.speed_test_controller.SpeedTestController"},
    #     "settings": {"episode_duration": 60 * 60 * 6, "sampling_time": 30, "n_episodes_play": 1, "plot_interval": 1},
    #     "environment_specific": {
    #         "scenario_time_begin": "2017-04-30 00:00",
    #         "scenario_time_end": "2017-05-30 18:00",
    #         "allow_policy_shaping": False,
    #         "allow_limiting_CHP_switches": False,
    #         "extended_observations": True,
    #         # winter. 20.01.2017
    #         # summer 07.07.2017
    #     },
    # }

    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_ETA_rule_based",
    #     config_overwrite=config_experiment_1,
    #     relpath_config="config/",
    # )

    # sim_time_start = time.time()
    # experiment_1.play("SpeedTest_TSCL", "experiment_1")
    # sim_time_elapsed = time.time() - sim_time_start
    # print("sim_time_elapsed", sim_time_elapsed)


if __name__ == "__main__":
    main()
