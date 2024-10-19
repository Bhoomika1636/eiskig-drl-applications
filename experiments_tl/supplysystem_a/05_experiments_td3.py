from __future__ import annotations

import pathlib

import numpy as np
from eta_utility import get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule
from stable_baselines3.common.noise import NormalActionNoise


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent
    n_actions = 3

    ##################
    #                #
    #    TRAINING    #
    #                #
    ##################

    config_experiment_1_learn = {
        "settings": {"n_environments": 4, "n_episodes_learn": 10008, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "action_noise": NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions)),
            "train_freq": 10,
            "buffer_size": 1000000,
            "policy_kwargs": {"net_arch": [500, 400, 300]},
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_td3",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )

    experiment_1.learn("td3_baseline", "experiment_1")

    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    # config_experiment_1_play = {
    #     "settings": {"n_environments": 1, "n_episodes_play": 1, "episode_duration": 259200},
    #     "environment_specific": {
    #         "temperature_cool_init_max": 288,
    #         "temperature_cool_init_min": 288,
    #         "temperature_heat_init_max": 342.5,
    #         "temperature_heat_init_min": 342.5,
    #         "scenario_time_begin": "2018-01-01 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "random_sampling": False,
    #         "scenario_files": [
    #             {
    #                 "path": "Factory_2018.csv",
    #                 "interpolation_method": "interpolate",
    #                 "resample_method": "asfreq",
    #                 "time_conversion_str": "%Y-%m-%d %H:%M",
    #                 "scale_factors": {
    #                     "power_electricity": 12.5,
    #                     "power_heat": 16.0,
    #                     "power_gas": 6.25,
    #                     "time_availability": 1,
    #                 },
    #             },
    #             {
    #                 "path": "EnergyMarkets_2018.csv",
    #                 "interpolation_method": "ffill",
    #                 "resample_method": "asfreq",
    #                 "time_conversion_str": "%Y-%m-%d %H:%M",
    #                 "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
    #             },
    #         ],
    #     },
    # }

    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_a_td3",
    #     config_overwrite=config_experiment_1_play,
    #     relpath_config="config/",
    # )

    # experiment_1.play("td3_1", "experiment_1")


if __name__ == "__main__":
    main()
