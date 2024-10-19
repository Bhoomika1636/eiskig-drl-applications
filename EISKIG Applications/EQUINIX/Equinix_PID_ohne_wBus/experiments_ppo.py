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
        "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
        "settings": {"n_environments": 16, "n_episodes_learn": 1000, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="equinix_pid_ohne_wBus_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )

    experiment_1.learn("ppo_training_6", "seed_123")


    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    # config_experiment_1_play = {
    #     "settings": {"n_environments": 1, "n_episodes_play": 1, "episode_duration": 86400 * 1, "plot_interval": 1},
    #     "agent_specific": {"policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]}},
    #     "environment_specific": {
    #         "temperature_cool_init_max": 288,
    #         "temperature_cool_init_min": 288,
    #         "temperature_heat_init_max": 342.5,
    #         "temperature_heat_init_min": 342.5,
    #         "scenario_time_begin": "2018-07-07 00:00",
    #         "scenario_time_end": "2018-12-27 00:00",
    #         "random_sampling": False,
    #         "variance_min": 1.0,
    #         "variance_max": 1.0,
    #         "variance_parameters": ["all"],
    #         "scenario_files": [
    #             {
    #                 "path": "Heat_2018.csv",
    #                 "interpolation_method": "interpolate",
    #                 "resample_method": "asfreq",
    #                 "time_conversion_str": "%Y-%m-%d %H:%M",
    #                 "scale_factors": {},
    #             },
    #             {
    #                 "path": "EnergyMarkets_2018.csv",
    #                 "interpolation_method": "ffill",
    #                 "resample_method": "asfreq",
    #                 "time_conversion_str": "%Y-%m-%d %H:%M",
    #                 "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
    #             },
    #             {
    #                 "path": "Weather_Frankfurt_2018.csv",
    #                 "interpolation_method": "interpolate",
    #                 "resample_method": "asfreq",
    #                 "time_conversion_str": "%Y-%m-%d %H:%M",
    #                 "scale_factors": {
    #                     "air_pressure": 1,
    #                     "air_temperature": 1,
    #                     "relative_air_humidity": 1,
    #                     "rain_indicator": 1,
    #                     "rainfall": 1,
    #                     "wind_direction": 1,
    #                     "wind_speed": 1,
    #                     "clouds": 1,
    #                     "global_radiation": 1,
    #                     "direct_radiation": 1,
    #                     "diffuse_radiation": 1,
    #                 },
    #             },
    #         ],
    #     },
    # }

    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="equinix_pid_ohne_wBus_ppo",
    #     config_overwrite=config_experiment_1_play,
    #     relpath_config="config/",
    # )

    # experiment_1.play("play_6", "seed_123")


if __name__ == "__main__":
    main()
