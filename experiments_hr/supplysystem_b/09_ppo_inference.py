from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

  
    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    config_play_on_P2 = {
        "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
        "settings": {"n_environments": 1, "n_episodes_play": 1, "episode_duration": 86400 * 10, "plot_interval": 1},
        "agent_specific": {"policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]}},
        "environment_specific": {
            "scenario_time_begin": "2018-01-01 00:00",
            "scenario_time_end": "2018-12-27 00:00",
            "random_sampling": False,
            "variance_min": 1.0,
            "variance_max": 1.0,
            "distribution": "fixed",
            "variance_parameters": ["all"],
            "scenario_files": [
                {
                    "path": "Factory_2018.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {
                        "power_electricity": 12.5,
                        "power_heat": 16.0,
                        "power_cold": 60.5,
                        "power_gas": 6.25,
                        "time_availability": 1,
                    },
                },
                {
                    "path": "EnergyMarkets_2018.csv",
                    "interpolation_method": "ffill",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
                },
                {
                    "path": "Weather_Frankfurt_2018.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {
                        "air_pressure": 1,
                        "air_temperature": 1,
                        "relative_air_humidity": 1,
                        "rain_indicator": 1,
                        "rainfall": 1,
                        "wind_direction": 1,
                        "wind_speed": 1,
                        "clouds": 1,
                        "global_radiation": 1,
                        "direct_radiation": 1,
                        "diffuse_radiation": 1,
                    },
                },
            ],
        },
    }


    experiment_P1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_play_on_P2,
        relpath_config="config/",
    )

    experiment_P1.play("PPO_on_P1", "PPO_P1")


    experiment_uniform = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_play_on_P2,
        relpath_config="config/",
    )

    experiment_uniform.play("PPO_on_uniform", "PPO_uniform_distribution_of_SA_parameters")


    experiment_normal = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_play_on_P2,
        relpath_config="config/",
    )

    experiment_normal.play("PPO_on_normal", "PPO_normal_distribution_around_P_ident")


    experiment_P_ident = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_play_on_P2,
        relpath_config="config/",
    )

    experiment_P_ident.play("PPO_on_identified", "PPO_P_ident")


    experiment_P2 = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_play_on_P2,
        relpath_config="config/",
    )

    experiment_P2.play("PPO_on_P2", "PPO_P2")



if __name__ == "__main__":
    main()
