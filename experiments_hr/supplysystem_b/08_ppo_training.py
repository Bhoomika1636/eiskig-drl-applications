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


    # This training will create an agent trained on the standard simulation parameters (parameterset "P1").
    config_experiment_P1_learn = {
        "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
        "settings": {"n_environments": 4, "n_episodes_learn": 8004, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cpu",  # "cuda" on systems with cuda installed
        },
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-27 00:00",
            "random_sampling": True,
            "variance_min": 1.0,
            "variance_max": 1.0,
            "distribution": "uniform",
            "variance_parameters": ["all"],
            "scenario_files": [
                {
                    "path": "Factory_2017.csv",
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
                    "path": "EnergyMarkets_2017.csv",
                    "interpolation_method": "ffill",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
                },
                {
                    "path": "Weather_Frankfurt_2017.csv",
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
        config_overwrite=config_experiment_P1_learn,
        relpath_config="config/",
    )

    experiment_P1.learn("PPO_on_P1", "PPO_P1")


    # This training will create an agent trained on a uniform distribution of the most sensitive parameters in the boundaries [0.8, 1.2]
    config_experiment_uniform_learn = {
        "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
        "settings": {"n_environments": 4, "n_episodes_learn": 8004, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cpu",  # "cuda" on systems with cuda installed
        },
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-27 00:00",
            "random_sampling": True,
            "variance_min": 0.8,
            "variance_max": 1.2,
            "distribution": "uniform",
            "variance_parameters": [
                "chp_variance",
                "compressionChiller_CT_variance",
                "coolingTower_open_variance",
                "idealPump1_variance",
            ],
            "scenario_files": [
                {
                    "path": "Factory_2017.csv",
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
                    "path": "EnergyMarkets_2017.csv",
                    "interpolation_method": "ffill",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
                },
                {
                    "path": "Weather_Frankfurt_2017.csv",
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

    experiment_uniform = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_experiment_uniform_learn,
        relpath_config="config/",
    )

    experiment_uniform.learn("PPO_on_uniform", "PPO_uniform_distribution_of_SA_parameters")


    # This training will create an agent trained on a normal distribution around P_ident
    config_experiment_normal_learn = {
        "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
        "settings": {"n_environments": 4, "n_episodes_learn": 8004, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cpu",  # "cuda" on systems with cuda installed
        },
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-27 00:00",
            "random_sampling": True,
            "variance_min": 0.8,
            "variance_max": 1.2,
            "distribution": "normal",
            "variance_parameters": [
                "chp_variance",
                "compressionChiller_CT_variance",
                "coolingTower_open_variance",
                "idealPump1_variance",
            ],
            "scenario_files": [
                {
                    "path": "Factory_2017.csv",
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
                    "path": "EnergyMarkets_2017.csv",
                    "interpolation_method": "ffill",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
                },
                {
                    "path": "Weather_Frankfurt_2017.csv",
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

    experiment_normal = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_experiment_normal_learn,
        relpath_config="config/",
    )

    experiment_normal.learn("PPO_on_normal", "PPO_normal_distribution_around_P_ident")


    # This training will create an agent trained on P_ident
    config_experiment_P_ident_learn = {
        "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
        "settings": {"n_environments": 4, "n_episodes_learn": 8004, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cpu",  # "cuda" on systems with cuda installed
        },
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-27 00:00",
            "random_sampling": True,
            "variance_min": 1.0,
            "variance_max": 1.0,
            "distribution": "fixed",
            "variance_parameters": ["all"],
            "scenario_files": [
                {
                    "path": "Factory_2017.csv",
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
                    "path": "EnergyMarkets_2017.csv",
                    "interpolation_method": "ffill",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
                },
                {
                    "path": "Weather_Frankfurt_2017.csv",
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

    experiment_P_ident = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_experiment_P_ident_learn,
        relpath_config="config/",
    )

    experiment_P_ident.learn("PPO_on_identified", "PPO_P_ident")


    # This training will create an agent trained on P2
    config_experiment_P2_learn = {
        "setup": {"policy_import": "common.CustomPolicies.CustomActorCriticPolicy"},
        "settings": {"n_environments": 4, "n_episodes_learn": 8004, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cpu",  # "cuda" on systems with cuda installed
        },
        "environment_specific": {
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-27 00:00",
            "random_sampling": True,
            "variance_min": 1.0,
            "variance_max": 1.0,
            "distribution": "fixed",
            "variance_parameters": ["all"],
            "scenario_files": [
                {
                    "path": "Factory_2017.csv",
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
                    "path": "EnergyMarkets_2017.csv",
                    "interpolation_method": "ffill",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
                },
                {
                    "path": "Weather_Frankfurt_2017.csv",
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

    experiment_P2 = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_ppo",
        config_overwrite=config_experiment_P2_learn,
        relpath_config="config/",
    )

    experiment_P2.learn("PPO_on_P2", "PPO_P2")


if __name__ == "__main__":
    main()
