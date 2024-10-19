from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    csv_path = (
        "experiments_hr/supplysystem_b/results/2018_360_days_random_P2/2018_360_days_all_random_000-01_episode.csv"
    )
    episode_length = 86400 * 15
    step_size = 180

    config_experiment_1 = {
        "settings": {"episode_duration": episode_length, "n_episodes_play": 1, "plot_interval": 1},  # needs to be known
        "environment_specific": {
            "temperature_cool_init_max": 288.15,  # TODO: Replace this by read-out from csv
            "temperature_cool_init_min": 288.15,  # TODO: Replace this by read-out from csv
            "temperature_heat_init_max": 353.15,  # TODO: Replace this by read-out from csv
            "temperature_heat_init_min": 353.15,  # TODO: Replace this by read-out from csv
            "scenario_time_begin": "2018-01-01 00:00",  # = day 46
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
        "agent_specific": {
            "csv_path": csv_path,
            "steps_per_episode": int(episode_length / step_size),
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_csv_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("csv_source_test", "Identified_day_1_long")


if __name__ == "__main__":
    main()
