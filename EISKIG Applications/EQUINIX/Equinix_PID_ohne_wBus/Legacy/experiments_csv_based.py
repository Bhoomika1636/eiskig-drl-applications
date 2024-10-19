from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    config_experiment_1 = {
        "settings": {"episode_duration": 86400 * 1, "n_episodes_play": 5, "plot_interval": 1},  # needs to be known
        "environment_specific": {
            "temperature_cool_init_max": 288,  # needs to be read manually from the csv
            "temperature_cool_init_min": 288,  # needs to be read manually from the csv
            "temperature_heat_init_max": 342.5,  # needs to be read manually from the csv
            "temperature_heat_init_min": 342.5,  # needs to be read manually from the csv
            "scenario_time_begin": "2018-01-01 00:00",  # needs to be known
            "scenario_time_end": "2018-12-27 00:00",  # needs to be known
            "random_sampling": False,
            "variance_min": 0.8,
            "variance_max": 1.2,
            "variance_parameters": ["chp_variance"],
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
            "csv_path": "C:/Users/h.ranzau_lokal/Daten/GitLab/experiments_hr/experiments_hr/supplysystem_b/results/Test_CSV_Player/seed_17_002-01_episode.csv",
            "steps_per_episode": 480 * 1,
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_b_csv_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("Test_CSV_Player_2", "TEST_CHP")


if __name__ == "__main__":
    main()
