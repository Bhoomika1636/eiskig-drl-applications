from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    config_experiment_1 = {
        "settings": {"episode_duration": 259200, "n_episodes_play": 1, "plot_interval": 1},
        "environment_specific": {
            "temperature_cool_init_max": 288,
            "temperature_cool_init_min": 288,
            "temperature_heat_init_max": 342.5,
            "temperature_heat_init_min": 342.5,
            "scenario_time_begin": "2018-03-17 00:00",
            "scenario_time_end": "2018-05-30 00:00",
            "random_sampling": False,
            "scenario_files": [
                {
                    "path": "Factory_2018.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {
                        "power_electricity": 12.5,
                        "power_heat": 16.0,
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
            ],
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_rule_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("rule_based_test", "experiment_1")


if __name__ == "__main__":
    main()
