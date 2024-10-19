from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx
import time
import sys
# Add the directory containing controllerFunctions.py to the Python module search path
sys.path.append('experiments_hr')

def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    config_experiment_1 = {
        "settings": {"episode_duration": 3600*12 ,"sampling_time": 120, "n_episodes_play": 1, "plot_interval": 1},
        "environment_specific": {
            "discretize_action_space": False,
            "up_down_discrete": False,
            "use_policy_shaping": True,
            "scenario_time_begin": "2019-01-21 22:00",
            "scenario_time_end": "2019-12-30 00:00",
            "random_sampling": False,
            "activate_load_recalc": True,
            "scenario_files": [
                {
                    "path": "Factory_2019.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%d.%m.%Y %H:%M",
                    "scale_factors": {
                        "q_cold": 1,
                        "q_cool": 1
                    },
                },
                {
                    "path": "EnergyMarkets_2019.csv",
                    "interpolation_method": "ffill",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%d.%m.%Y %H:%M",
                    "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
                }, 
                {
                    "path": "Weather_Wurzburg_2019.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%d.%m.%Y %H:%M",
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
                }
            ]
        }
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="BoschSystem_rule_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )
    t1 = time.time()
    experiment_1.play("rule_based_test2", "experiment_1")
    print("Time:", time.time()-t1)

if __name__ == "__main__":
    main()
