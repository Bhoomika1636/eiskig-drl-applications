from __future__ import annotations

import pathlib

from eta_utility import LOG_INFO, get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule
import time

import sys
import numpy as np
# Add the directory containing controllerFunctions.py to the Python module search path
sys.path.append('experiments_hr')

def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent
    # neew to fix np. seed as well
    seed = 123
    np.random.seed(seed)
    episode_duration = 3600*24*364
    n_env = 1 # keep at 1 to ensure same dates are taken for both strategies 
    n_episodes = 1
    config_rb = {
        "settings": {"episode_duration":episode_duration,"sampling_time": 120, "n_episodes_play": n_episodes*n_env, "plot_interval": 1, "seed": seed},
        "environment_specific": {
            "discretize_action_space": False,
            "up_down_discrete": False,
            "scenario_time_begin": "2019-01-01 00:00",
            # "scenario_time_begin": "2019-05-23 20:02",
            "scenario_time_end": "2019-12-31 23:00",
            "random_sampling": False,
            "use_policy_shaping": True,
            "no_overwrite": False,
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

    config_ppo_play = {
        "settings": {"episode_duration": episode_duration,"sampling_time": 120,"n_environments": n_env, "n_episodes_play": n_episodes, "plot_interval": 1,"seed": seed},
        # "agent_specific": {"policy_kwargs": {"net_arch": [256, dict(pi=[128, 64], vf=[128, 64])]},"device": "cuda"},
        "agent_specific": {"policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},"device": "cuda"},
        "environment_specific": {
            "discretize_action_space": True,
            "up_down_discrete": True,
            "use_policy_shaping": True,
            "scenario_time_begin": "2019-01-01 00:00",
            "scenario_time_end": "2019-12-31 23:00",
            "random_sampling": False,
            "no_overwrite": False,
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

    experiment_ppo = ETAx(
        root_path=root_path,
        config_name="BoschSystem_ppo",
        config_overwrite=config_ppo_play,
        relpath_config="config/",
    )

    experiment_rb = ETAx(
        root_path=root_path,
        config_name="BoschSystem_rule_based",
        config_overwrite=config_rb,
        relpath_config="config/",
    )

    # experiment_ppo.play("compare_ppo_rb3", "agent_2")
    experiment_ppo.play("compare_FY", "agent_5")
    # experiment_rb.play("compare_FY", "rb")

if __name__ == "__main__":
    main()