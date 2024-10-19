from __future__ import annotations
import pandas as pd
import pathlib

from datetime import datetime

from eta_utility import get_logger
from eta_utility.eta_x import ETAx

def load_and_check_csv(file_path: str, required_columns: list[str]) -> pd.DataFrame:
    df = pd.read_csv(file_path, sep=';')
    print(f"Loaded {file_path} with columns: {df.columns.tolist()}")
    for col in required_columns:
        if col not in df.columns:
            raise KeyError(f"Column '{col}' not found in {file_path}")
    return df


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    heat_df = load_and_check_csv(root_path /"data"/ "Heat_2017.csv", ["datetime", "Heat"])
    config_experiment_1 = {
        "settings": {"episode_duration": 86400 * 7, "n_episodes_play": 1, "plot_interval": 1, "sampling_time": 60},
        "environment_specific": {
            "scenario_time_begin": "2018-08-01 00:00",
            "scenario_time_end": "2019-01-01 00:00",
            "discretize_action_space": False,
            "random_params": False,
            "random_sampling": False,
            "scenario_files": [
                {
                    "path": "Heat_2018.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%Y-%m-%d %H:%M",
                    "scale_factors": {},
                    
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

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="equinix_pid_ohne_wBus_rule_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("JSON_SAVING_2", "Test")

    startTime = datetime.now()   
    print("startTime: ", startTime.strftime('%Y-%m-%d %H:%M:%S'))
    experiment_1.play("rule_based_summer_2018", "run_1")
    endTime = datetime.now()
    print("endTime: ", endTime.strftime('%Y-%m-%d %H:%M:%S'))
    runTime = endTime - startTime
    print("runTime: ", runTime)



if __name__ == "__main__":
    main()
