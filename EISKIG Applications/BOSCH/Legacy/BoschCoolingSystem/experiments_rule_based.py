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
        "settings": {"episode_duration": 3600*10,"sampling_time": 30, "n_episodes_play": 1, "plot_interval": 1},
        "environment_specific": {
            "scenario_time_begin": "2018-03-17 00:00",
            "scenario_time_end": "2018-05-30 00:00",
            "random_sampling": False,
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="BoschSystem_rule_based",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )
    t1 = time.time()
    experiment_1.play("rule_based_test", "experiment_1")
    print("Time:", time.time()-t1)

if __name__ == "__main__":
    main()
