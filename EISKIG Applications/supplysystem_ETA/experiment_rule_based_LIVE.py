from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    """
    CAUTION! This script is used for the live application of optmization algorithms. 

    As soon as this script is started, real systems are switched in the ETA factory, 
    which represents a corresponding risk if the application is not prepared accordingly. 
    This script should therefore not be started without prior consultation with an ETA Wimi!

    """

    # config_experiment_1 = {
    #     "settings": {"episode_duration": 60 * 60 * 26, "sampling_time": 30, "n_episodes_play": 1},
    #     "environment_specific": {},
    # }

    # experiment_1 = ETAx(
    #     root_path=root_path,
    #     config_name="supplysystem_ETA_live",
    #     config_overwrite=config_experiment_1,
    #     relpath_config="config/",
    # )

    # experiment_1.play("live_ETA_Application_rulebased", "experiment_1605_run2")


if __name__ == "__main__":
    main()
