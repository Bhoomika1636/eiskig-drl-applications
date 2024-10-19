from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    config_experiment_1 = {
        "settings": {"episode_duration": 60 * 60 * 26, "sampling_time": 30, "n_episodes_play": 1},
        "environment_specific": {},
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_ETA_live",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("live_ETA_Application_rulebased", "experiment_1605_run2")


if __name__ == "__main__":
    main()
