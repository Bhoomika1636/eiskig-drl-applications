from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    config_experiment_1 = {
        "settings": {"episode_duration": 60*60*48, "sampling_time": 180, "n_episodes_play": 1, "plot_interval": 1},
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="AFA_system",
        config_overwrite=config_experiment_1,
        relpath_config="config/",
    )

    experiment_1.play("rule_based_test_6hour", "experiment_3")


if __name__ == "__main__":
    main()
