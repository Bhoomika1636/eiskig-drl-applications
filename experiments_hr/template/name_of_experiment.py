from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    ##################
    #                #
    #    TRAINING    #
    #   (for DRLs)   #
    #                #
    ##################

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="name_of_config_file_in_config_folder",
        config_overwrite={},
        relpath_config="config/",
    )

    experiment_1.learn("experiment_name", "tensorboard_description")

    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="name_of_config_file_in_config_folder",
        config_overwrite={},
        relpath_config="config/",
    )

    experiment_1.play("experiment_name", "tensorboard_description")


if __name__ == "__main__":
    main()
