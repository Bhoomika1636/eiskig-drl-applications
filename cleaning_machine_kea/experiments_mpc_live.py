import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = pathlib.Path(__file__).parent

    experiment = ETAx(
        root_path=root_path, config_overwrite=None, relpath_config="config", config_name="cleaning_machine_kea_mpc"
    )

    experiment.play("test", "experiment_1")


if __name__ == "__main__":
    main()
