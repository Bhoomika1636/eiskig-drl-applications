from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx


def main() -> None:
    get_logger()
    root_path = get_path()

    # EXPERIMENTS WITH DRL CONTROLLER

    # This experiment just takes settings from the config file
    # Start learning process:
    experiment1 = ETAx(root_path, "damped_oscillator_ppo", relpath_config="config")
    experiment1.learn("learning_series_1", "run1")
    # Start playing process:
    experiment1.play("learning_series_1", "run1")

    # This experiment overwrites certain settings from the config file
    # Start learning process:
    experiment2 = ETAx(
        root_path,
        "damped_oscillator_ppo",
        {"settings": {"n_episodes_learn": 10, "n_environments": 4}},
        relpath_config="config",
    )
    experiment2.learn("learning_series_2", "run1")
    # Start playing process:
    experiment2 = ETAx(
        root_path, "damped_oscillator_ppo", {"settings": {"n_episodes_play": 3}}, relpath_config="config"
    )
    experiment2.play("learning_series_2", "run1")


def get_path() -> pathlib.Path:
    """Get the path of this file.

    :return: Path to this file.
    """
    return pathlib.Path(__file__).parent


def plot() -> None:
    """Load results from both runs and create a plot to compare them."""
    import matplotlib as mpl
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib import cm

    data = (
        pd.concat(
            (
                pd.read_csv("results/conventional_series/run1_000_01.csv", sep=";").add_prefix("conv_"),
                pd.read_csv("results/learning_series/run1_000_01.csv", sep=";").add_prefix("rl_"),
            ),
            axis=1,
        )
        .rolling(30)
        .mean()
    )

    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["font.size"] = "9"
    linestyles = ["--", "-"]

    def greys(x: int) -> tuple[int]:
        return cm.Greys(int(255 - ((255 - 100) / 3) * x))

    fig, ax = plt.subplots(1, 1, figsize=(5, 2))
    fig.set_tight_layout(True)

    x = data.index
    columns = {
        "mass deviation conventional": "conv_s",
        "input conventional": "conv_u",
        "mass deviation DRL": "rl_s",
        "input DRL": "rl_u",
    }

    lines: list[mpl.lines.Line2D] = []
    labels: list[str] = []
    for name, col in columns.items():
        hdl = ax.plot(x, data[col], color=greys(len(lines)), linestyle=linestyles[len(lines) % len(linestyles)])[0]
        lines.append(hdl)
        labels.append(name)

    ax.legend(lines, labels, loc="upper right")
    ax.yaxis.grid(color="gray", linestyle="dashed")

    ax.set_xlabel("time")
    ax.set_ylabel("distance")

    plt.savefig("training_results.png", dpi=300)
    plt.show()


if __name__ == "__main__":
    main()
