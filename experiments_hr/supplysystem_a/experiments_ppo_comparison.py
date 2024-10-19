from __future__ import annotations

import pathlib

from eta_utility import get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import CustomExtractor, LinearSchedule


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    ##################
    #                #
    # TRAINING  PURE #
    #                #
    ##################

    config_experiment_pure_learn = {
        "settings": {"n_environments": 4, "n_episodes_learn": 4006, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }

    experiment_pure = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_ppo",
        config_overwrite=config_experiment_pure_learn,
        relpath_config="config/",
    )

    # experiment_pure.learn("ppo_pure", "experiment_pure")

    ##################
    #                #
    # TRAINING  CNN  #
    #                #
    ##################

    config_experiment_cnn_learn_predictions = {
        "settings": {"n_environments": 4, "n_episodes_learn": 4006, "episode_duration": 259200, "plot_interval": 100},
        "environment_specific": {"variant": "extended_predictions"},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002),
            "device": "cuda",  # "cuda" on systems with cuda installed
            "policy_kwargs": {
                "features_extractor_class": CustomExtractor,
                "features_extractor_kwargs": {
                    "net_arch": [
                        {
                            "process": "Split1d",
                            "sizes": [None, 360],
                            "net_arch": [
                                [
                                    # nothing here, the non-prediction observations just get passed along
                                ],
                                [
                                    {"process": "Fold1d", "out_channels": 3},
                                    {
                                        "layer": "Conv1d",
                                        "out_channels": 12,
                                        "kernel_size": 3,
                                        "stride": 1,
                                        "padding": "valid",
                                    },
                                    {
                                        "layer": "Conv1d",
                                        "out_channels": 3,
                                        "kernel_size": 12,
                                        "stride": 6,
                                        "padding": "valid",
                                    },
                                    {
                                        "layer": "Conv1d",
                                        "out_channels": 2,
                                        "kernel_size": 3,
                                        "stride": 2,
                                        "padding": "valid",
                                    },
                                    {"layer": "MaxPool1d", "kernel_size": 2, "stride": 3},
                                    {"activation_func": "Tanh"},
                                    {"layer": "Flatten"},
                                ],
                            ],
                        }
                    ]
                },
                "net_arch": [500, dict(pi=[400, 300], vf=[400, 300])],
            },
        },
    }

    experiment_cnn = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_ppo",
        config_overwrite=config_experiment_cnn_learn_predictions,
        relpath_config="config/",
    )

    experiment_cnn.learn("ppo_cnn", "experiment_cnn")

    ##################
    #                #
    # TRAINING LSTM  #
    #                #
    ##################

    config_experiment_lstm_learn = {
        "settings": {"n_environments": 4, "n_episodes_learn": 4006, "episode_duration": 259200, "plot_interval": 100},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }

    experiment_lstm = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_ppo_lstm",
        config_overwrite=config_experiment_lstm_learn,
        relpath_config="config/",
    )

    experiment_lstm.learn("ppo_lstm", "experiment_lstm")


if __name__ == "__main__":
    main()
