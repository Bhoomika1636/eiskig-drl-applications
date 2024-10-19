from __future__ import annotations

import pathlib

from eta_utility import LOG_INFO, get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule
import time
import wandb
from wandb.integration.sb3 import WandbCallback


def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    # wandb setup
    # wandb.tensorboard.patch(root_logdir=f'C:\Gitlab\experimentshr_ast\experiments_hr\Askari_RL_Test1\results\RL_Test_WT_PPO')
    wandb.login() #70bf67ca693d81899954e5c546ca449e07406334
    run = wandb.init(
        project="RL_Test_WT_PPO",
        sync_tensorboard=True
    )
    
    ##################
    #                #
    #    TRAINING    #
    #                #
    ##################

    config_experiment_1_learn = {
        "settings": {"sampling_time": 30,"n_environments": 4, "n_episodes_learn": 500, "episode_duration": 2000, "plot_interval": 50},
        "agent_specific": {
            "learning_rate": LinearSchedule(0.0002, 0.00002).value,
            "batch_size": 256,
            "n_epochs": 4,
            # "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cpu",  # "cuda" on systems with cuda installed

        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )
    ts = time.time()
    experiment_1.learn(run.project, "experiment_"+run.id, callbacks=WandbCallback())
    ts2 =time.time()

    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    config_experiment_1_play = {
        "settings": {"episode_duration": 3000,"sampling_time": 30, "n_episodes_play": 1, "plot_interval": 1},
        "environment_specific": {
            "scenario_time_begin": "2018-03-17 00:00",
            "scenario_time_end": "2018-05-30 00:00",
            "random_sampling": False,
        }
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="supplysystem_a_ppo",
        config_overwrite=config_experiment_1_play,
        relpath_config="config/",
    )

    experiment_1.play("ppo_test", "experiment_1")
    print(ts2-ts)

    # End Wandbd
    run.finish()


if __name__ == "__main__":
    main()
