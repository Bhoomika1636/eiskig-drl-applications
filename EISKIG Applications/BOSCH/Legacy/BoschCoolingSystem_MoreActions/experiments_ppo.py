from __future__ import annotations

import pathlib

from eta_utility import LOG_INFO, get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule, vectorize_environment, WarmUpSchedule, WarmupCosineAnnealingLR
import time
import sys
# Add the directory containing controllerFunctions.py to the Python module search path
sys.path.append('experiments_hr')

def main() -> None:

    get_logger()
    root_path = pathlib.Path(__file__).parent

    ##################
    #                #
    #    TRAINING    #
    #                #
    ##################
    n_env = 16 
    worker_factor = int(n_env/8)
    config_experiment_1_learn = {
        "settings": {"sampling_time": 120,"n_environments": n_env, "n_episodes_learn":800, "episode_duration":3600*24*3, "plot_interval": 32/n_env, "save_model_every_x_episodes": 32/n_env},
        "environment_specific": {"discretize_action_space": True, "up_down_discrete": True, "use_policy_shaping": True, "plot_only_env1": True},
        "agent_specific": {
            "n_steps": 256,
            "batch_size": 256,
            "learning_rate": WarmupCosineAnnealingLR(0.00002, 0.00025*worker_factor, 0.00002, warmup_proportion=0.1), # WarmUpSchedule(0.00001, 0.00003,final_p_reached=0.8), #LinearSchedule(0.00025, 0.000025).value,
            "n_epochs": 4,
            "clip_range": 0.2*worker_factor,
            "policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]},
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="BoschSystem_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )

    run_description = "Same as agent 3 but CER*1.3 instead of *2 and more temperature importance(3->5, k_lin=0.1->1, T+/-0.5 instead of 1) -- also, no punishment for pump policy shaping, much higher reward shaping punishment5->150, includes T vorlauf main "
    ts = time.time()
    experiment_1.learn("ppo_test_newEnv_5", "agent_5_2", run_description=run_description, use_eval_cb=False, n_eval_envs=1, eval_freq=200000/experiment_1.config.settings.n_environments)
    ts2 =time.time()
    print(ts2-ts)
    ##################
    #                #
    #   EXECUTING    #
    #                #
    ##################

    config_experiment_1_play = {
        "settings": {"episode_duration": 3600*24*3,"sampling_time": 120,"n_environments": 4, "n_episodes_play": 1, "plot_interval": 1},
        "agent_specific": {"policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]}},
        "environment_specific": {
            "discretize_action_space": True,
            "up_down_discrete": True,
            "scenario_time_begin": "2019-01-02 00:00",
            "scenario_time_end": "2019-12-30 00:00",
            "random_sampling": True,
            "scenario_files": [
                {
                    "path": "Factory_2019.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%d.%m.%Y %H:%M",
                    "scale_factors": {
                        "q_cold": 2,
                        "q_cool": 2
                    },
                },
                {
                    "path": "EnergyMarkets_2019.csv",
                    "interpolation_method": "ffill",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%d.%m.%Y %H:%M",
                    "scale_factors": {"electrical_energy_price": 0.001, "gas_price": 0.001},
                }, 
                {
                    "path": "Weather_Wurzburg_2019.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%d.%m.%Y %H:%M",
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
                }
            ]
        }
    }

    experiment_1 = ETAx(
        root_path=root_path,
        config_name="BoschSystem_ppo",
        config_overwrite=config_experiment_1_play,
        relpath_config="config/",
    )
    ts = time.time()
    experiment_1.play("ppo_test_newEnv_5", "agent_5_2")
    ts2 =time.time()
    print(ts2-ts)

if __name__ == "__main__":
    main()
