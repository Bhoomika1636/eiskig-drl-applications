from __future__ import annotations
import glob
import os

import pathlib

from eta_utility import LOG_INFO, get_logger
from eta_utility.eta_x import ETAx
from eta_utility.eta_x.common import LinearSchedule, vectorize_environment, WarmUpSchedule, WarmupCosineAnnealingLR, WarmupPiecewiseConstantLR
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
    net_arch_fac = 1
    n_env = 16 
    worker_factor = int(n_env/8)
    total_timesteps = 6000000
    lr1 = (1-(777600/total_timesteps),net_arch_fac*0.000125*worker_factor)
    lr2 = (1-(1360800/total_timesteps),net_arch_fac*0.0000625*worker_factor)
    lr3 = (1-(2500000/total_timesteps),net_arch_fac*0.00002)
    config_experiment_1_learn = {
        "settings": {"sampling_time": 120,"n_environments": n_env, "n_episodes_learn":int(total_timesteps/2160), "episode_duration":3600*24*3, "plot_interval": 32/n_env, "save_model_every_x_episodes": 32/n_env, "seed": 123},
        "environment_specific": {"discretize_action_space": True, "up_down_discrete": True, "use_policy_shaping": True, "plot_only_env1": True},
        "agent_specific": {
            "n_steps": 256,
            "batch_size": 256,
            "learning_rate":WarmupPiecewiseConstantLR(0.000001,net_arch_fac*0.00025*worker_factor,[lr1,lr2,lr3], (278528/total_timesteps)),#WarmupCosineAnnealingLR(0.00002, 0.00025*worker_factor, 0.000001, warmup_proportion=0.06), # WarmUpSchedule(0.00001, 0.00003,final_p_reached=0.8), #LinearSchedule(0.00025, 0.000025).value,
            "n_epochs": 4,
            "clip_range": 0.2*worker_factor,
            #"policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]}, # [256, dict(pi=[128, 64], vf=[128, 64])] #This net arch is the original one, but wont work because of the latest sb3 version
            "device": "cuda",  # "cuda" on systems with cuda installed
        },
    }
# ca 100000 params vs 700000 params ca.
    experiment_1 = ETAx(
        root_path=root_path,
        config_name="BoschSystem_ppo",
        config_overwrite=config_experiment_1_learn,
        relpath_config="config/",
    )

    run_description = "see excel"
    ts = time.time()
    experiment_1.learn("ppo_red_3", "agent_13", run_description=run_description) 
    #use_eval_cb=False, n_eval_envs=1, eval_freq=200000/experiment_1.config.settings.n_environments only available for custom etax 
    ts2 =time.time()
    print(ts2-ts)

if __name__ == "__main__":
    main()
    
