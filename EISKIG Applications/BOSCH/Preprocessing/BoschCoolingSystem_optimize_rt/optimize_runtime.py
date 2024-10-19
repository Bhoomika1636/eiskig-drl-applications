from __future__ import annotations

import pathlib

from eta_utility import LOG_INFO, get_logger
from eta_utility.eta_x import ETAx
import time

import sys
import numpy as np
import pandas as pd
import os
from scipy.optimize import differential_evolution, OptimizeResult
# Add the directory containing controllerFunctions.py to the Python module search path
sys.path.append('experiments_hr')

def playEpisode(values = [120]):
    playEpisode.counter += 1
    simParams = {
        "coolingTowerCircuitFull.RiseTime": values[0],
        "coolingTowerCircuitFull.PID.k": values[1],
        "coolingTowerCircuitFull.PID.Ti": values[2],
        "Control_KKM_Circuit_simple.k": values[3],
        "Control_KKM_Circuit_simple.Ti":values[4],
        "fullHeatExchangerCircuit.riseTime_VR":values[5],
        "coolingWaterCircuitFull.idealPump_ETA_X_OperatingStrategy.k": values[6],
        "coolingWaterCircuitFull.idealPump_ETA_X_OperatingStrategy.Ti": values[7],
        "compressionChillerCircuit.riseTime": values[8], # for valve
        "compressionChillerCircuit.RiseTime": values[9], # KKM
        "compressionChillerCircuit.coldWaterCircuit.idealPump_ETA_X_OperatingStrategy.k": values[10],
        "compressionChillerCircuit.coldWaterCircuit.idealPump_ETA_X_OperatingStrategy.Ti": values[11],
        "compressionChillerCircuit.k_mixControl": values[12],
        "compressionChillerCircuit.Ti_mixControl": values[13]
    }
    get_logger()
    root_path = pathlib.Path(__file__).parent
    # neew to fix np. seed as well
    seed = 123
    np.random.seed(seed)
    episode_duration = 3600*3
    n_env = 1 # keep at 1 to ensure same dates are taken for both strategies 
    n_episodes = 1
    
    config_ppo_play = {
        "settings": {"episode_duration": episode_duration,"sampling_time": 120,"n_environments": n_env, "n_episodes_play": n_episodes, "plot_interval": 100,"seed": seed, "verbose": 2},
        "agent_specific": {"policy_kwargs": {"net_arch": [500, dict(pi=[400, 300], vf=[400, 300])]}},
        "environment_specific": {
            "discretize_action_space": True,
            "up_down_discrete": True,
            "use_policy_shaping": True,
            "scenario_time_begin": "2019-01-02 00:00",
            "scenario_time_end": "2019-12-30 00:00",
            "random_sampling": True,
            "sim_parameters": simParams,
            "scenario_files": [
                {
                    "path": "Factory_2019.csv",
                    "interpolation_method": "interpolate",
                    "resample_method": "asfreq",
                    "time_conversion_str": "%d.%m.%Y %H:%M",
                    "scale_factors": {
                        "q_cold": 1,
                        "q_cool": 1
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

    experiment_ppo = ETAx(
        root_path=root_path,
        config_name="BoschSystem_ppo",
        config_overwrite=config_ppo_play,
        relpath_config="config/",
    )
    t1 = time.time()
    try:
        experiment_ppo.play("compare_ppo_rb", "experiment_1")
        t2 = time.time()
    except:
        t2 = time.time() + 2000 # some high number to not count for the optimization
    print(f"Test {playEpisode.counter}, Runtime = {t2-t1} \n for input params: {simParams}")
    simParams['Runtime'] = t2-t1
    df = pd.DataFrame(simParams, index=[0])
    try:
        df_last = pd.read_csv(r"experiments_hr\BoschCoolingSystem_optimize_rt\results\log2.csv",index_col=0)
        df = pd.concat([df_last, df]).reset_index(drop=True)
    except:
        print("first call")
    df.to_csv(r"experiments_hr\BoschCoolingSystem_optimize_rt\results\log2.csv")

    return t2-t1

#callback : callable, `callback(xk, convergence=val)
def callback_ev(xk, convergence):
    x = xk
    update = convergence
    simParams = {
        "coolingTowerCircuitFull.RiseTime": xk[0],
        "coolingTowerCircuitFull.PID.k": xk[1],
        "coolingTowerCircuitFull.PID.Ti": xk[2],
        "Control_KKM_Circuit_simple.k": xk[3],
        "Control_KKM_Circuit_simple.Ti":xk[4],
        "fullHeatExchangerCircuit.riseTime_VR":xk[5],
        "coolingWaterCircuitFull.idealPump_ETA_X_OperatingStrategy.k": xk[6],
        "coolingWaterCircuitFull.idealPump_ETA_X_OperatingStrategy.Ti": xk[7],
        "compressionChillerCircuit.riseTime": xk[8], # for valve
        "compressionChillerCircuit.RiseTime": xk[9], # KKM
        "compressionChillerCircuit.coldWaterCircuit.idealPump_ETA_X_OperatingStrategy.k": xk[10],
        "compressionChillerCircuit.coldWaterCircuit.idealPump_ETA_X_OperatingStrategy.Ti": xk[11],
        "compressionChillerCircuit.k_mixControl": xk[12],
        "compressionChillerCircuit.Ti_mixControl": xk[13],
        "convergence": update
    }
    df = pd.DataFrame(simParams, index=[0])
    try:
        df_last = pd.read_csv(r"experiments_hr\BoschCoolingSystem_optimize_rt\results\log.csv",index_col=0)
        df = pd.concat([df_last, df]).reset_index(drop=True)
    except:
        print("first call")
    df.to_csv(r"experiments_hr\BoschCoolingSystem_optimize_rt\results\log.csv")
    return None
    
if __name__ == '__main__':
    file_path = r"experiments_hr\BoschCoolingSystem_optimize_rt\results\log.csv"
    if os.path.exists(file_path):
        os.remove(file_path)
    # set bounds for variable in order 
    bounds = [(60,420), (0.05, 20), (0.1,350), (0.05,20), (0.1,350), (10, 300), (0.05,20), (0.1,350), (10, 300), (10, 420), (0.05,20), (0.1,350), (0.05,20), (0.1,350)]
    # Number of total function evals: (maxiter + 1) * popsize * N, where N = len(x)
    playEpisode.counter = 0
    result = differential_evolution(playEpisode, bounds, callback=callback_ev, maxiter=1000, workers=1, popsize=15, seed = 123)
    print(result)
    print(playEpisode.counter)
    if result.success:
        callback_ev(result.x, 1)

    # simParams = {
    #     "coolingTowerCircuitFull.RiseTime": 240.06153573495595,
    #     "coolingTowerCircuitFull.PID.k": 10.000617806726256,
    #     "coolingTowerCircuitFull.PID.Ti": 57.60850195963685,
    #     "Control_KKM_Circuit_simple.k": 8.653713786867094,
    #     "Control_KKM_Circuit_simple.Ti":181.07795385145607,
    #     "fullHeatExchangerCircuit.riseTime_VR":172.89820731896546,
    #     "coolingWaterCircuitFull.idealPump_ETA_X_OperatingStrategy.k": 0.71899348267908,
    #     "coolingWaterCircuitFull.idealPump_ETA_X_OperatingStrategy.Ti": 224.792805651019,
    #     "compressionChillerCircuit.riseTime": 233.0471938530949, # for valve
    #     "compressionChillerCircuit.RiseTime": 359.9827912827821, # KKM
    #     "compressionChillerCircuit.coldWaterCircuit.idealPump_ETA_X_OperatingStrategy.k": 11.40585180780678,
    #     "compressionChillerCircuit.coldWaterCircuit.idealPump_ETA_X_OperatingStrategy.Ti": 189.4701613562949,
    #     "compressionChillerCircuit.k_mixControl": 0.0602859578999392,
    #     "compressionChillerCircuit.Ti_mixControl": 205.719919913004
    # }
# res 39.71963024139404: 
# 240.06153573495595, 1
# 10.000617806726256, 2
# 57.60850195963685, 3
# 8.653713786867094, 4
# 181.07795385145607, 5
# 172.89820731896546, 6
# 0.71899348267908,  7
# 224.792805651019,   8
# 233.0471938530949,9
# 359.9827912827821,10
# 11.40585180780678,11
# 189.4701613562949,12
# 0.0602859578999392,13
# 205.719919913004,14

# Reihenfolge parameter
# “coolingTowerCircuitFull.RiseTime”1
# “coolingTowerCircuitFull.PID.k”2
# “coolingTowerCircuitFull.PID.Ti”3
# “Control_KKM_Circuit_simple.k”4
# “Control_KKM_Circuit_simple.Ti”5
# “fullHeatExchangerCircuit.riseTime_VR”6
# “coolingWaterCircuitFull.idealPump_ETA_X_OperatingStrategy.k”7
# “coolingWaterCircuitFull.idealPump_ETA_X_OperatingStrategy.Ti”8
# “compressionChillerCircuit.riseTime” (für das Ventil)9
# “compressionChillerCircuit.RiseTime” (KKM)10
# “compressionChillerCircuit.coldWaterCircuit.idealPump_ETA_X_OperatingStrategy.k”11
# “compressionChillerCircuit.coldWaterCircuit.idealPump_ETA_X_OperatingStrategy.Ti”12
# “compressionChillerCircuit.k_mixControl”13
# “compressionChillerCircuit.Ti_mixControl”14