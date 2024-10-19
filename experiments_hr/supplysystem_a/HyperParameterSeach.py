import optuna
import gymnasium as gym
import numpy as np

import pathlib
from typing import Any, Dict
from stable_baselines3 import PPO
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
# from eta_utility.eta_x import ETAx
from common.Custom_ETAx import ETAx2
from eta_utility.eta_x.common import LinearSchedule
from optuna.visualization import plot_optimization_history, plot_param_importances, plot_rank
# from stable_baselines3.common.cmd_util import make_vec_env

# https://colab.research.google.com/github/araffin/rl-tutorial-jnrr19/blob/master/5_custom_gym_env.ipynb
# from custom_env import GoLeftEnv

def sample_all_ppo_params(trial):
    """ Learning hyperparamters we want to optimise"""

    n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32, 64, 128, 256, 512])
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate_start = trial.suggest_float("learning_rate_start", 0.0001, 0.001, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
    max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    vf_coef = trial.suggest_categorical("vf_coef", [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]) #trial.suggest_float("vf_coef", 0, 1)

    # TODO: account when using multiple envs
    # if batch_size > n_steps:
        # batch_size = n_steps

    net_arch = {
        "tiny": [64, 32, 32],
        "small": [128, 64, 64],
        "medium": [500, 400, 300],
    }[net_arch_type]

    # Display true values
    # trial.set_user_attr("n_steps", n_steps)
    # trial.set_user_attr("learning_rate", learning_rate)
    # trial.set_user_attr("net_arch", net_arch)

    n_env_sampled = trial.suggest_categorical("n_env", [4, 8, 16])

    return {
        "n_steps": n_steps,
        "batch_size":batch_size,
        "gamma": gamma,
        "learning_rate": LinearSchedule(learning_rate_start, 0.00002).value,
        "ent_coef":ent_coef,
        "clip_range":clip_range,
        "n_epochs":n_epochs,
        "gae_lambda":gae_lambda,
        "max_grad_norm":max_grad_norm,
        "vf_coef":vf_coef,
        "device": "cuda",
        "policy_kwargs" : {
            "net_arch": net_arch
            }
        
    }, n_env_sampled

def sample_td3_params(trial):#: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict) -> Dict[str, Any]:
    """
    Sampler for TD3 hyperparams.

    :param trial:
    :return:
    """
    gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64, 100, 128, 256, 512, 1024, 2048])
    buffer_size = trial.suggest_categorical("buffer_size", [int(1e4), int(1e5), int(1e6)])
    # Polyak coeff
    tau = trial.suggest_categorical("tau", [0.001, 0.005, 0.01, 0.02, 0.05, 0.08])

    train_freq = trial.suggest_categorical("train_freq", [1, 4, 8, 16, 32, 64, 128, 256, 512])
    gradient_steps = train_freq

    noise_type = trial.suggest_categorical("noise_type", ["ornstein-uhlenbeck", "normal", None])
    noise_std = trial.suggest_float("noise_std", 0, 1)

    # NOTE: Add "verybig" to net_arch when tuning HER
    net_arch_type = trial.suggest_categorical("net_arch", ["small", "medium", "big"])
    # activation_fn = trial.suggest_categorical('activation_fn', [nn.Tanh, nn.ReLU, nn.ELU, nn.LeakyReLU])

    net_arch = {
        "small": [64, 64],
        "medium": [256, 256],
        "big": [400, 300],
        # Uncomment for tuning HER
        # "verybig": [256, 256, 256],
    }[net_arch_type]

    hyperparams = {
        "gamma": gamma,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "buffer_size": buffer_size,
        "train_freq": train_freq,
        "gradient_steps": gradient_steps,
        "policy_kwargs": dict(net_arch=net_arch),
        "tau": tau,
    }
    n_actions = 3

    if noise_type == "normal":
        hyperparams["action_noise"] = NormalActionNoise(mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions))
    elif noise_type == "ornstein-uhlenbeck":
        hyperparams["action_noise"] = OrnsteinUhlenbeckActionNoise(
            mean=np.zeros(n_actions), sigma=noise_std * np.ones(n_actions)
        )

    # if additional_args["using_her_replay_buffer"]:
    #     hyperparams = sample_her_params(trial, hyperparams, additional_args["her_kwargs"])

    return hyperparams

def sample_her_params(trial: optuna.Trial, hyperparams: Dict[str, Any], her_kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """
    Sampler for HerReplayBuffer hyperparams.

    :param trial:
    :parma hyperparams:
    :return:
    """
    her_kwargs = her_kwargs.copy()
    her_kwargs["n_sampled_goal"] = trial.suggest_int("n_sampled_goal", 1, 5)
    her_kwargs["goal_selection_strategy"] = trial.suggest_categorical(
        "goal_selection_strategy", ["final", "episode", "future"]
    )
    hyperparams["replay_buffer_kwargs"] = her_kwargs
    return hyperparams

def sample_some_ppo_params(trial):
    """ Learning hyperparamters we want to optimise"""

    n_steps = trial.suggest_categorical("n_steps", [32, 64, 128, 256, 512, 1024, 2048])
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256, 512])
    net_arch_type = trial.suggest_categorical("net_arch", ["tiny", "small", "medium"])
    # gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
    learning_rate_start = trial.suggest_categorical("learning_rate_start", [0.0001, 0.0002, 0.0003, 0.0004])
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_categorical("clip_range", [0.1, 0.2, 0.3, 0.4])
    n_epochs = trial.suggest_categorical("n_epochs", [1, 5, 10, 20])
    gae_lambda = trial.suggest_categorical("gae_lambda", [0.9, 0.95, 0.98, 0.99, 1.0])
    # max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
    # vf_coef = trial.suggest_float("vf_coef", 0, 1)

    net_arch = {
        "tiny": [64, 32, 32],
        "small": [128, 64, 64],
        "medium": [500, 400, 300],
    }[net_arch_type]

    n_env_sampled = trial.suggest_categorical("n_env", [4, 8, 16, 32])

    return {
        "n_steps": n_steps,
        "batch_size":batch_size,
        # "gamma": gamma,
        "learning_rate": LinearSchedule(learning_rate_start, 0.00002).value,
        "ent_coef":ent_coef,
        "clip_range":clip_range,
        "n_epochs":n_epochs,
        "gae_lambda":gae_lambda,
        # "max_grad_norm":max_grad_norm,
        # "vf_coef":vf_coef,
        "device": "cuda",
        "policy_kwargs" : {
            "net_arch": net_arch
            }
        
    }, n_env_sampled


def optimize_ppo_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """

    Test_number = 6

    model_params, n_env_sampled = sample_some_ppo_params(trial)
    root_path = pathlib.Path(__file__).parent

    series_name = f"Hyperparameter_Search_PPO_Test_{Test_number}_Trail_{trial.number}"
    run_name = "experiment_1"

    config_model = {
        # "settings": {"n_environments": n_env_sampled, "n_episodes_learn": 2, "n_episodes_play": 1, "episode_duration": 259200/3, "plot_interval": 100},
        "settings": {"norm_wrapper_reward": True, "n_environments": n_env_sampled, "n_episodes_learn": 3000, "n_episodes_play": 10, "episode_duration": 259200, "plot_interval": 1000},
        "agent_specific": model_params,
        "environment_specific":{
            "scenario_time_begin": "2017-01-01 00:00",
            "scenario_time_end": "2017-12-27 00:00",
            "poweron_cost_combinedheatpower": 0.4,
            "poweron_cost_condensingboiler": 0.2,
            "poweron_cost_immersionheater": 0.1,
            "variant":"extended_observations"}}

    model = ETAx2(
        root_path=root_path,
        config_name="supplysystem_a_ppo_HyperparameterOpt",
        config_overwrite=config_model,
        relpath_config="config/")
    model.learn(series_name, run_name)

    #overwrite the existing config with test dataset values
    config_model["settings"]["norm_wrapper_reward"] = False
    config_model["settings"]["plot_interval"] = 1
    config_model["settings"]["n_environments"] = 1
    config_model["environment_specific"]["scenario_time_begin"] = "2018-01-01 00:00"
    config_model["environment_specific"]["scenario_time_end"] = "2018-12-27 00:00"

    model = ETAx2(
        root_path=root_path,
        config_name="supplysystem_a_ppo_HyperparameterOpt",
        config_overwrite=config_model,
        relpath_config="config/")

    mean_reward = model.play(series_name, run_name)

    return mean_reward

def optimize_td3_agent(trial):
    """ Train the model and optimize
        Optuna maximises the negative log likelihood, so we
        need to negate the reward here
    """

    Test_number = 1

    model_params = sample_td3_params(trial)
    n_env_sampled = 4
    root_path = pathlib.Path(__file__).parent

    series_name = f"Hyperparameter_Search_TD3_Test_{Test_number}_Trail_{trial.number}"
    run_name = "experiment_1"

    config_model = {
        "settings": {"n_environments": n_env_sampled, "n_episodes_learn": 2, "n_episodes_play": 1, "episode_duration": 259200/3, "plot_interval": 100},
        "agent_specific": model_params}

    model = ETAx2(
        root_path=root_path,
        config_name="supplysystem_a_td3",
        config_overwrite=config_model,
        relpath_config="config/")
    
    model.learn(series_name, run_name)
    mean_reward = model.play(series_name, run_name)

    return mean_reward

if __name__ == '__main__':
    study = optuna.create_study(sampler=TPESampler(), direction="maximize")  # ToDo: change direction="maximize" and delete -1* for reward; no pruner needed cause early stopping not possible yet
    try:
        study.optimize(optimize_ppo_agent, n_trials=15, n_jobs=2)
        # study.optimize(optimize_td3_agent, n_trials=2, n_jobs=2)
        # study.optimize(sample_td3_params(trial: optuna.Trial, n_actions: int, n_envs: int, additional_args: dict), n_trials=2, n_jobs=2)
    except KeyboardInterrupt:
        print('Interrupted by keyboard.')

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trial = study.best_trial
    print(f"  Value: {trial.value}")
    print("  Params: ")
    for key, value in trial.params.items():
        print(f"    {key}: {value}")
    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print(f"    {key}: {value}")

    # Write report
    study.trials_dataframe().to_csv("test_study_results.csv")
    fig1 = plot_optimization_history(study)
    fig2 = plot_param_importances(study)
    fig3 = plot_rank(study)

    fig1.show()
    fig2.show()
    fig3.show()