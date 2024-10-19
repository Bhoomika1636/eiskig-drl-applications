import json
import random
import sys
from pathlib import Path

# load local package
current_file_path = Path(__file__)
root_path = current_file_path.parent.parent
sys.path.append(str(root_path))
from common.PI_evaluation import PIEvaluation

# set correct directory and saving path name
directory = current_file_path.parent / "results" / "parameter_evaluation"

# function to store results
def save_json(run_name, sa_result):
    file_path_json = directory / str(run_name + ".json")
    if not directory.exists():
        directory.mkdir(parents=True)
    with open(file_path_json, "w") as json_file:
        json.dump(sa_result, json_file)


# setting variance limits for all variance parameters
all_variances = {
    "watertank_warm_variance": [0.8, 1.2],
    "watertank_cold_variance": [0.8, 1.2],
    "val_after_HeatPump_variance": [0.8, 1.2],
    "val_before_CoolingTower_variance": [0.8, 1.2],
    "heatExchanger_counterflow_variance": [0.8, 1.2],
    "idealPump2_variance": [0.8, 1.2],
    "idealPump3_variance": [0.8, 1.2],
    "idealPump4_variance": [0.8, 1.2],
    "idealPump5_variance": [0.8, 1.2],
    "idealPump1_variance": [0.8, 1.2],
    "idealPump_variance": [0.8, 1.2],
    "coolingTower_open_variance": [0.8, 1.2],
    "chp_variance": [0.8, 1.2],
    "condensingBoiler_variance": [0.8, 1.2],
    "immersionHeater_variance": [0.8, 1.2],
    "compressionChiller_Tanks_variance": [0.8, 1.2],
    "compressionChiller_CT_variance": [0.8, 1.2],
}

# choose init üarameters that should be set from csv values
csv_inits = {
    "T_start_warmwater": {"name": "s_temp_heat_storage_hi", "add": 0.0, "multiply": 1.0},
    "T_start_warmwater_mid": {"name": "s_temp_heat_storage_mid", "add": 0.0, "multiply": 1.0},
    "T_start_warmwater_low": {"name": "s_temp_heat_storage_lo", "add": 0.0, "multiply": 1.0},
    "T_start_coldwater_up": {"name": "s_temp_cold_storage_hi", "add": 0.0, "multiply": 1.0},
    "T_start_coldwater_mid": {"name": "s_temp_cold_storage_mid", "add": 0.0, "multiply": 1.0},
    "T_start_coldwater": {"name": "s_temp_cold_storage_lo", "add": 0.0, "multiply": 1.0},
}

# mapping csv columns to fmu inputs
csv_to_fmu_inputs = {
    "u_combinedheatpower": "u_cHP",
    "u_condensingboiler": "u_CondensingBoiler",
    "u_immersionheater": "u_ImmersionHeater",
    "d_production_electric_power": "u_electric_power_demand_production",
    "u_heatpump": "u_CompressionChiller",
    "u_coolingtower": "u_CoolingTower",
    "u_compressionchiller": "u_CompressionChiller_CT",
    "d_production_heat_power": "u_heat_power_demand_production",
    "d_production_cool_power": "u_heat_power_from_production",
    "d_weather_drybulbtemperature": "u_weather_DryBulbTemperature",
    "d_weather_time": "u_weather_Time",
    "d_weather_globalhorizontalradiation": "u_weather_GlobalHorizontalRadiation",
    "d_weather_opaqueskycover": "u_weather_OpaqueSkyCover",
    "d_weather_directnormalradiation": "u_weather_DirectNormalRadiation",
    "d_weather_diffusehorizontalradiation": "u_weather_DiffuseHorizontalRadiation",
    "d_weather_windspeed": "u_weather_WindSpeed",
    "d_weather_precip_depth": "u_weather_Precip_Depth",
    "d_weather_winddirection": "u_weather_WindDirection",
    "d_weather_precipitation": "u_weather_Precipitation",
    "d_weather_groundtemperature": "u_weather_GroundTemperature",
    "d_weather_relativehumidity": "u_weather_RelativeHumidity",
    "d_production_gas_power": "u_gas_power_demand_production",
}

fmu_actions = [
    "u_cHP",
    "u_CondensingBoiler",
    "u_ImmersionHeater",
    "u_electric_power_demand_production",
    "u_CompressionChiller",
    "u_CoolingTower",
    "u_CompressionChiller_CT",
]

# enter global PI result here
global_PI_result = {
    "chp_variance": 1.130931219809690,
    "compressionChiller_CT_variance": 1.169152493935210,
    "idealPump1_variance": 0.979281674723515,
    "coolingTower_open_variance": 1.168919400640050,
}


selected_keys = list(global_PI_result.keys())
identified_parameters = list(global_PI_result.values())


random.seed(17)
sample_size = 100
lower_bound = 1
upper_bound = 350

step_size = 180
episode_length = 86400

output_list = [
    {"Temp_storage_heat_upper": "s_temp_heat_storage_hi"},
    {"Temp_storage_cold_lower": "s_temp_cold_storage_lo"},
]


# Create a list of all possible numbers in the given range
all_numbers = list(range(lower_bound, upper_bound + 1))
# Shuffle the list to randomize the order of numbers
random.shuffle(all_numbers)
# Select the first 'sample_size' numbers after shuffling
PI_days = all_numbers[:sample_size]


eval_dict = {}
for item in output_list:  # Iterate through each dictionary in the list
    for key in item:  # Iterate through each key in the dictionary
        eval_dict[key] = {}  # Use the key to set a new key in eval_dict


for output_number in range(len(output_list)):
    fmu_output_to_csv_target = output_list[output_number]
    for day in PI_days:

        # choose most sensitive parameters and extract subset dict
        selected_variances = {key: all_variances[key] for key in selected_keys}
        print("[INFO] Most sensitive parameters:", selected_keys)

        # define run name
        run_name = "PIEvaluation"
        # perform PI
        pi_eval_result = PIEvaluation(
            path_to_fmu="experiments_hr/supplysystem_b/environment/supplysystem_b_variance.fmu",
            path_to_csv="experiments_hr/supplysystem_b/results/2017_360_days_P2/2017_360_days_P2_000-01_episode.csv",
            start_csv_index=int((day - 1) * (86400 / step_size)),
            csv_inits=csv_inits,
            variances=selected_variances,
            parameter_values=identified_parameters,
            step_size=step_size,
            episode_duration=episode_length,
            csv_to_fmu_inputs=csv_to_fmu_inputs,
            fmu_actions=fmu_actions,
            fmu_output_to_csv_target=fmu_output_to_csv_target,
            plotting=False,
        )

        eval_dict[list(fmu_output_to_csv_target.keys())[0]][day] = pi_eval_result
        # # print and save results
        print(pi_eval_result)

print(eval_dict)
save_json(run_name=run_name, sa_result=eval_dict)
