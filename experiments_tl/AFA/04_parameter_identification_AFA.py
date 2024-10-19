import json
import sys
from pathlib import Path

# load local package
current_file_path = Path(__file__)
root_path = current_file_path.parent.parent
sys.path.append(str(root_path))
from common.parameter_identification import fmuParameterIdentification

# set correct directory and saving path name
directory = current_file_path.parent / "results" / "parameter_identification"

# function to store results
def save_json(run_name, sa_result):
    file_path_json = directory / str(run_name + ".json")
    if not directory.exists():
        print(f"Creating directory: {directory}")
        directory.mkdir(parents=True)
    print(f"Saving results to: {file_path_json}")
    with open(file_path_json, "w") as json_file:
        json.dump(sa_result, json_file)


# setting variance limits for all variance parameters
all_variances = {
    "conductivity_outside_layer": [0.1, 20],     # [W/(m*K)]
    "heat_capacity_outside_layer": [1100, 8000],  # [J/(kg*K)]
    "thickness_outside_layer": [0.055, 0.5],    # [m]
    "area_wall_west": [8, 3*12],           # [m²]
    "area_roof_west": [100, 200],           # [m²]
    "density_outside_layer": [2000, 8000],         # [kg/m^3]
    "ebsilon_concrete": [0.3, 0.97],        # [-]
    "V_glycol_basement": [0.12, 0.15],       # [m³]
    "V_glycol_east": [0.02, 0.04],           # [m³]
    "V_glycol_west": [0.02, 0.04],           # [m³]
}

# choose init parameters that should be set from csv values

"""
There is only one .csv file, where everything relevant is stored
One column should be used to init the starting temperature, so the simulation and real system start from the same starting point

"""
# mapping csv columns to init values
csv_inits = {
    "T_start": {"name": "t_int_pu600", "add": 0.0, "multiply": 1.0},
}
"""
aus csv Datei wird erster Startwert ausgelesen und in fmu initialisiert (auf höchster Ebene in FMU)
{"FMU-Benennung"}:{"name":csv-Bezeichnung}

#TODO: fix initialization. 
Right now, init values need to be defined in the fmu to work properly.
init doesn't overwrite the init values defined in the fmu.
"""

# mapping csv columns to fmu inputs
"""
The real world data is stored in the .csv file, which should be input to the fmu
links:csv-Spaltenname, rechts: FMU-Name Input
diese Werte werden jeden Zeitschritt von csv an FMU geschrieben
Alles was in AFA_system als from_scenario=true deklariert ist.
"""
csv_to_fmu_inputs = {
    "air_temperature":"T_amb",
    "global_radiation":"Solar_irradiation",
    "controlpump":"PU600_onoff",
    "u_rv660":"RV660_onoff",
    "u_rv640":"RV640_onoff",
    "u_rv600":"RV600_onoff",
    #"wind_direction":"Wind_direction",
    "wind_speed":"Wind_speed"
}


#not necessary:
fmu_actions = [
    
]

#FMU id verwenden

experiment_list = [
    {
        "fmu_output_to_csv_target": {"T_in_PU600_T": "t_int_pu600"},
        "PI_days": {
            #0: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete"], #[5, 1100, 2.5, 0.93]
            #1: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete"]  #[4, 1000, 3, 0.93]
            #2: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete"]  #[1, 900, 2, 0.92]
            #3: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete"]  #[1, 900, 2, 0.92]
            #4: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete", "V_glycol_basement", "V_glycol_east", "V_glycol_west"]  #[5, 1100, 2.5, 0.93, 0.135088, 0.025858, 0.024849]
            #7: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete", "V_glycol_basement", "V_glycol_east", "V_glycol_west"]  #[5, 1100, 2.5, 0.93, 0.135088, 0.025858, 0.024849]
            #8: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete", "V_glycol_basement", "V_glycol_east", "V_glycol_west"]  #[5, 1100, 2.5, 0.93, 0.135088, 0.025858, 0.024849]
            #9: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete", "V_glycol_basement", "V_glycol_east", "V_glycol_west"]  #[5, 1100, 2.5, 0.93, 0.135088, 0.025858, 0.024849]; episode_duration 10800
            #10: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete", "V_glycol_basement", "V_glycol_east", "V_glycol_west"]  #[5, 1100, 2.5, 0.93, 0.135088, 0.025858, 0.024849]; episode_duration 10800, changed AFA_system.json (adjusted start time)
            #11: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete", "V_glycol_basement", "V_glycol_east", "V_glycol_west"]  #[5, 1100, 2.5, 0.93, 0.135088, 0.025858, 0.024849]; episode_duration 7200, starting at 10:00
            #12: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete", "V_glycol_basement", "V_glycol_east", "V_glycol_west"]  #[5, 1100, 2.5, 0.93, 0.135088, 0.025858, 0.024849]; starting at 10:00
            #13: ["conductivity_outside_layer", "heat_capacity_outside_layer", "density_outside_layer", "ebsilon_concrete", "V_glycol_basement", "V_glycol_east", "V_glycol_west"]  #[5, 1100, 2.5, 0.93, 0.135088, 0.025858, 0.024849]; starting at 16:00
            #14: ["conductivity_outside_layer", "heat_capacity_outside_layer",  "density_outside_layer", "ebsilon_concrete", "V_glycol_basement", "V_glycol_east", "V_glycol_west"]  #[0.1, 1100, 2.5, 0.93, 0.135088, 0.025858, 0.024849]; starting at 16:00; reduced conductivity
            #15: ["conductivity_outside_layer", "heat_capacity_outside_layer", "thickness_outside_layer", "area_wall_west", "area_roof_west",  "density_outside_layer", "ebsilon_concrete"]  #[1, 1100, 0.2, 10, 135, 2.5, 0.93]; episode_duration 7200; starting at 16:00; varying conductivity, thickness and areas
            #15: ["heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer"]  #[1, 1100, 0.2, 10, 135, 2.5, 0.93]; episode_duration 7200; starting at 16:00; varying conductivity, thickness and areas
            #16: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer"]  #[5, 1100, 12, 147, 2500]; episode_duration 3600; starting at 16:00; adjusted the unit of density to SI (kg/m³)
            #17: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[5, 1100, 12, 147, 2500, 0.93]; episode_duration 86400; starting at 16:00
            #18: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[5, 4000, 12, 147, 2500, 0.93]; episode_duration 86400; starting at 16:00
            #19: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[5, 4000, 12, 147, 2500, 0.93]; episode_duration 172800; starting at 16:00
            #20: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[0.1, 10], [1100, 8000], [1, 200], [1, 3*147], [2000, 8000], [0.5, 0.97]; [5, 4000, 12, 147, 2500, 0.93]; episode_duration 86400; starting at 16:00
            #21: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer"]  #[0.1, 10], [1100, 8000], [8, 3*12], [100, 3*147], [2000, 8000]; [5, 4000, 12, 147, 2500]; episode_duration 86400; starting at 16:00
            #22: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer"]  #[0.1, 20], [1100, 8000], [8, 3*12], [100, 200], [2000, 8000]; [5, 4000, 12, 147, 2500, 0.055]; episode_duration 86400; starting at 16:00
            #23: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer"]  #[0.1, 20], [1100, 8000], [8, 3*12], [100, 200], [2000, 8000]; [5, 4000, 12, 147, 2500, 0.055]; episode_duration 108000; starting at 10:00
            #24: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[0.1, 20], [1100, 8000], [8, 3*12], [100, 200], [2000, 8000], [0.8, 0.97]; [5, 4000, 12, 147, 2500, 0.94]; episode_duration 172800; starting at 10:00
            #25: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[0.1, 20], [1100, 8000], [8, 3*12], [100, 200], [2000, 8000], [0.5, 0.97]; [5, 1100, 12, 147, 2500, 0.94]; episode_duration 172800; starting at 10:00
            #26: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[0.1, 20], [1100, 8000], [8, 3*12], [100, 200], [2000, 8000], [0.5, 0.97]; [5, 1100, 12, 147, 2500, 0.94]; episode_duration 120000; starting at 10:00
            #27: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[0.1, 20], [1100, 8000], [8, 3*12], [100, 200], [2000, 8000], [0.5, 0.97]; [5, 1100, 12, 147, 2500, 0.94]; episode_duration 172800; starting at 10:00; interpolated solar data
            #28: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[0.1, 20], [1100, 8000], [8, 3*12], [100, 200], [2000, 8000], [0.3, 0.97]; [5, 1100, 12, 147, 2500, 0.94]; new data set (12.08.); episode_duration 86400; starting at 00:00; interpolated solar data
            29: ["conductivity_outside_layer", "heat_capacity_outside_layer", "area_wall_west", "area_roof_west", "density_outside_layer", "ebsilon_concrete"]  #[0.1, 20], [1100, 8000], [8, 3*12], [100, 200], [2000, 8000], [0.3, 0.97]; [5, 1100, 12, 147, 2500, 0.94]; episode_duration 172800; starting at 10:00; interpolated solar data; changed solar irradition in Dymola
        },        
    },

    """
    right now, it is necessary to create a new day for every experiment if you don't
    want to overwrite the plot files. A different solution could be implemented later.
    the code lines for creating and storing the plots can be found here:
        experiments_hr\common\parameter_identification.py line 174 ff.
    """
    #TODO

#     {
#         "fmu_output_to_csv_target": {"T_ext_PU600_T": "t_ext_pu600"},
#         "PI_days": {

            # room for more experiments

#             },
#     },
 ]



for experiment in experiment_list:
    fmu_output_to_csv_target = experiment["fmu_output_to_csv_target"]
    PI_days = experiment["PI_days"]
    
   
    step_size = 600

    for day in list(PI_days.keys()):

        # choose most sensitive parameters and extract subset dict+                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                               
        selected_keys = PI_days[day]
        selected_variances = {key: all_variances[key] for key in selected_keys}
        print("[INFO] Most sensitive parameters:", selected_keys)
        print("day: ", day)
        
        # define run name
        run_name = "day_" + str(day) + "_" + str(next(iter(fmu_output_to_csv_target.values()))) 
        
        # perform PI
            
        pi_result = fmuParameterIdentification(
            initial_guess=[5, 1100, 12, 147, 2500, 0.94],
            diff_step=0.1, # how much the PI varies the parameters in each episode
            plot=True,
            path_to_fmu="experiments_hr/AFA/environment/AFA_erweitert_new_solar_1208_0900.fmu", # _wind_dir implicates that the wind direction is considered
            path_to_csv="experiments_hr/AFA/data/Aufzeichnung_1208_0900 (interpolated solar).csv", 
            # start_csv_index=int((day - 1) * (86400 / step_size)), # can be always 1
            start_csv_index=0, # csv headlines are not regarded as a line
            csv_inits=csv_inits,
            variances=selected_variances,
            step_size=step_size, 
            episode_duration=1200, #Dymola sim time for one episode
            csv_to_fmu_inputs=csv_to_fmu_inputs,
            fmu_actions=fmu_actions,
            fmu_output_to_csv_target=fmu_output_to_csv_target,
            PI_days=list(PI_days.keys())
        )

        #fmuParameterIdentification.saveplot(PI_days=list(PI_days.keys()))

        # print and save results
        #print(pi_result)
        print("[INFO] PI result:", pi_result)
        save_json(run_name=run_name, sa_result=pi_result)
