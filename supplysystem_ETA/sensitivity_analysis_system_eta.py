import json
import sys
from pathlib import Path

# load local package
current_file_path = Path(__file__)
root_path = current_file_path.parent.parent
sys.path.append(str(root_path))
from common.sensitivity_analysis import fmuSensitivityAnalysis

if __name__ == "__main__":

    # set correct directory and saving path name
    directory = current_file_path.parent / "results" / "sensitivity_analysis"

    # function to store results
    def save_json(run_name, sa_result):
        file_path_json = directory / str(run_name + ".json")
        if not directory.exists():
            directory.mkdir(parents=True)
        with open(file_path_json, "w") as json_file:
            json.dump(sa_result, json_file)

    # setting variance limits for all variance parameters
    variances = {
        "HydraulischeSwitch_variance": [0.8, 1.2],
        "SV146_variance_dp": [0.8, 1.2],
        "SV146_variance_riseTime": [0.8, 1.2],
        "PU_HeatPump_OperatingStrategy_variance": [0.8, 1.0],
        "HeatPump_variance_P": [0.8, 1.2],
        "HeatPump_variance_dp": [0.8, 1.2],
        "HydraulicSwitch_variance": [0.8, 1.2],
        "SV605_variance_dp": [0.8, 1.2],
        "SV605_variance_riseTime": [0.8, 1.2],
        "SV235_variance_dp": [0.8, 1.2],
        "SV235_variance_riseTime": [0.8, 1.2],
        "SV246_variance_dp": [0.8, 1.2],  # This is creating unstability in the simulation when at exactly 1.0
        "SV246_variance_riseTime": [0.8, 1.2],
        "SV_XX_variance_dp": [0.8, 1.2],
        "SV_XX_variance_riseTime": [0.8, 1.2],
        "RV600_variance_dp": [0.8, 1.2],
        "RV600_variance_riseTime": [0.8, 1.2],
        "PWT6_variance": [0.8, 1.2],
        "PU600_OperatingStrategy_variance": [0.8, 1.2],
        "PU235_OperatingStrategy_variance": [0.8, 1.2],
        "PU_HeatPump_HNLT_OperatingStrategy_variance": [0.8, 1.0],
        "PU215_OperatingStrategy_variance": [0.8, 1.2],
        "aFA_simple_2_1_variance": [0.8, 1.2],
        "HVFA_CN_795_variance": [0.8, 1.2],
        "HVFA_CN_796_variance": [0.8, 1.2],
        "HVFA_HNLT_variance": [0.8, 1.2],
        "SV138_variance_dp": [0.8, 1.2],
        "SV138_variance_riseTime": [0.8, 1.2],
        "SV_CN_HVFA_pressuredrop_variance_dp": [0.8, 1.2],
        "SV_CN_HVFA_pressuredrop_variance_riseTime": [0.8, 1.2],
        "SV_HNLT_HVFA_AFA_Correction_variance_dp": [0.8, 1.2],
        "SV_HNLT_HVFA_AFA_Correction_variance_riseTime": [0.8, 1.2],
        "SV_HNLT_HVFA_pressuredrop_variance_dp": [0.8, 1.2],
        "SV_HNLT_HVFA_pressuredrop_variance_riseTime": [0.8, 1.2],
        "RV105_variance_dp": [0.8, 1.2],
        "RV105_variance_riseTime": [0.8, 1.2],
        "SV105_variance_dp": [0.8, 1.2],
        "SV105_variance_riseTime": [0.8, 1.2],
        "Consumer_Producer_Switch_variance_dp": [0.8, 1.2],
        "Consumer_Producer_Switch_variance_riseTime": [0.8, 1.2],
        "Consumer_Producer_Switch1_variance_dp": [0.8, 1.2],
        "Consumer_Producer_Switch1_variance_riseTime": [0.8, 1.2],
        "SV106_variance_dp": [0.8, 1.2],
        "SV106_variance_riseTime": [0.8, 1.2],
        "RV205_variance_dp": [0.8, 1.2],
        "RV205_variance_riseTime": [0.8, 1.2],
        "SV205_variance_dp": [0.8, 1.2],
        "SV205_variance_riseTime": [0.8, 1.2],
        "SV206_variance_dp": [0.8, 1.2],
        "SV206_variance_riseTime": [0.8, 1.2],
        "PWT4_variance": [0.8, 1.2],
        "PWT5_variance": [0.8, 1.2],
        "PU138_OperatingStrategy_variance": [0.8, 1.0],
        "PU105_OperatingStrategy_variance": [0.8, 1.2],
        "PU205_OperatingStrategy_variance": [0.8, 1.2],
        "eChiller_variance": [0.8, 1.2],
        "VSI_variance": [0.8, 1.2],
        "HydraulicSwitch_HNHT_variance": [0.8, 1.2],
        "SV305_variance_dp": [0.8, 1.2],
        "SV305_variance_riseTime": [0.8, 1.2],
        "SV315_variance_dp": [0.8, 1.2],
        "SV315_variance_riseTime": [0.8, 1.2],
        "SV331_variance_dp": [0.8, 1.2],
        "SV331_variance_riseTime": [0.8, 1.2],
        "SV322_variance_dp": [0.8, 1.2],
        "SV322_variance_riseTime": [0.8, 1.2],
        "SV321_variance_dp": [0.8, 1.2],
        "SV321_variance_riseTime": [0.8, 1.2],
        "SV307_variance_dp": [0.8, 1.2],
        "SV307_variance_riseTime": [0.8, 1.2],
        "SV306_variance_dp": [0.8, 1.2],
        "SV306_variance_riseTime": [0.8, 1.2],
        "RV215_variance_dp": [0.8, 1.2],
        "RV215_variance_riseTime": [0.8, 1.2],
        "RV322_variance_dp": [0.8, 1.2],
        "RV322_variance_riseTime": [0.8, 1.2],
        "RV321_variance_dp": [0.8, 1.2],
        "RV321_variance_riseTime": [0.8, 1.2],
        "PWT1_variance": [0.8, 1.2],
        "PU315_OperatingStrategy_variance": [0.8, 1.2],
        "PU307_OperatingStrategy_variance": [0.8, 1.2],
        "PU306_OperatingStrategy_variance": [0.8, 1.2],
        "PU331_OperatingStrategy_variance": [0.8, 1.0],
        "PU322_OperatingStrategy_variance": [0.8, 1.0],
        "PU321_OperatingStrategy_variance": [0.8, 1.0],
        "CondensingBoiler_variance_P": [0.8, 1.2],
        "CondensingBoiler_variance_dp": [0.8, 1.2],
        "CHP1_variance_P": [0.8, 1.2],
        "CHP1_variance_dp": [0.8, 1.2],
        "CHP2_variance_P": [0.8, 1.2],
        "CHP2_variance_dp": [0.8, 1.2],
    }

    # choosing init parameters that should be set from csv values
    csv_inits = {
        "T_start_HNHT_up": {"name": "HNHT_Buffer_fUpperTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_HNHT": {"name": "HNHT_Buffer_fMidTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_HNHT_low": {"name": "HNHT_Buffer_fLowerTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_HNLT_up": {"name": "HNLT_Buffer_fUpperTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_HNLT": {"name": "HNLT_Buffer_fMidTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_HNLT_low": {"name": "HNLT_Buffer_fLowerTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_CN_up": {"name": "CN_Buffer_fUpperTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_CN": {"name": "CN_Buffer_fMidTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_CN_low": {"name": "CN_Buffer_fLowerTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_VSI": {"name": "HNHT_VSI_fUpperTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_VSI_mid": {"name": "HNHT_VSI_fMidTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_VSI_low": {"name": "HNHT_VSI_fLowerTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_HVFA_HNLT_up": {"name": "HNLT_HVFA_fUpperTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_HVFA_HNLT": {"name": "HNLT_HVFA_fLowerTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_HVFA_CN_up": {"name": "CN_HVFA_fUpperTemperature", "add": 273.15, "multiply": 1.0},
        "T_start_HVFA_CN": {"name": "CN_HVFA_fLowerTemperature", "add": 273.15, "multiply": 1.0},
    }

    # mapping csv columns to fmu inputs
    csv_to_fmu_inputs = {
        "time_daytime": "time_daytime",
        "d_HNHT_prod_heat_demand_consumer": "HNHT_prod_heat_demand_consumer",
        "d_HNLT_prod_heat_demand_consumer": "HNLT_prod_heat_demand_consumer",
        "d_HNLT_prod_heat_demand_producer": "HNLT_prod_heat_demand_producer",
        "d_CN_prod_heat_demand_consumer": "CN_prod_heat_demand_consumer",
        "weather_T_amb": "weather_T_amb",
        "weather_T_Ground_1m": "weather_T_Ground_1m",
        "time_month": "time_month",
        "bSetStatusOn_HeatExchanger1": "bSetStatusOn_HeatExchanger1",
        "bSetStatusOn_CHP1": "bSetStatusOn_CHP1",
        "bSetStatusOn_CHP2": "bSetStatusOn_CHP2",
        "bSetStatusOn_CondensingBoiler": "bSetStatusOn_CondensingBoiler",
        "bSetStatusOn_VSIStorage": "bSetStatusOn_VSIStorage",
        "bLoading_VSISystem": "bLoading_VSISystem",
        "bSetStatusOn_HVFASystem_HNLT": "bSetStatusOn_HVFASystem_HNLT",
        "bLoading_HVFASystem_HNLT": "bLoading_HVFASystem_HNLT",
        "bSetStatusOn_eChiller": "bSetStatusOn_eChiller",
        "bSetStatusOn_HVFASystem_CN": "bSetStatusOn_HVFASystem_CN",
        "bLoading_HVFASystem_CN": "bLoading_HVFASystem_CN",
        "bSetStatusOn_OuterCapillaryTubeMats": "bSetStatusOn_OuterCapillaryTubeMats",
        "bSetStatusOn_HeatPump": "bSetStatusOn_HeatPump",
    }

    fmu_actions = [
        "bSetStatusOn_HeatExchanger1",
        "bSetStatusOn_CHP1",
        "bSetStatusOn_CHP2",
        "bSetStatusOn_CondensingBoiler",
        "bSetStatusOn_VSIStorage",
        "bLoading_VSISystem",
        "bSetStatusOn_HVFASystem_HNLT",
        "bLoading_HVFASystem_HNLT",
        "bSetStatusOn_eChiller",
        "bSetStatusOn_HVFASystem_CN",
        "bLoading_HVFASystem_CN",
        "bSetStatusOn_OuterCapillaryTubeMats",
        "bSetStatusOn_HeatPump",
    ]

    # perform tests
    SA_days = [1]
    step_size = 30

    # TODO: Continue here, there are still errors!
    for day in SA_days:
        # define run name
        run_name = "day_" + str(day)
        # perform SA
        sa_result = fmuSensitivityAnalysis(
            path_to_fmu="experiments_hr/supplysystem_ETA/environment/supplysystem_ETA_variance_dymola_solvers.fmu",
            path_to_csv="experiments_hr/supplysystem_ETA/results/Three_days_2017_ETA/3_days_000-01_episode.csv",
            start_csv_index=int((day - 1) * (86400 / step_size)),
            csv_inits=csv_inits,
            variances=variances,
            fmu_actions=fmu_actions,
            output_names=["HNHT_Buffer_fMidTemperature", "HNLT_Buffer_fMidTemperature", "CN_Buffer_fMidTemperature"],
            sampling_method="saltelli",
            N=8,
            multiprocessingCores=4,
            step_size=step_size,
            episode_duration=60 * 60 * 3,
            csv_to_fmu_inputs=csv_to_fmu_inputs,
        )
        # print and save results
        print(sa_result)
        save_json(run_name=run_name, sa_result=sa_result)
