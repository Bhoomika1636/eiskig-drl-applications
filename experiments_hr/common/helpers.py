from eta_utility.eta_x.envs import StateVar
import pandas as pd
import os 
import glob

def purgeWasteFiles(targetfolderpath, targetsubfolders):
    for subfolder in targetsubfolders:
        folderpath = targetfolderpath + '/' + subfolder + '/'
        all_files_csv = glob.glob(folderpath +'*episode.csv')
        all_files_html = glob.glob(folderpath+ '*episode.html')
        # Loop through the list of files and delete each one but the last
        for filename in all_files_csv:
            if os.path.exists(filename):
                os.remove(filename)
        # Loop through the list of files and delete each one but the last
        for filename in all_files_html:
            if os.path.exists(filename):
                os.remove(filename)

# helperfunction to convert CSV to state_var_tuple
def getStateVars(pathToCSV, decimal=",", sep=";", includeWeather=False, includeEnergyPrice=False, includeDemand=False):
    df = pd.read_csv(
        pathToCSV,
        sep=sep,
        decimal=decimal,
        true_values=["True"],
        false_values=["False"],
    )
    stateVarList = []

    for i, row in df.iterrows():
        stateVar = StateVar.from_dict(row.dropna())
        stateVarList.append(stateVar)
    if includeWeather:
        stateVarList = addStateVarsWeather(stateVarList)
    if includeEnergyPrice:
        stateVarList = addEnergyPrices(stateVarList)
    if includeDemand:
        stateVarList = addDemand(stateVarList)
    state_var_tuple = tuple(stateVarList)
    return state_var_tuple


# helperfunction to convert CSV to state_var_tuple
def addStateVarsWeather(stateVarList):
    weatherStates = [
        StateVar(
            name="d_weather_drybulbtemperature",
            ext_id="u_weather_DryBulbTemperature",
            scenario_id="air_temperature",  # [°C]
            from_scenario=True,
            is_agent_observation=True,
            is_ext_input=True,
            low_value=-20,
            high_value=45,
        ),
        StateVar(
            name="d_weather_relativehumidity",
            ext_id="u_weather_RelativeHumidity",
            scenario_id="relative_air_humidity",  # [%]
            from_scenario=True,
            is_agent_observation=True,
            is_ext_input=True,
            low_value=0,
            high_value=100,
        ),
        StateVar(
            name="d_weather_time",
            ext_id="u_weather_Time",
            is_ext_input=True,
            low_value=0,
            high_value=31968000,
        ),
        StateVar(
            name="d_weather_globalhorizontalradiation",
            ext_id="u_weather_GlobalHorizontalRadiation",
            scenario_id="global_radiation",  # [W/m2]
            from_scenario=True,
            is_ext_input=True,
            low_value=0,
            high_value=1000,
        ),
        StateVar(
            name="d_weather_opaqueskycover",
            ext_id="u_weather_OpaqueSkyCover",
            scenario_id="clouds",  # [*100 %]
            from_scenario=True,
            is_ext_input=True,
            low_value=0,
            high_value=1,
        ),
        StateVar(
            name="d_weather_directnormalradiation",
            ext_id="u_weather_DirectNormalRadiation",
            scenario_id="direct_radiation",  # [W/m2]
            from_scenario=True,
            is_ext_input=True,
            low_value=0,
            high_value=1000,
        ),
        StateVar(
            name="d_weather_diffusehorizontalradiation",
            ext_id="u_weather_DiffuseHorizontalRadiation",
            scenario_id="diffuse_radiation",  # [W/m2]
            from_scenario=True,
            is_ext_input=True,
            low_value=0,
            high_value=1000,
        ),
        StateVar(
            name="d_weather_windspeed",
            ext_id="u_weather_WindSpeed",
            scenario_id="wind_speed",  # [m/s]
            from_scenario=True,
            is_ext_input=True,
            low_value=0,
            high_value=30,
        ),
        StateVar(
            name="d_weather_precip_depth",
            ext_id="u_weather_Precip_Depth",
            scenario_id="rainfall",  # [mm]
            from_scenario=True,
            is_ext_input=True,
            low_value=-1000,
            high_value=1000,
        ),
        StateVar(
            name="d_weather_winddirection",
            ext_id="u_weather_WindDirection",
            scenario_id="global_radiation",  # [deg]
            from_scenario=True,
            is_ext_input=True,
            low_value=0,
            high_value=360,
        ),
        StateVar(
            name="d_weather_precipitation",
            ext_id="u_weather_Precipitation",
            scenario_id="rain_indicator",  # [1 = rain, 0 = no rain]
            from_scenario=True,
            is_ext_input=True,
            low_value=0,
            high_value=1,
        ),
        StateVar(
            name="d_weather_groundtemperature",
            ext_id="u_weather_GroundTemperature",
            scenario_id="air_temperature",  # [°C]
            from_scenario=True,
            is_ext_input=True,
            low_value=-20,
            high_value=45,
        ),
    ]
    stateVarList.extend(weatherStates)
    return stateVarList


def addEnergyPrices(stateVarList):
    states = [
        StateVar(
            name="s_price_electricity",
            scenario_id="electrical_energy_price",
            from_scenario=True,
            is_agent_observation=True,
            low_value=-10,
            high_value=10,
        )  # to obtain €/kWh
    ]
    stateVarList.extend(states)
    return stateVarList


def addDemand(stateVarList):
    # disturbances
    # states = [
    #     StateVar(
    #         name="d_production_cool_power",
    #         ext_id="",
    #         scenario_id="Q_Cold[W]",
    #         from_scenario=True,
    #         is_ext_input=False,
    #         is_agent_observation=False,
    #         low_value=0.0,
    #         high_value=150000.0,
    #     ),
    #     StateVar(
    #         name="d_production_electric_power",
    #         ext_id="u_electric_power_demand_production",
    #         scenario_id="Q_Cool[W]",
    #         from_scenario=True,
    #         is_ext_input=True,
    #         is_agent_observation=False,
    #         low_value=0.0,
    #         high_value=300000.0,
    #     ),
    # ]
    # stateVarList.extend(states)
    return stateVarList


# a = getStateVars("experiments_hr\common\BoschStateVars.csv", includeWeather=True)
# for i in a:
#     print('-')
#     print(i)
#     print('-')
