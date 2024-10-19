from functools import partial
from time import sleep

import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
from scipy.optimize import curve_fit
from pathlib import Path


current_file_path = Path(__file__)

def fmuParameterIdentification(
    initial_guess: list,
    diff_step,
    plot: bool,
    path_to_fmu,
    path_to_csv,
    start_csv_index,
    csv_inits,
    variances,
    step_size,
    episode_duration,
    csv_to_fmu_inputs,
    fmu_actions,
    fmu_output_to_csv_target,
    PI_days: list,
):
    """
    Parameter Identification an einer FMU welche Custom Inputs pro Zeitschritt übergeben bekommt

    Args:
        fmu_filename: Name der FMU Datei oder Pfad zu dieser
        ouptus: Outputs der FMU in einer Liste
        output_index: Positon der Outputs, an welchen die SA durchgeführt werden soll
        parameters: Parameter auf welche die PI angewandt werden soll
        y: Erwartetes Ergebnis der FMU Simulation
        initial_guess: Initiale Schätzung der Parameter
        start_time: Startzeitpunkt
        stop_time: Endzeitpunkt
        step_size: Größe der Zeitschritte
        bounds: Intervall an Werten, welche die Parameter annehmen können
        pdPath: Pfad oder Dateiname der CSV Datei
        pdCols: Cols welche aus der CSV ausgelesen sollen(Reihenfolg)
        inputOrder: Reihenfolger in welcher die Inputs in die FMU gegeben sollen(gleicher Name wie die Cols in der CSV)
        csvIndex: Startindex, ab welchem die CSV ausgelesen soll
        diff_step:0.1 oder 0.01

    Returns:
        keine Richtige Rückgabe, Ergebnisse werden geprintet
    """

    # read sensitive parameters from variance info
    sensitive_parameters = list(variances.keys())

    target_csv_output = list(fmu_output_to_csv_target.values())[0]
    fmu_output = list(fmu_output_to_csv_target.keys())

    # x create list of time steps
    x = [z for z in range(0, episode_duration, step_size)]

    # read fmu and csv
    model_description = read_model_description(path_to_fmu)
    #print(list(csv_to_fmu_inputs.keys()))
    csv_inputs = pd.read_csv(path_to_csv, sep=";", usecols=list(csv_to_fmu_inputs.keys()))

    # get temporary directory of unzipped fmu
    fmu_temp_dir = extract(path_to_fmu)

    # Custom Inputs/Outputs/Varianzparameter werden aus der FMU ausgelesen
    fmu_inputs = [v for v in model_description.modelVariables if v.causality == "input"]

    all_fmu_parameters = [v for v in model_description.modelVariables if v.causality == "parameter"]
    # Create a dictionary for quick lookups by name
    parameters_dict = {param.name: param for param in all_fmu_parameters}
    # Find parameters that are specified in sensitive_parameters
    fmu_variance_parameters = [parameters_dict[name] for name in sensitive_parameters if name in parameters_dict]

    # Create list of all variance parameters and inputs for the FMU
    fmu_params_inputs = fmu_variance_parameters + fmu_inputs

    # get initialization values from csv and creating the desired dictionary by mapping the column names to the original keys

    # Extract the column names for usecols in pd.read_csv()
    column_names = [details["name"] for details in csv_inits.values()]

    # Reading the initialization values from the CSV
    init_values = pd.read_csv(path_to_csv, sep=";", usecols=column_names).iloc[start_csv_index]
    #TODO: check various column names

    # Creating a reversed dictionary for mapping CSV column names back to the original keys in csv_inits
    reversed_csv_inits = {details["name"]: key for key, details in csv_inits.items()}

    # Applying the specified 'add' and 'multiply' operations to the initialization values
    csv_init_values = {
        reversed_csv_inits[col]: (init_values[col] + csv_inits[reversed_csv_inits[col]]["add"])
        * csv_inits[reversed_csv_inits[col]]["multiply"]
        for col in init_values.index
    }

    f = partial(
        get_system_response,
        csv_inputs=csv_inputs,
        fmu_output=fmu_output,
        fmu_params_inputs=fmu_params_inputs,
        fmu_actions=fmu_actions,
        model_description=model_description,
        fmu_temp_dir=fmu_temp_dir,
        episode_duration=episode_duration,
        step_size=step_size,
        start_csv_index=start_csv_index,
        csv_init_values=csv_init_values,
        csv_to_fmu_inputs=csv_to_fmu_inputs,
    )

    # Creating the list of tuples again with the updated dictionary
    bounds = [tuple(values[i] for values in variances.values()) for i in range(len(next(iter(variances.values()))))]

    # load target values from csv
    y = pd.read_csv(path_to_csv, sep=";", usecols=[target_csv_output]).iloc[
        start_csv_index : int(start_csv_index + (episode_duration / step_size)), 0
    ]

    print("[INFO] Performing parameter identification, please wait...")
    # perform curve_fit as many times as needed
    popt, pcov = curve_fit(f=f, xdata=x, ydata=y, p0=initial_guess, bounds=bounds, diff_step=diff_step)

    # create output dict
    output_dict = {
        fmu_output[0]: {sensitive_parameters[i]: popt[i] for i in range(len(sensitive_parameters))},
        "pcov of " + str(fmu_output[0]): str(pcov),
    }

    # plot here
    if plot:
        directory = current_file_path.parent.parent / "AFA" / "results" / "parameter_identification"
        

        for day in PI_days:
            # create list of situations to compare
            compare_parameters = {
                 "base": initial_guess,
                # "minimum": [0.8] * len(popt),
                # "maximum": [1.2] * len(popt),
                "identified": popt,
            }
            # fill plot dict
            plot_dict = {}
            for i in range(len(compare_parameters)):
                # add args depending on popt length
                args = ()
                for pos in range(len(popt)):
                    args = args + (list(compare_parameters.values())[i][pos],)
                # fill plot dict
                plot_dict[list(compare_parameters.keys())[i]] = f(
                    x,
                    *args,
                    csv_inputs=csv_inputs,
                    fmu_output=fmu_output,
                    fmu_params_inputs=fmu_params_inputs,
                    model_description=model_description,
                    fmu_temp_dir=fmu_temp_dir,
                    episode_duration=episode_duration,
                    step_size=step_size,
                    start_csv_index=start_csv_index,
                    csv_init_values=csv_init_values,
                    csv_to_fmu_inputs=csv_to_fmu_inputs,
                )

            #print("plot_dict= ", plot_dict)
            plt.figure(figsize=(10, 5))  # Optional: Sets the figure size

            # Plotting both lists
            #plt.plot(x, plot_dict["maximum"], label="max", color="darkgrey", linestyle="--")
            #plt.plot(x, plot_dict["minimum"], label="min", color="lightgrey", linestyle="--")
            plt.plot(x, y, label="real", color="black")
            plt.plot(x, plot_dict["base"], label="initial guess", color="red")
            plt.plot(x, plot_dict["identified"], label="identified", color="green")

            # Adding labels and title
            plt.xlabel("Index")
            plt.ylabel("Values")
            plt.title("Parameteridentification of " + str(fmu_output[0]))
            plt.legend()  # This adds a legend using the labels provided in the plot calls

            # Ensure the output directory exists
            if not directory.exists():
                directory.mkdir(parents=True)
            
            # Save the plot
            #for i in fmu_output:
            #print("fmu_output= ", fmu_output)
            #plot_filename = os.path.join(directory, f"parameter_identification_{str(fmu_output[1])}.png")
            
            # for i in PI_days[i]:
            #     print(PI_days[i])
            #     plot_filename = os.path.join(directory, f"parameter_identification_day_{str(PI_days[i])}.png")
            #     print(plot_filename)
            #     plt.savefig(plot_filename)

            #for day in PI_days:
            plot_filename = os.path.join(directory, f"parameter_identification_day_{day}.png")
            plt.savefig(plot_filename) #TODO Querformat

            # Display the plot
            plt.show()
    
    return output_dict



def get_system_response(
    xTime,
    *args,
    csv_inputs,
    fmu_output,
    fmu_params_inputs,
    fmu_actions,
    model_description,
    fmu_temp_dir,
    episode_duration: float,
    step_size: float,
    start_csv_index: int,
    csv_init_values: dict,
    csv_to_fmu_inputs: dict,
):
    """
    FMU Simulation mit Custom Inputs und Multiprocessing

    Args:
        args: Übergebene Zufallswerte für die Varianzparameter für die aktuelle Instanz
        csv_inputs:Inputs, welche jeden Zeitschritt gesetzt werden
        outputs:Outputs der FMU
        fmu_params_inputs: Liste für die Values die übergeben werden
        model_decription: Model Beschreibung der FMU
        step_size: Schrittgröße der Zeitschritt während der FMU Simulation
        episode_duration: Endzeitpunkt
        inputOrder: Reihenfolger in welcher die csv_inputs in die FMU gegeben sollen(gleicher Name wie die Cols in der CSV)
        start_csv_index: Startindex, ab welchem die CSV ausgelesen soll

    Returns:
        iterationvalue: Liste mit den Varianzen der Outputs
    """

    # transform parameter touple from curve_fit into list
    print("[INFO] Trying parameters:", args)
    args = list(args)

    # setup fmu
    fmu = FMU2Slave(
        guid=model_description.guid,
        unzipDirectory=fmu_temp_dir,
        modelIdentifier=model_description.coSimulation.modelIdentifier,
        instanceName="fmu_instance",
    )

    fmu.instantiate()
    current_time = 0.0
    fmu.setupExperiment(startTime=current_time)

    # set initialization values (mostl likely temperatures)
    print(csv_init_values)
    # for value_name, value_ in csv_init_values.items():
    #     tempValue = [v for v in model_description.modelVariables if v.name == value_name]
    #     print(tempValue)
    #     fmu.setReal([tempValue[0].valueReference], [value_])

    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    fmu_episode_outputs = []  # prepare list to collect step outputs from the episode
    current_csv_index = start_csv_index  # set current index to start index

    # get the indices of the selected outputs for the sensitivity analysis
    #print(model_description.modelVariables)
    all_outputs = [v for v in model_description.modelVariables if v.causality == "output"]
    selected_outputs = [output for output in all_outputs if output.name in fmu_output]

    # create flipped dictiary for csv -> fmu linkage
    csv_colums_for_fmu_input = {value: key for key, value in csv_to_fmu_inputs.items()}

    # simulate one episode
    while current_time < episode_duration:
        # setting fmu variance parameters (stay constant) and inputs (change)
        for counter_of_params, fmu_param_info in enumerate(fmu_params_inputs):
            if fmu_param_info.causality == "parameter":
                # set values in fmu
                fmu.setReal([fmu_param_info.valueReference], [args[counter_of_params]])
            elif fmu_param_info.causality == "input":
                #print(fmu_param_info)
                #print(csv_inputs.axes)
                #print(current_time)
                current_value = csv_inputs.loc[
                    current_csv_index + int(fmu_param_info.name in fmu_actions),
                    csv_colums_for_fmu_input[fmu_param_info.name],
                ]
                # set values in fmu
                if fmu_param_info.type == "Real":
                    fmu.setReal([fmu_param_info.valueReference], [current_value])
                elif fmu_param_info.type == "Boolean":
                    fmu.setBoolean([fmu_param_info.valueReference], [int(current_value)])

        # Collecting and storing output values
        fmu_step_outputs = [fmu.getReal([v.valueReference])[0] for v in selected_outputs]
        fmu_episode_outputs.append(fmu_step_outputs[0])

        # perform step and collect step outputs of the chosen
        fmu.doStep(currentCommunicationPoint=current_time, communicationStepSize=step_size)

        print(current_time)
        current_time += step_size
        current_csv_index += 1

    # terminate fmu after episode simulation
    fmu.terminate()
    fmu.freeInstance()

    # return list of chosen fmu output
    return np.array(fmu_episode_outputs)
