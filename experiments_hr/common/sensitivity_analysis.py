from functools import partial
from multiprocessing import Pool
from typing import Callable, Dict

import numpy as np
import pandas as pd
from fmpy import extract, read_model_description
from fmpy.fmi2 import FMU2Slave
from SALib.analyze import sobol
from SALib.sample import latin, saltelli
from SALib.sample import sobol as so
from tqdm import tqdm


def create_samples(problem: dict, method: str, N: int):
    """
    Creates a sample based on the specified method.

    Parameters:
    - problem: dict specifying the problem definition.
    - method: str indicating the sampling method to use.
    - N: int, the number of samples to generate.

    Returns:
    - samples: A matrix of size (N*(D+2) x D), where D is the number of variable parameters (as spezified in problem)
    """

    # Map method names to their corresponding functions
    method_map: Dict[str, Callable] = {
        "saltelli": lambda problem, N: saltelli.sample(problem, N, calc_second_order=False),  # prefered option!
        "sobol": lambda problem, N: so.sample(problem, N, calc_second_order=False),
        "latin": lambda problem, N: latin.sample(problem, N),
    }

    if method in method_map:
        samples = method_map[method](problem, N)
    else:
        raise ValueError(f"Sampling method '{method}' is not supported.")
    return samples


def get_system_response(
    current_sample: list,
    csv_inputs,
    output_names,
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
        current_sample: Übergebene Zufallswerte für die Varianzparameter für die aktuelle Instanz
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
    for value_name, value_ in csv_init_values.items():
        tempValue = [v for v in model_description.modelVariables if v.name == value_name]

        fmu.setReal([tempValue[0].valueReference], [value_])

    fmu.enterInitializationMode()
    fmu.exitInitializationMode()

    fmu_episode_outputs = []  # prepare list to collect step outputs from the episode
    current_csv_index = start_csv_index  # set current index to start index

    # get the indices of the selected outputs for the sensitivity analysis
    all_outputs = [v for v in model_description.modelVariables if v.causality == "output"]
    selected_outputs = [output for output in all_outputs if output.name in output_names]

    # create flipped dictiary for csv -> fmu linkage
    csv_colums_for_fmu_input = {value: key for key, value in csv_to_fmu_inputs.items()}

    # simulate one episode
    while current_time < episode_duration:
        # setting fmu variance parameters (stay constant) and inputs (change)
        for counter_of_params, fmu_param_info in enumerate(fmu_params_inputs):
            if fmu_param_info.causality == "parameter":
                # set values in fmu
                fmu.setReal([fmu_param_info.valueReference], [current_sample[counter_of_params]])
            elif fmu_param_info.causality == "input":
                current_value = csv_inputs.loc[
                    current_csv_index + int(fmu_param_info.name in fmu_actions),
                    csv_colums_for_fmu_input[fmu_param_info.name],
                ]
                current_value = csv_inputs.loc[current_csv_index, csv_colums_for_fmu_input[fmu_param_info.name]]
                # set values in fmu
                if fmu_param_info.type == "Real":
                    fmu.setReal([fmu_param_info.valueReference], [current_value])
                elif fmu_param_info.type == "Boolean":
                    fmu.setBoolean([fmu_param_info.valueReference], [int(current_value)])

        # Collecting and storing output values
        fmu_step_outputs = [fmu.getReal([v.valueReference])[0] for v in selected_outputs]

        # perform step and collect step outputs of the chosen
        fmu.doStep(currentCommunicationPoint=current_time, communicationStepSize=step_size)
        fmu_episode_outputs.append(fmu_step_outputs)

        current_time += step_size
        current_csv_index += 1

    # terminate fmu after episode simulation
    fmu.terminate()
    fmu.freeInstance()

    # summarize system response and squeeze all step values for each output respectively using the variance # TODO: Use other methods?
    return [np.var(episode_outputs) for episode_outputs in zip(*fmu_episode_outputs)]


def fmuSensitivityAnalysis(
    path_to_fmu: str,
    path_to_csv: str,
    start_csv_index: int,
    csv_inits: dict,
    variances: dict,
    fmu_actions: list,
    sampling_method: str,
    N: int,
    multiprocessingCores: int,
    output_names: list,
    step_size: int,
    episode_duration: int,
    csv_to_fmu_inputs: dict,
):
    """
    Senstivitätsanalyse an einer FMU welche Custom Inputs pro Zeitschritt übergeben bekommt

    Args:
        path_to_fmu: Name der FMU Datei oder Pfad zu dieser
        problem: Dictonary im welchem alle Varianzparameter aufgelistet sind und im welchem Intervalle diese Variieren dürfen(SALib)
        sampling_method: Methode welche zum erstellen der Sample benutzt werden soll(sobol, saltelli oder latin) Man muss beachten, dass man bestimmt Kombination von Sample und analyzer nehmen muss(Saltelli und Sobol nur mit Sobol analyze/Latin funktioniert nicht mit Sobol)
        N: Anzahl der Samples die erstellt werden sollen/Formel für Tortal-Order: N * (D+2) D=Anzahl der Varianzparameter
        ouptus: Outputs der FMU in einer Liste
        step_size: Schrittgröße der Zeitschritt während der FMU Simulation
        episode_duration: Endzeitpunkt
        multiprocessingCores: Anzahl der Cores welche fürs Multiprocessing verwendet werden sollen
        path_to_csv: Pfad oder Dateiname der CSV Datei
        inputOrder: Reihenfolger in welcher die Inputs in die FMU gegeben sollen(gleicher Name wie die Cols in der CSV)
        start_csv_index: Startindex, ab welchem die CSV ausgelesen soll

    Returns:
        keine Richtige Rückgabe, Ergebnisse werden geprintet

    Beispiel für einen Aufruf der Methode in SATestB.py
    """

    # create problem dict in the shape necessary for SALib
    problem = {"num_vars": len(variances), "names": list(variances.keys()), "bounds": list(variances.values())}

    # create necessary number of samples depending on the problem and N
    samples = create_samples(problem=problem, method=sampling_method, N=N)

    # Model_Description der FMU und die relevanten Zeilen der CSV werden ausgelesen
    model_description = read_model_description(path_to_fmu)
    csv_inputs = pd.read_csv(path_to_csv, sep=";", usecols=list(csv_to_fmu_inputs.keys()))

    # Custom Inputs/Outputs/Varianzparameter werden aus der FMU ausgelsen
    fmu_inputs = [v for v in model_description.modelVariables if v.causality == "input"]

    all_fmu_parameters = [v for v in model_description.modelVariables if v.causality == "parameter"]
    # create a dictionary for quick lookups by name
    parameters_dict = {param.name: param for param in all_fmu_parameters}
    # find parameters that are specified in problem
    fmu_variance_parameters = [parameters_dict[name] for name in problem["names"] if name in parameters_dict]

    # create list of all variance parameters and inputs for the fmu
    fmu_params_inputs = fmu_variance_parameters + fmu_inputs

    # get temporary directory of unzipped fmu
    fmu_temp_dir = extract(path_to_fmu)

    # Initialize Y with empty lists for each output name
    Y = [[] for _ in output_names]

    # get initialization values from csv and creating the desired dictionary by mapping the column names to the original keys

    # Extract the column names for usecols in pd.read_csv()
    column_names = [details["name"] for details in csv_inits.values()]

    # Reading the initialization values from the CSV
    init_values = pd.read_csv(path_to_csv, sep=";", usecols=column_names).iloc[start_csv_index]

    # Creating a reversed dictionary for mapping CSV column names back to the original keys in csv_inits
    reversed_csv_inits = {details["name"]: key for key, details in csv_inits.items()}

    # Applying the specified 'add' and 'multiply' operations to the initialization values
    csv_init_values = {
        reversed_csv_inits[col]: (init_values[col] + csv_inits[reversed_csv_inits[col]]["add"])
        * csv_inits[reversed_csv_inits[col]]["multiply"]
        for col in init_values.index
    }

    # Use multiprocessing to parallelize the get_system_response function
    with Pool(processes=multiprocessingCores) as pool:
        system_response = pool.imap(
            func=partial(
                get_system_response,
                csv_inputs=csv_inputs,
                output_names=output_names,
                fmu_params_inputs=fmu_params_inputs,
                fmu_actions=fmu_actions,
                model_description=model_description,
                fmu_temp_dir=fmu_temp_dir,
                episode_duration=episode_duration,
                step_size=step_size,
                start_csv_index=start_csv_index,
                csv_init_values=csv_init_values,
                csv_to_fmu_inputs=csv_to_fmu_inputs,
            ),
            iterable=samples,
        )

        # collect results with progress bar
        X = list(tqdm(system_response, total=len(samples)))

    # rearrange results into Y
    for output_number in range(len(output_names)):
        Y[output_number] = [x[output_number] for x in X]

    # analyze and print results for each output
    sa_all_outputs = {}
    for i, output_name in enumerate(output_names):
        sa_one_output = {}
        Si = sobol.analyze(problem=problem, Y=np.array(Y[i]), calc_second_order=False)
        for param_name, st_value in zip(problem["names"], Si["ST"]):
            sa_one_output[param_name] = st_value
        sa_all_outputs[output_name] = sa_one_output

    # TODO: Add error bar functionality!

    return sa_all_outputs
