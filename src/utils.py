import json
import os
import subprocess as sub

import pandas as pd
import yaml


def add_to_dict(
    dictionary: dict, key: str, data, dict_len: int = 0, verbose: bool = False
):
    """add value to dictionary"""
    if "__" in key:
        key = key.replace("__", "_")

    if key not in dictionary:
        dictionary[key] = []

        # pad dict entries that did not exist to ensure proper
        # lineup when converting dict to dataframe
        if dict_len > 0:
            dictionary[key].extend([-1] * dict_len)

    dictionary[key].append(data)

    if verbose:
        print(f"[D]\t{key}: {data}")


def mkdir_if_not_exists(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)


def exec_bash_cmd(cmd: str, log_file: str = None):
    if log_file is not None:
        with open(log_file, mode="w", encoding="utf-8") as fd:
            return sub.run(cmd, shell=True, check=True, stdout=fd, text=True)
    else:
        return sub.run(cmd, shell=True, capture_output=True, check=True)


# load experimental parameters
def load_params_from_yaml(
    exp_parameters_file: str,
    verbose: bool = False,
):
    print(f"[D] Loading parameters from:\n[D] \t{exp_parameters_file}")
    with open(f"{exp_parameters_file}", "rb") as f:
        parameters = yaml.load(f, Loader=yaml.FullLoader)
    if verbose:
        print("[D] Experiment parameters:")
        print(json.dumps(parameters, indent=4))

    return parameters


def save_results(path: str, file_name: str, results: pd.DataFrame, ext: str = None):
    """save test/retrain results to file"""
    if not os.path.exists(path):
        os.makedirs(path)

    if ext is not None and f".{ext}" not in file_name:
        file_name = f"{file_name}.{ext}"

    if ".pkl" in file_name or (ext is not None and "pkl" in ext):
        results.to_pickle(path + f"{file_name}")
    elif ".json" in file_name or (ext is not None and "json" in ext):
        results.to_json(path + f"{file_name}", index=False)
    elif ".csv" in file_name or (ext is not None and "csv" in ext):
        results.to_csv(path + f"{file_name}", index=False)
    else:
        print(f"[E] Unknown file extension format. Could not save file\n\t{file_name}")

    print(f"[D] Test results and metrics saved to {path}{file_name}")
