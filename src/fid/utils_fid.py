import json
import os

import pandas as pd

from finetune.utils_finetune import load_dataset


def get_finetune_instants(dataset, time_interval: int, time_interval_type: str):
    # load dataset
    if isinstance(dataset, str):
        print(f"[D] Loading {dataset} dataset")
        raw_dataset = load_dataset(dataset).get_raw_dataset()
    elif isinstance(dataset, pd.DataFrame):
        raw_dataset = dataset

    if "time" in time_interval_type:
        min_range = raw_dataset["week_number"].min()
        max_range = raw_dataset["week_number"].max()
    else:
        min_range = raw_dataset.index[0]
        max_range = raw_dataset.index[-1]

    if min_range != 0:
        finetune_instants = [0] + list(range(min_range, max_range, time_interval))
    else:
        finetune_instants = [-1] + list(range(min_range, max_range, time_interval))
    last_finetune = max_range - time_interval
    print(f"[D] {len(finetune_instants)} finetune instants:\n\t{finetune_instants}")

    return finetune_instants, last_finetune


def load_file(path, file_name):
    try:
        with open(path + file_name, encoding="utf8") as tmp_file:
            f = json.load(tmp_file)
        df = pd.json_normalize(f, sep="-")
        max_col_len = df.apply(lambda col: len(col[0])).max()
        for col in list(df.columns):
            if len(df[col][0]) < max_col_len:
                print(
                    f"[W] column {col}: len={len(df[col][0])} < max-col-len={max_col_len}. Adding -1"
                )
                df[col][0] = df[col].to_list()[0] + [-1]
        df = df.explode(list(df.columns)).reset_index(drop=True).infer_objects()
        if "curr_week" in list(df.columns):
            df = df.astype(
                {
                    "curr_week": "int32",
                    "curr_week_start_index": "int32",
                    "curr_week_end_index": "int32",
                }
            )
        return df
    except FileNotFoundError:
        print(f"[E] Could not find file {file_name} in dir {path}")
        return -1


def load_data(
    path: str,
    params: dict,
    finetune_instants: list,
):
    print(f"[D] Trying to load data from {path}")

    data = {}
    for finetune_instant in finetune_instants:
        if finetune_instant in [0, -1]:
            key = "finetuneInstant_noRetrain"
        else:
            key = f"finetuneInstant_{finetune_instant}"

        if "hk-news" in params["dataset"]:
            key = key + (
                f"-dataset_{params['dataset']}-"
                f"timeIntervalType_{params['time_interval_type']}-"
                f"timeInterval_{params['time_interval']}-"
                f"finetuneType_{params['finetune_type']}"
            )
        else:
            key = key + (
                f"-dataset_{params['dataset']}-fixedTestSetEval-"
                f"timeIntervalType_{params['time_interval_type']}-"
                f"timeInterval_{params['time_interval']}-"
                f"finetuneType_{params['finetune_type']}"
                "-percentNewData_1.0-percentOldData_0.0"
            )
        for file_name in os.listdir(path):
            if key in file_name:
                print(f"[D] Loading file {file_name}")  # , end="\r")
                data[file_name[:-12]] = load_file(path, file_name)

    return data
