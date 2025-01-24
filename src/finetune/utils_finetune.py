#!/usr/bin/env python

import json
import os
from datetime import timedelta

import pandas as pd
from datasets import Dataset, DatasetDict

from finetune.dataset import HknewsDataset, OpusDataset
from finetune.hf_utils import tokenize_dataset
from constants import WEEK_LENGTH
from utils import exec_bash_cmd, mkdir_if_not_exists


def load_dataset(name: str, fixed_test_set: bool = False):
    """
    Load (create) the desired MT dataset
    """

    if "hk-news" in name:
        dataset = HknewsDataset(name, fixed_test_set)

    if "opus_eng_fra" in name:
        dataset = OpusDataset(name, fixed_test_set)

    return dataset


def get_path_to_latest_finetuned_model(finetuned_models_dir: str):
    for model_file in os.listdir(finetuned_models_dir):
        if "finetuned_model" in model_file and "latest" in model_file:
            print(
                f"[D] Retrieving finetuned model path: {finetuned_models_dir + model_file}"
            )
            return finetuned_models_dir + model_file

    return None


def update_latest_finetuned_model_name(finetuned_models_dir: str):
    for model_dir in os.listdir(finetuned_models_dir):
        if "finetuned_model" in model_dir and "latest" in model_dir:
            old_model_dir = finetuned_models_dir + model_dir
            new_model_dir = finetuned_models_dir + model_dir.replace("-latest", "")
            # create new dir
            mkdir_if_not_exists(new_model_dir)
            # rename files (i.e., move them) to new dir
            for model_file in os.listdir(old_model_dir):
                os.rename(
                    src=old_model_dir + "/" + model_file,
                    dst=new_model_dir + "/" + model_file,
                )
            # delete "renamed" dir
            os.rmdir(old_model_dir)


def _get_time_based_split(
    curr_time, finetune_start_index: int, finetune_period: int, dataset: pd.DataFrame
):
    # test first interval of no fine-tune ==> train set doesn't matter
    if curr_time < dataset["timestamp"].min():
        finetune_start_time = curr_time
        finetune_end_time = dataset["timestamp"].min()
    else:
        finetune_start_time = dataset.at[finetune_start_index, "timestamp"]
        finetune_end_time = curr_time + timedelta(
            days=int(finetune_period) * WEEK_LENGTH
        )

    filtered_df = dataset.loc[dataset["timestamp"] < finetune_end_time]
    finetune_end_index = 0
    if len(filtered_df) > 0:
        finetune_end_index = filtered_df.index[-1] + 1

    test_finetune_end_time = finetune_end_time + timedelta(days=WEEK_LENGTH)

    print(f"[D] Fine-tune interval start time: {finetune_start_time}")
    print(f"[D] Fine-tune interval end time: {finetune_end_time}")
    print(f"[D] Test fine-tune start time: {finetune_end_time}")
    print(f"[D] Test fine-tune end time: {test_finetune_end_time}")

    return (
        finetune_start_time,
        finetune_end_time,
        finetune_end_index,
        test_finetune_end_time,
    )


def _get_test_set(
    dataset: Dataset,
    dataset_df,
    finetune_end_index,
    test_finetune_end_time,
    time_interval_type,
    finetune_period: int,
):
    if dataset.is_test_set_fixed():
        test_dataset = dataset.get_test_set()
        # if the test set is fixed, we only need to
        # finetune and evaluate once
        if isinstance(test_dataset, dict):
            for key, item in test_dataset.items():
                if item.num_rows == 0:
                    del test_dataset[key]
            if len(test_dataset) == 0:
                test_dataset = None
        else:
            if len(test_dataset) == 0:
                test_dataset = None
    else:
        test_df = dataset_df[
            dataset_df.index.isin(
                list(range(finetune_end_index, dataset_df.tail(1).index.item() + 1, 1))
            )
        ]
        if "time" in time_interval_type:
            test_df = test_df.loc[test_df["timestamp"] < test_finetune_end_time].drop(
                columns=dataset.get_columns_to_drop()
            )
        else:
            test_df = test_df.head(int(finetune_period)).copy()

        print(f"[D] Using {0.1 * 100}% of {len(test_df)} test samples")
        if len(test_df) > 0:
            print(
                f"[D] test-set start index: {test_df.index[0]}   test-set end index: {test_df.index[-1]}"
            )
        test_dataset = Dataset.from_pandas(df=test_df.sample(frac=0.1, random_state=42))
        test_dataset = test_dataset.remove_columns(["__index_level_0__"])

    return test_dataset, finetune_end_index


def get_new_old_data(
    curr_idx: int,
    finetune_end_idx: int,
    dataset: Dataset,
):
    dataset_df = dataset.get_raw_dataset()

    old_data = dataset_df[
        dataset_df.index.isin(list(range(dataset_df.index[0], curr_idx, 1)))
    ].copy()

    new_data = dataset_df[
        dataset_df.index.isin(list(range(curr_idx, finetune_end_idx, 1)))
    ].copy()

    return old_data, new_data


def split_dataset(
    time_tracker_dict: dict,
    finetune_period: int,
    finetune_type: str,
    time_interval_type: str,
    dataset: Dataset,
):
    dataset_df = dataset.get_raw_dataset()

    if "base" in finetune_type:  # fine-tune with all data seen until now
        finetune_start_index = dataset_df.index[0]
    else:  # incremental ==> fine-tune only with new data
        finetune_start_index = time_tracker_dict["curr_index"]

    if "sentence" in time_interval_type:  # finetune based on number of new sentences
        # time does not matter in this case
        finetune_start_time = finetune_end_time = test_finetune_end_time = None

        if time_tracker_dict["curr_index"] == -1:
            finetune_end_index = 0
        else:
            # get the next #finetune_period sentences
            filtered_df = dataset_df[
                dataset_df.index.isin(
                    list(
                        range(
                            time_tracker_dict["curr_index"], dataset_df.index[-1] + 1, 1
                        )
                    )
                )
            ].head(int(finetune_period))

            # finetune_end_index will be used by range which returns values in
            # the interval [a; b[, which excludes b. We want to include b, so
            # it is necessary to sum 1 to b
            finetune_end_index = filtered_df.index[-1] + 1

    else:  # finetune based on how much time has passed
        (
            finetune_start_time,
            finetune_end_time,
            finetune_end_index,
            test_finetune_end_time,
        ) = _get_time_based_split(
            curr_time=time_tracker_dict["curr_time"],
            finetune_start_index=finetune_start_index,
            finetune_period=finetune_period,
            dataset=dataset_df,
        )

    train_df = dataset_df[
        dataset_df.index.isin(list(range(finetune_start_index, finetune_end_index, 1)))
    ]
    if len(train_df) > 0:
        print(
            f"[D] Train data: start index = {train_df.index[0]}   end index = {train_df.index[-1]}"
        )
        assert train_df.index[-1] == (
            finetune_end_index - 1
        ), "[E] Train set end index does not match finetune end index!"
    else:
        print("[W] No training data in current split")
    train_dataset = Dataset.from_pandas(
        df=train_df.drop(columns=dataset.get_columns_to_drop())
    )
    train_dataset = train_dataset.remove_columns(["__index_level_0__"])

    test_dataset, finetune_end_index = _get_test_set(
        dataset=dataset,
        dataset_df=dataset_df,
        finetune_end_index=finetune_end_index,
        test_finetune_end_time=test_finetune_end_time,
        time_interval_type=time_interval_type,
        finetune_period=finetune_period,
    )

    return (
        {
            "finetune_start_time": finetune_start_time,
            "finetune_end_time": finetune_end_time,
            "finetune_end_index": finetune_end_index,
        },
        {"train_split": train_dataset, "test_split": test_dataset},
    )


def tokenize_data(
    train_split,
    test_split,
    tokenizer,
    parameters,
):
    tokenized_train_data = tokenize_dataset(
        train_split,
        tokenizer,
        parameters,
    )

    if isinstance(test_split, dict):
        for key, item in test_split.items():
            print(f"[D] {key}")
            test_split[key] = tokenize_dataset(
                item,
                tokenizer,
                parameters,
            )
        tokenized_test_data = test_split
    else:
        tokenized_test_data = tokenize_dataset(
            test_split,
            tokenizer,
            parameters,
        )

    tokenized_datasets = DatasetDict(
        {
            "train": tokenized_train_data,
            "test": tokenized_test_data,
        }
    )
    print(f"[D] Data split: {tokenized_datasets}")

    return tokenized_datasets


def evaluate_model(
    train_set,
    test_set,
    trainer,
    parameters,
    finetune_duration=None,
):
    if isinstance(test_set, dict):
        eval_dict = {}
        for dataset_name in test_set.keys():
            print(f"[D] {dataset_name}")
            test_dataset = test_set[dataset_name]
            curr_eval_dict = trainer.evaluate(
                test_dataset,
                max_length=parameters["max_target_length"],
                metric_key_prefix=f"{dataset_name}",
            )
            curr_eval_dict[f"{dataset_name}-num_test_examples"] = test_dataset.num_rows
            for key, val in curr_eval_dict.items():
                if key not in eval_dict:
                    eval_dict[key] = val
    else:
        eval_dict = trainer.evaluate(
            max_length=parameters["max_target_length"], metric_key_prefix=""
        )
        eval_dict["num_test_examples"] = test_set.num_rows

    if finetune_duration is None:
        eval_dict["epoch"] = -1
        eval_dict["finetune_duration"] = -1
        eval_dict["num_train_examples"] = -1
    else:
        eval_dict["finetune_duration"] = finetune_duration
        eval_dict["num_train_examples"] = train_set.num_rows

    return eval_dict


def save_finetune_results(
    results: dict,
    results_file_name: str,
    results_dir: str,
    version: int,
):
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    for res_file in os.listdir(results_dir):
        if results_file_name in res_file:
            os.rename(
                src=results_dir + res_file,
                dst=results_dir + res_file.replace("-latest", f"-v{version}"),
            )
    with open(f"{results_dir}{results_file_name}", "w", encoding="utf8") as file:
        file.write(json.dumps(results))

    old_file = results_dir + results_file_name.replace("-latest", f"-v{version}")
    if os.path.isfile(old_file):
        exec_bash_cmd(cmd=f"rm {old_file}")
