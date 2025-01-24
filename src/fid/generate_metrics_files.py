"""
To generate the Finetune Impact Dataset (FID), it is necessary to:

1. finetune models at each "time" interval and evaluate the finetuned
    model on all consecutive "time" intervals, saving all metrics,
    including the sets of "old" and "new" data
    (Note: "time" can be defined either as:
    - weeks ==> finetune_periods = 1 implies finetune each week
    - sentences ==> finetune_periods = 1000 implies finetune after 1000
    new sentences have been received)

2. iterate over every possible combination of finetune intervals and
    compute the desired features (e.g. ratio of sentence overlap)

This script implements step 1.

To deploy these experiments in batch in the Pittsburgh SuperComputer Center
use script deploy_psc_exp.py
"""

import argparse
import json
import time
from datetime import timedelta

import pandas as pd
import torch
from huggingface_hub import login, logout
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from utils_fid import get_finetune_instants

from finetune.dataset import MyDataset
from finetune.finetune import new_finetune_round
from finetune.utils_finetune import load_dataset, save_finetune_results
from metrics.metrics import load_metrics
from constants import (
    FINETUNE_EXP_PARAMS_FILE,
    FINETUNED_MODELS_DIR,
    FID_DIR,
    WEEK_LENGTH,
)
from utils import add_to_dict, load_params_from_yaml, mkdir_if_not_exists


def init_time_tracker_dict(
    settings: dict, finetune_instant: int, raw_dataset: pd.DataFrame
):
    if settings["exp_parameters"]["time_interval_type"] == "time":
        time_tracker_key = "curr_week"

        weeks_passed = int(finetune_instant) - raw_dataset["week_number"].min()
        curr_time = raw_dataset["timestamp"].min() + timedelta(
            days=int(weeks_passed) * WEEK_LENGTH
        )
        start_time = raw_dataset.loc[raw_dataset["timestamp"] >= curr_time]
        end_time = raw_dataset["week_number"].max()
        curr_index = start_time.head(1).index.item()
        curr_week = start_time.week_number.min()

        time_tracker_dict = {
            "curr_time": curr_time,
            "curr_index": curr_index,
            "curr_week": curr_week,
            "end_time": end_time,
        }

        if finetune_instant == 0:
            time_tracker_dict["curr_week"] = 0

    else:  # fine-tune based on number of sentences
        time_tracker_key = "curr_index"
        if finetune_instant == -1:
            curr_index = -1
        elif finetune_instant == 0:
            # there are no sentences with which to fine-tune
            # get index of the first element
            curr_index = raw_dataset.index[0]
        else:
            # there is "finetune_instant * time_interval sentences"
            # with which to fine-tune
            # get index of sentence "finetune_instant * time_interval sentences"
            num_sents = int(finetune_instant)
            curr_index = num_sents

        time_tracker_dict = {
            "curr_index": curr_index,
            "end_time": raw_dataset.index[-1],
        }

    print(f"[D] start time = {time_tracker_dict[time_tracker_key]}")

    return time_tracker_dict, time_tracker_key


def finetune_and_evaluate(settings: dict, finetune_instant: int, dataset: MyDataset):
    # keep track of how long the entire run takes
    settings["run_start_time"] = str(int(time.time()))

    # initially we want to have finetune disabled:
    #   - it is enabled when it is time to finetune
    #   - it is disabled after the first time it executes
    settings["exp_parameters"]["finetune_on"] = False

    # load model and tokenizer
    print("[D] Loading pre-trained huggingface model")
    model = AutoModelForSeq2SeqLM.from_pretrained(settings["model_checkpoint"])
    model = model.to(settings["device"])

    # create dir to save finetuned models to
    mkdir_if_not_exists(FINETUNED_MODELS_DIR)

    time_tracker_dict, time_tracker_key = init_time_tracker_dict(
        settings=settings,
        finetune_instant=finetune_instant,
        raw_dataset=dataset.get_raw_dataset(),
    )

    results_dict = {}
    while time_tracker_dict[time_tracker_key] < time_tracker_dict["end_time"]:
        print(
            "#" * 50
            + f"  INTERVAL  finetune instant = {finetune_instant} --- Time {time_tracker_dict}  "
            + "#" * 50
        )

        # only fine-tune if:
        #   - the current time is that of a finetune instant
        #   - no fine-tune has been performed yet
        #   -
        if (time_tracker_dict[time_tracker_key] == finetune_instant) and (
            "exp_parameters_finetune_on" not in results_dict
            or False in results_dict["exp_parameters_finetune_on"]
        ):
            settings["exp_parameters"]["finetune_on"] = True
            print("[D] Going to FINE-TUNE!")

        (
            time_tracker_dict["curr_time"],
            time_tracker_dict["curr_index"],
            finetune_res,
            finetuned_model_name,
        ) = new_finetune_round(
            dataset=dataset,
            time_tracker_dict=time_tracker_dict,
            finetune_period=int(settings["exp_parameters"]["time_interval"]),
            model=model,
            settings=settings,
        )

        # update base model -- it is now finetuned
        if settings["exp_parameters"]["finetune_on"]:
            settings["exp_parameters"]["finetune_on"] = False
        if finetuned_model_name != "":
            print(
                "[D] Model has been fine-tuned! Updating base model to use fine-tuned model"
            )
            model = AutoModelForSeq2SeqLM.from_pretrained(
                FINETUNED_MODELS_DIR + finetuned_model_name
            )
            model = model.to(settings["device"])

        if finetune_res is not None:
            for key, value in finetune_res.items():
                add_to_dict(results_dict, key, value)

        # save results of model evaluation (pre- and post- finetune)
        exp_params = (
            f"dataset_{settings['exp_parameters']['dataset']}"
            f"-timeIntervalType_{settings['exp_parameters']['time_interval_type']}"
            f"-timeInterval_{settings['exp_parameters']['time_interval']}"
            f"-finetuneType_{settings['exp_parameters']['finetune_type']}"
            f"-percentNewData_{settings['exp_parameters']['percent_new_data']}"
            f"-percentOldData_{settings['exp_parameters']['percent_old_data']}"
        )

        if finetune_instant == settings["finetune_instants"][0]:
            results_file_name = f"finetuneInstant_noRetrain-{exp_params}-latest.json"
        else:
            results_file_name = (
                f"finetuneInstant_{finetune_instant}-{exp_params}-latest.json"
            )
        save_finetune_results(
            results=results_dict,
            results_file_name=results_file_name,
            results_dir=FID_DIR,
            version=len(results_dict[list(results_dict.keys())[0]]) - 1,
        )

        if settings["exp_parameters"]["time_interval_type"] == "time":
            # update current week
            time_tracker_dict["curr_week"] = (
                dataset.get_raw_dataset()
                .loc[[time_tracker_dict["curr_index"]]]["week_number"]
                .to_list()[0]
            )

            if (
                time_tracker_dict["curr_week"]
                >= dataset.get_raw_dataset()["week_number"].max()
            ):
                break


def main(parameters: dict):
    print(json.dumps(parameters, indent=4))

    # load dataset
    print(f"[D] Loading {parameters['dataset']} dataset")
    dataset = load_dataset(
        f"{parameters['dataset']}-timeInterval_{parameters['time_interval']}-seed_{1}-{parameters['source_lang']}-{parameters['target_lang']}",
        fixed_test_set=False,
    )

    if "time" in parameters["time_interval_type"] and not dataset.has_time_columns():
        print(
            f"[W] Dataset {parameters['dataset']} does not have time columns.\n",
            "[W] Defaulting to:\n\ttime_interval_type == sentence\n\ttime_interval == 10000 sentences",
        )
        parameters["time_interval"] = 10000
        parameters["time_interval_type"] = "sentence"

    finetune_instants = get_finetune_instants(
        dataset=dataset.get_raw_dataset(),
        time_interval=parameters["time_interval"],
        time_interval_type=parameters["time_interval_type"],
    )[0]

    loaded_metrics = False
    while not loaded_metrics:
        metrics, error = load_metrics(parameters)
        if error:
            # some other process completed and logged out
            # so we need to login again
            # login to huggingface to access model comet22-kiwi
            login(parameters["huggingface_token"], add_to_git_credential=True)
        else:
            loaded_metrics = True
            del parameters["huggingface_token"]

    settings = {
        "finetune_instants": finetune_instants,
        "exp_parameters": parameters,
        "model_checkpoint": f"{parameters['model_checkpoint']}-{parameters['source_lang']}-{parameters['target_lang']}",
        "metrics": metrics,
    }
    # select device to run on
    if torch.backends.mps.is_available():
        settings["device"] = "mps"  # return torch.device("mps")
    elif parameters["gpu_on"]:
        settings["device"] = "cuda:0" if torch.cuda.is_available() else "cpu"
    else:
        settings["device"] = "cpu"

    settings["tokenizer"] = AutoTokenizer.from_pretrained(
        settings["model_checkpoint"],
        return_tensors="pt",
    )

    # create dir to save results to
    mkdir_if_not_exists(FID_DIR)

    if parameters["finetune_instant"] is not None:
        print(
            f"[D] Starting to evaluate finetune instant {parameters['finetune_instant']}"
        )
        finetune_and_evaluate(settings, parameters["finetune_instant"], dataset)
    else:
        for count, finetune_instant in enumerate(finetune_instants):
            print(
                f"[D] Round {count} --- starting to evaluate finetune instant {finetune_instant}"
            )
            settings["exp_parameters"]["finetune_instant"] = finetune_instant
            finetune_and_evaluate(settings, finetune_instant, dataset)


if __name__ == "__main__":
    # Instantiate the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-ti",
        "--time_interval",
        help="<Required> Set time_interval period.\n\t1 time_interval period = 1 week if time_interval_type=time\n\totherwise specify number of sentences",
        required=True,
        type=int,
    )

    parser.add_argument(
        "--finetune_instant",
        help="<Optional> When set, finetune the model when this moment is reached. Otherwise, gather metrics for all finetune instants",
        required=False,
        type=int,
        default=None,
    )

    parser.add_argument(
        "--finetune_type",
        help="<Optional> Type of fine-tune. DEFAULT = incremental\n\tbase: always fine-tune base hugging face model with all the data seen until now\n\tincremental: fine-tune the previously fine-tuned model with new data gathered since",
        required=False,
        type=str,
        default="incremental",
    )

    parser.add_argument(
        "--time_interval_type",
        help="<Optional> Type of time_interval periodicity. DEFAULT = time\n\ttime: time_interval based on how much time has passed since the last adaptation\n\tsentence: time_interval when the specified number of new sentences has been received",
        required=False,
        type=str,
        default="time",
    )

    parser.add_argument(
        "--percent_new_data",
        help="<Optional> Percentage of new data to use for fine-tuning the model. Default = 1.0. Range=[0.0; 1.0]",
        required=False,
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--percent_old_data",
        help="<Optional> Percentage of old data to (re-)use when fine-tuning the model. Default = 0.0. Range=[0.0; 1.0]",
        required=False,
        type=float,
        default=0.0,
    )

    gpu_group = parser.add_mutually_exclusive_group()
    gpu_group.add_argument(
        "--gpu_off",
        help="<Optional> Do not use GPU even if available",
        required=False,
        action="store_true",
    )

    gpu_group.add_argument(
        "--gpu_on",
        help="<Optional> Use GPU if available",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "--huggingface_token",
        help="<Optional> Token to login to HuggingFace. This is required to access model comet22-kiwi",
        required=False,
        type=str,
    )

    prog_args = parser.parse_args()
    print(prog_args)

    # login to huggingface to access model comet22-kiwi
    login(prog_args.huggingface_token, add_to_git_credential=True)

    # load finetune experimental parameters
    exp_parameters = load_params_from_yaml(
        exp_parameters_file=FINETUNE_EXP_PARAMS_FILE,
    )
    exp_parameters["huggingface_token"] = prog_args.huggingface_token
    exp_parameters["time_interval"] = prog_args.time_interval
    exp_parameters["finetune_type"] = prog_args.finetune_type
    exp_parameters["time_interval_type"] = prog_args.time_interval_type
    exp_parameters["gpu_on"] = not prog_args.gpu_off
    exp_parameters["finetune_instant"] = (
        prog_args.finetune_instant
        if (
            isinstance(prog_args.finetune_instant, int)
            and prog_args.finetune_instant >= -1
        )
        else None
    )
    exp_parameters["percent_new_data"] = prog_args.percent_new_data
    exp_parameters["percent_old_data"] = prog_args.percent_old_data

    main(exp_parameters)

    # logout from huggingface
    logout()
