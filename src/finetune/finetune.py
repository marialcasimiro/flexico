#!/usr/bin/env python

import json
import os
import time
from datetime import datetime, timedelta

import pandas as pd
import transformers
from datasets import Dataset
from transformers import AutoModelForSeq2SeqLM, Seq2SeqTrainingArguments

from finetune.dataset import MyDataset
from finetune.hf_utils import get_hf_trainer, get_hf_training_args
from finetune.utils_finetune import (
    evaluate_model,
    get_new_old_data,
    split_dataset,
    tokenize_data,
)
from constants import FINETUNED_MODELS_DIR, WEEK_LENGTH

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print(f"Transformers version: {transformers.__version__}")


def finetune(
    start_at,
    model: AutoModelForSeq2SeqLM,
    training_args: Seq2SeqTrainingArguments,
    tokenized_datasets,
    settings: dict,
):
    print("[D] Fine-tuning model")

    parameters = settings["exp_parameters"]

    if "time" in parameters["time_interval_type"]:
        start = f"startTime_{start_at}"
    else:
        start = f"startIndex_{start_at}"
    finetuned_model_name = (
        f"finetuned_model-dataset_{parameters['dataset']}-{start}-"
        f"finetuneInstant_{parameters['finetune_instant']}-"
        f"timeIntervalType_{parameters['time_interval_type']}-"
        f"timeInterval_{parameters['time_interval']}-"
        f"finetuneType_{parameters['finetune_type']}-"
        f"percentNewData_{parameters['percent_new_data']}-"
        f"percentOldData_{parameters['percent_old_data']}"
    )

    # if fine-tuned model already exists, we just load it and evaluate it
    # else we finetune it
    do_finetune = False
    try:
        model = AutoModelForSeq2SeqLM.from_pretrained(
            FINETUNED_MODELS_DIR + finetuned_model_name
        )
        print("[D] Fine-tuned model exists ==> loading!")
    except (ValueError, EnvironmentError):
        print("[E] Fine-tuned model does not exist yet ==> fine-tuning.")
        do_finetune = True

    if "base" in parameters["finetune_type"]:
        print("[D] Resetting model to base hugging face model for finetuning")
        model = AutoModelForSeq2SeqLM.from_pretrained(settings["model_checkpoint"])
        model = model.to(settings["device"])
    trainer = get_hf_trainer(
        model=model,
        training_args=training_args,
        tokenized_datasets=tokenized_datasets,
        metrics=settings["metrics"],
        tokenizer=settings["tokenizer"],
        comet_batch_size=parameters["comet_batch_size"],
    )
    if do_finetune:
        timer_start = time.time()
        trainer.train()
        finetune_duration = time.time() - timer_start
        # update_latest_finetuned_model_name(FINETUNED_MODELS_DIR)
        trainer.save_model(output_dir=FINETUNED_MODELS_DIR + finetuned_model_name)
        trainer.save_state()
    else:
        finetune_duration = "loaded pre-trained model"

    # evalute fine-tuned model
    print("[D] Evaluating fine-tuned model")
    eval_dict = evaluate_model(
        train_set=tokenized_datasets["train"],
        test_set=tokenized_datasets["test"],
        trainer=trainer,
        parameters=parameters,
        finetune_duration=finetune_duration,
    )

    return eval_dict, finetuned_model_name


def filter_train_set(
    parameters,
    curr_idx,
    finetune_end_idx,
    dataset,
):
    old_data, new_data = get_new_old_data(
        curr_idx=curr_idx,
        finetune_end_idx=finetune_end_idx,
        dataset=dataset,
    )

    train_df = new_data.sample(frac=parameters["percent_new_data"], random_state=42)
    train_df = pd.concat([train_df, old_data], ignore_index=True)
    if parameters["percent_old_data"] < 1.0:
        train_df = pd.concat(
            [
                train_df,
                old_data.sample(frac=parameters["percent_old_data"], random_state=42),
            ],
            ignore_index=True,
        )

    return Dataset.from_pandas(df=train_df.drop(columns=dataset.get_columns_to_drop()))


def new_finetune_round(
    dataset: MyDataset,
    time_tracker_dict: dict,
    finetune_period: int,
    model: AutoModelForSeq2SeqLM,
    settings: dict,
):
    parameters = settings["exp_parameters"]
    finetuned_model_name = ""

    finetune_time_dict, splits = split_dataset(
        time_tracker_dict=time_tracker_dict,
        finetune_period=finetune_period,
        finetune_type=parameters["finetune_type"],
        time_interval_type=parameters["time_interval_type"],
        dataset=dataset,
    )

    if parameters["percent_new_data"] != 1.0 or parameters["percent_old_data"] != 0.0:
        splits["train_split"] = filter_train_set(
            parameters=parameters,
            curr_idx=time_tracker_dict["curr_index"],
            finetune_end_idx=finetune_time_dict["finetune_end_index"],
            dataset=dataset,
        )

    if splits["test_split"] is None:
        return (
            finetune_time_dict["finetune_end_time"],
            finetune_time_dict["finetune_end_index"],
            None,
            finetuned_model_name,
        )

    # apply pre-processing
    tokenized_datasets = tokenize_data(
        train_split=splits["train_split"],
        test_split=splits["test_split"],
        tokenizer=settings["tokenizer"],
        parameters=parameters,
    )

    # define training arguments
    training_args = get_hf_training_args(parameters)

    print(
        f"[D] no_cuda={training_args.no_cuda}   n_gpus={training_args.n_gpu}   device={training_args.device}"
    )

    if parameters["time_interval_type"] == "time":
        results_dict = {
            "exp_parameters": parameters,
            "curr_time": datetime.timestamp(time_tracker_dict["curr_time"]),
            "curr_time_str": str(time_tracker_dict["curr_time"]),
            "curr_week": str(
                dataset.get_raw_dataset()
                .loc[[time_tracker_dict["curr_index"]]]["week_number"]
                .to_list()[0]
            ),
            "curr_week_start_index": str(time_tracker_dict["curr_index"]),
            "curr_week_end_index": str(finetune_time_dict["finetune_end_index"]),
            "finetune_period_start_time": str(
                finetune_time_dict["finetune_start_time"]
            ),
            "finetune_period_end_time": str(finetune_time_dict["finetune_end_time"]),
            "finetune_eval_period_start_time": str(
                finetune_time_dict["finetune_end_time"]
            ),
            "finetune_eval_period_end_time": str(
                finetune_time_dict["finetune_end_time"] + timedelta(days=WEEK_LENGTH)
            ),
        }
        finetune_start = time_tracker_dict["curr_time"]
    else:
        results_dict = {
            "exp_parameters": parameters,
            "curr_start_index": str(time_tracker_dict["curr_index"]),
            "curr_end_index": str(finetune_time_dict["finetune_end_index"]),
        }
        finetune_start = time_tracker_dict["curr_index"]

    # fine-tune model if it is on and if there is data
    if parameters["finetune_on"] and tokenized_datasets["train"].num_rows == 0:
        print("[W] Skipping fine-tuning because there is no train-data")
    if parameters["finetune_on"] and tokenized_datasets["train"].num_rows > 0:
        eval_dict, finetuned_model_name = finetune(
            start_at=finetune_start,
            model=model,
            training_args=training_args,
            tokenized_datasets=tokenized_datasets,
            settings=settings,
        )
        results_dict["eval"] = eval_dict
    else:
        trainer = get_hf_trainer(
            model=model,
            training_args=training_args,
            tokenized_datasets=tokenized_datasets,
            metrics=settings["metrics"],
            tokenizer=settings["tokenizer"],
            comet_batch_size=parameters["comet_batch_size"],
        )

        eval_dict = evaluate_model(
            train_set=tokenized_datasets["train"],
            test_set=tokenized_datasets["test"],
            trainer=trainer,
            parameters=parameters,
            finetune_duration=None,
        )

        results_dict["eval"] = eval_dict

    print("[D] CURRENT ROUND RESULTS:")
    print(json.dumps(results_dict, indent=4))

    return (
        finetune_time_dict["finetune_end_time"],
        finetune_time_dict["finetune_end_index"],
        pd.json_normalize(results_dict, sep="_").to_dict(orient="records")[0],
        finetuned_model_name,
    )
