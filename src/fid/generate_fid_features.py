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

This script implements step 2.
"""

import argparse
import itertools
import os
import time

import pandas as pd

from fid.fid_features import (
    compute_ngram_freq_dist_features,
    compute_sent_embedding_features,
    compute_sent_fuzzy_score_features,
    compute_sent_overlap_ratio_features,
    count_words_by_language,
    get_metric_value,
)
from fid.utils_fid import get_finetune_instants, load_data, load_file
from finetune.dataset import MyDataset
from finetune.utils_finetune import load_dataset
from constants import FID_TMP_FILES_DIR, GEN_FID_PARAMS_FILE, TMP_METRICS_DIR
from utils import add_to_dict, load_params_from_yaml, mkdir_if_not_exists


def compute_content_aware_features(
    features_to_compute: list,
    features_dict: dict,
    doc1,
    doc2,
    new_data_type: str,
    test_set_name: str = None,
):
    compute = False
    if "all" in features_to_compute or "contentAware" in features_to_compute:
        compute = True

    if compute or "sent_overlap_ratio" in features_to_compute:
        compute_sent_overlap_ratio_features(
            features_dict, doc1, doc2, new_data_type, test_set_name, VERBOSE
        )

    if compute or "sent_fuzzy_score" in features_to_compute:
        compute_sent_fuzzy_score_features(
            features_dict, doc1, doc2, new_data_type, test_set_name, VERBOSE
        )

    if compute or "ngram_freq_dist_diff" in features_to_compute:
        compute_ngram_freq_dist_features(
            features_dict, doc1, doc2, new_data_type, test_set_name, VERBOSE
        )

    if compute or "sent_embedding_cluster_dist" in features_to_compute:
        compute_sent_embedding_features(
            features_dict, doc1, doc2, new_data_type, test_set_name, VERBOSE
        )


def compute_sys_perf_features(
    features_dict: dict,
    curr_finetune: dict,
    prev_finetune: dict,
    dataset: MyDataset,
    dataset_name: str,
):
    curr_finetune_metrics = curr_finetune["metrics"]
    prev_finetune_metrics = prev_finetune["metrics"]

    # save metrics related features
    # for all metrics, save their values:
    #   - now (before this finetune)
    #   - in the following time interval WITH finetune
    #   - in the following time interval WITHOUT finetune

    if "hk-news" in dataset_name:
        # load files with eval metrics for the fixed test sets
        split_idx = len(curr_finetune["metadata"].split("timeIntervalType")[0])
        curr_fixed_metrics_df = load_file(
            path=f"{TMP_METRICS_DIR}{dataset_name}/",
            file_name=curr_finetune["metadata"][:split_idx]
            + "fixedTestSetEval-"
            + curr_finetune["metadata"][split_idx:]
            + "-latest.json",
        )

        split_idx = len(prev_finetune["metadata"].split("timeIntervalType")[0])
        prev_fixed_metrics_df = load_file(
            path=f"{TMP_METRICS_DIR}{dataset_name}/",
            file_name=prev_finetune["metadata"][:split_idx]
            + "fixedTestSetEval-"
            + prev_finetune["metadata"][split_idx:]
            + "-latest.json",
        )
    elif "opus_eng_fra" in dataset_name:
        curr_fixed_metrics_df = curr_finetune_metrics
        prev_fixed_metrics_df = prev_finetune_metrics
        prev_metrics = load_file(
            path=f"{TMP_METRICS_DIR}{dataset_name}/",
            file_name=prev_finetune["metadata"].replace("fixedTestSetEval-", "")
            + "-latest.json",
        )

    for metric in curr_finetune_metrics["exp_parameters_metrics"][0]:
        ## FOR THE FIXED TEST-SETS
        # if the files exist
        if isinstance(curr_fixed_metrics_df, pd.DataFrame) and isinstance(
            prev_fixed_metrics_df, pd.DataFrame
        ):
            for test_set in dataset.get_test_set_names():
                # "current" model
                add_to_dict(
                    dictionary=features_dict,
                    key=f"curr_{test_set}_{metric}",
                    data=prev_fixed_metrics_df.at[0, f"eval_{test_set}_{metric}"],
                    verbose=VERBOSE,
                )
                # fine-tuned model
                add_to_dict(
                    dictionary=features_dict,
                    key=f"target_{test_set}_{metric}",
                    data=curr_fixed_metrics_df.at[0, f"eval_{test_set}_{metric}"],
                    verbose=VERBOSE,
                )

        # get current metric value (e.g. current chrf, comet)
        if "hk-news" in dataset_name:
            ## FOR THE CONSECUTIVE WEEK
            # before any adaptation
            metric_value = get_metric_value(
                data=prev_finetune_metrics,
                metric=f"curr_{metric}",
                start_time=curr_finetune_metrics.at[0, "curr_time_str"],
                end_time=curr_finetune_metrics.at[0, "finetune_period_end_time"],
                dataset_name=dataset_name,
            )

            add_to_dict(
                dictionary=features_dict,
                key=f"curr_{metric}",
                data=metric_value,
                verbose=VERBOSE,
            )

            # get target metric value after finetune
            metric_value = get_metric_value(
                data=curr_finetune_metrics,
                metric=f"target_{metric}_finetune",
                start_time=curr_finetune_metrics.at[
                    0, "finetune_eval_period_start_time"
                ],
                end_time=curr_finetune_metrics.at[0, "finetune_eval_period_end_time"],
                dataset_name=dataset_name,
            )
            add_to_dict(
                dictionary=features_dict,
                key=f"target_{metric}_finetune",
                data=metric_value,
                verbose=VERBOSE,
            )

            # get target metric value after NO finetune
            metric_value = get_metric_value(
                data=prev_finetune_metrics,
                metric=f"target_{metric}_noFinetune",
                start_time=curr_finetune_metrics.at[
                    0, "finetune_eval_period_start_time"
                ],
                end_time=curr_finetune_metrics.at[0, "finetune_eval_period_end_time"],
                dataset_name=dataset_name,
            )
            add_to_dict(
                dictionary=features_dict,
                key=f"target_{metric}_noFinetune",
                data=metric_value,
                verbose=VERBOSE,
            )

        elif "opus_eng_fra" in dataset_name:
            ## FOR THE CONSECUTIVE WEEK
            # before any adaptation
            metric_value = get_metric_value(
                data=prev_metrics,
                metric=f"curr_{metric}",
                start_time=str(
                    int(curr_finetune_metrics.at[0, "curr_start_index"])
                    - int(
                        prev_finetune["metadata"]
                        .split("timeInterval_")[1]
                        .split("-")[0]
                    )
                ),
                end_time=curr_finetune_metrics.at[0, "curr_start_index"],
                dataset_name=dataset_name,
            )
            add_to_dict(
                dictionary=features_dict,
                key=f"curr_{metric}",
                data=metric_value,
                verbose=VERBOSE,
            )


def compute_basic_data_features(
    features_dict: dict,
    old_data: pd.DataFrame,
    new_data: pd.DataFrame,
    finetune_data: pd.DataFrame,
    dataset: MyDataset,
):
    # save amount of old and new data
    add_to_dict(
        dictionary=features_dict,
        key="amount_old_data",
        data=len(old_data.index),
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="amount_new_data",
        data=len(new_data.index),
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="total_data",
        data=features_dict["amount_new_data"][-1]
        + features_dict["amount_old_data"][-1],
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="amount_finetune_data",
        data=features_dict["total_data"][-1],
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="ratio_new_old_data",
        data=features_dict["amount_new_data"][-1]
        / (
            features_dict["amount_old_data"][-1]
            if features_dict["amount_old_data"][-1] > 0
            else 1.0
        ),
        verbose=VERBOSE,
    )

    prod = itertools.product(
        [
            list(a)
            for a in zip(
                ["source", "target"],
                [dataset.get_source_lang(), dataset.get_target_lang()],
            )
        ],
        [
            list(a)
            for a in zip(
                [old_data, new_data, finetune_data],
                ["old_data", "new_data", "finetune_data"],
            )
        ],
    )

    for col, lang, data, data_name in [
        list(itertools.chain.from_iterable(a)) for a in prod
    ]:
        word_count = count_words_by_language(text=data[col].to_numpy(), language=lang)
        add_to_dict(
            dictionary=features_dict,
            key=f"count_{data_name}_{lang}_words_total",
            data=word_count[1],
            verbose=VERBOSE,
        )
        add_to_dict(
            dictionary=features_dict,
            key=f"count_{data_name}_{lang}_words_trimmed",
            data=word_count[2],
            verbose=VERBOSE,
        )

    for lang in [dataset.get_source_lang(), dataset.get_target_lang()]:
        add_to_dict(
            dictionary=features_dict,
            key=f"ratio_new_old_data_{lang}_words_total",
            data=(
                features_dict[f"count_new_data_{lang}_words_total"][-1]
                / features_dict[f"count_old_data_{lang}_words_total"][-1]
            ),
            verbose=VERBOSE,
        )
        add_to_dict(
            dictionary=features_dict,
            key=f"ratio_new_old_data_{lang}_words_trimmed",
            data=(
                features_dict[f"count_new_data_{lang}_words_trimmed"][-1]
                / features_dict[f"count_old_data_{lang}_words_trimmed"][-1]
            ),
            verbose=VERBOSE,
        )


def get_finetune_new_old_data_opus_en_fr(
    dataset: MyDataset,
    curr_finetune_metrics: pd.DataFrame,
    prev_finetune_metrics: pd.DataFrame,
):
    raw_dataset = dataset.get_raw_dataset()

    data_sets = {}

    # OLD DATA
    if "base" in prev_finetune_metrics.at[0, "exp_parameters_finetune_type"]:
        old_data_start_index = raw_dataset.index[0]
    else:  # incremental fine-tuning
        old_data_start_index = int(prev_finetune_metrics.at[0, "curr_start_index"])

    old_data_end_index = int(prev_finetune_metrics.at[0, "curr_start_index"])
    old_data_index_list = list(range(old_data_start_index, old_data_end_index, 1))

    data_sets["old_data"] = raw_dataset[
        raw_dataset.index.isin(old_data_index_list)
    ].copy()

    # NEW DATA
    new_data_idxs = {
        "start": int(old_data_end_index),  # coincides with old_data_end_index
        "end": int(curr_finetune_metrics.at[0, "curr_start_index"]),
    }
    new_data_index_list = list(range(new_data_idxs["start"], new_data_idxs["end"], 1))

    data_sets["new_data"] = raw_dataset[
        raw_dataset.index.isin(new_data_index_list)
    ].copy()

    if VERBOSE:
        print(
            f"[D] old-data index: {min(old_data_index_list)}-{max(old_data_index_list)}"
        )
        print(
            f"[D] New data: start index={new_data_idxs['start']}    end index={new_data_idxs['end']}"
        )

    # FINE-TUNE DATA
    if "base" in curr_finetune_metrics.at[0, "exp_parameters_finetune_type"]:
        finetune_data_index_list = list(
            range(old_data_start_index, new_data_idxs["end"], 1)
        )
    else:
        finetune_data_index_list = list(
            range(new_data_idxs["start"], new_data_idxs["end"], 1)
        )

    data_sets["finetune_data"] = raw_dataset[
        raw_dataset.index.isin(finetune_data_index_list)
    ].copy()

    return data_sets


def compute_opus_en_fr_basic_features(
    features_dict: dict,
    dataset: MyDataset,
    curr_finetune: dict,
    prev_finetune: dict,
    data_sets: dict,
):
    curr_finetune_metrics = curr_finetune["metrics"]
    prev_finetune_metrics = prev_finetune["metrics"]

    time_interval = int(curr_finetune_metrics.at[0, "exp_parameters_time_interval"])

    # compute basic features
    if VERBOSE:
        print("[D] Computing basic features!")
    basic_features_start_time = time.time()

    add_to_dict(
        dictionary=features_dict,
        key="curr_finetune_start_id",
        data=int(curr_finetune_metrics.at[0, "curr_start_index"]),
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="prev_finetune_start_id",
        data=int(prev_finetune_metrics.at[0, "curr_start_index"]),
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="finetune_delta",
        data=features_dict["curr_finetune_start_id"][-1]
        - features_dict["prev_finetune_start_id"][-1],
        verbose=VERBOSE,
    )

    add_to_dict(
        dictionary=features_dict,
        key="new_data_original_opus_dataset_name",
        data=list(data_sets["new_data"]["original_opus_dataset_name"].unique())[0],
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="old_data_original_opus_dataset_name",
        data=list(data_sets["old_data"]["original_opus_dataset_name"].unique()),
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="prev_chunk_original_opus_dataset_name",
        data=list(
            data_sets["old_data"]
            .tail(time_interval)["original_opus_dataset_name"]
            .unique()
        )[0],
        verbose=VERBOSE,
    )

    # save data features:
    #   - #old data, #new data, #total data
    #   - ratio new/old data
    #   - #new words source & target lang
    compute_basic_data_features(
        features_dict=features_dict,
        old_data=data_sets["old_data"],
        new_data=data_sets["new_data"],
        finetune_data=data_sets["finetune_data"],
        dataset=dataset,
    )

    compute_sys_perf_features(
        features_dict=features_dict,
        curr_finetune=curr_finetune,
        prev_finetune=prev_finetune,
        dataset=dataset,
        dataset_name="opus_eng_fra",
    )

    add_to_dict(
        dictionary=features_dict,
        key="basic_features_total_time",
        data=time.time() - basic_features_start_time,
    )
    if VERBOSE:
        print(
            f"[D] Basic features computed in {features_dict['basic_features_total_time'][-1]} secs!"
        )


def get_finetune_new_old_data_hksar_news(
    dataset: MyDataset,
    curr_finetune_metrics: pd.DataFrame,
    prev_finetune_metrics: pd.DataFrame,
):
    raw_dataset = dataset.get_raw_dataset()
    # these are the start and end times of the current time interval
    # which has the new data for finetune
    #   - finetune_period_start_time = curr_finetune_metrics.at[0, "curr_time_str"]
    #   - finetune_period_end_time = curr_finetune_metrics.at[0, "finetune_period_end_time"]

    data_sets = {}

    # OLD DATA
    data_sets["old_data"] = raw_dataset.loc[
        (raw_dataset["timestamp"] >= raw_dataset["timestamp"].min())
        & (
            raw_dataset["timestamp"]
            < prev_finetune_metrics.at[0, "finetune_period_end_time"]
        )
    ].copy()

    if VERBOSE:
        print(
            f"[D] New data start index = {prev_finetune_metrics.at[0, 'curr_week_end_index']}    new data end index = {curr_finetune_metrics.at[0, 'curr_week_end_index']}"
        )

    # FINE-TUNE DATA
    data_sets["finetune_data"] = raw_dataset.loc[
        (raw_dataset["timestamp"] >= raw_dataset["timestamp"].min())
        & (
            raw_dataset["timestamp"]
            < curr_finetune_metrics.at[0, "finetune_period_end_time"]
        )
    ].copy()

    # NEW DATA
    # this is all the new data ==> not the whole finetune data used
    data_sets["new_data"] = raw_dataset[
        raw_dataset.index.isin(
            list(
                range(
                    int(prev_finetune_metrics.at[0, "curr_week_end_index"]),
                    int(curr_finetune_metrics.at[0, "curr_week_end_index"]),
                    1,
                )
            )
        )
    ].copy()

    return data_sets


def compute_hksar_news_basic_features(
    features_dict: dict,
    dataset: MyDataset,
    curr_finetune: dict,
    prev_finetune: dict,
    data_sets: dict,
):
    # compute basic features

    curr_finetune_metrics = curr_finetune["metrics"]
    prev_finetune_metrics = prev_finetune["metrics"]

    if VERBOSE:
        print("[D] Computing basic features!")
    basic_features_start_time = time.time()
    add_to_dict(
        dictionary=features_dict,
        key="period_start_timestamp",
        data=curr_finetune_metrics.at[0, "curr_time_str"],
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="period_end_timestamp",
        data=curr_finetune_metrics.at[0, "finetune_period_end_time"],
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="prev_finetune_period_start_timestamp",
        data=prev_finetune_metrics.at[0, "curr_time_str"],
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="prev_finetune_period_end_timestamp",
        data=prev_finetune_metrics.at[0, "finetune_period_end_time"],
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="curr_finetune_week",
        data=curr_finetune_metrics.at[0, "exp_parameters_finetune_instant"],
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="prev_finetune_week",
        data=prev_finetune_metrics.at[0, "exp_parameters_finetune_instant"],
        verbose=VERBOSE,
    )
    add_to_dict(
        dictionary=features_dict,
        key="finetune_delta",
        data=features_dict["curr_finetune_week"][-1]
        - features_dict["prev_finetune_week"][-1],
        verbose=VERBOSE,
    )

    # save data features:
    #   - #old data, #new data, #total data
    #   - ratio new/old data
    #   - #new words source & target lang
    compute_basic_data_features(
        features_dict=features_dict,
        old_data=data_sets["old_data"],
        new_data=data_sets["new_data"],
        finetune_data=data_sets["finetune_data"],
        dataset=dataset,
    )

    compute_sys_perf_features(
        features_dict=features_dict,
        curr_finetune=curr_finetune,
        prev_finetune=prev_finetune,
        dataset=dataset,
        dataset_name="hk-news",
    )

    basic_features_total_time = time.time() - basic_features_start_time
    add_to_dict(features_dict, "basic_features_total_time", basic_features_total_time)
    if VERBOSE:
        print(f"[D] Basic features computed in {basic_features_total_time} secs!")


def compute_features(
    features_to_compute: list,
    features_dict: dict,
    dataset: MyDataset,
    curr_finetune: dict,
    prev_finetune: dict,
    compute_basic_features_func,
    data_sets: dict,
):
    if "all" in features_to_compute or "basic" in features_to_compute:
        compute_basic_features_func(
            features_dict=features_dict,
            dataset=dataset,
            curr_finetune=curr_finetune,
            prev_finetune=prev_finetune,
            data_sets=data_sets,
        )

    fixed_test_sets = dataset.get_test_set(test_set_format="pandas")
    if (
        "all" in features_to_compute
        or "new_data" in features_to_compute
        or "new_data_contentAware" in features_to_compute
    ):
        compute_content_aware_features(
            features_to_compute=features_to_compute,
            features_dict=features_dict,
            doc1=data_sets["old_data"]["source"].to_numpy(),
            doc2=data_sets["new_data"]["source"].to_numpy(),
            new_data_type="new_data",
        )

        for test_set_name, test_set in fixed_test_sets.items():
            compute_content_aware_features(
                features_to_compute=features_to_compute,
                features_dict=features_dict,
                doc1=test_set["source"].to_numpy(),
                doc2=data_sets["new_data"]["source"].to_numpy(),
                new_data_type="new_data",
                test_set_name=test_set_name,
            )

    if (
        "all" in features_to_compute
        or "finetune_data" in features_to_compute
        or "finetune_data_contentAware" in features_to_compute
    ):
        compute_content_aware_features(
            features_to_compute=features_to_compute,
            features_dict=features_dict,
            doc1=data_sets["old_data"]["source"].to_numpy(),
            doc2=data_sets["finetune_data"]["source"].to_numpy(),
            new_data_type="finetune_data",
        )

        for test_set_name, test_set in fixed_test_sets.items():
            compute_content_aware_features(
                features_to_compute=features_to_compute,
                features_dict=features_dict,
                doc1=test_set["source"].to_numpy(),
                doc2=data_sets["finetune_data"]["source"].to_numpy(),
                new_data_type="finetune_data",
                test_set_name=test_set_name,
            )


def skip_finetune_instant(
    params, finetune_metadata, finetune_type, last_finetune_instant
):
    # this corresponds to the first instant (no retrain case)
    # so there is no previous instant against which to compare
    if "no_finetune" in finetune_metadata or "noRetrain" in finetune_metadata:
        return True

    finetune_instant = finetune_metadata.split("-")[0].split("_")[1]
    if (
        "all" not in params[f"{finetune_type}_finetune_instant"]
        and finetune_instant != params[f"{finetune_type}_finetune_instant"]
    ):  # skip curr_finetune when only computing features for a specific one
        return True

    if "curr" in finetune_type:
        # if this corresponds to the last instant, there is no
        # following week against which to compare
        if f"finetuneInstant_{last_finetune_instant}" in finetune_metadata:
            return True

    if (
        "percent_new_data" in params
        and f"percentNewData_{params['percent_new_data']}" not in finetune_metadata
    ):
        return True

    if (
        "percent_old_data" in params
        and f"percentOldData_{params['percent_old_data']}" not in finetune_metadata
    ):
        return True

    return False


def get_new_old_data_sets(
    get_finetune_new_old_data_func,
    dataset: MyDataset,
    curr_finetune: dict,
    prev_finetune: dict,
):
    curr_finetune_metadata = curr_finetune["metadata"]
    curr_metrics = curr_finetune["metrics"]
    prev_metrics = prev_finetune["metrics"]

    # get the sets of fine-tune data, new data, and old data
    # - fine-tune type == incremental ==> fine-tune_data == new_data
    # - fine-tune type == base ==> fine-tune_data == old_data + new_data
    data_sets = get_finetune_new_old_data_func(
        dataset=dataset,
        curr_finetune_metrics=curr_metrics,
        prev_finetune_metrics=prev_metrics,
    )

    if (
        "percentNewData" in curr_finetune_metadata
        and "percentOldData" in curr_finetune_metadata
    ):
        percent_new_data = float(
            curr_finetune_metadata.split("percentNewData_")[1].split("-")[0]
        )
        percent_old_data = float(
            curr_finetune_metadata.split("percentOldData_")[1].split("-")[0]
        )
        new_data = data_sets["new_data"].sample(frac=percent_new_data, random_state=42)
        finetune_data = pd.concat([new_data, data_sets["old_data"]], ignore_index=True)
        if percent_old_data < 1.0:
            new_data = pd.concat(
                [
                    new_data,
                    data_sets["old_data"].sample(
                        frac=percent_old_data, random_state=42
                    ),
                ],
                ignore_index=True,
            )
            finetune_data = pd.concat(
                [
                    finetune_data,
                    data_sets["old_data"].sample(
                        frac=percent_old_data, random_state=42
                    ),
                ],
                ignore_index=True,
            )

        data_sets["new_data"] = new_data
        data_sets["finetune_data"] = finetune_data

    data_sets["old_data"] = data_sets["old_data"].sample(frac=0.1, random_state=42)
    data_sets["new_data"] = data_sets["new_data"].sample(frac=0.1, random_state=42)
    data_sets["finetune_data"] = data_sets["finetune_data"].sample(
        frac=0.1, random_state=42
    )

    return data_sets


def generate_fid(
    fid_params: list,
    data: dict,
    dataset: MyDataset,
    last_finetune_instant: int,
):
    if "hk-news" in fid_params["dataset"]:
        get_finetune_new_old_data_func = get_finetune_new_old_data_hksar_news
        compute_basic_features_func = compute_hksar_news_basic_features
    if "opus" in fid_params["dataset"]:
        get_finetune_new_old_data_func = get_finetune_new_old_data_opus_en_fr
        compute_basic_features_func = compute_opus_en_fr_basic_features

    # features will be appended to this dict as they are computed
    features_dict = {}

    # compare finetune and no finetune until last timestamp
    for curr_finetune_metadata, curr_metrics in data.items():
        if skip_finetune_instant(
            fid_params, curr_finetune_metadata, "curr", last_finetune_instant
        ):
            print(f"Skipping... : {curr_finetune_metadata}")
            continue

        # compute features
        for prev_finetune_metadata, prev_metrics in data.items():
            if skip_finetune_instant(
                fid_params, prev_finetune_metadata, "prev", last_finetune_instant
            ):
                print(f"Skipping... : {prev_finetune_metadata}")
                continue

            print(
                "\n ---------------------------------------------------- NEW FINETUNE INTERVAL ----------------------------------------------------"
            )
            print(f"[D] Current finetune: {curr_finetune_metadata}")
            print(f"[D] Previous finetune: {prev_finetune_metadata}")
            # we only want to analyze models that were finetuned before the current timestamp
            curr_finetune_instant = curr_finetune_metadata.split("-")[0].split("_")[1]
            prev_finetune_instant = prev_finetune_metadata.split("-")[0].split("_")[1]
            if int(prev_finetune_instant) >= int(curr_finetune_instant):
                print(
                    f"[D] Skipping because prev_finetune_instant {prev_finetune_instant} >= curr_finetune_instant {curr_finetune_instant}"
                )
                continue

            compute_features_start_time = time.time()

            # get the sets of fine-tune data, new data, and old data
            # - fine-tune type == incremental ==> fine-tune_data == new_data
            # - fine-tune type == base ==> fine-tune_data == old_data + new_data
            data_sets = get_new_old_data_sets(
                get_finetune_new_old_data_func=get_finetune_new_old_data_func,
                dataset=dataset,
                curr_finetune={
                    "metadata": curr_finetune_metadata,
                    "metrics": curr_metrics,
                },
                prev_finetune={
                    "metadata": prev_finetune_metadata,
                    "metrics": prev_metrics,
                },
            )

            compute_features(
                features_to_compute=fid_params["fid_features"],
                features_dict=features_dict,
                dataset=dataset,
                curr_finetune={
                    "metrics": curr_metrics,
                    "metadata": curr_finetune_metadata,
                },
                prev_finetune={
                    "metadata": prev_finetune_metadata,
                    "metrics": prev_metrics,
                },
                compute_basic_features_func=compute_basic_features_func,
                data_sets=data_sets,
            )
            add_to_dict(
                features_dict,
                "compute_features_total_time",
                time.time() - compute_features_start_time,
            )

            add_to_dict(features_dict, "curr_finetune", curr_finetune_instant)
            add_to_dict(features_dict, "prev_finetune", prev_finetune_instant)

    return pd.DataFrame.from_dict(features_dict)


def main(fid_params: dict):
    # check if results file already exists: if yes, skip
    results_file_dir = FID_TMP_FILES_DIR + f"{fid_params['dataset']}/"
    mkdir_if_not_exists(results_file_dir)
    results_file_name = (
        f"fid-dataset_{fid_params['dataset']}-"
        f"features_{fid_params['fid_features']}-"
        f"currFinetune_{fid_params['curr_finetune_instant']}-"
        f"prevFinetune_{fid_params['prev_finetune_instant']}-"
        f"timeIntervalType_{fid_params['time_interval_type']}-"
        f"timeInterval_{fid_params['time_interval']}-"
        f"finetuneType_{fid_params['finetune_type']}"
    )

    if "percent_new_data" in fid_params:
        results_file_name = (
            results_file_name + f"-percentNewData_{fid_params['percent_new_data']}"
        )

    if "percent_old_data" in fid_params:
        results_file_name = (
            results_file_name + f"-percentOldData_{fid_params['percent_old_data']}"
        )
    results_file = results_file_dir + results_file_name + ".pkl"

    if not fid_params["overwrite"] and os.path.isfile(results_file):
        print("#" * 50)
        print(f"[D] Skipping output file: {results_file}")
        print("[D] File already exists")
        return

    # load dataset
    print(f"[D] Loading {fid_params['dataset']} dataset")
    dataset = load_dataset(
        f"{fid_params['dataset']}-timeInterval_{fid_params['time_interval']}-seed_{1}"
        f"-{fid_params['source_lang']}-{fid_params['target_lang']}",
        fixed_test_set=True,
    )

    finetune_instants, last_finetune_instant = get_finetune_instants(
        dataset=dataset.get_raw_dataset(),
        time_interval=fid_params["time_interval"],
        time_interval_type=fid_params["time_interval_type"],
    )

    if (
        "all" not in fid_params["curr_finetune_instant"]
        and "all" not in fid_params["prev_finetune_instant"]
    ):
        finetune_instants = [
            int(fid_params["prev_finetune_instant"]),
            int(fid_params["curr_finetune_instant"]),
        ]
    print(
        f"[D] Computing features for the following finetune instants:\n\t{finetune_instants}"
    )

    # load all metrics files
    print("[D] Loading pre-computed metrics files")
    data = load_data(
        path=TMP_METRICS_DIR + f"{fid_params['dataset']}/",
        params=fid_params,
        finetune_instants=finetune_instants,
    )

    # compute fid metrics
    results = generate_fid(
        fid_params=fid_params,
        data=data,
        dataset=dataset,
        last_finetune_instant=last_finetune_instant,
    )

    if not results.empty:
        results["dataset"] = fid_params["dataset"]
        results["timeIntervalType"] = fid_params["time_interval_type"]
        results["timeInterval"] = fid_params["time_interval"]
        results["finetuneType"] = fid_params["finetune_type"]

        # save results
        results.to_pickle(results_file)
        print(f"[D] FID saved to {results_file}")
    else:
        print("[W] Empty results file: nothing to save!")


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
        "-d",
        "--dataset",
        help="<Optional> Dataset to be used. DEFAULT = hk-news\n\tOptions: ['kh-news', 'opus_eng_fra']",
        required=False,
        type=str,
        default="hk-news",
    )

    parser.add_argument(
        "--finetune_type",
        help="<Optional> Type of fine-tune. DEFAULT = incremental\n\tOptions: ['base', 'incremental']\n\tbase: always fine-tune base hugging face model with all the data seen until now\n\tincremental: fine-tune the previously fine-tuned model with new data gathered since",
        required=False,
        type=str,
        default="incremental",
    )

    parser.add_argument(
        "--time_interval_type",
        help="<Optional> Type of time_interval periodicity. DEFAULT = time\n\tOptions: ['time', 'sentence']\n\ttime: time_interval based on how much time has passed since the last adaptation\n\tsentence: time_interval when the specified number of new sentences has been received",
        required=False,
        type=str,
        default="time",
    )

    parser.add_argument(
        "--fid_features",
        help="<Optional> Which FID features to compute. DEFAULT = all\n\tOptions: ['all', 'contentAware', 'basic', 'sent_overlap_ratio', 'sent_fuzzy_score', 'ngram_freq_dist_diff', 'sent_embedding_cluster_dist']\n\tcontentAware: all except basic",
        required=False,
        type=str,
        default="all",
    )

    parser.add_argument(
        "--curr_finetune",
        help="<Optional> Compute FID features for only this finetune instant as current finetune instant",
        required=False,
        type=str,
        default="all",
    )

    parser.add_argument(
        "--prev_finetune",
        help="<Optional> Compute FID features for only this finetune instant as previous finetune instant",
        required=False,
        type=str,
        default="all",
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

    parser.add_argument(
        "-v",
        "--verbose",
        help="<Optional> Verbosity",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-o",
        "--overwrite",
        help="<Optional> Whether to overwrite existing (pre-computed) FID files",
        required=False,
        action="store_true",
    )

    prog_args = parser.parse_args()
    print(prog_args)

    parameters = load_params_from_yaml(exp_parameters_file=GEN_FID_PARAMS_FILE)

    args = {
        "dataset": prog_args.dataset,
        "source_lang": parameters["source_lang"],
        "target_lang": parameters["target_lang"],
        "time_interval": prog_args.time_interval,
        "finetune_type": prog_args.finetune_type,
        "time_interval_type": prog_args.time_interval_type,
        "fid_features": prog_args.fid_features,
        "curr_finetune_instant": prog_args.curr_finetune,
        "prev_finetune_instant": prog_args.prev_finetune,
        "overwrite": prog_args.overwrite,
    }

    global VERBOSE
    VERBOSE = prog_args.verbose

    if prog_args.percent_new_data < 1.0:
        args["percent_new_data"] = prog_args.percent_new_data

    if prog_args.percent_old_data > 0.0:
        args["percent_old_data"] = prog_args.percent_old_data

    main(args)
