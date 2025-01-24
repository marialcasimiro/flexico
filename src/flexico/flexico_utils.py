import argparse

import numpy as np
import pandas as pd

from src.flexico.fip_factory import HKNewsFIP_factory, OpusFIP_factory
from constants import BASE_DATA_DIR, FID_DIR

# chrf and sacrebleu are in the range [0, 100]
# comet22 and comet22-qe are in the range [0, 1]
# use this rescaling to guarantee the same range
MT_METRIC_RESCALING = {
    "comet22": 100,
    "comet22-qe": 100,
    "chrf": 1,
    "sacrebleu": 1,
}

NUM_RANDOM_RUNS = 10


def get_amount_finetune_data(df: pd.DataFrame, curr_time: int, last_finetune: int):
    # the amount of fine-tune data should depend only on the current-time
    # however, due to rounding differences, sometimes it differs by 1
    # so we also specify the previous-time (i.e., last_finetune)
    finetune_data = df.loc[
        (df.curr_finetune == curr_time) & (df.prev_finetune == last_finetune)
    ]["amount_finetune_data"].unique()

    if len(finetune_data) == 0:
        finetune_data = [
            df.loc[df.curr_finetune == curr_time]["amount_finetune_data"].unique()[-1]
        ]

    assert (
        len(finetune_data) == 1
    ), f"[E] More than one option for finetune-data!   {finetune_data}   curr_finetune={curr_time}   last_finetune={last_finetune}"
    return finetune_data[0]


def get_finetune_cost(amount_finetune_data: float, dataset: str):
    if "hk-news" in dataset:
        # finetune_cost = 0.005174547088922856 * amount_finetune_data + -2.112023463967214
        finetune_cost = (
            1.43737419136746e-06 * amount_finetune_data + -0.0005866731844353444
        )
    elif "opus" in dataset:
        # finetune_cost = 0.006258860762517377 * amount_finetune_data + 1.5792866597605797
        finetune_cost = (
            1.738572434032605e-06 * amount_finetune_data + 0.00043869073882248506
        )
    else:
        print(f"[W] Unknown dataset {dataset} -- using hk-news equation!")
        # finetune_cost = 0.005174547088922856 * amount_finetune_data + -2.112023463967214
        finetune_cost = (
            1.43737419136746e-06 * amount_finetune_data + -0.0005866731844353444
        )

    return finetune_cost


def get_tactic_cost(
    tactic: str,
    delta_threshold: float,
    finetune_cost: float,
    weighted_delta: float,
):
    if "nop" in tactic:
        tactic_cost = 0
        if weighted_delta >= delta_threshold:
            tactic_cost = weighted_delta * finetune_cost * 2
    else:
        tactic_cost = 0
        if weighted_delta < delta_threshold:
            tactic_cost = (delta_threshold - weighted_delta) * finetune_cost * 2

    return tactic_cost


def init_fips(
    exp_args: dict,
    fid_df: pd.DataFrame,
    fip_model_type: str,
    fid_type: str,
    feature_set: str,
    metadata: dict,
    test_sets: list,
    verbose: bool = False,
):
    # create FIP factory
    fip_factory = metadata["fipFactory"](
        fid=fid_df,
        model_type=fip_model_type,
        fid_type=fid_type,
        test_sets=test_sets,
        verbose=verbose,
    )
    # initialize FIPs
    exp_args["fips"] = {}
    if "generic" in fid_type:  # init general FIPs
        for mt_metric in exp_args["fipTargetMetrics"]:
            exp_args["fips"][mt_metric], features = fip_factory.build_fip(
                feature_set=feature_set,
                target=f"delta-target_test_set_{mt_metric}",
            )
    else:  # init specific FIPs
        for mt_metric in exp_args["fipTargetMetrics"]:
            exp_args["fips"][mt_metric] = {}
            for test_set in test_sets:
                exp_args["fips"][mt_metric][test_set], features = fip_factory.build_fip(
                    feature_set=feature_set,
                    target=f"delta-target_{test_set}_{mt_metric}",
                )
    exp_args["fip_features"] = features


def get_daily_topic_weights(
    curr_time: int,
    topic_weight_type: str,
    daily_weights: dict,
    test_sets,
):
    topic_weights = {}

    if "random" in topic_weight_type:
        # random topic weights per day
        np.random.seed(curr_time)
        rg = np.random.default_rng(seed=curr_time)

        for day in ["1", "2", "3", "4", "5", "6", "7"]:
            topic_weights[day] = {}
            cum_weight = 0  # cumulative weight
            for test_set in test_sets:
                weight = rg.uniform(0, len(test_sets) - cum_weight)
                topic_weights[day][test_set] = weight
                cum_weight += weight

            if cum_weight < len(test_sets):
                target_test_set = np.random.choice(test_sets)
                topic_weights[day][target_test_set] = topic_weights[day][
                    target_test_set
                ] + (len(test_sets) - cum_weight)

            assert round(
                sum([topic_weights[day][test_set] for test_set in test_sets]), 10
            ) == len(test_sets), f"[E] daily weights do not add up to {len(test_sets)}"
    else:
        # get topic weights of the previous week
        # it is unrealistic to assume that we know the exact
        # weights for the current week
        week = curr_time
        if "delayed" in topic_weight_type:
            week = curr_time - 1  # get weights from the previous week

        for day in daily_weights[str(week)].keys():
            topic_weights[day] = {}
            for test_set in test_sets:
                topic_weights[day][test_set] = daily_weights[str(week)][day][test_set][
                    "percent"
                ]

    return topic_weights


def _get_base_arg_parser():
    # Instantiate the parser
    parser = argparse.ArgumentParser(formatter_class=argparse.RawTextHelpFormatter)

    parser.add_argument(
        "-d",
        "--dataset",
        help="Dataset to use. \n\tDEFAULT = hk-news",
        required=True,
        type=str,
        default="hk-news",
    )

    parser.add_argument(
        "-v",
        "--verbose",
        help="<Optional> Verbosity. \n\tDEFAULT = False",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-fc_type",
        "--finetune_cost_type",
        help="<Optional> Whether to use a fixed cost or a liner function of the fine-tune data. DEFAULT = fixed",
        required=False,
        nargs="+",
        type=str,
        default=["fixed"],
    )

    parser.add_argument(
        "-fc",
        "--finetune_costs",
        help=("<Optional> List of fine-tune costs to test.\n\tDEFAULT = 5"),
        nargs="+",
        type=int,
        default=[5],
    )

    parser.add_argument(
        "-t",
        "--thresholds",
        help=("<Optional> List of thresholds for delta.\n\tDEFAULT = 0.5"),
        nargs="+",
        type=float,
        default=[0.5],
    )

    def parse_multiple_lists(arg):
        # Split the argument by semicolon to get individual lists
        list_strings = arg.split(";")
        # Convert each list string into a list of strings
        return [lst.split(",") for lst in list_strings]

    parser.add_argument(
        "--fip_metrics",
        help=(
            "<Optional> List of FIP target MT metrics to test."
            "\n\tDEFAULT = comet22"
            "\n\tavailable-metrics = [comet22, comet22_qe, chrf, sacrebleu]"
        ),
        type=parse_multiple_lists,
        default="comet22",
    )

    parser.add_argument(
        "--topic_weights_types",
        help=(
            "<Optional> Set of types of topic weights to test."
            "\n\tDEFAULT = delayed"
            "\n\tavailable-types = [random, real, delayed]"
        ),
        nargs="+",
        type=str,
        default=["delayed"],
    )

    return parser


def get_flexico_parser():
    parser = _get_base_arg_parser()

    parser.add_argument(
        "-p",
        "--prism",
        help="<Optional> Use PRISM model checker to decide whether to finetune. \n\tDEFAULT = False",
        required=False,
        action="store_true",
    )

    parser.add_argument(
        "-b",
        "--baselines",
        help=(
            "<Optional> List of baselines to test."
            "\n\tDEFAULT = flexico"
            "\n\tOPTIONS = [flexico, noFinetune, finetune@1st, periodic, reactive, exponential, random, optimum]"
        ),
        required=False,
        nargs="+",
        type=str,
        default=["flexico"],
    )

    parser.add_argument(
        "--fip_models",
        help=(
            "<Optional> List of FIP models to test."
            "\n\tDEFAULT = rf"
            "\n\tOPTIONS = [rf, xgb, lin, mlp]"
        ),
        nargs="+",
        type=str,
        default=["rf"],
    )

    parser.add_argument(
        "--fid_types",
        help=(
            "<Optional> Type of FID dataset to load."
            "\n\tDEFAULT = generic"
            "\n\tOPTIONS = [generic, specific]"
        ),
        nargs="+",
        type=str,
        default=["generic"],
    )

    parser.add_argument(
        "-fip_f",
        "--fip_features",
        help=(
            "<Optional> Feature sets for the FIP models."
            "\n\tDEFAULT = all"
            "\n\tOPTIONS = [all, basic, contentAware, sys_perf, basic_sys_perf]"
        ),
        nargs="+",
        type=str,
        default=["all"],
    )

    return parser.parse_args()


def get_opt_baseline_parser():
    parser = _get_base_arg_parser()

    parser.add_argument(
        "--default_tactic",
        help="<Optional> Execute the default tactic if the cost of executing either tactic is the same. OPTIONS = [nop, finetune]. DEFAULT = NOP",
        required=False,
        type=str,
        default="nop",
    )

    # the default dataset is HK-NEWS, so the default lookahead is:
    # HK-NEWS[TEST_END] - HK-NEWS[TEST_START] = 157 - 106 = 51
    parser.add_argument(
        "-la",
        "--lookahead",
        help="<Optional> Look-ahead / horizon. DEFAULT = 51",
        required=False,
        nargs="+",
        type=int,
        default=[51],
    )

    return parser.parse_args()


def get_dataset_metadata(dataset: str):
    if "hk-news" in dataset:
        dataset_metadata = {
            "dataset": "hk-news",
            "TEST_SETS": [
                "Finance",
                "Entertainment",
                "TravelAndtourism",
                "HealthAndwellness",
                "Sports",
                "Environment",
                "Governance",
            ],
            "TIME_INTERVAL": 1,
            "TEST_START": 106,  # week with number 106 is the first week of 1999
            "TEST_END": 157,  # week with number 156 is the last week of 1999
            "time_interval_type": "time",
            "finetune_type": "base",
            "fipFactory": HKNewsFIP_factory,
            # for all baselines except noFinetune, assume there is
            # a first fine-tune with the first bit of available data
            "lastFinetune": 27,
            "lastFinetune_noFinetune_baseline": 0,
            "weekly_dist_file": f"{BASE_DATA_DIR}hk_news-weekly_topic_distribution.json",  # topic weight distribution per week
            "daily_dist_file": f"{BASE_DATA_DIR}hk_news-daily_topic_distribution.json",  # topic weight distribution per day
        }

        dataset_metadata["base_hf_model_perf_file"] = (
            f"{FID_DIR}tmp_metrics/{dataset}/finetuneInstant_0-dataset_{dataset}-"
            f"fixedTestSetEval-timeIntervalType_{dataset_metadata['time_interval_type']}-timeInterval_{dataset_metadata['TIME_INTERVAL']}-"
            f"finetuneType_{dataset_metadata['finetune_type']}-latest.json"
        )
    else:  # DATASET is OPUS
        dataset_metadata = {
            "dataset": "opus",
            "TEST_SETS": [
                "elitr",
                "elrc",
                "euBookshop",
                "php",
                "tanzil",
                "tedx-fr",
                "tedx-fr_ca",
            ],
            "TIME_INTERVAL": 10000,
            "TEST_START": 730000,
            "TEST_END": 1040000,
            "time_interval_type": "sentence",
            "finetune_type": "base",
            "fipFactory": OpusFIP_factory,
            # for all baselines except noFinetune, assume there is
            # a first fine-tune with the first bit of available data
            "lastFinetune": 10000,
            "lastFinetune_noFinetune_baseline": -1,
        }

        dataset_metadata["base_hf_model_perf_file"] = (
            f"{FID_DIR}tmp_metrics/{dataset}/finetuneInstant_noRetrain-dataset_{dataset}-"
            f"fixedTestSetEval-timeIntervalType_{dataset_metadata['time_interval_type']}-timeInterval_{dataset_metadata['TIME_INTERVAL']}-"
            f"finetuneType_{dataset_metadata['finetune_type']}-latest.json"
        )

    base_fid_file = (
        f"aid-finetune_data"
        f"-dataset_{dataset}-timeInterval_{dataset_metadata['TIME_INTERVAL']}"
        f"-timeIntervalType_{dataset_metadata['time_interval_type']}-"
        f"finetuneType_{dataset_metadata['finetune_type']}.csv"
    )
    # this is the generic fid file, to build generic FIPs
    dataset_metadata["fid_file"] = f"{FID_DIR}general_{base_fid_file}"

    # this is the specific fid file, to build specific FIPs
    dataset_metadata["specific_fid_file"] = f"{FID_DIR}{base_fid_file}"

    return dataset_metadata
