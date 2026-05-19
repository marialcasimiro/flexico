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

AVAILABILITY_ZONES = [
    'asia-south1', 
    'australia-southeast1', 
    'europe-central2', 
    'me-west1', 
    'us-east1', 
]

SPOT_INSTANCE_TYPE = "n1-standard-2"

GPU_TDP = 0.3  # kW consumed by the spot instance with GPU (NVIDIA Tesla V100) at max load


SPOT_INSTANCE_DATA = BASE_DATA_DIR + "spotLake_data/gcp/gcp-parsed.csv"
ENERGY_DATA = BASE_DATA_DIR + "region_carbon_over_time-parsed.csv"


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

def estimate_finetune_duration(amount_finetune_data: float, dataset: str):
    if "hk-news" not in dataset:
        print(f"[W] Unknown dataset {dataset} -- using hk-news equation!")
    return (0.00507 * amount_finetune_data + 2.93) / 3600  # in hours


def get_finetune_cost(amount_finetune_data: float, dataset: str, cost_type: str):
    if "opus" in dataset:
        finetune_cost = 1.738572434032605e-06 * amount_finetune_data + 0.00043869073882248506
    else:
        if "hk-news" not in dataset:
            print(f"[W] Unknown dataset {dataset} -- using hk-news equation!")
        estimated_finetune_duration = 0.00517 * amount_finetune_data - 2.11  # in seconds
        
        # based on this calculator: https://calculator.green-algorithms.org/
        # 1 NVIDIA tesla V100 GPU with 32GB hosted in PA, USA consumes 2.70 gCO2e and 8.11 Wh per minute
        # assuming the consumption rate remains constant, we can compute the 
        # gCO2e and Wh for each finetune based on how long it took

        if 'gco2e' in cost_type.lower():
            finetune_cost = (estimated_finetune_duration * 2.70) / 60
        elif 'wh' in cost_type.lower():
            finetune_cost = (estimated_finetune_duration * 8.11) / 60
        else:
            print(f"[W] Unknown finetune cost type {cost_type} -- using Wh!")
            finetune_cost = (estimated_finetune_duration * 8.11) / 60

    return finetune_cost


def get_tactic_cost(
    tactic: str,
    finetune_cost: float,
    delta_threshold: float,
    fip_delta: float,
):
    tactic_cost = 0
    if "nop" in tactic and fip_delta >= delta_threshold:    # missed oportunity cost
        tactic_cost = fip_delta * 2 * finetune_cost
    
    if "finetune" in tactic and fip_delta < delta_threshold:     # regret cost
        tactic_cost = (delta_threshold - fip_delta) * 2 * finetune_cost

    return tactic_cost


def get_az_cost(finetuneDuration: float, availability_zones: dict, az: str):
    return (
        availability_zones[az]['spot_price'] 
        * (1 - availability_zones[az]['gcfe']/100)
        * availability_zones[az]['grid_co2']
        * finetuneDuration * GPU_TDP
    )


def get_opt_az(finetuneDuration: float, availability_zones: dict):

    min_az_cost = np.inf
    selected_az = ''
    for az in availability_zones:
        az_cost = get_az_cost(finetuneDuration, availability_zones, az)
        if az_cost < min_az_cost:
            min_az_cost = az_cost
            selected_az = az
    
    return selected_az, min_az_cost

def get_az_data(spot_df, energy_df, curr_week):
    spot_data = spot_df.loc[
        spot_df['continuous_week'] == curr_week
    ]
    energy_data = energy_df.loc[
        (energy_df['day'] >= spot_data['day'].min())
        & (energy_df['day'] <= spot_data['day'].max())
    ]
    
    if len(energy_data) == 0: 
        # there is no data for this exact week
        # let us get the last data collected 
        # (closest to the desired date)
        last_day = energy_df.loc[
            energy_df['day'] <= spot_data['day'].min()
        ]['day'].max()
        last_energy_week = energy_df.loc[
            energy_df['day'] == last_day
        ]['continuous_week'].to_numpy()[0]
        energy_data = energy_df.loc[
            energy_df.continuous_week == last_energy_week
        ]

    az_data = {}
    for az in AVAILABILITY_ZONES:
        az_data[az] = {
            'spot_price': np.mean(spot_data.loc[spot_data.Region == az]['Spot Price'].to_numpy()),
            'gcfe': np.mean(energy_data.loc[energy_data['Google Cloud Region'] == az]['G_CFE'].to_numpy()),
            'grid_co2': np.mean(energy_data.loc[energy_data['Google Cloud Region'] == az]['grid_CO2'].to_numpy()),
        }

    return az_data


def get_burnt_co2(
    adapt: bool, missedOportunityCost: float, finetuneRegretCost: float, energy_consumption: float
):
    # check CO2 emissions
    # if tactic = nop & finetuneRegretCost >= missedOportunityCost ==> 0 CO2 emissions
    correct_co2 = 0
    incorrect_co2 = 0
    unburnt_co2 = 0
    if adapt:
        if finetuneRegretCost < missedOportunityCost:   # necessary CO2
            correct_co2 = energy_consumption
        if finetuneRegretCost >= missedOportunityCost:  # unnecessary/wrong CO2
            incorrect_co2 = energy_consumption
    else:
        if finetuneRegretCost < missedOportunityCost:
            # ==> missed-oportunity ==> you should have burned CO2
            unburnt_co2 = energy_consumption

    return correct_co2, incorrect_co2, unburnt_co2


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
        f"fid-finetune_data"
        f"-dataset_{dataset}-timeInterval_{dataset_metadata['TIME_INTERVAL']}"
        f"-timeIntervalType_{dataset_metadata['time_interval_type']}-"
        f"finetuneType_{dataset_metadata['finetune_type']}.csv"
    )
    # this is the generic fid file, to build generic FIPs
    dataset_metadata["fid_file"] = f"{FID_DIR}general_{base_fid_file}"

    # this is the specific fid file, to build specific FIPs
    dataset_metadata["specific_fid_file"] = f"{FID_DIR}{base_fid_file}"

    return dataset_metadata
