import itertools
import os
import sys
import time

import numpy as np
import pandas as pd

from src.flexico.flexico_utils import (
    MT_METRIC_RESCALING,
    NUM_RANDOM_RUNS,
    get_amount_finetune_data,
    get_dataset_metadata,
    get_finetune_cost,
    get_flexico_parser,
    get_tactic_cost,
    init_fips,
)
from src.flexico.scenarioA.prism_utils import check_adaptation_with_prism
from utils import mkdir_if_not_exists

# """
# RUN adaptiveMT WITH THE FOLLOWING COMMAND:
# time python3.8 src/flexico/scenarioA/run_adaptiveMT_framework.py
#     -d hk-news
#     --fip_metrics "comet22,chrf"
#     -fc 1 5 10 15 20 25
#     -t 0.1 0.15 0.2 0.25 0.3 0.35 0.4 0.45 0.5 0.55 0.6 0.65 0.7 0.75 0.8 0.85 0.9 0.95 1.0
#     -t 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     -b optimum flexico periodic-2 random random-75 exponential reactive-85 sentence sentence-2000
#     --fid_types generic
# """


def get_system_utility(
    tactic: str,
    metrics_delta: dict,
    exp_params: dict,
):
    # System utility
    total_cost = 0

    for mt_metric in exp_params["fipTargetMetrics"]:
        weighted_delta = metrics_delta[mt_metric]["weighted_avg"]
        total_cost += get_tactic_cost(
            tactic=tactic,
            delta_threshold=exp_params["deltaT"],
            finetune_cost=exp_params["finetuneCost"],
            weighted_delta=weighted_delta,
        )

    if "finetune" in tactic:
        total_cost += exp_params["finetuneCost"]

    return total_cost


def get_curr_metric(
    fid_df: pd.DataFrame,
    metric: str,
    curr_time: int,
    last_finetune: int,
):
    return (
        fid_df.loc[
            (fid_df.curr_finetune == curr_time)
            & (fid_df.prev_finetune == last_finetune)
        ][f"curr_{metric}"]
        .copy()
        .reset_index(drop=True)
        .loc[0]
        * MT_METRIC_RESCALING[metric]
    )


def compute_adaptation_benefits(
    exp_params: dict,
    curr_time: int,
    last_finetune: int,
    fid_df: pd.DataFrame,
):
    # compute finetune delta
    topics_delta_dict = {}
    for mt_metric in exp_params["fipTargetMetrics"]:
        topics_delta_dict[mt_metric] = {}
        topics_delta_dict[mt_metric]["weighted_avg"] = 0
        for test_set in TEST_SETS:
            if "test-set" in fid_df.columns:  # generic fips
                test_set_data = (
                    fid_df.loc[
                        (fid_df.curr_finetune == curr_time)
                        & (fid_df.prev_finetune == last_finetune)
                        & (fid_df["test-set"] == test_set)
                    ]
                    .copy()
                    .reset_index(drop=True)
                )
            else:
                test_set_data = (
                    fid_df.loc[
                        (fid_df.curr_finetune == curr_time)
                        & (fid_df.prev_finetune == last_finetune)
                    ]
                    .copy()
                    .reset_index(drop=True)
                )

            if "flexico" in exp_params["baseline"]:
                # get fip model that predicts for the
                # metric under analysis
                fip = exp_params["fips"][mt_metric]
                if isinstance(fip, dict):
                    fip = fip[test_set]
                fip_features = test_set_data[exp_params["fip_features"]]
                assert len(fip_features) == 1, "[E] More than one FID sample!"

                # get fip prediction (expected delta in target metric)
                delta = fip.predict(fip_features)[0] * MT_METRIC_RESCALING[mt_metric]
                preds = []
                for estimator in fip.estimators_:
                    preds.append(
                        estimator.predict(fip_features.to_numpy())[0]
                        * MT_METRIC_RESCALING[mt_metric]
                    )
                assert round(np.mean(preds), 7) == round(
                    delta, 7
                ), f"[E] Average of individual preds {np.mean(preds)} != {delta} fip pred!"
                delta_stdev = np.std(preds)
            else:
                if "test-set" in fid_df.columns:  # generic fips
                    target_col = f"delta-target_test_set_{mt_metric}"
                else:
                    target_col = f"delta-target_{test_set}_{mt_metric}"
                delta = (
                    test_set_data[target_col].loc[0] * MT_METRIC_RESCALING[mt_metric]
                )
                delta_stdev = 0

            topics_delta_dict[mt_metric][test_set] = {
                "perf": delta,
                "stdev": delta_stdev,
            }
            topics_delta_dict[mt_metric]["weighted_avg"] += delta
        topics_delta_dict[mt_metric]["weighted_avg"] = topics_delta_dict[mt_metric][
            "weighted_avg"
        ] / len(TEST_SETS)

    return topics_delta_dict


def plan_adaptation(
    exp_params: dict,
    curr_time: int,
    last_finetune: int,
    fid_df: pd.DataFrame,
    adaptation_benefits: dict,
):
    formal_verification_time = -1
    if USE_PRISM:
        curr_metrics = {}
        for metric in MT_METRIC_RESCALING:
            curr_metrics[metric] = get_curr_metric(
                fid_df=fid_df,
                metric=metric,
                curr_time=curr_time,
                last_finetune=last_finetune,
            )

        adapt, formal_verification_time = check_adaptation_with_prism(
            exp_params=exp_params,
            curr_time=curr_time,
            curr_metrics=curr_metrics,
            adaptation_beneftis=adaptation_benefits,
            test_sets=TEST_SETS,
            verbose=VERBOSE,
        )

    else:
        # System utility
        total_cost_nop = get_system_utility(
            tactic="nop",
            metrics_delta=adaptation_benefits,
            exp_params=exp_params,
        )
        total_cost_finetune = get_system_utility(
            tactic="finetune",
            metrics_delta=adaptation_benefits,
            exp_params=exp_params,
        )
        adapt = bool(total_cost_finetune < total_cost_nop)

    if VERBOSE:
        print(f"[D] Predicted performance: {adaptation_benefits}")
        if not USE_PRISM:
            print(
                f"[D] total-cost-nop={total_cost_nop}   total-cost-finetune={total_cost_finetune}"
            )
        print(f"[D] ==> fine-tuning: {str(adapt).upper()}")

    return adapt, formal_verification_time


def _periodic_adaptation(baseline: str, curr_time: int, last_finetune: int):
    if curr_time == TEST_START:
        adapt = True
    else:
        adapt = True
        if "-" in baseline:
            period = int(baseline.split("-")[1])
            adapt = curr_time - last_finetune == period

    return adapt


def _exponential_adaptation(
    baseline: str, curr_time: int, last_finetune: int, num_adaptations: int
):
    if curr_time == TEST_START:
        adapt = True
    else:
        base = 2
        if "-" in baseline:
            base = int(baseline.split("-")[1])
        adapt = curr_time - last_finetune == pow(base, num_adaptations - 1)

    return adapt


def _random_adaptation(baseline: str, random_generator):
    rand = random_generator.uniform(0, 1)
    prob = 50
    if "-" in baseline:
        prob = int(baseline.split("-")[1])
    adapt = bool(rand > (1.0 - float(prob / 100)))
    return adapt


def _reactive_adaptation(
    baseline: str,
    curr_time: int,
    last_finetune: int,
    fip_target_metrics: list,
    fid_df: pd.DataFrame,
    curr_perf: dict,
):
    adapt = False
    for mt_metric in fip_target_metrics:
        react_threshold = curr_perf[mt_metric]["weighted_avg"]
        if "-" in baseline:
            react_threshold = int(baseline.split("-")[1])
        # avg target metrics perf on the current data
        # fine-tune if the target metric in the current data is below some threshold
        curr_target_metric = get_curr_metric(
            fid_df, mt_metric, curr_time, last_finetune
        )
        if bool(curr_target_metric < react_threshold):
            adapt = True
            break
    return adapt


def _sentence_adaptation(
    baseline: str, fid_df: pd.DataFrame, curr_time: int, last_finetune: int
):
    react_sentences = 1000
    if "-" in baseline:
        react_sentences = int(baseline.split("-")[1])

    curr_new_sentences = (
        fid_df.loc[
            (fid_df.curr_finetune == curr_time)
            & (fid_df.prev_finetune == last_finetune)
        ]["amount_new_data"]
        .copy()
        .reset_index(drop=True)
        .loc[0]
    )
    adapt = bool(curr_new_sentences >= react_sentences)
    return adapt


def analyze_plan_adaptation(
    exp_params: dict,
    fid_df: pd.DataFrame,
    curr_time: int,
    last_finetune: int,
    curr_perf: dict,
    num_adaptations: int,
):
    baseline = exp_params["baseline"]
    formal_verification_time = -1

    if "noFinetune" in baseline or "finetune@1st" in baseline:
        adapt = False
    elif "periodic" in baseline:
        adapt = _periodic_adaptation(baseline, curr_time, last_finetune)
    elif "exponential" in baseline:
        adapt = _exponential_adaptation(
            baseline, curr_time, last_finetune, num_adaptations
        )
    elif "random" in baseline:
        adapt = _random_adaptation(baseline, exp_params["randomGenerator"])
    elif "reactive" in baseline:
        adapt = _reactive_adaptation(
            baseline,
            curr_time,
            last_finetune,
            exp_params["fipTargetMetrics"],
            fid_df,
            curr_perf,
        )
    elif "sentence" in baseline:
        adapt = _sentence_adaptation(baseline, fid_df, curr_time, last_finetune)
    elif "flexico" in baseline or "opt" in baseline or "optimum" in baseline:
        adaptation_benefits = compute_adaptation_benefits(
            exp_params=exp_params,
            curr_time=curr_time,
            last_finetune=last_finetune,
            fid_df=fid_df,
        )
        adapt, formal_verification_time = plan_adaptation(
            exp_params=exp_params,
            curr_time=curr_time,
            last_finetune=last_finetune,
            fid_df=fid_df,
            adaptation_benefits=adaptation_benefits,
        )

    return adapt, formal_verification_time


def get_mt_performance(
    perf_type: str,
    exp_params: dict,
    fid_df: pd.DataFrame,
    no_finetune_metrics: pd.DataFrame,
    curr_time: int,
    last_finetune: int,
):
    perf_dict = {}
    for mt_metric in exp_params["fipTargetMetrics"]:
        perf_dict[mt_metric] = {}
        perf_dict[mt_metric]["weighted_avg"] = 0
        for test_set in TEST_SETS:
            if "noFinetune" in exp_params["baseline"]:
                curr_perf = no_finetune_metrics[f"eval_{test_set}_{mt_metric}"].loc[0]
            else:
                if "test-set" in fid_df.columns:  # generic fips
                    curr_perf = (
                        fid_df.loc[
                            (fid_df.curr_finetune == curr_time)
                            & (fid_df.prev_finetune == last_finetune)
                            & (fid_df["test-set"] == test_set)
                        ][f"curr_test_set_{mt_metric}"]
                        .copy()
                        .reset_index(drop=True)
                        .loc[0]
                    )
                else:
                    curr_perf = (
                        fid_df.loc[
                            (fid_df.curr_finetune == curr_time)
                            & (fid_df.prev_finetune == last_finetune)
                        ][f"curr_{test_set}_{mt_metric}"]
                        .copy()
                        .reset_index(drop=True)
                        .loc[0]
                    )

            if "curr" in perf_type:
                perf = curr_perf
            else:  # real delta performance (target_perf - curr_perf)
                if "test-set" in fid_df.columns:  # generic fips
                    target_perf = (
                        fid_df.loc[
                            (fid_df.curr_finetune == curr_time)
                            & (fid_df["test-set"] == test_set)
                        ][f"target_test_set_{mt_metric}"]
                        .copy()
                        .reset_index(drop=True)
                        .loc[0]
                    )
                else:
                    target_perf = (
                        fid_df.loc[(fid_df.curr_finetune == curr_time)][
                            f"target_{test_set}_{mt_metric}"
                        ]
                        .copy()
                        .reset_index(drop=True)
                        .loc[0]
                    )
                perf = target_perf - curr_perf

            perf = perf * MT_METRIC_RESCALING[mt_metric]
            perf_dict[mt_metric][test_set] = {
                "perf": perf,
                "stdev": 0,
            }
            perf_dict[mt_metric]["weighted_avg"] += perf
        perf_dict[mt_metric]["weighted_avg"] = perf_dict[mt_metric][
            "weighted_avg"
        ] / len(TEST_SETS)

    return perf_dict


def main(
    exp_params: dict,
    fid_df: pd.DataFrame,
    no_finetune_metrics: pd.DataFrame,
):
    adaptation_moments = []

    res_dict = {}
    res_dict["formalVerificationLatencies"] = []
    res_dict["mapeLatencies"] = []
    res_dict["adaptations"] = []
    res_dict["cost"] = []

    last_finetune = exp_params["lastFinetune"]
    # SIMULATE FRAMEWORK
    for curr_time in range(TEST_START, TEST_END, TIME_INTERVAL):
        if VERBOSE:
            print(f"------------------- curr-time {curr_time} -------------------")

        latency_start_time = time.time()
        # MONITOR: compute current performance
        curr_test_set_perf = get_mt_performance(
            perf_type="curr",
            exp_params=exp_params,
            fid_df=fid_df,
            no_finetune_metrics=no_finetune_metrics,
            curr_time=curr_time,
            last_finetune=last_finetune,
        )

        # update fine-tune cost
        if "linear" in exp_params["finetuneCostType"]:
            exp_params["finetuneCost"] = get_finetune_cost(
                get_amount_finetune_data(fid_df, curr_time, exp_params["lastFinetune"]),
                DATASET,
            )

        # ANALYZE & PLAN: check whether to fine-tune
        adapt, formal_verification_time = analyze_plan_adaptation(
            exp_params=exp_params,
            fid_df=fid_df,
            curr_time=curr_time,
            last_finetune=last_finetune,
            curr_perf=curr_test_set_perf,
            num_adaptations=sum(res_dict["adaptations"]),
        )

        # EXECUTE
        real_delta = get_mt_performance(
            perf_type="real_delta",
            exp_params=exp_params,
            fid_df=fid_df,
            no_finetune_metrics=no_finetune_metrics,
            curr_time=curr_time,
            last_finetune=last_finetune,
        )
        curr_cost = 0
        curr_tactic = "nop"
        if adapt:
            if VERBOSE:
                print("[D] ### FINE-TUNING ###")
            last_finetune = curr_time
            adaptation_moments.append(curr_time)
            curr_tactic = "finetune"

        curr_cost = get_system_utility(
            tactic=curr_tactic,
            metrics_delta=real_delta,
            exp_params=exp_params,
        )

        # update results dict
        res_dict["mapeLatencies"].append(time.time() - latency_start_time)
        res_dict["formalVerificationLatencies"].append(formal_verification_time)
        res_dict["adaptations"].append(adapt)
        res_dict["cost"].append(curr_cost)
        if VERBOSE:
            print(f"[D] Real performance: {real_delta}")
            print(f"[D] Current cost: {curr_cost}")
            print(f"[D] Total cumulative cost: {sum(res_dict['cost'])}")

    print(f"[D] Executed {sum(res_dict['adaptations'])} fine-tunings!")
    print(f"[D]\tFine-tuned {len(adaptation_moments)} times @ {adaptation_moments}")
    print(f"[D]\tTotal-cost is {sum(res_dict['cost'])}")

    res_dict["totalAdaptations"] = sum(res_dict["adaptations"])
    res_dict["totalCost"] = sum(res_dict["cost"])

    return res_dict


def create_experiments_list(
    program_args, mt_dataset_metadata, generic_fid_df, specific_fid_df
):
    prod = itertools.product(
        list(set(program_args.baselines)),
        list(set(program_args.finetune_costs)),
        list(program_args.fip_metrics),
        list(set(program_args.thresholds)),
        list(program_args.finetune_cost_type),
        list(set(program_args.fid_types)),
    )

    experiments_list = []
    for b, fc, fip_m, t, fc_type, fid_type in prod:
        if "noFinetune" in b:
            lf = mt_dataset_metadata["lastFinetune_noFinetune_baseline"]
        else:
            lf = mt_dataset_metadata["lastFinetune"]

        if "flexico" in b:
            for fip_model_type, feature_set in itertools.product(
                list(set(program_args.fip_models)),
                list(set(program_args.fip_features)),
            ):
                exp_args = {
                    "baseline": b,
                    "finetuneCostType": fc_type,
                    "finetuneCost": fc,
                    "deltaT": t,
                    "fipTargetMetrics": fip_m,
                    "lastFinetune": lf,
                }
                exp_args["fipModel"] = fip_model_type  # [rf, lin, xgb] model
                exp_args["fidType"] = fid_type  # specific or generic FIP
                exp_args["fipFeatureSet"] = feature_set

                init_fips(
                    exp_args=exp_args,
                    fid_df=generic_fid_df if "generic" in fid_type else specific_fid_df,
                    fip_model_type=fip_model_type,
                    fid_type=fid_type,
                    feature_set=feature_set,
                    metadata=mt_dataset_metadata,
                    test_sets=TEST_SETS,
                    verbose=VERBOSE,
                )

                experiments_list.append(exp_args)
        else:
            exp_args = {
                "baseline": b,
                "finetuneCostType": fc_type,
                "finetuneCost": fc,
                "deltaT": t,
                "fipTargetMetrics": fip_m,
                "lastFinetune": lf,
                "fipModel": "None",
                "fipFeatureSet": "None",
                "fidType": fid_type,
            }
            experiments_list.append(exp_args)
    return experiments_list


if __name__ == "__main__":
    prog_args = get_flexico_parser()

    global VERBOSE
    global USE_PRISM
    global DATASET
    global TEST_SETS
    global TIME_INTERVAL
    global TEST_START
    global TEST_END

    VERBOSE = bool(prog_args.verbose)
    USE_PRISM = bool(prog_args.prism)
    DATASET = prog_args.dataset

    dataset_metadata = get_dataset_metadata(DATASET)
    TEST_SETS = dataset_metadata["TEST_SETS"]
    TIME_INTERVAL = dataset_metadata["TIME_INTERVAL"]
    TEST_START = dataset_metadata["TEST_START"]
    TEST_END = dataset_metadata["TEST_END"]

    # INITIALIZE RESULTS DICTS
    results_dict = {
        "runID": [],
        "baseline": [],
        "finetuneCostType": [],
        "finetuneCost": [],
        "deltaT": [],
        "fipTargetMetrics": [],
        "fipModel": [],
        "fidType": [],
        "fipFeatureSet": [],
    }

    # load adaptation impact dataset
    generic_fid = pd.read_csv(dataset_metadata["fid_file"])
    specific_fid = pd.read_csv(dataset_metadata["specific_fid_file"])

    # load base hugging-face model performance a.k.a. no-finetune
    noFinetune_metrics = pd.read_json(dataset_metadata["base_hf_model_perf_file"])

    # if VERBOSE:
    #     print(f"[D] FID columns:\n{list(fid.columns)}")

    # create experiments' list
    print("[D] Creating list of experiments...")
    exp_list = create_experiments_list(
        prog_args, dataset_metadata, generic_fid, specific_fid
    )

    # run all experiments
    for count, args in enumerate(exp_list):
        print(
            "#" * 60
            + f" EXPERIMENT {count + 1} / {len(exp_list)} ({round((count+1)/len(exp_list)*100, 2)}%) "
            + "#" * 60
        )
        fc_print = f"finetune-cost-type={args['finetuneCostType']}"
        if "linear" not in args["finetuneCostType"]:
            fc_print = fc_print + f"   finetune-cost={args['finetuneCost']}"
        print(
            f"[D] dataset={DATASET}   baseline={args['baseline']}   {fc_print}   "
            f"delta-threshold={args['deltaT']}   "
            f"fip-metrics={args['fipTargetMetrics']}   fip-model={args['fipModel']}   "
            f"fid-type={args['fidType']}   fip-feature-set={args['fipFeatureSet']}"
        )

        if "random" in args["baseline"]:
            num_runs = NUM_RANDOM_RUNS
        else:
            num_runs = 1

        for seed in range(1, num_runs + 1):
            print("-" * 30 + f" run # {seed} / {num_runs} " + "-" * 30)
            args["randomGenerator"] = np.random.default_rng(seed=seed)
            fid = generic_fid if "generic" in args["fidType"] else specific_fid
            res = main(args, fid, noFinetune_metrics)

            # update results dict
            results_dict["runID"].append(seed)
            for k, v in res.items():
                if k not in results_dict:
                    results_dict[k] = []
                results_dict[k].append(v)

            for arg_name, arg_val in args.items():
                if arg_name in results_dict:
                    results_dict[arg_name].append(arg_val)

    # save results
    res_dir = f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/framework_results/"
    # create dir if it does not exist yet
    mkdir_if_not_exists(res_dir)
    res_file = "adaptiveMT-base"
    for arg_name, arg_val in vars(prog_args).items():
        if "verbose" in arg_name:
            continue

        # snake-case to camel-case
        if "_" in arg_name:
            arg_name = arg_name.split("_")[0] + "".join(
                x.capitalize() or "_" for x in arg_name.split("_")[1:]
            )

        if not isinstance(arg_val, list):
            res_file = res_file + f"-{arg_name}_{arg_val}"
    res_file = res_file + ".csv"
    print(f"[D] Results saved to:\n\t{res_dir + res_file}")
    pd.DataFrame(results_dict).to_csv(res_dir + res_file, index=False)
