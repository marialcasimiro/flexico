#!/usr/bin/env python3

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

from src.flexico.flexico_utils_scenariosC_D import (
    MT_METRIC_RESCALING,
    SPOT_INSTANCE_TYPE,
    AVAILABILITY_ZONES,
    SPOT_INSTANCE_DATA,
    ENERGY_DATA,
    GPU_TDP,
    get_amount_finetune_data,
    get_daily_topic_weights,
    get_dataset_metadata,
    get_opt_baseline_parser,
    get_tactic_cost,
    get_opt_az,
    get_az_data,
    get_burnt_co2,
    estimate_finetune_duration,
)
from constants import TMP_METRICS_DIR


# """
# RUN WITH THE FOLLOWING COMMAND:
# time python src/flexico/scenarioD/run_global_optimum_baseline.py
#     -d hk-news
#     --fip_metrics "comet22,chrf"
#     -t 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
#     -la 5
# """



def load_metrics_files(path: str):
    files = {}

    for f in os.listdir(path):
        if (
            "finetuneInstant" in f
            and "fixedTestSetEval" in f
            and (
                f.endswith("finetuneType_base-latest.json")
                or f.endswith(
                    "finetuneType_base-percentNewData_1.0-percentOldData_0.0-latest.json"
                )
            )
        ):
            if "noRetrain" in f:
                key = "instant--1"
            else:
                key = f"instant-{int(f.split('-')[0].split('_')[1])}"
            files[key] = pd.read_json(path + f)

    return files


def get_system_utility(tactics: dict, node: dict):
    # System utility
    total_cost = 0

    finetune_duration = estimate_finetune_duration(node.finetune_data, DATASET)
    _, finetune_cost = get_opt_az(
        finetune_duration, 
        node.availability_zones,
    )

    adapt = False
    for day in node.topic_weights.keys():
        for metric in FIP_METRICS:
            weighted_delta = 0
            for test_set in TEST_SETS:
                weighted_delta += (
                    node.topic_weights[day][test_set] * node.delta[metric][test_set]
                )
            if tactics[day]:
                adapt = True
            total_cost += get_tactic_cost(
                tactic="finetune" if adapt else "nop",
                finetune_cost=finetune_cost,
                delta_threshold=DELTA_THRESHOLD,
                fip_delta=weighted_delta,
            )

    if adapt:
        total_cost += finetune_cost

    return total_cost


def get_metric(
    metrics: pd.DataFrame,  # dataframe that contains the metrics for a given model
    target_metric: str,  # metric whose value we want to know
):
    curr_perf = {}
    for test_set in TEST_SETS:
        curr_perf[test_set] = (
            metrics[f"eval_{test_set}_{target_metric}"].loc[0]
            * MT_METRIC_RESCALING[target_metric]
        )
    return curr_perf

def get_deltas(
    curr_time: int,
    start_date: int,
    prev_metrics: dict,
    metrics_files_dict,
):
    curr_metrics = {}
    deltas_dict = {}
    for metric in FIP_METRICS:
        curr_metrics[metric] = get_metric(
            metrics_files_dict[f"instant-{curr_time}"], metric
        )
        deltas_dict[metric] = {}
        for test_set in TEST_SETS:
            deltas_dict[metric][test_set] = round(
                curr_metrics[metric][test_set] - prev_metrics[metric][test_set], 5
            )
            if deltas_dict[metric][test_set] == 0:
                # delta is 0 when the current tactic is nop
                # but to compute the missing opportunity
                # cost we need the finetune delta
                deltas_dict[metric][test_set] = round(
                    get_metric(
                        metrics_files_dict[f"instant-{start_date + TACTIC_LATENCY * TIME_INTERVAL}"],
                        metric,
                    )[test_set]
                    - prev_metrics[metric][test_set],
                    5,
                )

    return deltas_dict, curr_metrics

class Node:
    def __init__(self, children, metric, delta, depth, last_finetune, finetune_data, availability_zones, topic_weights, tactics_per_day):
        self.children = children
        self.metric = metric
        self.delta = delta  # how much we expect to improve/degrade if we finetune
        self.depth = depth
        self.last_finetune = last_finetune
        self.finetune_data = finetune_data
        self.availability_zones = availability_zones
        self.topic_weights = topic_weights
        self.tactics_per_day = tactics_per_day

    def print_tree(self):
        self._print_tree_recursive(self, 0)

    def _print_tree_recursive(self, node, depth):
        if node is not None:
            print("    " * depth, end="")
            if node.children is None:
                print(
                    f"(Leaf Node) {FIP_METRICS}: {node.metric}   delta={node.delta}\t(last-finetune={node.last_finetune})"
                )
            else:
                print(
                    f"{FIP_METRICS}: {node.metric}   delta={node.delta}\t(last-finetune={node.last_finetune})"
                )
                for child in node.children:
                    self._print_tree_recursive(child, depth + 1)
                

def lowestCostLeaf(tree_node, cost, depth):
    if tree_node.children[0] is None:
        if VERBOSE:
            print(f"leaf cost = {cost}")
        return [cost, None, {}]

    tactics_per_day = None
    tactics_dict = None
    path_cost = np.inf
    for child_node in tree_node.children:
        node_cost = get_system_utility(child_node.tactics_per_day, tree_node)
        childCost, _, childTacticsDict = lowestCostLeaf(
            child_node, cost + node_cost, depth + 1
        )
        if childCost < path_cost:
            path_cost = childCost
            tactics_per_day = child_node.tactics_per_day
            tactics_dict = childTacticsDict

    tactics_dict[depth] = {"tactics": tactics_per_day, "path_cost": path_cost}

    if depth == 0:
        return [path_cost, tactics_per_day, tactics_dict]

    return [path_cost, None, tactics_dict]


def buildTree(
    start_date,
    last_finetune,
    finetune_ongoing,
    finetune_state,
    tactics_per_day,
    depth,
    metrics_files_dict,
    fid_df,
    spot_instance_df, 
    energy_df,
    weights_dist,
):
    # before any update due to the current tactic's execution
    # compute the performance metric for the "non-updated" (current) model
    prev_metrics_dict = {}
    for metric in FIP_METRICS:
        prev_metrics_dict[metric] = get_metric(
            metrics_files_dict[f"instant-{last_finetune}"], metric
        )

    if [
        day for day, tactic in tactics_per_day.items() if tactic
    ] and not finetune_ongoing:
        finetune_ongoing = True
        finetune_state = TACTIC_LATENCY

    if finetune_ongoing:
        if finetune_state == 0:
            last_finetune = start_date + TACTIC_LATENCY * TIME_INTERVAL
            finetune_ongoing = False
        else:
            finetune_state -= 1

    deltas_dict, curr_metrics_dict = get_deltas(
        curr_time=last_finetune,
        start_date=start_date,
        prev_metrics=prev_metrics_dict,
        metrics_files_dict=metrics_files_dict,
    )
        
    if VERBOSE:
        print(
            f"[curr-time={start_date + depth * TIME_INTERVAL}] "
            f"tactic={tactics_per_day} "
            f"metrics-file=instant-{last_finetune} "
            f"depth={depth} "
            f"max-depth={LOOKAHEAD}"
        )

    curr_time = start_date + depth * TIME_INTERVAL
    finetune_data = get_amount_finetune_data(fid_df, curr_time, last_finetune)
    spot_instance_data_weeks = spot_instance_df.continuous_week.unique()[-(END_DATE-START_DATE+1):]
    availability_zones=get_az_data(
        spot_instance_df, energy_df, spot_instance_data_weeks[curr_time-START_DATE]
    )
    topic_weights = get_daily_topic_weights(
        curr_time=curr_time,
        topic_weight_type=WEIGHT_TYPE,
        daily_weights=weights_dist,
        test_sets=TEST_SETS,
    )

    children_nodes = []
    if depth == LOOKAHEAD or (start_date + (depth * TIME_INTERVAL)) == END_DATE:
        for tactic_comb in range(8):
            children_nodes.append(None)
    else:
        # 8 tactic combinations:
        # - adapt at each day of the week ==> 7 combinations
        # - never adapt ==> 1 combination
        for tactic_comb in range(8):
            tactics = dict.fromkeys(topic_weights.keys(), False)
            if tactic_comb > 0:
                tactics[f"{tactic_comb}"] = True

            children_nodes.append(
                buildTree(
                    start_date=start_date,
                    last_finetune=last_finetune,
                    finetune_ongoing=finetune_ongoing,
                    finetune_state=finetune_state,
                    tactics_per_day=tactics,
                    depth=depth + 1,
                    metrics_files_dict=metrics_files_dict,
                    fid_df=fid_df,
                    spot_instance_df=spot_instance_df, 
                    energy_df=energy_df,
                    weights_dist=weights_dist,
                )
            )
    return Node(
        children_nodes,
        curr_metrics_dict,
        deltas_dict,
        depth,
        last_finetune,
        finetune_data,
        availability_zones,
        topic_weights,
        tactics_per_day,
    )


def main(metricsFiles, fid_df, res_dict, spot_instance_df, energy_df, daily_weights):
    adaptations = []
    adaptationsWeekDay = []
    finetune_instants = []
    real_metric_val = []
    cost = []
    az = []       # which availability zone was selected
    az_cost = []  # spot instance price for the selected az
    az_gcfe = []  # google-CFE for the selected az
    az_co2 = []   # grid co2 for the selected az
    finetune_durations = []
    correct_co2 = []
    incorrect_co2 = []
    unburnt_co2 = []
    nopSysU = []
    finetuneSysU = []


    last_finetune = LAST_FINETUNE
    spot_instance_data_weeks = spot_instance_df.continuous_week.unique()[-(END_DATE-START_DATE+1):]
    for curr_time in range(START_DATE, END_DATE, TIME_INTERVAL):
        #if VERBOSE:
        print("#" * 50 + f" curr-time={curr_time} " + "#" * 50)

        metrics_val_dict = {}
        curr_mt_perf = {}
        deltas_dict = {}
        for metric in FIP_METRICS:
            # performance of the current MT model evaluated on the fixed test-sets
            metrics_val_dict[metric] = get_metric(
                metricsFiles[f"instant-{last_finetune}"], metric
            )
            # performance of the MT model finetuned now evaluated on the fixed test-sets
            curr_mt_perf[metric] = get_metric(
                metricsFiles[f"instant-{curr_time}"], metric
            )
            deltas_dict[metric] = {}
            for test_set in TEST_SETS:
                deltas_dict[metric][test_set] = round(
                    curr_mt_perf[metric][test_set] - metrics_val_dict[metric][test_set],
                    5,
                )
        real_metric_val.append(metrics_val_dict)

        daily_topic_weights = get_daily_topic_weights(
            curr_time=curr_time,
            topic_weight_type=WEIGHT_TYPE,
            daily_weights=daily_weights,
            test_sets=TEST_SETS,
        )

        global LOOKAHEAD
        if curr_time > END_DATE - LOOKAHEAD and LOOKAHEAD > 0:
            LOOKAHEAD = LOOKAHEAD - 1
            # continue
        if curr_time > END_DATE - LOOKAHEAD and LOOKAHEAD == 0:
            continue

        children_nodes = []
        # 8 tactic combinations:
        # - adapt at each day of the week ==> 7 combinations
        # - never adapt ==> 1 combination
        for tactic_comb in range(8):
            tactics = dict.fromkeys(daily_topic_weights.keys(), False)
            if tactic_comb > 0:
                tactics[f"{tactic_comb}"] = True

            children_nodes.append(
                buildTree(
                    start_date=curr_time,
                    last_finetune=last_finetune,
                    finetune_ongoing=False,
                    finetune_state=0,
                    tactics_per_day=tactics,
                    depth=1,
                    metrics_files_dict=metricsFiles,
                    fid_df=fid_df,
                    spot_instance_df=spot_instance_df, 
                    energy_df=energy_df,
                    weights_dist=daily_weights,
                )
            )

        root = Node(
            children=children_nodes,
            metric=metrics_val_dict,
            delta=deltas_dict,
            depth=0,
            last_finetune=last_finetune,
            finetune_data=get_amount_finetune_data(fid_df, curr_time, last_finetune),
            availability_zones=get_az_data(
                spot_instance_df, energy_df, spot_instance_data_weeks[curr_time-START_DATE]
            ),
            topic_weights=daily_topic_weights,
            tactics_per_day=dict.fromkeys(daily_topic_weights.keys(), False),
        )

        if VERBOSE:
            root.print_tree()

        nextTactic = lowestCostLeaf(root, 0, 0)
        tactics_per_day = nextTactic[1]
        adaptation_day = [k for k, v in tactics_per_day.items() if v]
        curr_cost = get_system_utility(tactics_per_day, root)
        selected_az, _ = get_opt_az(
            estimate_finetune_duration(root.finetune_data, DATASET), 
            root.availability_zones,
        )
        
        if adaptation_day:
            if VERBOSE:
                print(f"Going to FINETUNE --- leaf-cost={nextTactic[0]}")
            adaptations.append(True)
            adaptationsWeekDay.append(int(adaptation_day[0]))
            last_finetune = curr_time
            finetune_instants.append(curr_time)
        else:
            adaptations.append(False)
            adaptationsWeekDay.append(0)

        # check CO2 emissions
        # consider the fine-tune child that leads to the lowest overall cost
        finetune_child_cost = np.inf
        for child in root.children:
            child_cost = lowestCostLeaf(child, get_system_utility(child.tactics_per_day, root), 1)[0]
            if any(child.tactics_per_day.values()):
                finetune_child_cost = min(finetune_child_cost, child_cost)
            else:
                nop_child_cost = child_cost
        az_data = root.availability_zones[selected_az]
        finetune_duration = estimate_finetune_duration(root.finetune_data, DATASET)
        correctCO2, incorrectCO2, unburntCO2 = get_burnt_co2(
            adapt=adaptations[-1],
            missedOportunityCost=nop_child_cost,
            finetuneRegretCost=finetune_child_cost - (az_data['spot_price'] * finetune_duration),
            energy_consumption=(1-az_data['gcfe']/100) * az_data['grid_co2'] * finetune_duration * GPU_TDP
        )

        # save data
        cost.append(curr_cost)
        az.append(selected_az)
        az_cost.append(root.availability_zones[selected_az]['spot_price'])
        az_gcfe.append(root.availability_zones[selected_az]['gcfe'])
        az_co2.append(root.availability_zones[selected_az]['grid_co2'])
        finetune_durations.append(finetune_duration)
        correct_co2.append(correctCO2)
        incorrect_co2.append(incorrectCO2)
        unburnt_co2.append(unburntCO2)
        nopSysU.append(nop_child_cost)
        finetuneSysU.append(finetune_child_cost)

        if (END_DATE - START_DATE - (LOOKAHEAD * TIME_INTERVAL)) == 0:
            print(f"Optimal strategy: {nextTactic[2]}")

    print(f"[D] Executed {sum(adaptations)} fine-tunings!")
    print(f"[D]\tFine-tuned {len(finetune_instants)} times @ {finetune_instants}")
    print(f"[D]\tTotal-cost is {np.cumsum(cost)[-1]}")
    
    res_dict["adaptations"].append(adaptations)
    res_dict["adaptationsWeekDay"].append(adaptationsWeekDay)
    res_dict["totalAdaptations"].append(len(finetune_instants))
    res_dict["totalCost"].append(np.cumsum(cost)[-1])
    res_dict["real_metric_val"].append(real_metric_val)
    res_dict["cost"].append(cost)
    res_dict["az"].append(az)
    res_dict["az_cost"].append(az_cost)
    res_dict["az_gcfe"].append(az_gcfe)
    res_dict["az_co2"].append(az_co2)
    totalSpotCost = 0
    totalEnergyCost = 0
    for idx, adaptation in enumerate(adaptations):
        if adaptation:
            totalSpotCost += az_cost[idx] * finetune_durations[idx]
            totalEnergyCost += (
                (1-az_gcfe[idx]/100) * az_co2[idx] * finetune_durations[idx] * GPU_TDP
            )
    res_dict['totalSpotCost'].append(totalSpotCost)
    res_dict['totalEnergyCost'].append(totalEnergyCost)

    res_dict["correctCO2"].append(correct_co2)
    res_dict["incorrectCO2"].append(incorrect_co2)
    res_dict["unburntCO2"].append(unburnt_co2)
    res_dict["totalCorrectCO2"].append(sum(correct_co2))
    res_dict["totalIncorrectCO2"].append(sum(incorrect_co2))
    res_dict["totalUnburntCO2"].append(sum(unburnt_co2))
    res_dict["nopSysU"].append(nopSysU)
    res_dict["finetuneSysU"].append(finetuneSysU)

    print(f"totalCorrectCO2={sum(correct_co2)}   totalEnergy={totalEnergyCost}")

if __name__ == "__main__":
    prog_args = get_opt_baseline_parser()

    global DEFAULT_TACTIC
    DEFAULT_TACTIC = prog_args.default_tactic.lower()
    global VERBOSE
    VERBOSE = bool(prog_args.verbose)
    global DATASET
    DATASET = prog_args.dataset

    if "opus" in DATASET:
        print(f"[D] Dataset {DATASET} currently not set-up for this use-case.")
        sys.exit(0)

    dataset_metadata = get_dataset_metadata(DATASET)

    global TIME_INTERVAL
    TIME_INTERVAL = dataset_metadata["TIME_INTERVAL"]
    global TEST_SETS
    TEST_SETS = dataset_metadata["TEST_SETS"]
    global START_DATE
    START_DATE = dataset_metadata["TEST_START"]
    global END_DATE
    END_DATE = dataset_metadata["TEST_END"]
    global LAST_FINETUNE
    LAST_FINETUNE = dataset_metadata["lastFinetune"]

    # load adaptation impact dataset
    fid = pd.read_csv(dataset_metadata["fid_file"])

    # load metrics files
    metrics_files = load_metrics_files(
        path=f"{TMP_METRICS_DIR}{DATASET.lower()}/"
    )

    # load energy and spot instance data
    spot_instance_df = pd.read_csv(SPOT_INSTANCE_DATA)
    spot_instance_df = spot_instance_df.loc[
        (spot_instance_df.Region.isin(AVAILABILITY_ZONES))
        & (spot_instance_df.InstanceType==SPOT_INSTANCE_TYPE)
    ]
    energy_df = pd.read_csv(ENERGY_DATA)
    energy_df = energy_df.loc[energy_df['Google Cloud Region'].isin(AVAILABILITY_ZONES)]

    # load daily topic distributions
    with open(dataset_metadata["daily_dist_file"], encoding="utf-8") as f_in:
        daily_dist = json.load(f_in)

    # initialize results dict
    results_dict = {
        "baseline": [],
        "lookahead": [],
        "deltaT": [],
        "fipTargetMetrics": [],
        "adaptations": [],
        "totalAdaptations": [],
        "totalCost": [],
        "real_metric_val": [],
        "cost": [],
        "az": [],
        "az_cost": [],
        "az_gcfe": [],
        "az_co2": [],
        "totalSpotCost": [],
        "totalEnergyCost": [],
        "correctCO2": [],
        "incorrectCO2": [],
        "unburntCO2": [],
        "totalCorrectCO2": [],
        "totalIncorrectCO2": [],
        "totalUnburntCO2": [],
        "nopSysU": [],
        "finetuneSysU": [],
        "topicWeightType": [],
        "adaptationsWeekDay": [],
    }

    # create experiments' list
    prod = itertools.product(
        list(set(prog_args.lookahead)),
        list(set(prog_args.thresholds)),
        list(prog_args.fip_metrics),
        list(set(prog_args.topic_weights_types)),
    )
    exp_list = []
    for la, delta_t, fip_m, tw_type in prod:
        exp_list.append(
            {
                "lookahead": la,
                "deltaT": delta_t,
                "fipTargetMetrics": fip_m,
                "topicWeightType": tw_type,
            }
        )

    # run all experiments
    for exp_counter, exp_args in enumerate(exp_list):
        # update global variables
        global LOOKAHEAD
        LOOKAHEAD = exp_args["lookahead"]
        global DELTA_THRESHOLD
        DELTA_THRESHOLD = exp_args["deltaT"]
        global FIP_METRICS
        FIP_METRICS = exp_args["fipTargetMetrics"]
        global TACTIC_LATENCY
        TACTIC_LATENCY = 0
        global WEIGHT_TYPE
        WEIGHT_TYPE = exp_args["topicWeightType"]

        print(
            "#" * 75
            + f" EXPERIMENT {exp_counter + 1} / {len(exp_list)} ({round((exp_counter + 1)/len(exp_list)*100, 2)}%) "
            + "#" * 75
        )
        
        print(
            f"la={LOOKAHEAD}   DELTA-T={DELTA_THRESHOLD}   "
            f"FIP-METRIC={FIP_METRICS}   finetune-latency={TACTIC_LATENCY}   topic_weights_type={WEIGHT_TYPE}"
        )

        main(metrics_files, fid, results_dict, spot_instance_df, energy_df, daily_dist)

        results_dict["baseline"].append(f"opt-{exp_args['lookahead']}")
        for arg_name, arg_val in exp_args.items():
            results_dict[arg_name].append(arg_val)

    # save results
    res_dir = f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/framework_results/"
    res_file = (
        f"{res_dir}optimum_baseline-topicWeightsDynamicCost-dataset_{DATASET}-default_{DEFAULT_TACTIC}.csv"
    )
    pd.DataFrame(results_dict).to_csv(res_file, index=False)
    print(f"[D] Results saved to:\n\t{res_dir + res_file}")
