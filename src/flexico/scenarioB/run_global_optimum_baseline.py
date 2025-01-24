#!/usr/bin/env python3

import itertools
import json
import os
import sys

import numpy as np
import pandas as pd

from src.flexico.flexico_utils import (
    MT_METRIC_RESCALING,
    get_amount_finetune_data,
    get_daily_topic_weights,
    get_dataset_metadata,
    get_finetune_cost,
    get_opt_baseline_parser,
    get_tactic_cost,
)
from constants import TMP_METRICS_DIR

METRICS_DIR = TMP_METRICS_DIR + "hk-news/"


def load_metrics_files(path: str):
    files = {}

    for f in os.listdir(path):
        if (
            "finetuneInstant" in f
            and "fixedTestSetEval" in f
            and f.endswith("finetuneType_base-latest.json")
        ):
            key = f"instant-{int(f.split('-')[0].split('_')[1])}"
            files[key] = pd.read_json(path + f)

    return files


def get_system_utility(tactics: dict, node: dict):
    # System utility
    total_cost = 0

    if FINETUNE_COST_TYPE == "linear":
        finetune_cost = get_finetune_cost(node.finetune_data, DATASET)
    else:
        finetune_cost = TACTIC_COST

    adapt = False
    for day in node.topic_weights.keys():
        for mt_metric in FIP_METRICS:
            weighted_delta = 0
            for test_set in TEST_SETS:
                weighted_delta += (
                    node.topic_weights[day][test_set] * node.delta[mt_metric][test_set]
                )

            if tactics[day]:
                adapt = True
            total_cost += get_tactic_cost(
                tactic="finetune" if adapt else "nop",
                delta_threshold=DELTA_THRESHOLD,
                finetune_cost=finetune_cost,
                weighted_delta=weighted_delta,
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
                        metrics_files_dict[f"instant-{start_date + TACTIC_LATENCY}"],
                        metric,
                    )[test_set]
                    - prev_metrics[metric][test_set],
                    5,
                )

    return deltas_dict, curr_metrics


class Node:
    def __init__(
        self,
        children,
        metric,
        delta,
        topic_weights,
        depth,
        last_finetune,
        finetune_data,
        tactics_per_day,
    ):
        self.children = children
        self.metric = metric
        self.delta = delta  # how much we expect to improve/degrade if we finetune
        self.topic_weights = topic_weights
        self.depth = depth
        self.last_finetune = last_finetune
        self.finetune_data = finetune_data
        self.tactics_per_day = tactics_per_day

    def print_tree(self):
        self._print_tree_recursive(self, 0)

    def _print_tree_recursive(self, node, depth):
        if node is not None:
            print("    " * depth, end="")
            if node.left is None:
                print(
                    f"(Leaf Node) {FIP_METRICS}: {node.metric}   delta={node.delta}\t(last-finetune={node.last_finetune})"
                )
            else:
                print(
                    f"{FIP_METRICS}: {node.metric}   delta={node.delta}\t(last-finetune={node.last_finetune})"
                )
                self._print_tree_recursive(node.left, depth + 1)
                self._print_tree_recursive(node.right, depth + 1)


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
    weights_dist,
    fid_df,
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
            last_finetune = start_date + TACTIC_LATENCY
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
            f"[curr-time={start_date+depth}] "
            f"tactic={tactics_per_day} "
            f"metrics-file=instant-{last_finetune} "
            f"depth={depth} "
            f"max-depth={LOOKAHEAD}"
        )

    curr_time = start_date + depth
    finetune_data = get_amount_finetune_data(fid_df, curr_time, last_finetune)
    topic_weights = get_daily_topic_weights(
        curr_time=curr_time,
        topic_weight_type=WEIGHT_TYPE,
        daily_weights=weights_dist,
        test_sets=TEST_SETS,
    )

    children_nodes = []
    if depth == LOOKAHEAD or start_date + depth == END_DATE:
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
                    weights_dist=weights_dist,
                    fid_df=fid_df,
                )
            )
    return Node(
        children_nodes,
        curr_metrics_dict,
        deltas_dict,
        topic_weights,
        depth,
        last_finetune,
        finetune_data,
        tactics_per_day,
    )


def main(metricsFiles, daily_weights, fid_df, res_dict):
    adaptations = []
    adaptationsWeekDay = []
    finetune_instants = []
    real_metric_val = []
    cost = []
    last_finetune = 27

    for curr_time in range(START_DATE, END_DATE):
        if VERBOSE:
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
                    weights_dist=daily_weights,
                    fid_df=fid_df,
                )
            )

        root = Node(
            children=children_nodes,
            metric=metrics_val_dict,
            delta=deltas_dict,
            topic_weights=daily_topic_weights,
            depth=0,
            last_finetune=last_finetune,
            finetune_data=get_amount_finetune_data(fid_df, curr_time, last_finetune),
            tactics_per_day=dict.fromkeys(daily_topic_weights.keys(), False),
        )

        if VERBOSE:
            root.print_tree()

        nextTactic = lowestCostLeaf(root, 0, 0)
        tactics_per_day = nextTactic[1]
        adaptation_day = [k for k, v in tactics_per_day.items() if v]
        curr_cost = get_system_utility(tactics_per_day, root)
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

        cost.append(curr_cost)

        if END_DATE - START_DATE - LOOKAHEAD == 0:
            print(f"Optimal strategy: {nextTactic[2]}")

    print(f"[D] Executed {sum(adaptations)} fine-tunings!")
    print(f"[D]\tFine-tuned {len(finetune_instants)} times @ {finetune_instants}")
    print(f"[D]\tTotal-cost is {np.cumsum(cost)[-1]}")

    res_dict["adaptationsWeekDay"].append(adaptationsWeekDay)
    res_dict["adaptations"].append(adaptations)
    res_dict["totalAdaptations"].append(sum(adaptations))
    # res_dict["real_metric_val"].append(real_metric_val)
    res_dict["cost"].append(cost)
    res_dict["totalCost"].append(sum(cost))


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

    global TEST_SETS
    TEST_SETS = dataset_metadata["TEST_SETS"]
    global START_DATE
    START_DATE = dataset_metadata["TEST_START"]
    global END_DATE
    END_DATE = dataset_metadata["TEST_END"]

    # load adaptation impact dataset
    fid = pd.read_csv(dataset_metadata["fid_file"])

    # load metrics files
    metrics_files = load_metrics_files(path=METRICS_DIR)

    # load daily topic distributions
    with open(dataset_metadata["daily_dist_file"], encoding="utf-8") as f_in:
        daily_dist = json.load(f_in)

    # initialize results dict
    results_dict = {
        "runID": [],
        "baseline": [],
        "lookahead": [],
        "finetuneCostType": [],
        "finetuneCost": [],
        "deltaT": [],
        "fipTargetMetrics": [],
        "fipModel": [],
        "fipFeatureSet": [],
        "topicWeightType": [],
        "adaptationsWeekDay": [],
        "adaptations": [],
        "totalAdaptations": [],
        "cost": [],
        "totalCost": [],
    }

    # create experiments' list
    prod = itertools.product(
        list(set(prog_args.lookahead)),
        list(set(prog_args.thresholds)),
        list(prog_args.fip_metrics),
        list(set(prog_args.finetune_costs)),
        list(set(prog_args.finetune_cost_type)),
        list(set(prog_args.topic_weights_types)),
    )
    exp_list = []
    for la, delta_t, fip_m, fc, fc_type, tw_type in prod:
        exp_list.append(
            {
                "lookahead": la,
                "deltaT": delta_t,
                "fipTargetMetrics": fip_m,
                "finetuneCost": fc,
                "finetuneCostType": fc_type,
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
        global TACTIC_COST
        TACTIC_COST = exp_args["finetuneCost"]
        global TACTIC_LATENCY
        TACTIC_LATENCY = 0
        global FINETUNE_COST_TYPE
        FINETUNE_COST_TYPE = exp_args["finetuneCostType"]
        global WEIGHT_TYPE
        WEIGHT_TYPE = exp_args["topicWeightType"]

        print(
            "#" * 75
            + f" EXPERIMENT {exp_counter + 1} / {len(exp_list)} ({round((exp_counter + 1)/len(exp_list)*100, 2)}%) "
            + "#" * 75
        )
        fc_print = f"finetune-cost-type={FINETUNE_COST_TYPE}"
        if "linear" not in FINETUNE_COST_TYPE:
            fc_print = fc_print + f"   finetune-cost={TACTIC_COST}"
        print(
            f"la={LOOKAHEAD}   DELTA-T={DELTA_THRESHOLD}   FIP-METRIC={FIP_METRICS}   "
            f"{fc_print}   finetune-latency={TACTIC_LATENCY}   topic_weights_type={WEIGHT_TYPE}"
        )

        main(metrics_files, daily_dist, fid, results_dict)

        # UPDATE RESULTS DICT
        results_dict["runID"].append(1)
        results_dict["baseline"].append(f"opt-{exp_args['lookahead']}")
        results_dict["fipModel"].append("None")
        results_dict["fipFeatureSet"].append("None")
        for arg_name, arg_val in exp_args.items():
            results_dict[arg_name].append(arg_val)

    # SAVE RESULTS
    res_dir = f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/framework_results/"
    res_file = f"{res_dir}optimum_baseline-topicWeights-dataset_{DATASET}-default_{DEFAULT_TACTIC}.csv"
    pd.DataFrame(results_dict).to_csv(res_file, index=False)
    print(f"[D] Results saved to:\n\t{res_dir + res_file}")
