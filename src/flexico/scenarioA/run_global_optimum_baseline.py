#!/usr/bin/env python3

import itertools
import os
import sys

import numpy as np
import pandas as pd

from src.flexico.flexico_utils import (
    MT_METRIC_RESCALING,
    get_amount_finetune_data,
    get_dataset_metadata,
    get_finetune_cost,
    get_opt_baseline_parser,
    get_tactic_cost,
)
from constants import TMP_METRICS_DIR


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


def get_system_utility(tactic: str, node: dict):
    # System utility
    total_cost = 0

    if FINETUNE_COST_TYPE == "linear":
        finetune_cost = get_finetune_cost(node.finetune_data, DATASET)
    else:
        finetune_cost = TACTIC_COST

    for metric in FIP_METRICS:
        total_cost += get_tactic_cost(
            tactic=tactic,
            delta_threshold=DELTA_THRESHOLD,
            finetune_cost=finetune_cost,
            weighted_delta=node.delta[metric],
        )

    if "finetune" in tactic:
        total_cost += finetune_cost

    return total_cost


def get_metric(
    metrics: pd.DataFrame,  # dataframe that contains the metrics for a given model
    target_metric: str,  # metric whose value we want to know
):
    curr_perf = 0
    for test_set in TEST_SETS:
        curr_perf += (
            metrics[f"eval_{test_set}_{target_metric}"].loc[0]
            * MT_METRIC_RESCALING[target_metric]
        )
    return round(curr_perf / len(TEST_SETS), 5)


class Node:
    def __init__(self, left, right, metric, delta, depth, last_finetune, finetune_data):
        self.left = left
        self.right = right
        self.metric = metric
        self.delta = delta  # how much we expect to improve/degrade if we finetune
        self.depth = depth
        self.last_finetune = last_finetune
        self.finetune_data = finetune_data

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


def lowestCostLeaf(tree, cost, depth):
    nop_cost = get_system_utility("nop", tree)
    finetune_cost = get_system_utility("finetune", tree)

    if tree.left is None and tree.right is None:
        if VERBOSE:
            print(f"leaf cost = {cost}")
        return [cost, None, {}]

    leftCost, _, left_tactics_dict = lowestCostLeaf(
        tree.left, cost + finetune_cost, depth + 1
    )
    rightCost, _, right_tactics_dict = lowestCostLeaf(
        tree.right, cost + nop_cost, depth + 1
    )

    if "nop" in DEFAULT_TACTIC:
        if leftCost < rightCost:
            tactic = "finetune"
            tactics_dict = left_tactics_dict
            path_cost = leftCost
        else:
            tactic = "nop"
            tactics_dict = right_tactics_dict
            path_cost = rightCost
    else:
        if leftCost <= rightCost:
            tactic = "finetune"
            tactics_dict = left_tactics_dict
            path_cost = leftCost
        else:
            tactic = "nop"
            tactics_dict = right_tactics_dict
            path_cost = rightCost

    tactics_dict[depth] = {"tactic": tactic, "path_cost": path_cost}

    assert (
        min(leftCost, rightCost) == path_cost
    ), f"[E] min(leftCost, rightCost) != tactic_cost <==> {min(leftCost, rightCost)} != {path_cost}"

    if depth == 0:
        return [path_cost, tactic, tactics_dict]

    return [min(leftCost, rightCost), None, tactics_dict]


def buildTree(
    start_date,
    last_finetune,
    finetune_ongoing,
    finetune_state,
    tactic,
    depth,
    metrics_files_dict,
    fid_df,
):
    # before any update due to the current tactic's execution
    # compute the performance metric for the "non-updated" (current) model
    prev_metrics_dict = {}
    for metric in FIP_METRICS:
        prev_metrics_dict[metric] = get_metric(
            metrics_files_dict[f"instant-{last_finetune}"], metric
        )

    if (tactic == "finetune") and not finetune_ongoing:
        finetune_ongoing = True
        finetune_state = TACTIC_LATENCY

    if finetune_ongoing:
        if finetune_state == 0:
            last_finetune = start_date + TACTIC_LATENCY * TIME_INTERVAL
            finetune_ongoing = False
        else:
            finetune_state -= 1

    curr_metrics_dict = {}
    deltas_dict = {}
    for metric in FIP_METRICS:
        curr_metrics_dict[metric] = get_metric(
            metrics_files_dict[f"instant-{last_finetune}"], metric
        )
        deltas_dict[metric] = round(
            curr_metrics_dict[metric] - prev_metrics_dict[metric], 5
        )
        if deltas_dict[metric] == 0:
            # delta is 0 when the current tactic is nop
            # but to compute the missing opportunity
            # cost we need the finetune delta
            deltas_dict[metric] = round(
                get_metric(
                    metrics_files_dict[
                        f"instant-{start_date + TACTIC_LATENCY*TIME_INTERVAL}"
                    ],
                    metric,
                )
                - prev_metrics_dict[metric],
                5,
            )

    if VERBOSE:
        print(
            f"[curr-time={start_date+depth*TIME_INTERVAL}] "
            f"tactic={tactic} "
            f"metrics-file=instant-{last_finetune} "
            f"depth={depth} "
            f"max-depth={LOOKAHEAD}"
        )

    curr_time = start_date + depth * TIME_INTERVAL
    finetune_data = get_amount_finetune_data(fid_df, curr_time, last_finetune)
    if depth == LOOKAHEAD or (start_date + (depth * TIME_INTERVAL)) == END_DATE:
        return Node(
            None,
            None,
            curr_metrics_dict,
            deltas_dict,
            depth,
            last_finetune,
            finetune_data,
        )
    else:
        leftNode = buildTree(
            start_date,
            last_finetune,
            finetune_ongoing,
            finetune_state,
            "finetune",
            depth + 1,
            metrics_files_dict,
            fid_df,
        )
        rightNode = buildTree(
            start_date,
            last_finetune,
            finetune_ongoing,
            finetune_state,
            "nop",
            depth + 1,
            metrics_files_dict,
            fid_df,
        )
        return Node(
            leftNode,
            rightNode,
            curr_metrics_dict,
            deltas_dict,
            depth,
            last_finetune,
            finetune_data,
        )


def main(metricsFiles, fid_df, res_dict):
    adaptations = []
    finetune_instants = []
    real_metric_val = []
    cost = []
    last_finetune = LAST_FINETUNE

    for curr_time in range(START_DATE, END_DATE, TIME_INTERVAL):
        if VERBOSE:
            print("#" * 50 + f" curr-time={curr_time} " + "#" * 50)

        metrics_val_dict = {}
        deltas_dict = {}
        for metric in FIP_METRICS:
            metrics_val_dict[metric] = get_metric(
                metricsFiles[f"instant-{last_finetune}"], metric
            )
            deltas_dict[metric] = round(
                get_metric(metricsFiles[f"instant-{curr_time}"], metric)
                - get_metric(metricsFiles[f"instant-{last_finetune}"], metric),
                5,
            )
        real_metric_val.append(metrics_val_dict)

        global LOOKAHEAD
        if curr_time > END_DATE - LOOKAHEAD and LOOKAHEAD > 0:
            LOOKAHEAD = LOOKAHEAD - 1
            # continue
        if curr_time > END_DATE - LOOKAHEAD and LOOKAHEAD == 0:
            continue

        root = Node(
            left=buildTree(
                start_date=curr_time,
                last_finetune=last_finetune,
                finetune_ongoing=False,
                finetune_state=0,
                tactic="finetune",
                depth=1,
                metrics_files_dict=metricsFiles,
                fid_df=fid_df,
            ),
            right=buildTree(
                start_date=curr_time,
                last_finetune=last_finetune,
                finetune_ongoing=False,
                finetune_state=0,
                tactic="nop",
                depth=1,
                metrics_files_dict=metricsFiles,
                fid_df=fid_df,
            ),
            metric=metrics_val_dict,
            delta=deltas_dict,
            depth=0,
            last_finetune=last_finetune,
            finetune_data=get_amount_finetune_data(fid_df, curr_time, last_finetune),
        )

        if VERBOSE:
            root.print_tree()

        nextTactic = lowestCostLeaf(root, 0, 0)
        curr_cost = 0
        if nextTactic[1] == "finetune":
            if VERBOSE:
                print(f"Going to FINETUNE --- leaf-cost={nextTactic[0]}")
            adaptations.append(True)
            last_finetune = curr_time
            finetune_instants.append(curr_time)
            curr_cost = get_system_utility("finetune", root)
        else:
            adaptations.append(False)
            curr_cost = get_system_utility("nop", root)

        cost.append(curr_cost)

        if (END_DATE - START_DATE - (LOOKAHEAD * TIME_INTERVAL)) == 0:
            print(f"Optimal strategy: {nextTactic[2]}")

    print(f"[D] Executed {sum(adaptations)} fine-tunings!")
    print(f"[D]\tFine-tuned {len(finetune_instants)} times @ {finetune_instants}")
    print(f"[D]\tTotal-cost is {np.cumsum(cost)[-1]}")

    res_dict["adaptations"].append(adaptations)
    res_dict["totalAdaptations"].append(len(finetune_instants))
    res_dict["totalCost"].append(np.cumsum(cost)[-1])
    res_dict["real_metric_val"].append(real_metric_val)
    res_dict["cost"].append(cost)


if __name__ == "__main__":
    prog_args = get_opt_baseline_parser()

    global DEFAULT_TACTIC
    DEFAULT_TACTIC = prog_args.default_tactic.lower()
    global VERBOSE
    VERBOSE = bool(prog_args.verbose)
    global DATASET
    DATASET = prog_args.dataset

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

    # initialize results dict
    results_dict = {
        "baseline": [],
        "lookahead": [],
        "deltaT": [],
        "fipTargetMetrics": [],
        "finetuneCostType": [],
        "finetuneCost": [],
        "adaptations": [],
        "totalAdaptations": [],
        "totalCost": [],
        "real_metric_val": [],
        "cost": [],
    }

    # create experiments' list
    prod = itertools.product(
        list(set(prog_args.lookahead)),
        list(set(prog_args.thresholds)),
        list(prog_args.fip_metrics),
        list(set(prog_args.finetune_costs)),
        list(prog_args.finetune_cost_type),
    )
    exp_list = []
    for la, delta_t, fip_m, fc, fc_type in prod:
        exp_list.append(
            {
                "lookahead": la,
                "deltaT": delta_t,
                "fipTargetMetric": fip_m,
                "finetuneCost": fc,
                "finetuneCostType": fc_type,
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
        FIP_METRICS = exp_args["fipTargetMetric"]
        global TACTIC_COST
        TACTIC_COST = exp_args["finetuneCost"]
        global TACTIC_LATENCY
        TACTIC_LATENCY = 0
        global FINETUNE_COST_TYPE
        FINETUNE_COST_TYPE = exp_args["finetuneCostType"]

        print(
            "#" * 75
            + f" EXPERIMENT {exp_counter + 1} / {len(exp_list)} ({round((exp_counter + 1)/len(exp_list)*100, 2)}%) "
            + "#" * 75
        )
        fc_print = f"finetune-cost-type={FINETUNE_COST_TYPE}"
        if "linear" not in FINETUNE_COST_TYPE:
            fc_print = fc_print + f"   finetune-cost={TACTIC_COST}"
        print(
            f"la={LOOKAHEAD}   DELTA-T={DELTA_THRESHOLD}   FIP-METRIC={FIP_METRICS}   {fc_print}   finetune-latency={TACTIC_LATENCY}"
        )

        main(metrics_files, fid, results_dict)

        results_dict["baseline"].append(f"opt-{exp_args['lookahead']}")
        for arg_name, arg_val in exp_args.items():
            results_dict[arg_name].append(arg_val)

    # save results
    res_dir = f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/framework_results/"
    res_file = (
        f"{res_dir}optimum_baseline-base-dataset_{DATASET}-default_{DEFAULT_TACTIC}.csv"
    )
    pd.DataFrame(results_dict).to_csv(res_file, index=False)
    print(f"[D] Results saved to:\n\t{res_dir + res_file}")
