#!/usr/bin/env python3

import itertools
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
# time python src/flexico/scenarioC/run_global_optimum_baseline.py
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


def get_system_utility(tactic: str, node: dict):
    # System utility
    total_cost = 0

    finetune_duration = estimate_finetune_duration(node.finetune_data, DATASET)
    opt_az, finetune_cost = get_opt_az(
        finetune_duration, 
        node.availability_zones,
    )

    for metric in FIP_METRICS:
        total_cost += get_tactic_cost(
            tactic=tactic,
            finetune_cost=finetune_cost,
            delta_threshold=DELTA_THRESHOLD,
            fip_delta=node.delta[metric],
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
    def __init__(self, left, right, metric, delta, depth, last_finetune, finetune_data, availability_zones):
        self.left = left
        self.right = right
        self.metric = metric
        self.delta = delta  # how much we expect to improve/degrade if we finetune
        self.depth = depth
        self.last_finetune = last_finetune
        self.finetune_data = finetune_data
        self.availability_zones = availability_zones

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
    spot_instance_df, 
    energy_df,
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
    spot_instance_data_weeks = spot_instance_df.continuous_week.unique()[-(END_DATE-START_DATE+1):]
    availability_zones=get_az_data(
        spot_instance_df, energy_df, spot_instance_data_weeks[curr_time-START_DATE]
    )
    if depth == LOOKAHEAD or (start_date + (depth * TIME_INTERVAL)) == END_DATE:
        return Node(
            None,
            None,
            curr_metrics_dict,
            deltas_dict,
            depth,
            last_finetune,
            finetune_data,
            availability_zones,
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
            spot_instance_df, 
            energy_df,
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
            spot_instance_df, 
            energy_df,
        )
        return Node(
            leftNode,
            rightNode,
            curr_metrics_dict,
            deltas_dict,
            depth,
            last_finetune,
            finetune_data,
            availability_zones
        )


def main(metricsFiles, fid_df, res_dict, spot_instance_df, energy_df):
    adaptations = []
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
                spot_instance_df=spot_instance_df, 
                energy_df=energy_df,
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
                spot_instance_df=spot_instance_df, 
                energy_df=energy_df,
            ),
            metric=metrics_val_dict,
            delta=deltas_dict,
            depth=0,
            last_finetune=last_finetune,
            finetune_data=get_amount_finetune_data(fid_df, curr_time, last_finetune),
            availability_zones=get_az_data(
                spot_instance_df, energy_df, spot_instance_data_weeks[curr_time-START_DATE]
            )
        )

        if VERBOSE:
            root.print_tree()

        nextTactic = lowestCostLeaf(root, 0, 0)
        curr_cost = 0
        selected_az, _ = get_opt_az(
            estimate_finetune_duration(root.finetune_data, DATASET), 
            root.availability_zones,
        )
        
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

        # check CO2 emissions
        finetune_duration = estimate_finetune_duration(root.finetune_data, DATASET)
        nop_sys_u = lowestCostLeaf(root.right, get_system_utility("nop", root), 1)[0]
        finetune_sys_u = lowestCostLeaf(root.left, get_system_utility("finetune", root), 1)[0]
        az_data = root.availability_zones[selected_az]
        correctCO2, incorrectCO2, unburntCO2 = get_burnt_co2(
            adapt=adaptations[-1],
            missedOportunityCost=nop_sys_u,
            finetuneRegretCost=finetune_sys_u - (az_data['spot_price'] * finetune_duration),
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
        nopSysU.append(nop_sys_u)
        finetuneSysU.append(finetune_sys_u)

        

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
    }

    # create experiments' list
    prod = itertools.product(
        list(set(prog_args.lookahead)),
        list(set(prog_args.thresholds)),
        list(prog_args.fip_metrics),
    )
    exp_list = []
    for la, delta_t, fip_m in prod:
        exp_list.append(
            {
                "lookahead": la,
                "deltaT": delta_t,
                "fipTargetMetrics": fip_m,
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

        print(
            "#" * 75
            + f" EXPERIMENT {exp_counter + 1} / {len(exp_list)} ({round((exp_counter + 1)/len(exp_list)*100, 2)}%) "
            + "#" * 75
        )
        
        print(
            f"la={LOOKAHEAD}   DELTA-T={DELTA_THRESHOLD}   "
            f"FIP-METRIC={FIP_METRICS}   finetune-latency={TACTIC_LATENCY}   "
        )

        main(metrics_files, fid, results_dict, spot_instance_df, energy_df)

        results_dict["baseline"].append(f"opt-{exp_args['lookahead']}")
        for arg_name, arg_val in exp_args.items():
            results_dict[arg_name].append(arg_val)

    # save results
    res_dir = f"{os.path.dirname(os.path.abspath(sys.argv[0]))}/framework_results/"
    res_file = (
        f"{res_dir}optimum_baseline-dynamicCost-dataset_{DATASET}-default_{DEFAULT_TACTIC}.csv"
    )
    pd.DataFrame(results_dict).to_csv(res_file, index=False)
    print(f"[D] Results saved to:\n\t{res_dir + res_file}")
