import os
import subprocess
import sys
import time
import pydotplus
import numpy as np

from flexico.flexico_utils_scenariosC_D import MT_METRIC_RESCALING, get_opt_az, get_az_cost
from constants import PRISM, PRISM_DIR

PRISM_MODEL = f"{PRISM_DIR}system_model-scenario_d.prism"
PRISM_PROPS = f"{PRISM_DIR}properties.props"
# variables in PRISM have to be ints,
# so we multiply to get less rounding errors
PRISM_MULTIPLY_FACTOR = 10


def getStrInBetween(line, str1, str2):
    idx1 = line.find(str1)
    idx2 = line.find(str2)

    return line[idx1 + len(str1) : idx2]


def _create_prism_cmd(
    exp_params: dict,
    curr_time: int,
    curr_metrics: dict,
    new_data: int,
    adaptation_beneftis: dict,
    test_sets: list,
    availability_zones: dict,
    daily_topic_weights: dict,
):
    strat_file_dir = f"{PRISM_DIR}strat_dot_files/"
    if not os.path.exists(strat_file_dir):
        os.makedirs(strat_file_dir)

    stratFile = f"{strat_file_dir}tactic_{curr_time}.dot"
    prismCommand = (
        f"{PRISM} {PRISM_MODEL} {PRISM_PROPS} -const MULTIPLY_FACTOR={PRISM_MULTIPLY_FACTOR},"
        f"deltaThreshold={exp_params['deltaT']*PRISM_MULTIPLY_FACTOR},numNewsTopics={len(test_sets)},"
        f"CURR_NEW_DATA={new_data},"
    )
    for mt_metric in MT_METRIC_RESCALING:
        if mt_metric in exp_params["fipTargetMetrics"]:
            metric_val = f"{round(curr_metrics[mt_metric]*PRISM_MULTIPLY_FACTOR)},"
        else:
            metric_val = "-1,"
        prismCommand = (
            prismCommand + f"INIT_{mt_metric.upper().replace('-', '_')}={metric_val}"
        )
    
    for az_id, az in enumerate(availability_zones):
        prismCommand = (
            prismCommand + 
            f"az{az_id+1}_finetune_cost={get_az_cost(exp_params['finetuneDuration'], availability_zones, az)},"
        )

    fip_topic_deltas = ""
    topic_weights = ""  # by default, all test-sets have the same weight
    for test_set in test_sets:
        # topic weights are constant for all metrics
        # (they are independent from the metrics)
        for day in daily_topic_weights.keys():
            topic_weights = (
                topic_weights
                + f"weight_{test_set}_day_{day}={daily_topic_weights[day][test_set]},"
            )
        for mt_metric in MT_METRIC_RESCALING:
            if mt_metric in exp_params["fipTargetMetrics"]:
                delta = f"{adaptation_beneftis[mt_metric][test_set]['perf']*PRISM_MULTIPLY_FACTOR},"
            else:
                delta = "-1,"

            fip_topic_deltas = (
                fip_topic_deltas
                + f"fipDelta{test_set}_{mt_metric.replace('-', '_')}={delta}"
            )

    prismCommand = (
        prismCommand
        + fip_topic_deltas
        + topic_weights[:-1]
        + f" -explicit -prop 2 -exportstrat {stratFile}"
    )

    return prismCommand, stratFile


def _parse_prism_results(stratFile):
    graph = pydotplus.graph_from_dot_file(stratFile)

    nodes_to_edges_dict = {}
    first_edge = None
    for edge in graph.get_edges():
        if not first_edge:
            first_edge = edge
        source_node = edge.get_source().strip('"')
        nodes_to_edges_dict[source_node] = edge

    doneTactic = ""
    tactics_per_day = {}
    go_finetune = False
    selected_az = -1
    edge = first_edge
    edge_label = edge.get_label()
    # Pydot labels may be stored with extra quotes, lets strip them
    if edge_label is not None:
        edge_label = edge_label.strip('"')
    while 'endExecution' not in edge_label:
        if ":" in edge_label:
            if 'Az' in edge_label:
                az = int(edge_label[-1]) - 1
            if ("finetune_start" in edge_label or "finetune_complete" in edge_label) or (
                "nop_start" in edge_label or "nop_complete" in edge_label
            ):
                doneTactic = edge_label
        
            source_node = edge.get_source().strip('"')
            source_node_list = graph.get_node(source_node)
            if source_node_list:
                node = source_node_list[0]
                clockTime = getStrInBetween(node.get_label(), "(", ",")
                if clockTime < "8":
                    if ("nop_start" in doneTactic) or "nop_complete" in doneTactic:
                        tactics_per_day[clockTime] = False
                    elif not go_finetune and (
                        "finetune_start" in doneTactic
                        or "finetune_complete" in doneTactic
                    ):
                        tactics_per_day[clockTime] = True
                        go_finetune = True
                        selected_az = az
        
        dest_node = edge.get_destination().strip('"')
        # next edge is the one that has dest_node as source
        edge = nodes_to_edges_dict[dest_node]
        edge_label = edge.get_label()
        # Pydot labels may be stored with extra quotes, lets strip them
        if edge_label is not None:
            edge_label = edge_label.strip('"')

    return go_finetune, selected_az, tactics_per_day


def check_adaptation_with_prism(
    exp_params: dict,
    curr_time: int,
    curr_metrics: dict,
    new_data: int,
    adaptation_beneftis: dict,
    test_sets: list,
    availability_zones: dict,
    daily_topic_weights: dict,
    verbose: bool = False,
):
    prismCommand, stratFile = _create_prism_cmd(
        exp_params=exp_params,
        curr_time=curr_time,
        curr_metrics=curr_metrics,
        new_data=new_data,
        adaptation_beneftis=adaptation_beneftis,
        test_sets=test_sets,
        availability_zones=availability_zones,
        daily_topic_weights=daily_topic_weights,
    )

    if verbose:
        print(f"[D] Running prism command:\n{prismCommand}")
    else:
        # | grep 'ç' so that prism doesn't print anything
        prismCommand = prismCommand + " | grep 'ç'"

    formal_verification_start_time = time.time()
    subprocess.run(prismCommand, shell=True, check=False)
    formal_verification_time = time.time() - formal_verification_start_time

    if not os.path.isfile(stratFile):
        print(f"[E] {stratFile} does not exist")
        sys.exit(1)

    adapt, selected_az, tactics_per_day = _parse_prism_results(stratFile)

    # when prism decides NOP is the optimal tactic, it basically selects 
    # the availability zone randomly because the cost of the availability 
    # zone has no impact on system utility in this case. To prevent this 
    # random choice, we force the selection of the AZ with minimum cost.
    if not adapt and selected_az == -1:
        selected_az, _ = get_opt_az(
            exp_params['finetuneDuration'], 
            availability_zones, 
        )
        print(f"[D] selected AZ: {selected_az}")
    else:
        selected_az = list(availability_zones.keys())[selected_az]

    return adapt, formal_verification_time, selected_az, tactics_per_day
