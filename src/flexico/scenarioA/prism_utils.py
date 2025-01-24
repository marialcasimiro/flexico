import os
import subprocess
import sys
import time

from flexico.flexico_utils import MT_METRIC_RESCALING
from constants import PRISM_DIR, PRISM

PRISM_MODEL = f"{PRISM_DIR}system_model-scenario_a.prism"
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
    adaptation_beneftis: dict,
    test_sets: list,
):
    strat_file_dir = f"{PRISM_DIR}strat_dot_files/"
    if not os.path.exists(strat_file_dir):
        os.makedirs(strat_file_dir)

    stratFile = f"{strat_file_dir}tactic_{curr_time}.dot"
    prismCommand = (
        f"{PRISM} {PRISM_MODEL} {PRISM_PROPS} -const MULTIPLY_FACTOR={PRISM_MULTIPLY_FACTOR},"
        f"FINETUNE_COST={exp_params['finetuneCost']},"
        f"deltaThreshold={exp_params['deltaT']*PRISM_MULTIPLY_FACTOR},numNewsTopics={len(test_sets)},"
    )
    for mt_metric in MT_METRIC_RESCALING:
        if mt_metric in exp_params["fipTargetMetrics"]:
            metric_val = f"{round(curr_metrics[mt_metric]*PRISM_MULTIPLY_FACTOR)},"
        else:
            metric_val = "-1,"
        prismCommand = (
            prismCommand + f"INIT_{mt_metric.upper().replace('-', '_')}={metric_val}"
        )

    fip_topic_deltas = ""
    topics_info = ""  # by default, all test-sets have the same weight
    for test_set in test_sets:
        # topic weights are constant for all metrics
        # (they are independent from the metrics)
        topics_info = topics_info + f"weight_{test_set}={1},"
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
        + topics_info[:-1]
        + f" -explicit -prop 2 -exportstrat {stratFile}"
    )

    return prismCommand, stratFile


def _parse_prism_results(stratFile):
    with open(stratFile, "r", encoding="utf-8") as fpIn:
        # Ignore two first lines (type of probabilistic model and node shape)
        fpIn.readline()
        fpIn.readline()
        line = fpIn.readline()
        doneTactic = ""
        go_finetune = False
        while line:
            line = line.rstrip()  # Strip trailing spaces and newline
            if ":" in line:  # this line contains an action
                tactic = (
                    getStrInBetween(line, "label=", ",").replace('"', "").split(":")[1]
                )
                if ("finetune_start" in tactic or "finetune_complete" in tactic) or (
                    "nop_start" in tactic or "nop_complete" in tactic
                ):
                    doneTactic = tactic
                line = fpIn.readline()  # ignore this line: node shape
                line = fpIn.readline()  # ignore this line: transition info
                line = fpIn.readline()  # this line has clock info
                line = line.rstrip()

            clockTime = getStrInBetween(line, "(", ",")
            if clockTime == "2":
                if ("nop_start" in doneTactic) or "nop_complete" in doneTactic:
                    go_finetune = False
                elif (
                    "finetune_start" in doneTactic or "finetune_complete" in doneTactic
                ):
                    go_finetune = True

            # Ignore the next two lines (shape and unused label)
            line = fpIn.readline()
    return go_finetune


def check_adaptation_with_prism(
    exp_params: dict,
    curr_time: int,
    curr_metrics: dict,
    adaptation_beneftis: dict,
    test_sets: list,
    verbose: bool = False,
):
    prismCommand, stratFile = _create_prism_cmd(
        exp_params=exp_params,
        curr_time=curr_time,
        curr_metrics=curr_metrics,
        adaptation_beneftis=adaptation_beneftis,
        test_sets=test_sets,
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

    adapt = _parse_prism_results(stratFile)

    return adapt, formal_verification_time
