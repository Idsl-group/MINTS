# -*- coding: utf-8 -*-

"""
This python file can be used to perform empirical analysis.
Please goto the last time and uncomment the function based on the (empirical) analysis that has to performed.

"""


import pathlib
import random
import statistics
import sys
import time
from itertools import accumulate
from operator import add

import pandas as pd
import rstr
from tabulate import tabulate

"""# Mine for properties

## Trace Generation
"""


def getTimeVariability_Synthetic(trace="", windowTimeLengthVariability=5):
    deltaDistribution = []
    deltaDistribution.extend(
        list((random.randint(1, max(1, windowTimeLengthVariability)) for e in range(1, len(trace) + 1))))

    time = list(accumulate(deltaDistribution, add))
    return time


# Get trace and the associated event time. This follows a strict regex pattern
# pattern has to be a valid regex
# variability is the variational delta between neighboring event
def getTrace(pattern='ab+ac', windowTimeLengthVariability=5):
    trace = rstr.xeger(pattern)
    time = getTimeVariability_Synthetic(trace, windowTimeLengthVariability)

    print("Synthetic: Trace generated & Trace time generated successfully")

    return trace, time


def getTimeShift(time):
    time_new = [0]

    # Complex but short
    # time_shift = [0]
    # time_shift.extend(time[0:len(time)-1])
    # time_new = list(0 if i == 0 else time[i]-time_shift[i] for i in list(range(0, len(time))))

    # Simple but long
    for i in range(1, len(time)):
        time_diff = max(0, time[i] - time[i - 1])
        time_new.append((time_diff))

    return time_new


def completeTrace(pattern='ab+ac', windowTimeLengthVariability=15, local_clock=False):
    trace, time = "", [];

    trace, time = getTrace(pattern, windowTimeLengthVariability)

    if local_clock:
        time = getTimeShift(time)

    return trace, time


"""#### Hyperparameter"""

STD_VAL = 0.675  # wWthin the interquartile range
K = 10  # Depth
P_THRESHOLD = 0.20  # Probability threshold to extract dominant properties

VARIABILITY_WINDOW_LENGTH = 5  # for synthetic trace generation
ENABLE_DEBUG = False;  # Enables debug

"""#### Generate Synthetic trace"""


def generateTrace(regex_trace, variability_window_length=VARIABILITY_WINDOW_LENGTH, local_clock=False):
    trace, trace_time = completeTrace(regex_trace, variability_window_length, local_clock)
    trace_size = len(trace)
    print("Trace length : ", trace_size)

    return (list(trace), trace_time)


"""## Create Timed Trie

### Define
"""

global_node_count = 0


class TrieNode(object):
    def __init__(self, id: int, char: str):
        self.node_id = "Node_" + str(id)
        self.char = char
        self.children = []
        self.count = 1
        self.t_min = sys.maxsize
        self.t_max = 0
        self.t_mean = 0
        self.t_list = []
        self.t_var = 0
        self.dropped = 0
        self.prob = 0.0  # probability of occuring during transition
        self.prob_pattern = 0.0  # probability of pattern occuring
        self.tree_height = 0  # height of the tree considering the present node as root
        self.is_end = True


def traverseAndBuild(node: TrieNode, timedTrace: (list(), list()), pos: int):
    if pos >= len(timedTrace[0]):
        return

    global global_node_count
    node.is_end = False
    # print("timed_trace ", timed_trace)

    event, time = timedTrace[0][pos: pos + 1], timedTrace[1][pos: pos + 1]
    # print("event, time, pos ", event, time, pos)
    event, time = event[0], time[0]
    found = False
    doTimeCheck = True if pos < len(timedTrace[0]) - 1 else False

    for child in node.children:
        if child.char == event:  # check with character only
            if doTimeCheck == False or (doTimeCheck == True and time >= child.t_min and time <= child.t_max):
                found = True
                if doTimeCheck == False:
                    child.count += 1
                    child.t_min = min(child.t_min, time)
                    child.t_max = max(child.t_max, time)
                    child.t_list.append(time)
                traverseAndBuild(child, timedTrace, pos + 1)
                return

    if not found and doTimeCheck == False:  # only create for last element in the trace
        global_node_count += 1
        newNode = TrieNode(global_node_count, event)
        newNode.t_min = time
        newNode.t_max = time
        newNode.t_list.append(time)
        newNode.count = 1
        node.children.append(newNode)
        traverseAndBuild(newNode, timedTrace, pos + 1)


def evaluateProb(node: TrieNode, d: int, current_d: int, trace_size=1):
    if current_d > d:
        return

    # find pattern probability
    # height_of_currentTree = node.tree_height;
    # total_patterns = sum(trace_size-(k-1) for k in list(range(current_d+1, height_of_currentTree + 2))) # We need all possible pattern sizes from 1 -> current_d+1
    # total_patterns = sum(trace_size-(k-1) for k in list(range(current_d+1, height_of_currentTree + 2))) # We need all possible pattern sizes from 1 -> current_d+1
    node.prob_pattern = round(float(node.count / (trace_size - current_d + 1)), 2)

    # find transition probability
    tot_count = 0 if len(node.children) > 0 else 1
    _inner_count_list = []

    for _child in node.children:
        tot_count += _child.count
        _inner_count_list.append((_child.count))

    # TODO: Look into this. This block is for fail-safe but the necessity of this should never happen
    if len(node.children) > 0 and tot_count == 0:
        tot_count = 1

    for _child in node.children:
        try:
            _child.prob = round(float(_child.count / tot_count), 2)
        except ZeroDivisionError:
            print("ZeroDivisionError ")
            print(
                "node.children count:{0}  tot_count:{1}  count_list:{2}".format(str(len(node.children)), str(tot_count),
                                                                                _inner_count_list))
            raise ZeroDivisionError

    for _child in node.children:
        evaluateProb(_child, d, current_d + 1, trace_size)


def evaluateHeightOfTree(node: TrieNode, d: int, current_d: int):
    max_child_height = 0;
    for _child in node.children:
        max_child_height = max(max_child_height, evaluateHeightOfTree(_child, d, current_d + 1))

    node.tree_height = max_child_height + 1;
    return node.tree_height


def evaluateAtDepth(node: TrieNode, d: int, current_d: int):
    if d <= 1:
        return

    global STD_VAL
    if current_d < d - 1:
        for child in node.children:
            evaluateAtDepth(child, d, current_d + 1)
    else:
        node.t_mean = statistics.mean(node.t_list)
        node.t_var = statistics.pstdev(node.t_list)
        _var = STD_VAL * node.t_var
        node.t_min, node.t_max = node.t_mean - _var, node.t_mean + _var
        node.t_min, node.t_max = round(node.t_min), round(node.t_max)
        # print("node.t_min ", node.t_min, "  node.t_max  ", node.t_max)

        # Take inner quartile
        node.count = sum(ele >= node.t_min and ele <= node.t_max for ele in node.t_list)
        node.dropped = len(node.t_list) - node.count


def buildGraph(timed_trace, K: int = 3) -> TrieNode:
    global global_node_count
    global_node_count += 1

    root = TrieNode(global_node_count, "*")
    for k in list(range(1, K + 1)):
        # print("depth ------ ", k)
        for i in list(range(0, len(timed_trace[0]) + 1 - k)):
            sub_trace_event = timed_trace[0][i: i + k]
            sub_trace_time = timed_trace[1][i: i + k]
            sub_trace_time = getTimeShift(sub_trace_time)  # Get reset time shit
            # print("from ", str(i), " to ", str(i+k), " ", sub_trace_event)

            sub_trace = (sub_trace_event, sub_trace_time)
            # print(sub_trace)
            traverseAndBuild(root, sub_trace, 0)
        evaluateAtDepth(root, k, 0)

    evaluateHeightOfTree(root, k, 0)
    evaluateProb(root, k, 0, len(timed_trace[0]))
    return root


"""### Perform Mining"""


def build_timed_trie(timed_trace, K):
    start = time.time()

    timed_trie = buildGraph(timed_trace, K)
    time_diff = round(time.time() - start, 2)
    print("Timed Trie build complete")
    print("Timed Elapsed : ", str(time_diff), " sec")

    return (timed_trie, time_diff)


def generateTraceAndBuildTrie(regex, K, variability_window=5, local_clock=False, hasFullTrace=False,
                              full_Timed_Trace=()):
    if hasFullTrace == False:
        timed_trace = generateTrace(regex, variability_window, local_clock)
    else:
        timed_trace = full_Timed_Trace

    timed_trie, elapsed_time = build_timed_trie(timed_trace, K)

    trace_length = len(timed_trace[0])
    albhabet_size = len(set(timed_trace[0]))

    return trace_length, albhabet_size, K, elapsed_time


""" #### Empiral Analysis"""

# regex_trace = "ab+ac"
regex_trace = "(c{1,3}(ac{1,3}bc{1,3}){1,3}){10,12}"
# regex_trace = "([a-e]{100,200}){100,200}"

# regex_trace_response = "([bz]*(a[a-cz]*b[bz]*)){1,2}"
regex_trace_response_restricted = "([bz]{0,2}(a[a-cz]{0,5}b[bz]{0,2})){3,15}"
# regex_trace_alternating = "[cz]*(a[ca]*b[cz]*)*"
# regex_trace_alternating_restricted = "[c-z]{0,2}(a[c-z]{0,2}b[c-z]{0,2}){2,3}"

# generateTraceAndBuildTrie(regex_trace, K, VARIABILITY_WINDOW_LENGTH, False)

# Sub-List format -> [length, alpbhabet, K, time]
empirical_result_var_len = []
empirical_result_var_alpbhabet = []
empirical_result_var_K = []


# ----------------------------------------------------------------------------------------
def empiral_evaluation_for_len(K=5, variability_window_length=10, TRACE_LENGTH=5000):
    global empirical_result_var_len
    print("Empiral Evaluation for_len started")

    regex_trace = "(c{1,3}(ac{1,3}bc{1,3}){1,3}){10,15}"

    fullTrace = ""
    hasFullTrace = True

    for i in list(range(10, TRACE_LENGTH, 10)):
        _regex_trace = regex_trace

        while len(fullTrace) <= i:
            fullTrace += rstr.xeger(_regex_trace)

        fullTrace = fullTrace[: i]
        fullTime = getTimeVariability_Synthetic(fullTrace, variability_window_length)
        full_timed_trace = (list(fullTrace), fullTime)
        print("Trace length ", len(fullTrace))

        empirical_result_var_len.append(
            generateTraceAndBuildTrie(_regex_trace, K, variability_window_length, False, hasFullTrace,
                                      full_timed_trace))

    print("Completed")


def invokeEmpiricalEvaluationWithLength(K, VARIABILITY_WINDOW_LENGTH, TRACE_LENGTH=5000):
    empiral_evaluation_for_len(K, VARIABILITY_WINDOW_LENGTH, TRACE_LENGTH)
    empirical_result_var_len_DF = pd.DataFrame(empirical_result_var_len,
                                               columns=["Length", "Alphabet", "Depth", "Time (sec)"])
    empirical_result_var_len_DF.sort_values(by=["Length"], ascending=True, inplace=True)

    folder_path = str(pathlib.Path().absolute()) + "/Result/"
    file_name = "synthetic_empirical_result_var_len.csv"
    print("Saving to " + folder_path + file_name)

    empirical_result_var_len_DF.to_csv(folder_path + file_name)

    print(tabulate(empirical_result_var_len_DF, headers='keys', tablefmt='psql'))


# ----------------------------------------------------------------------------------------
def empiral_evaluation_for_depth(variability_window_length=10):
    global empirical_result_var_len
    print("Empiral Evaluation for_depth started")

    regex_trace = "(c{1,3}(ac{1,3}bc{1,3}){1,3}){10,15}"

    fullTrace = ""
    hasFullTrace = True

    while len(fullTrace) <= 1000000:
        fullTrace += rstr.xeger(regex_trace)

    fullTrace = fullTrace[:1000000]
    fullTime = getTimeVariability_Synthetic(fullTrace, variability_window_length)
    full_timed_trace = (list(fullTrace), fullTime)
    print("Trace length ", len(fullTrace))

    for i in list(range(3, 33, 3)):
        print("For K ", i)
        empirical_result_var_K.append(
            generateTraceAndBuildTrie(regex_trace, i, variability_window_length, False, hasFullTrace,
                                      full_timed_trace))

    print("Completed")


def invokeEmpiricalEvaluationWithDepth(VARIABILITY_WINDOW_LENGTH):
    empiral_evaluation_for_depth(VARIABILITY_WINDOW_LENGTH)
    empirical_result_var_K_DF = pd.DataFrame(empirical_result_var_K,
                                             columns=["Length", "Alphabet", "Depth", "Time (sec)"])
    empirical_result_var_K_DF.sort_values(by=["Length"], ascending=True, inplace=True)

    folder_path = str(pathlib.Path().absolute()) + "/Result/"
    file_name = "synthetic_empirical_result_var_depth.csv"
    print("Saving to " + folder_path + file_name)

    empirical_result_var_K_DF.to_csv(folder_path + file_name)
    print(tabulate(empirical_result_var_K_DF, headers='keys', tablefmt='psql'))


# ----------------------------------------------------------------------------------------
def empiral_evaluation_for_alpbhabet(K=3, variability_window_length=10):
    global empirical_result_var_len
    print("Empiral Evaluation for_alpbhabet started")

    # regex_trace = "(c{1,3}(ac{1,3}bc{1,3}){1,3}){10,15}"
    regex_trace = "[a-zA-Z]"

    fullTrace = ""
    hasFullTrace = True
    fixed_len = 10000

    for i in list(range(3, 51, 1)):
        fullTrace = ""
        while len(set(fullTrace)) < i:
            fullTrace += rstr.xeger(regex_trace)

        current_uniqiue_alpbhabet = list(set(fullTrace))
        while len(fullTrace) <= fixed_len:
            fullTrace += current_uniqiue_alpbhabet[random.randint(1, len(current_uniqiue_alpbhabet)) - 1]

        fullTrace = fullTrace[:10000]
        fullTime = getTimeVariability_Synthetic(fullTrace, variability_window_length)
        full_timed_trace = (list(fullTrace), fullTime)
        print("Trace length ", len(fullTrace))

        print("For Alpbhabet ", len(set(fullTrace)))
        empirical_result_var_alpbhabet.append(
            generateTraceAndBuildTrie(regex_trace, K, variability_window_length, False, hasFullTrace,
                                      full_timed_trace))

    print("Completed")


def invokeEmpiricalEvaluationWithAlpbhabet(K, VARIABILITY_WINDOW_LENGTH):
    empiral_evaluation_for_alpbhabet(K, VARIABILITY_WINDOW_LENGTH)
    empirical_result_var_Alphabet_DF = pd.DataFrame(empirical_result_var_alpbhabet,
                                                    columns=["Length", "Alphabet", "Depth", "Time (sec)"])
    empirical_result_var_Alphabet_DF.sort_values(by=["Length"], ascending=True, inplace=True)

    folder_path = str(pathlib.Path().absolute()) + "/Result/"
    file_name = "synthetic_empirical_result_var_alpbhabet.csv"
    print("Saving to " + folder_path + file_name)

    empirical_result_var_Alphabet_DF.to_csv(folder_path + file_name)

    print(tabulate(empirical_result_var_Alphabet_DF, headers='keys', tablefmt='psql'))

# ---------------------------------------------EMPIRICAL ANALYSIS-------------------------------------------


#  Uncomment for Trace Length vs Time
# invokeEmpiricalEvaluationWithLength(K, VARIABILITY_WINDOW_LENGTH)

# Uncomment for Alpbabet Size vs Time
# invokeEmpiricalEvaluationWithDepth(VARIABILITY_WINDOW_LENGTH)

# Uncomment for Depth of Timed Trie vs Time
# invokeEmpiricalEvaluationWithAlpbhabet(5, VARIABILITY_WINDOW_LENGTH)
