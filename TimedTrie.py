import statistics

import numpy as np
from Utility import Util

from TrieNode import TrieNode

class TimedTrie:

    def __init__(self, params):
        self.ENABLE_IQR = getattr(params, 'enable_iqr', True)
        self.STD_VAL = getattr(params, 'std_threshold', 0.675)  # Uses only if ENABLE_IQR is False
        self.K = getattr(params, 'depth', 5)  # Depth
        self.P_THRESHOLD = getattr(params, 'property_ext_threshold',
                                   0.45)  # Probability threshold to extract dominant properties

        self.ENABLE_DEBUG = getattr(params, 'DEBUG', False);  # Enables debug
        self.ENABLE_STRICT_TEMPORAL_MATCH = getattr(params, 'strict_event_prob', True);  # Src -> Des must contain

        self.global_node_count = 0

    def traverseAndBuild(self, node: TrieNode, timedTrace: (list(), list()), pos: int):
        if pos >= len(timedTrace[0]):
            return

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
                    self.traverseAndBuild(child, timedTrace, pos + 1)
                    return

        if not found and doTimeCheck == False:  # only create for last element in the trace
            self.global_node_count += 1
            newNode = TrieNode(self.global_node_count, event)
            newNode.t_min = time
            newNode.t_max = time
            newNode.t_list.append(time)
            newNode.count = 1
            node.children.append(newNode)
            self.traverseAndBuild(newNode, timedTrace, pos + 1)

    def evaluateProb(self, node: TrieNode, d: int, current_d: int):
        if current_d > d:
            return

        # find pattern probability
        # height_of_currentTree = node.tree_height;
        # total_patterns = sum(trace_size-(k-1) for k in list(range(current_d+1, height_of_currentTree + 2))) # We need all possible pattern sizes from 1 -> current_d+1
        # total_patterns = sum(trace_size-(k-1) for k in list(range(current_d+1, height_of_currentTree + 2))) # We need all possible pattern sizes from 1 -> current_d+1
        node.prob_pattern = round(float(node.count / (self.trace_size - current_d + 1)), 2)

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
                    "node.children count:{0}  tot_count:{1}  count_list:{2}".format(str(len(node.children)),
                                                                                    str(tot_count),
                                                                                    _inner_count_list))
                raise ZeroDivisionError

        for _child in node.children:
            self.evaluateProb(_child, d, current_d + 1)

    def evaluateHeightOfTree(self, node: TrieNode, d: int, current_d: int):
        max_child_height = 0;
        for _child in node.children:
            max_child_height = max(max_child_height, self.evaluateHeightOfTree(_child, d, current_d + 1))

        node.tree_height = max_child_height + 1;
        return node.tree_height

    def evaluateAtDepth(self, node: TrieNode, d: int, current_d: int):
        if d <= 1:
            return

        if current_d < d - 1:
            for child in node.children:
                self.evaluateAtDepth(child, d, current_d + 1)
        else:
            node.t_mean = statistics.mean(node.t_list)
            if self.ENABLE_IQR:
                _var = np.percentile(node.t_list, [25, 75])
                node.t_min, node.t_max = node.t_mean - _var[0], node.t_mean + _var[1]
            else:
                node.t_var = statistics.pstdev(node.t_list)
                _var = self.STD_VAL * node.t_var
                node.t_min, node.t_max = node.t_mean - _var, node.t_mean + _var

            node.t_min, node.t_max = round(node.t_min), round(node.t_max)
            # print("node.t_min ", node.t_min, "  node.t_max  ", node.t_max)

            # Take inner quartile
            node.count = sum(ele >= node.t_min and ele <= node.t_max for ele in node.t_list)
            node.dropped = len(node.t_list) - node.count

    def buildGraph(self, timed_trace, K: int = 3) -> TrieNode:
        self.global_node_count = 1
        self.trace_size = len(timed_trace[0])

        root = TrieNode(self.global_node_count, "*")
        for k in list(range(1, K + 1)):
            # print("depth ------ ", k)
            for i in list(range(0, len(timed_trace[0]) + 1 - k)):
                sub_trace_event = timed_trace[0][i: i + k]
                sub_trace_time = timed_trace[1][i: i + k]
                sub_trace_time = Util.getTimeShift(sub_trace_time)  # Get reset time shit
                # print("from ", str(i), " to ", str(i+k), " ", sub_trace_event)

                sub_trace = (sub_trace_event, sub_trace_time)
                # print(sub_trace)
                self.traverseAndBuild(root, sub_trace, 0)
            self.evaluateAtDepth(root, k, 0)

        self.evaluateHeightOfTree(root, k, 0)
        self.evaluateProb(root, k, 0)
        return root
