import copy

import TrieNode
import TimedTrie
import copy
import pandas as pd
from tabulate import tabulate

import TimedTrie
import TrieNode


class PatternMiner:
    def __init__(self, timedTrie: TrieNode, timed_trace, timed_trie_model: TimedTrie):
        self.extractedPatternList = [];
        self.extractedPatternList_str_count = [];

        self.timed_trie_model = timed_trie_model
        self.timedTrie = timedTrie
        self.timed_trace = timed_trace  # the entire trace
        self.max_depth = self.timed_trie_model.K

    def extractPattern(self, node: TrieNode, current_d: int = 0, current_path: list = []):
        if current_d > self.max_depth:
            return;

        currentPattern = ""

        for child in node.children:
            if child.prob >= self.timed_trie_model.P_THRESHOLD or current_d == 0:
                new_path = current_path + [child]
                self.extractedPatternList.append((copy.copy(new_path)))

                inner_patterns = self.extractPattern(child, current_d + 1, copy.copy(new_path))

    # Find the number of matching patterns
    def findMatchingPatterns(self, pattern_to_test):
        count = 0

        current_pattern = []
        for ele in self.timed_trace[0]:
            if len(current_pattern) == len(pattern_to_test):
                current_pattern = current_pattern[1:]

            current_pattern.append(ele)

            # this needs to be done separately so that append and checking are divided and not mixed up
            if current_pattern == pattern_to_test:
                count += 1

        return count

    # Find the support and confidence
    def evaluateSupportAndConfidenceForList(self, pattern_to_test):
        count_for_exact_pattern = self.findMatchingPatterns(pattern_to_test);
        count_of_one_less_pattern = self.findMatchingPatterns(pattern_to_test[:-1])

        total_count = len(self.timed_trace[0])

        # print("count_for_exact_pattern-{0} count_of_one_less_pattern-{1} total_count-{2}".format(count_for_exact_pattern, count_of_one_less_pattern, total_count))

        support = count_for_exact_pattern / (total_count - (len(pattern_to_test[0]) - 1))
        confidence = 1 if len(pattern_to_test) == 1 else count_for_exact_pattern / count_of_one_less_pattern

        return [round(support, 5), round(confidence, 5)];

    def printFromPatternList(self, returnPatternCount=False):

        for _ele in self.extractedPatternList:
            _currentPattern = ""
            _plain_pattern = []
            for idx, inner_ele in enumerate(_ele):
                if idx > 0:
                    _currentPattern = _currentPattern + "[" + str(inner_ele.t_min) + "," + str(inner_ele.t_max) + "]"
                else:
                    _currentPattern = _currentPattern + "[-INF," + str(inner_ele.t_max) + "]"

                _plain_pattern.append(inner_ele.char)
                _currentPattern += inner_ele.char + " "

            "### Calculate Support and Confidence (if-required)"
            support_conf = self.evaluateSupportAndConfidenceForList((_plain_pattern))

            if returnPatternCount == True:
                return [_currentPattern, inner_ele.count, support_conf[0],
                        support_conf[1]]  # so that the callee can make changes accordingly
            else:
                self.extractedPatternList_str_count.append([_currentPattern, inner_ele.count, support_conf[0], support_conf[1]])

    def printPatternList_pretty(self):

        self.printFromPatternList(False)

        self.extractedPatternList_DF = pd.DataFrame(self.extractedPatternList_str_count,
                                               columns=["Pattern", "Count", "Support", "Confidence"])
        self.extractedPatternList_DF.sort_values(by=["Count"], ascending=False, inplace=True)

        print(tabulate(self.extractedPatternList_DF, headers='keys', tablefmt='psql'))

    """
        Save Result
    """
    def save_result(self):
        self.extractedPatternList_DF.to_csv("./Result/dominant_patterns.csv");

    """#Pruned Trie Model"""
    # Get a pruned and clean version of the state machine
    def buildGraphPrunedTrie(self) -> TrieNode:

        extracted_pattern = set(tuple(item) for item in self.extractedPatternList)

        global_node_count_pruned = 1

        root = TrieNode(global_node_count_pruned, "*")

        for _idx, _val in enumerate(extracted_pattern):
            _temp = root;
            for _ele in _val:
                found = False;
                for _child in _temp.children:
                    if _child.char == _ele.char:
                        _temp = _child
                        found = True
                        break;

                if found == False:
                    global_node_count_pruned += 1
                    newNode = TrieNode(global_node_count_pruned, _ele.char)
                    newNode.t_min = _ele.t_min
                    newNode.t_max = _ele.t_max
                    newNode.t_list = _ele.t_list
                    newNode.count = _ele.count
                    newNode.prob = _ele.prob
                    newNode.prob_pattern = _ele.prob_pattern
                    newNode.tree_height = _ele.tree_height
                    newNode.is_end = _ele.is_end
                    newNode.dropped = _ele.dropped

                    _temp.children.append(newNode)
                    _temp = newNode

        return root