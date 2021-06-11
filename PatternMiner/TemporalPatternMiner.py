import re

import PatternMiner.TemporalPatternTemplate as temporalTemplate
from PatternMiner import DominantPatternMiner
import TimedTrie
from Utility import Util

import pandas as pd
from tabulate import tabulate

import TrieNode
import TimedTrie

class TemporalPatternMiner:
    def __init__(self, timedTrie: TrieNode, timed_trace, timed_trie_model: TimedTrie, dominant_pattern_miner_model: DominantPatternMiner):

        self.temporal_patterns_pat_str_count = []

        self.timed_trie_model = timed_trie_model
        self.timedTrie = timedTrie
        self.timed_trace = timed_trace  # the entire trace

        self.PATTERN_MATCHED_TEMPORAL = {
            "RESPONSE_PATTERN_REGEX": [],
            "ALTERNATING_PATTERN_REGEX": [],
            "MULTIEFFECT_PATTERN_REGEX": [],
            "MULTICAUSE_PATTERN_REGEX": []
        }

        self.dominant_pattern_miner_model = dominant_pattern_miner_model
        self.extracted_pattern = self.dominant_pattern_miner_model.extractedPatternList
        self.Alphabets = sorted(list(set(self.timed_trace[0])))


    """#### Begin extraction of temporal properties """

    def find_matching_pattern(self, interested_pattern: str = "RESPONSE_PATTERN_REGEX"):

        pattern_regex = temporalTemplate.PATTERN_REGEX_DIC[interested_pattern]

        for alpha_1 in self.Alphabets:
            for alpha_2 in self.Alphabets:
                if alpha_1 != alpha_2:

                    matching_regex = pattern_regex.format(alpha_1, alpha_2)
                    if self.timed_trie_model.ENABLE_DEBUG:
                        print("For ", interested_pattern, "   Matching with ", matching_regex)
                    _currentPattern = ""
                    for _ele in self.extracted_pattern:
                        _cp = [s.char for s in _ele]
                        _currentPattern = ''.join(_cp)

                        try:
                            if re.fullmatch(matching_regex, _currentPattern):
                                if self.timed_trie_model.ENABLE_DEBUG:
                                    print("Matched with ", _currentPattern)

                                if self.timed_trie_model.ENABLE_STRICT_TEMPORAL_MATCH:
                                    if Util.doesStringContains(_currentPattern, alpha_1) and Util.doesStringContains(
                                            _currentPattern, alpha_2):
                                        self.PATTERN_MATCHED_TEMPORAL[interested_pattern].append(
                                            [(alpha_1, alpha_2), _ele])
                                else:
                                    self.PATTERN_MATCHED_TEMPORAL[interested_pattern].append([(alpha_1, alpha_2), _ele])

                        except Exception as e:
                            print(e)
                            print("matching_regex ", matching_regex, "  _currentPattern ", _currentPattern)

    def extractTemporalPatterns(self):
        for key, value in temporalTemplate.PATTERN_REGEX_DIC.items():
            self.find_matching_pattern(key)

    def printAllTemporalPatterns(self):
        for key, value in self.PATTERN_MATCHED_TEMPORAL.items():
            if self.timed_trie_model.ENABLE_DEBUG:
                print("For Temporal Pattern : ", key)

            for _pat in value:
                result_pattern = "{0}  -    {1}".format(_pat[0][0], _pat[0][1])

                if self.timed_trie_model.ENABLE_DEBUG:
                    print(result_pattern, " : -----------")

                _pattern_count = [key, result_pattern] + self.dominant_pattern_miner_model.printFromPatternList(True)  # Print the pattern
                self.temporal_patterns_pat_str_count.append(_pattern_count)

            if self.timed_trie_model.ENABLE_DEBUG:
                print("--------")

    def printAllTemporalPatterns_pretty(self):
        self.printAllTemporalPatterns()

        self.temporalPatternList_DF = pd.DataFrame(self.temporal_patterns_pat_str_count,
                                              columns=["Type", "FromToEvent", "Pattern", "Count", "Support",
                                                       "Confidence"])
        self.temporalPatternList_DF.sort_values(by=["Count"], ascending=False, inplace=True)

        print(tabulate(self.temporalPatternList_DF, headers='keys', tablefmt='psql'))

    """
        Save Result
    """
    def save_result(self):
        self.temporalPatternList_DF.to_csv("./Result/temporal_patterns.csv");