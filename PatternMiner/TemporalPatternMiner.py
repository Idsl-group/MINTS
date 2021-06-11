import re

import TemporalPatternTemplate as temporalTemplate
import TimedTrie
from Utility import Util


class TemporalPatternMiner:
    def __init__(self, timedTrie: TimedTrie, timed_trace):

        self.temporal_patterns_pat_str_count = []

        self.timedTrie = timedTrie
        self.timed_trace = timed_trace  # the entire trace

        self.PATTERN_MATCHED_TEMPORAL = {
            "RESPONSE_PATTERN_REGEX": [],
            "ALTERNATING_PATTERN_REGEX": [],
            "MULTIEFFECT_PATTERN_REGEX": [],
            "MULTICAUSE_PATTERN_REGEX": []
        }

        self.Alphabets = sorted(list(set(self.timed_trace[0])))

    """#### Begin extraction of temporal properties """

    def find_matching_pattern(self, extracted_pattern: list = [], interested_pattern: str = "RESPONSE_PATTERN_REGEX"):

        pattern_regex = temporalTemplate.PATTERN_REGEX_DIC[interested_pattern]

        for alpha_1 in self.Alphabets:
            for alpha_2 in self.Alphabets:
                if alpha_1 != alpha_2:

                    matching_regex = pattern_regex.format(alpha_1, alpha_2)
                    if self.timedTrie.ENABLE_DEBUG:
                        print("For ", interested_pattern, "   Matching with ", matching_regex)
                    _currentPattern = ""
                    for _ele in extracted_pattern:
                        _cp = [s.char for s in _ele]
                        _currentPattern = ''.join(_cp)

                        try:
                            if re.fullmatch(matching_regex, _currentPattern):
                                if self.timedTrie.ENABLE_DEBUG:
                                    print("Matched with ", _currentPattern)

                                if self.timedTrie.ENABLE_STRICT_TEMPORAL_MATCH:
                                    if Util.doesStringContains(_currentPattern, alpha_1) and Util.doesStringContains(
                                            _currentPattern, alpha_2):
                                        self.PATTERN_MATCHED_TEMPORAL[interested_pattern].append(
                                            [(alpha_1, alpha_2), _ele])
                                else:
                                    self.PATTERN_MATCHED_TEMPORAL[interested_pattern].append([(alpha_1, alpha_2), _ele])

                        except Exception as e:
                            print(e)
                            print("matching_regex ", matching_regex, "  _currentPattern ", _currentPattern)

    def find_all_temporal_patterns(self, extracted_pattern: list = []):
        for key, value in temporalTemplate.PATTERN_REGEX_DIC.items():
            self.find_matching_pattern(extracted_pattern, key)

    def printAllTemporalPatterns(self):
        for key, value in self.PATTERN_MATCHED_TEMPORAL.items():
            if self.timedTrie.ENABLE_DEBUG:
                print("For Temporal Pattern : ", key)

            for _pat in value:
                result_pattern = "{0}  -    {1}".format(_pat[0][0], _pat[0][1])

                if self.timedTrie.ENABLE_DEBUG:
                    print(result_pattern, " : -----------")

                _pattern_count = [key, result_pattern] + self.printFromPatternList([_pat[1]], [], True)  # Print the pattern
                self.temporal_patterns_pat_str_count.append(_pattern_count)

            if self.timedTrie.ENABLE_DEBUG:
                print("--------")
