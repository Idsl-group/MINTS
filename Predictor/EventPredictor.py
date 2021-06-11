import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tabulate import tabulate

import TimedTrie
import TrieNode
from Utility import Util


class EventPredictor:

    def __init__(self, params, timedTrie: TrieNode, timed_trace, timed_trie_model: TimedTrie):
        self.VARIABILITY_OF_LOOKBACK = getattr(params, 'variable_lookback', True)
        self.PLOT_PROBABILITY_ALPBHABET_MAX_LENGTH = getattr(params, 'ignore_event_prob_plot_len_above',
                                                             15);  # Plot fails for large alpbhabet size

        self.timed_trie_model = timed_trie_model
        self.timed_trace = timed_trace  # the entire trace
        self.Alphabets = sorted(list(set(self.timed_trace[0])))
        self.timedTrie = timedTrie

    def evaluate_time_probability_using_count(self):
        for index, val in enumerate(self.time_probability):
            _count_list_total = sum(val)
            if _count_list_total > 0:
                _prob_list = [round(float(e / _count_list_total), 2) for e in val]
                self.time_probability[index] = _prob_list

    def fill_time_probability_mat(self, root: TrieNode, sub_sub_trace):
        pos = 0;
        sub_sub_trace_timeShifted = Util.getTimeShift(sub_sub_trace[1])

        while pos < len(sub_sub_trace[0]):
            _e, _t = sub_sub_trace[0][pos], sub_sub_trace_timeShifted[pos]
            for _child in root.children:
                if _child.char == _e and _child.t_min <= _t <= _child.t_max:
                    root = _child
                    if pos == len(sub_sub_trace[0]) - 1:
                        # Now fill the
                        # for ___child in _child.children:
                        for time_step in list(range(len(self.time_probability))):
                            for __child in _child.children:
                                all_time_transitions = __child.t_list
                                _pos_of_char_in_Alphabet = self.Alphabets.index(__child.char)
                                _count = len(list(filter(lambda _x: _x == time_step + 1, all_time_transitions)))
                                self.time_probability[time_step][_pos_of_char_in_Alphabet] += _count

                    break;  # break for

            pos += 1

        return

    def fill_time_probability_mat_on_lookBack(self, sub_sub_trace, variable_lookback: bool = True):

        subTrace_len = len(sub_sub_trace[0])
        root = self.timedTrie

        for _k in list(range(subTrace_len, 0, -1)):
            if variable_lookback == False and _k < subTrace_len:
                break;

            sub_sub_trace_event = sub_sub_trace[0][-_k:]
            sub_sub_trace_time = sub_sub_trace[1][-_k:]

            sub_sub_trace_timeEvent = (sub_sub_trace_event, sub_sub_trace_time)

            if self.timed_trie_model.ENABLE_DEBUG:
                print("k ", str(_k), "   filling --- ", sub_sub_trace_timeEvent)

            self.fill_time_probability_mat(root, sub_sub_trace_timeEvent)

        if self.timed_trie_model.ENABLE_DEBUG:
            print("Count list ", self.time_probability)
        self.evaluate_time_probability_using_count()

    def predict(self, subtrace_timed):
        self.tp_max = Util.get_time_length_based_on_lookback(self.timedTrie, subtrace_timed, self.VARIABILITY_OF_LOOKBACK)  # Look into this
        self.time_probability = np.zeros((self.tp_max, len(self.Alphabets)))

        self.fill_time_probability_mat_on_lookBack(subtrace_timed)

        time_probability_DF = pd.DataFrame(self.time_probability, columns=self.Alphabets, index=list(range(1, self.tp_max + 1)))
        time_probability_DF.index.name = "Time (t)"
        print(tabulate(time_probability_DF, headers='keys', tablefmt='psql'))

    """#Plot Area Timed_Area graph"""

    def plot_time_probability(self, save_figure=False):

        if len(self.Alphabets) > self.PLOT_PROBABILITY_ALPBHABET_MAX_LENGTH:
            print("Skipping plotting of probability graph since alphabet size more than {0}".format(
                self.timedTrie.PLOT_PROBABILITY_ALPBHABET_MAX_LENGTH))
        else:
            y = np.vstack(self.time_probability)
            x = np.arange(len(self.Alphabets))  # label location
            # barWidth = 0.2
            barWidth = float(1.0 / (2 * len(self.Alphabets)))

            fig, ax = plt.subplots()
            ax.set_ylabel('Probability')
            ax.set_xlabel("Time (sec)")
            ax.set_title('Probablity of Event transition over time')

            r = np.arange(len(y.T[0]))
            for idx, val in enumerate(y.T):
                _label = self.Alphabets[idx]
                plt.bar(r, val, width=barWidth, label=_label)

                # Set position of bar on X axis
                r = [x + barWidth for x in r]

            plt.xticks([r + barWidth for r in range(len(y))], list(range(1, len(self.time_probability) + 1)))
            plt.legend()

            if save_figure == True:
                plt.savefig("./Result/event_prediction_plot.png")

            plt.show()

