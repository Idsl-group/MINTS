import pandas as pd

import TrieNode

def getNodeId(node_id: str):
    if len(node_id.split("_")) > 1:
        return str(int(node_id.split("_")[1]) - 1);
    return "-1"


def getStateId(node_id: str):
    return "q_" + getNodeId((node_id));


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

    contains_negative = [True if _ele < 0 else False for _ele in time_new]
    if True in contains_negative:
        print("This list contains negative value ", time)
        print("Processed list ", time_new)

    return time_new


def doesStringContains(str="", whichWord=""):
    return str.find(whichWord) > -1


def preprocess_input_file(file_path=""):
    """
        Read and preprocess input file

        :param file_path: Input file path
        :return: Tuple of event and time
    """

    if file_path == "":
        print("Provide an input file path")
        return

    trace_df = pd.read_csv(file_path)

    time = list(map(int, list(trace_df.loc[:, "Time"].values)))
    event = list(trace_df.loc[:, "Event"].values)

    return (list(event), time)


def determine_time_length(timed_trie: TrieNode, sub_sub_trace=[]):
    t_max = -1;
    pos = 0;
    sub_sub_trace_timeShifted = getTimeShift(sub_sub_trace[1])

    root = timed_trie

    while pos < len(sub_sub_trace[0]):
        _e, _t = sub_sub_trace[0][pos], sub_sub_trace_timeShifted[pos]
        for _child in root.children:
            if _child.char == _e and _child.t_min <= _t and _child.t_max >= _t:

                if pos == len(sub_sub_trace[0]) - 1:
                    for ___child in _child.children:
                        t_max = max(___child.t_list)

                root = _child
                break;  # break for

        pos += 1

    return t_max;


def get_time_length_based_on_lookback(timed_trie: TrieNode, subTrace, variable_lookback: bool = True):
    t_max = 0;
    subTrace_len = len(subTrace[0])

    for _k in list(range(subTrace_len, 0, -1)):
        if variable_lookback == False and _k < subTrace_len:
            break;

        sub_sub_trace = subTrace[0][-_k:]
        sub_sub_trace_time = subTrace[1][-_k:]

        __t_max_sub = determine_time_length(timed_trie, (sub_sub_trace, sub_sub_trace_time))
        t_max = max(t_max, __t_max_sub)

    return t_max
