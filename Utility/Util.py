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
