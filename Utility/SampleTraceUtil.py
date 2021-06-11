import random

def getRandomSubtrace(depth, timed_trace):
    trace_size = len(timed_trace[0])

    k = random.randint(1, depth);

    i_pos = random.randint(0, trace_size - depth)
    sub_event = timed_trace[0][i_pos:i_pos + k]
    sub_time = timed_trace[1][i_pos:i_pos + k]

    return (sub_event, sub_time)