import numpy as np
import pandas as pd
import argparse
import Utility.Util as util
from TimedTrie import TimedTrie
from PatternMiner import DominantPatternMiner, TemporalPatternMiner

"""
    Read input file path from argument
"""

parser = argparse.ArgumentParser(description='Temporal pattern extractor')
parser.add_argument('--inputpath', help='Event file path. Must be a valid .csv format', required=True)
args = parser.parse_args()

timed_trace = util.preprocess_input_file(args.inputpath)

if timed_trace == None:
    pass

# print("Trace Length ", len(timed_trace[0]))
# print("Unique Events ", len(set(timed_trace[0])))


"""
    Define Params
"""
params = {
    'depth': 5,
    'enable_iqr' : True,
    'std_threshold' : 0.675,
    'property_ext_threshold': 0.5,
    'strict_event_prob': True,
    # 'DEBUG': False
}


"""
    Build Model
"""
timed_trie_model = TimedTrie(params)
timed_automata = timed_trie_model.buildGraph(timed_trace)

"""
    Visualize
"""


"""
    Extract Pattern
"""

# Dominant Patterns
dominant_pattern_miner_model = DominantPatternMiner.PatternMiner(timed_automata, timed_trace)
dominant_pattern_miner_model.extractPattern(timed_automata)
dominant_pattern_miner_model.printPatternList_pretty()

# Temporal Patterns
temporal_pattern_miner_model = TemporalPatternMiner.TemporalPatternMiner(timed_automata, timed_trace)
temporal_pattern_miner_model.extractTemporalPatterns()
temporal_pattern_miner_model.printAllTemporalPatterns_pretty()


"""
    
"""