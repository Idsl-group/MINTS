import numpy as np
import pandas as pd
import argparse
import Utility.Util as util
from TimedTrie import TimedTrie
from PatternMiner import DominantPatternMiner, TemporalPatternMiner
from Utility import SampleTraceUtil, TimedTrieVisualizer
from Predictor import EventPredictor

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
   Event Prediction 
"""

# Replace the following line with a trace for which the next event has to be predicted
sample_sub_trace = SampleTraceUtil(params.depth, timed_trace)

event_predicted_model = EventPredictor.EventPredictor(timed_automata, timed_trace)
event_predicted_model.predict(sample_sub_trace)
event_predicted_model.plot_time_probability()

"""
    Visualize
"""

TimedTrieVisualizer.renderTrie(timed_automata, "State diagram", True)
TimedTrieVisualizer.renderTrie(timed_automata, "Transition diagram", False)

# Only for DOMINANT PATTERN Visualization (this is pruned version)
# pruned_timed_automata = dominant_pattern_miner_model.buildGraphPrunedTrie() # with dominant patterns
# TimedTrieVisualizer.renderTrie(pruned_timed_automata, "timed_trie_pruned_state_diagram", True)
# TimedTrieVisualizer.renderTrie(pruned_timed_automata, "timed_trie_pruned", False)

