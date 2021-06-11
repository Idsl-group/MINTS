import numpy as np
import pandas as pd
import argparse
import Utility.Util as util
from TimedTrie import TimedTrie
from PatternMiner import DominantPatternMiner, TemporalPatternMiner
from Utility import SampleTraceUtil, TimedTrieVisualizer
from Predictor import EventPredictor

# ==================   PART 1: Necessary imports / read trace / preprocessing (i.e only selecting event and timestamp) ==================

"""
    Read input file path from argument
"""

parser = argparse.ArgumentParser(description='Temporal pattern extractor')
parser.add_argument('--inputpath', help='Event file path. Must be a valid .csv format', required=True)
args = parser.parse_args()

timed_trace = util.preprocess_input_file(args.inputpath)

if timed_trace == None:
    pass

print("Trace Length ", len(timed_trace[0]))
print("Unique Events ", len(set(timed_trace[0])))


# ==================   PART 2: Build Model ==================
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


# ==================   PART 3: Extract Pattern (Dominant / Temporal) ==================
"""
    Extract Pattern
"""

# Dominant Patterns
dominant_pattern_miner_model = DominantPatternMiner.PatternMiner(timed_automata, timed_trace, timed_trie_model)
dominant_pattern_miner_model.extractPattern(timed_automata)
dominant_pattern_miner_model.printPatternList_pretty()

# Temporal Patterns
temporal_pattern_miner_model = TemporalPatternMiner.TemporalPatternMiner(timed_automata, timed_trace, timed_trie_model, dominant_pattern_miner_model)
temporal_pattern_miner_model.extractTemporalPatterns()
temporal_pattern_miner_model.printAllTemporalPatterns_pretty()


# ==================   PART 4: Event Prediction ==================
"""
   Event Prediction 
"""

# Replace the following line with a trace for which the next event has to be predicted
sample_sub_trace = SampleTraceUtil.getRandomSubtrace(params['depth'], timed_trace)
print("Event Prediction")
print("Predict subsequent event for Trace: ", sample_sub_trace)

event_predicted_model = EventPredictor.EventPredictor(params, timed_automata, timed_trace, timed_trie_model)
event_predicted_model.predict(sample_sub_trace)
event_predicted_model.plot_time_probability(True)


# ==================   PART 5: How to Visualize the Timed Automata ==================
"""
    Visualize
"""

TimedTrieVisualizer.renderTrie(timed_automata, "State diagram", True)
TimedTrieVisualizer.renderTrie(timed_automata, "Transition diagram", False)

# Only for DOMINANT PATTERN Visualization (this is pruned version)
# pruned_timed_automata = dominant_pattern_miner_model.buildGraphPrunedTrie() # with dominant patterns
# TimedTrieVisualizer.renderTrie(pruned_timed_automata, "timed_trie_pruned_state_diagram", True)
# TimedTrieVisualizer.renderTrie(pruned_timed_automata, "timed_trie_pruned", False)


# ==================   (Misc.) PART 6: Save Result ==================
"""
    Save results
"""
dominant_pattern_miner_model.save_result()
temporal_pattern_miner_model.save_result()