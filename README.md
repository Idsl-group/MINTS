# MINTS: Unsupervised Temporal Specification Miner

Welcome. This repository contains implementation for our paper called 'MINTS: Unsupervised Temporal Specification Miner' 

## Objective
* Build a Time Automaton (TA) using event-time series trace.
* Extract Dominant Properties from the TA
* Extract Temporal Properties from the TA
* Provide capability for event prediction using TA and prediction_trace

## Package Requirement

**OS (tested on)**: Ubuntu 18.04.5 LTS x86_64 \
**System Spec (tested on)** Intel i7-2630QM (8) @ 2.900GHz, 8 GB DDR3, 256 GB SSD \
**Language:** Python 3 \
**Libraries**

```
    1. rstr (For synthetic trace generation)
    2. pyvis (Interactive Python Visualizer)
    3. graphviz pygraphviz (For drawing the Timed Automaton)
    4. seaborn, tabulate (For ploting result)
```

The above packages can be silently installed using
```
    !pip install -q graphviz seaborn tabulate rstr pyvis
    !apt-get -qq install graphviz graphviz-dev -y && pip install -q pygraphviz;
```

## Folder Structure
```
    ├── Dataset     [Contains the Dataset for pattern extraction. Sample trace files provided]
    │   ├── synthetic_trace2.csv
    │   └── synthetic_trace.csv
    ├── LICENSE
    ├── PatternMiner        [Dominant and Temporal Pattern Miners]
    │   ├── DominantPatternMiner.py
    │   ├── TemporalPatternMiner.py
    │   └── TemporalPatternTemplate.py
    ├── Predictor           [Event Predictor]
    │   ├── EventPredictor.py
    ├── README.md
    ├── Result              [Results are stored in this folder. Some common files are dominant/temporal pattern csv, State diagram, Transition diagram and Event Prediction plot]
    │   ├── dominant_patterns.csv
    │   ├── event_prediction_plot.png
    │   ├── State diagram.gv
    │   ├── State diagram.gv.png
    │   ├── temporal_patterns.csv
    │   ├── Transition diagram.gv
    │   └── Transition diagram.gv.png
    ├── temporalMiner.py    [Sample file providing example on the usage of the model]
    ├── TimedTrie.py        [Base class for TimedTrie]
    ├── TrieNode.py         [Base class for TrieNode]
    └── Utility             [Util folder]
        ├── SampleTraceUtil.py
        ├── TimedTrieVisualizer.py
        └── Util.py
    
    8 directories, 12 files
```

Sample file (`temporalMiner.py`) is divided into 5 parts. They are as follow:

1. Necessary imports, reading trace file, preprocessing
2. Build Model
3. Extract Pattern (Dominant and Temporal)
4. Event Prediction (using a trace for which the next subsequent event has to be predicted)
5. Visualize the Timed Automaton.
6. Save result (miscellaneous)

## Usage
For quick usage, the following command can be executed

` python temporalMiner.py --inputpath Dataset/synthetic_trace.csv `
or \
` python temporalMiner.py --inputpath Dataset/hexcopter_trace.csv `

For additional tweaking with the model, please see `temporalMiner.py`. It contains more description on the how to use each module. Major building blocks are to 
* Read trace file
* Build Timed Trie Model
* Extract patterns
* Visualize
* Perform event prediction (if there is a necessity to predict the next subsequent event)

## Result
#### Hexcopter Result

Five interesting dominant properties extracted from the Hexcopter trace (`hexcopter_trace.csv`)
![Dominant Properties](Result/Five%20Dominant%20Property%20Hexcopter.png)

Few Temporal properties extracted from the Hexcopter trace (`hexcopter_trace.csv`)
![Temporal Properties](Result/Temporal%20Property%20Hexcopter.png)

### Synthetic Result

#### Complexity analysis

[Please note, with every execution the results will change however it should follow the below trend. The following results are an average of 10 trials]

![Trace Length vs Time](Result/Trace%20Length%20vs%20Time.png)

![Alpbabet Size vs Time](Result/Alpbabet%20Size%20vs%20Time.png)

![Depth of Timed Trie vs Time](Result/Depth%20of%20Timed%20Trie%20vs%20Time.png)


## License
Restricted License. Cannot be replicated or used without prior authorization.