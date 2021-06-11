import sys

class TrieNode(object):
    def __init__(self, id: int, char: str):
        self.node_id = "Node_" + str(id)
        self.char = char
        self.children = []
        self.count = 1
        self.t_min = sys.maxsize
        self.t_max = 0
        self.t_mean = 0
        self.t_list = []
        self.t_var = 0
        self.dropped = 0
        self.prob = 0.0  # probability of occuring during transition
        self.prob_pattern = 0.0  # probability of pattern occuring
        self.tree_height = 0  # height of the tree considering the present node as root
        self.is_end = True