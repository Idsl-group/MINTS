from graphviz import Digraph

import TrieNode
from Utility import Util

def traverseAndCreaseDigraph(dot: Digraph, node: TrieNode, state_graph=False):
    parent_id = node.char if not state_graph else Util.getStateId(node.node_id)

    if not node.children:
        dot.attr('node', shape="doublecircle")
    else:
        dot.attr('node', shape="circle")

    dot.node(node.node_id, parent_id)

    tot_count = 1
    for _child in node.children:
        tot_count += _child.count

    for child in node.children:

        prefix = "" if not state_graph else (child.char + " , ");
        edge_label = prefix + "[" + str(child.t_min) + "," + str(child.t_max) + "]";
        if state_graph:
            edge_label += " , " + str(child.prob) + ", " + str(child.prob_pattern)
        else:
            edge_label += " { Count : " + str(child.count) + " Dropped : " + str(child.dropped) + " }";

        child_id = child.char if not state_graph else Util.getStateId(child.node_id)

        if not child.children:
            dot.attr('node', shape="doublecircle")
        else:
            dot.attr('node', shape="circle")

        dot.node(child.node_id, child_id)

        dot.edge(node.node_id, child.node_id, label=edge_label)
        traverseAndCreaseDigraph(dot, child, state_graph)


def visualizeTrie(timed_trie: TrieNode, state_graph=False):
    dot = Digraph(comment='Timed Trie', format='png')
    traverseAndCreaseDigraph(dot, timed_trie, state_graph)
    return dot


def renderTrie(timed_trie: TrieNode, diagram_name="timed_trie", state_graph=False):
    timed_trie_viz = visualizeTrie(timed_trie, state_graph)
    timed_trie_viz.render(diagram_name + '.gv', directory="./Result", view=True)
    return timed_trie_viz
