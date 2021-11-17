import networkx as nx
import ql
import numpy as np
import time
import matplotlib.pyplot as plt
import itertools

def main():
    new_net = nx.DiGraph()
    list_nodes = [i for i in range(1,8)]
    actions = list_nodes
    # Defining nodes neighbohrs 
    next_a = [2,3]
    next_b = [1, 4, 5]
    next_c = [1, 6, 7]
    next_d = [2]
    next_e = [2]
    next_f = [3]
    next_g = [3]

    new_net.add_nodes_from(list_nodes)
    new_net.nodes()

    ws = [25, 32, 10, 7]

    list_links = [(1, 2, ws[0]), (1, 3, ws[1]), (2, 4, ws[2]), (2, 5, ws[2]), (3, 6, ws[3]), (3, 7, ws[3])]
    new_net.add_weighted_edges_from(list_links)
    new_net.edges()

    new_net.nodes[1]['pos'] = (2, 2)
    new_net.nodes[2]['pos'] = (1, 1)
    new_net.nodes[3]['pos'] = (3, 1)
    new_net.nodes[4]['pos'] = (0, 0)
    new_net.nodes[5]['pos'] = (1, 0)
    new_net.nodes[6]['pos'] = (3, 0)
    new_net.nodes[7]['pos'] = (4, 0)

    node_posterior = nx.get_node_attributes(new_net, 'pos')
    nx.draw_networkx(new_net, node_posterior, node_size=450)
    arc_weight = nx.get_edge_attributes(new_net, 'height')
    nx.draw_networkx_edge_labels(new_net, node_posterior, edge_labels=arc_weight)
    plt.show()
    container = [actions, [6,7]]
    activator = list(itertools.product(*container))
    print(activator)

if __name__ == "__main__":
    main()
