"""
Maximum flow problem.
Two algorithms are implemented: Ford-Fulkerson (concretely, Edmonds-Karp, because it uses a breadth-first search algo
to find a path) and Dinic's algorithm.
"""


import networkx as nx
from matplotlib import pyplot as plt
import random
import math
from time import time
import numpy as np


def main():
    # generate some complete graphs
    graphs = []
    for i in range(5, 20):
        graphs.append(nx.complete_graph(i).to_directed())

    # assign random capacity (throughput) and current flow (initialize with 0) to all edges
    for graph in graphs:
        for (u, v) in graph.edges():
            graph.edges[u, v]['capacity'] = random.randint(1, 30)
            graph.edges[u, v]['flow'] = 0

    # in case you wnat to visualize graph
    '''pos = nx.spring_layout(graphs[0])
        labels = nx.get_edge_attributes(graphs[0], 'capacity')
        nx.draw_networkx(graphs[0], pos, with_labels=True, font_weight='bold')
        nx.draw_networkx_edge_labels(graphs[0], pos, edge_labels=labels)

        plt.show()'''

    print("Examples:")
    print("Graph 1: ")
    print("Solution by Ford-Fulkerson algorithm: ", ford_fulkerson(graphs[1], 0, 3))
    print("Solution by Dinic's algorithm: ", dinic(graphs[1], 0, 3))
    print("Solution by Networkx built-in function: ", nx.algorithms.flow.maximum_flow(graphs[1], 0, 3)[0])

    print("Graph 2: ")
    print("Solution by Ford-Fulkerson algorithm: ", ford_fulkerson(graphs[5], 0, 8))
    print("Solution by Dinic's algorithm: ", dinic(graphs[5], 0, 8))
    print("Solution by Networkx built-in function: ", nx.algorithms.flow.maximum_flow(graphs[5], 0, 8)[0])


    # Synthetic testing.

    # lists of lists [time, max_flow, nodes_quantity]
    test_metrics_ford_fulkerson = []
    test_metrics_dinic = []
    test_metrics_newtowrx = []
    for graph in graphs:
        nodes_number = len(graph.nodes)
        print("Number of nodes: ", len(graph.nodes))

        start_time = time()
        max_flow = ford_fulkerson(graph, 0, 3)
        finish_time = time()
        test_metrics_ford_fulkerson.append([max_flow, finish_time - start_time, nodes_number])
        print(max_flow)

        start_time = time()
        max_flow = dinic(graph, 0, 3)
        finish_time = time()
        test_metrics_dinic.append([max_flow, finish_time - start_time, nodes_number])
        print(max_flow)

        start_time = time()
        max_flow = nx.algorithms.flow.maximum_flow(graph, 0, 3)[0]
        finish_time = time()
        test_metrics_newtowrx.append([max_flow, finish_time - start_time, nodes_number])
        print(max_flow)

    # compare performance of algorithms
    test_metrics_ford_fulkerson = np.array(test_metrics_ford_fulkerson)
    test_metrics_dinic = np.array(test_metrics_dinic)
    test_metrics_newtowrx = np.array(test_metrics_newtowrx)

    # plot time performance
    plt.plot(test_metrics_ford_fulkerson[:, 2], test_metrics_ford_fulkerson[:, 1], color='green', label='Ford-Fulkerson')
    plt.plot(test_metrics_dinic[:, 2], test_metrics_dinic[:, 1], color='blue', label="Dinic's algorithm")
    plt.plot(test_metrics_newtowrx[:, 2], test_metrics_newtowrx[:, 1], color='red', label='NetworkX built-in function (Preflow-Push algo)')
    plt.legend()
    plt.xlabel('Number of nodes')
    plt.ylabel('Performance time, seconds')
    plt.show()




def ford_fulkerson(graph: nx.Graph, source_node, sink_node):
    graph = graph.copy()
    max_flow = 0

    path_exist, path = find_path(graph, source_node, sink_node)
    #print(path)

    while path_exist:
        # get possible flows on every edge via formula: capacity - flow
        all_possible_flows = []
        for i in range(len(path) - 1):
            possible_flow = graph.edges[path[i], path[i+1]]['capacity'] - graph.edges[path[i], path[i+1]]['flow']
            all_possible_flows.append(possible_flow)

        min_possible_flow = min(all_possible_flows)
        # update flow and reverse-edges flow
        for i in range(len(path) - 1):
            graph.edges[path[i], path[i + 1]]['flow'] += min_possible_flow

            if not graph.has_edge(path[i + 1], path[i]):
                graph.add_edge(path[i + 1], path[i])
                graph.edges[path[i + 1], path[i]]['capacity'] = graph.edges[path[i], path[i+1]]['capacity']
                graph.edges[path[i + 1], path[i]]['flow'] = 0

            graph.edges[path[i + 1], path[i]]['flow'] = graph.edges[path[i], path[i + 1]]['flow']

        max_flow += min_possible_flow

        path_exist, path = find_path(graph, source_node, sink_node)

    return max_flow


def find_path(graph: nx.Graph, start_node, finish_node):
    """
    Breadth First Search for path between 2 given nodes. Also, checks if the capacity of a path more than 0.
    :return boolean (path to sink is found), path (list of nodes)
    """

    graph = graph.copy()
    nx.set_node_attributes(graph, False, name='visited')

    path_references = [-1] * len(graph.nodes)
    queue = [start_node]

    while queue:
        current_node = queue.pop(0)
        for neighbor in graph.neighbors(current_node):
            residual = graph.edges[current_node, neighbor]['capacity'] - graph.edges[current_node, neighbor]['flow']
            if not graph.nodes[neighbor]['visited'] and residual > 0:
                queue.append(neighbor)
                graph.nodes[neighbor]['visited'] = True
                path_references[neighbor] = current_node

            graph.nodes[current_node]['visited'] = True

    # get the path
    path = []
    if graph.nodes[finish_node]['visited']:
        current_node = finish_node
        while current_node is not start_node:
            path.append(current_node)
            current_node = path_references[current_node]
        path.append(start_node)

    path.reverse()

    return graph.nodes[finish_node]['visited'], path


def dinic(graph: nx.Graph, source_node, sink_node):
    graph = graph.copy()
    nx.set_node_attributes(graph, math.inf, name='level')

    max_flow = 0
    path_exist, path = find_path_dinic(graph, source_node, sink_node)

    while path_exist:
        flow_to_add = depth_first_search(graph, source_node, sink_node, math.inf, list())
        while flow_to_add > 0:
            max_flow += flow_to_add
            flow_to_add = depth_first_search(graph, source_node, sink_node, math.inf, list())

        path_exist, path = find_path_dinic(graph, source_node, sink_node)

    return max_flow



def find_path_dinic(graph: nx.Graph, start_node, finish_node):
    """
    Upgraded find_path function. Adds levels to nodes while finding a path.
    """

    graph = graph.copy()
    nx.set_node_attributes(graph, False, name='visited')

    path_references = [-1] * len(graph.nodes)
    queue = [start_node]
    graph.nodes[start_node]['level'] = 0

    while queue:
        current_node = queue.pop(0)
        for neighbor in graph.neighbors(current_node):
            x = graph.edges[current_node, neighbor]['capacity']
            y = graph.edges[current_node, neighbor]['flow']
            residual = graph.edges[current_node, neighbor]['capacity'] - graph.edges[current_node, neighbor]['flow']
            if not graph.nodes[neighbor]['visited'] and residual > 0 and \
                    graph.nodes[neighbor]['level'] >= graph.nodes[current_node]['level'] + 1:
                queue.append(neighbor)
                graph.nodes[neighbor]['visited'] = True
                graph.nodes[neighbor]['level'] = graph.nodes[current_node]['level'] + 1
                path_references[neighbor] = current_node

            graph.nodes[current_node]['visited'] = True

    # get the path
    path = []
    if graph.nodes[finish_node]['visited']:
        current_node = finish_node
        while current_node is not start_node:
            path.append(current_node)
            current_node = path_references[current_node]
        path.append(start_node)

    path.reverse()

    return graph.nodes[finish_node]['visited'], path


def depth_first_search(graph, current_node, finish_node, current_flow, path):
    if current_node == finish_node:
        return current_flow

    path.append(current_node)

    available_neighbors = [n for n in graph.neighbors(current_node) if n not in path]

    for neighbor in available_neighbors:
        if graph.nodes[neighbor]['level'] == graph.nodes[current_node]['level'] + 1:
            residual = graph.edges[current_node, neighbor]['capacity'] - graph.edges[current_node, neighbor]['flow']
            if residual > 0:
                current_flow = min(current_flow, residual)
                additional_flow = depth_first_search(graph, neighbor, finish_node, current_flow, path)
                if additional_flow > 0:
                    graph.edges[current_node, neighbor]['flow'] += additional_flow

                    if not graph.has_edge(neighbor, current_node):
                        graph.add_edge(neighbor, current_node)
                        graph.edges[neighbor, current_node]['capacity'] = graph.edges[current_node, neighbor]['capacity']
                        graph.edges[neighbor, current_node]['flow'] = 0

                    graph.edges[neighbor, current_node]['flow'] = graph.edges[current_node, neighbor]['flow']

                    return additional_flow
    return 0


if __name__ == "__main__":
    main()
