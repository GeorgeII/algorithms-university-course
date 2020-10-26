import networkx as nx
from matplotlib import pyplot as plt
import random


def main():
    # generate some complete graphs
    graphs = []
    for i in range(4, 11):
        graphs.append(nx.complete_graph(i).to_directed())

    # assign random capacity (throughput) and current flow (initialize with 0) to all edges
    for graph in graphs:
        for (u, v) in graph.edges():
            graph.edges[u, v]['capacity'] = random.randint(1, 30)
            graph.edges[u, v]['flow'] = 0

    print(graphs[0].edges)
    #ford_fulkerson(graphs[1], 0, 3)

    G = nx.Graph()
    G.add_nodes_from([0,1,2,3,4,5])
    G.add_edge(0, 1)
    G.edges[0, 1]['capacity'] = 4
    G.add_edge(1, 2)
    G.edges[1, 2]['capacity'] = 4
    G.add_edge(2, 5)
    G.edges[2, 5]['capacity'] = 2
    G.add_edge(0, 4)
    G.edges[0, 4]['capacity'] = 3
    G.add_edge(3, 4)
    G.edges[3, 4]['capacity'] = 6
    G.add_edge(4, 5)
    G.edges[4, 5]['capacity'] = 6
    G.add_edge(2, 3)
    G.edges[2, 3]['capacity'] = 3
    for (u, v) in G.edges():
        G.edges[u, v]['flow'] = 0
    print(ford_fulkerson(G, 0, 5))
    print(nx.algorithms.flow.maximum_flow(G, 0, 5))

    G2 = nx.Graph()
    G2.add_nodes_from([0, 1, 2, 3, 4, 5])
    G2.add_edge(0, 1)
    G2.edges[0, 1]['capacity'] = 16
    G2.add_edge(1, 2)
    G2.edges[1, 2]['capacity'] = 10
    G2.add_edge(1, 3)
    G2.edges[1, 3]['capacity'] = 12
    G2.add_edge(2, 1)
    G2.edges[2, 1]['capacity'] = 4
    G2.add_edge(2, 4)
    G2.edges[2, 4]['capacity'] = 14
    G2.add_edge(0, 2)
    G2.edges[0, 2]['capacity'] = 13
    G2.add_edge(3, 2)
    G2.edges[3, 2]['capacity'] = 9
    G2.add_edge(3, 5)
    G2.edges[3, 5]['capacity'] = 20
    G2.add_edge(4, 5)
    G2.edges[4, 5]['capacity'] = 4
    G2.add_edge(4, 3)
    G2.edges[4, 3]['capacity'] = 7
    for (u, v) in G2.edges():
        G2.edges[u, v]['flow'] = 0
    print(ford_fulkerson(G2, 0, 5))
    print(nx.algorithms.flow.maximum_flow(G2, 0, 5))

    pos = nx.spring_layout(G2)
    labels = nx.get_edge_attributes(G2, 'capacity')
    nx.draw_networkx(G2, pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(graphs[0], pos, edge_labels=labels)

    plt.show()
    '''pos = nx.spring_layout(graphs[0])
    labels = nx.get_edge_attributes(graphs[0], 'capacity')
    nx.draw_networkx(graphs[0], pos, with_labels=True, font_weight='bold')
    nx.draw_networkx_edge_labels(graphs[0], pos, edge_labels=labels)

    plt.show()'''


def ford_fulkerson(graph: nx.Graph, source_node, sink_node):
    max_flow = 0

    path_exist, path = find_path(graph, source_node, sink_node)
    print(path)

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


if __name__ == "__main__":
    main()
