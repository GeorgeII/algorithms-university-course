import networkx as nx
from matplotlib import pyplot as plt
import random
import numpy as np


def main():
    # generate some complete graphs
    graphs = []
    for i in range(4, 10):
        graphs.append(nx.complete_graph(i))

    # generate 3 more simpler graphs with more nodes
    graphs.append(nx.wheel_graph(10))
    graphs.append(nx.wheel_graph(11))
    graphs.append(nx.connected_caveman_graph(3, 4))

    print('Number of graphs: ', len(graphs))

    # assign random weights to edges
    for graph in graphs:
        for (u, v) in graph.edges():
            graph.edges[u, v]['weight'] = random.randint(1, 15)

    # visualize generated graphs
    '''for graph in graphs:
        plt.figure(figsize=(10, 10))

        pos = nx.spring_layout(graph)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx(graph, pos, with_labels=True, font_weight='bold')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        #plt.show()
        plt.savefig(str(graph.number_of_nodes()) + ' nodes.png')
    '''
    # solving and synthetic testing
    #print(bruteforce(graphs[5], 1))
    ant_colony(graphs[0], 0)


def bruteforce(graph: nx.Graph, start_and_end_node):
    current_node = start_and_end_node

    nodes_left = set(graph.nodes)
    path = [start_and_end_node]

    all_solutions = []

    bruteforce_helper(graph, start_and_end_node, current_node=current_node, path=path, path_length=0,
                                           nodes_left=nodes_left, all_solutions=all_solutions)

    best_path_length, best_path = min(all_solutions, key=lambda t: t[0])
    number_of_all_paths = len(all_solutions)

    return best_path_length, best_path, number_of_all_paths


def bruteforce_helper(graph, start_and_end_node, current_node, path, path_length, nodes_left, all_solutions):
    nodes_left = nodes_left.copy()
    #print(nodes_left)
    #print(current_node)

    nodes_left.remove(current_node)

    if len(nodes_left) == 0:
        # we have to return to the start node. Check if our last node is connected to the start node. If it's not - return.
        if start_and_end_node in graph.neighbors(current_node):
            path.append(start_and_end_node)
            path_length += graph.get_edge_data(current_node, start_and_end_node)['weight']
            all_solutions.append((path_length, path))
        # if there's no more nodes - we are done anyway.
        return

    neighbors = [n for n in graph.neighbors(current_node)]

    for next_node in neighbors:
        if next_node in nodes_left:
            new_path = path.copy()
            new_path.append(next_node)
            new_length = path_length + graph.get_edge_data(current_node, next_node)['weight']

            bruteforce_helper(graph, start_and_end_node, current_node=next_node, path=new_path, path_length=new_length,
                       nodes_left=nodes_left, all_solutions=all_solutions)
    return


def ant_colony(graph: nx.Graph, start_and_end_node, number_of_ants=3, alpha=1, beta=1,
               pheromone_evaporation_coefficient=0.2, pheromone_by_ant_coefficient=1):
    """
    One critical restriction is required for this algorithm: a graph must be complete.
    Every ant finds a Hamiltonian path leaving pheromone on an edge after every iteration.
    """

    # copy the original graph so we can add some info to nodes and edges.
    graph = graph.copy()

    # initialize pheromone for all edges
    nx.set_edge_attributes(graph, 1, 'pheromone')

    # create ants
    ants_current_nodes = np.random.choice(graph.number_of_nodes(), number_of_ants, replace=False)
    print(ants_current_nodes)

    # list of sets. Each set controls nodes that are left for a particular ant
    ants_nodes_left = []
    for ant_node in ants_current_nodes:
        nodes_left = set(graph.nodes)
        nodes_left.remove(ant_node)
        ants_nodes_left.append(nodes_left)

    probability_table_for_every_ant = []
    ants_next_nodes = []

    for ant in ants_current_nodes:
        ant_neighbors = [n for n in graph.neighbors(ant)]
        aggregated_pheromone_for_all_neighbors = np.array([graph.get_edge_data(ant, next_node)['pheromone'] for next_node in ant_neighbors])
        probability_for_neighbors = []

        for neighbor in ant_neighbors:
            # Heuristic for state transition. Usually, it's 1/d, where d is the Euclidean distance to the next node.
            # But as we have no Euclidean distance determined in this task we just put it to be 1.
            eta = 1

            # Attractiveness of the next node based on the pheromone.
            tau = graph.get_edge_data(ant, neighbor)['pheromone']

            numerator = tau ** alpha * eta ** beta
            denominator = np.sum(aggregated_pheromone_for_all_neighbors**alpha * tau**beta)
            probability_to_move_to_this_neighbor = numerator / denominator

            probability_for_neighbors.append(probability_to_move_to_this_neighbor)

        # Store all important information in the following format: (current node (ant), neighbors : their probabilities).
        # It must be done to guarantee that it works even if graph.neighbors(ant) returns an unordered list.
        info_of_ant = (ant, dict(zip(ant_neighbors, probability_for_neighbors)))
        probability_table_for_every_ant.append(info_of_ant)
        print(info_of_ant)

        next_node = np.random.choice(ant_neighbors, 1, p=probability_for_neighbors)[0]
        print(next_node)
        ants_next_nodes.append(next_node)


    # Global update of pheromones.

    # evaporation on all edges
    print(graph.edges.data('pheromone'))
    for i in range(len(graph.nodes)):
        for j in range(i + 1, len(graph.nodes)):
            graph.get_edge_data(i, j)['pheromone'] *= 1 - pheromone_evaporation_coefficient
    print(graph.edges.data('pheromone'))

    # add pheromone to an edge which was chosen by an ant
    for i in range(len(ants_current_nodes)):
        edge = graph.get_edge_data(ants_current_nodes[i], ants_next_nodes[i])
        edge['pheromone'] += pheromone_by_ant_coefficient / edge['weight']
    print(graph.edges.data('pheromone'))



def ant_colony_helper(graph: nx.Graph, start_and_end_node, number_of_ants, alpha, beta,
               pheromone_evaporation_coefficient, pheromone_by_ant_coefficient):
    pass


if __name__ == "__main__":
    main()
