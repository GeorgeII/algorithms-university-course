"""
Traveling salesman problem.
Three algorithms are implemented to solve it: bruteforce (works for graphs which have Hamiltonian cycles at a given
node, not only complete graphs), ant colony algorithm and genetic algorithm (Inver-over variation of genetic algorithm,
to be precise).
"""

import networkx as nx
from matplotlib import pyplot as plt
import random
import numpy as np
from time import time


def main():
    # generate some complete graphs
    graphs = []
    for i in range(4, 11):
        graphs.append(nx.complete_graph(i))

    print('Number of graphs: ', len(graphs))

    # assign random weights to edges
    for graph in graphs:
        for (u, v) in graph.edges():
            graph.edges[u, v]['weight'] = random.randint(1, 30)

    # visualize generated graphs
    '''for graph in graphs:
        plt.figure(figsize=(10, 10))

        pos = nx.spring_layout(graph)
        labels = nx.get_edge_attributes(graph, 'weight')
        nx.draw_networkx(graph, pos, with_labels=True, font_weight='bold')
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=labels)

        plt.show()
        #plt.savefig(str(graph.number_of_nodes()) + ' nodes.png')
    '''

    # solve non-complete graphs via bruteforce
    non_complete_graphs = []
    non_complete_graphs.append(nx.wheel_graph(10))
    non_complete_graphs.append(nx.wheel_graph(11))
    non_complete_graphs.append(nx.connected_caveman_graph(3, 4))
    for graph in non_complete_graphs:
        for (u, v) in graph.edges():
            graph.edges[u, v]['weight'] = random.randint(1, 30)
        print("Non-complete graph: ", bruteforce(graph, 0))

    print("Bruteforce: ", bruteforce(graphs[5], 1))
    print("Ant colony: ", ant_colony(graphs[5], 1, number_of_ants=100, alpha=0.5, beta=0.5, pheromone_evaporation_coefficient=0.1))
    print("Genetic: ", genetic_algorithm(graphs[5], 1, iterations_number=100, population_size=20))

    # Synthetic testing.

    # lists of lists [time, path_length, nodes_quantity]
    test_metrics_bruteforce = []
    test_metrics_ant_colony = []
    test_metrics_genetic = []
    for graph in graphs:
        nodes_number = len(graph.nodes)
        start_time = time()
        length, _, _ = bruteforce(graph, 0)
        finish_time = time()
        test_metrics_bruteforce.append([length, finish_time - start_time, nodes_number])

        start_time = time()
        length, _, _ = ant_colony(graph, 0, number_of_ants=100, alpha=0.5, beta=0.5, pheromone_evaporation_coefficient=0.1)
        finish_time = time()
        test_metrics_ant_colony.append([length, finish_time - start_time, nodes_number])

        start_time = time()
        length, _ = genetic_algorithm(graph, 0, iterations_number=100, population_size=20)
        finish_time = time()
        test_metrics_genetic.append([length, finish_time - start_time, nodes_number])

    # compare performance of all algorithms on 4-10 nodes.
    test_metrics_bruteforce = np.array(test_metrics_bruteforce)
    test_metrics_ant_colony = np.array(test_metrics_ant_colony)
    test_metrics_genetic = np.array(test_metrics_genetic)

    # plot time performance
    plt.plot(test_metrics_bruteforce[:, 2], test_metrics_bruteforce[:, 1], color='green', label='bruteforce')
    plt.plot(test_metrics_ant_colony[:, 2], test_metrics_ant_colony[:, 1], color='blue', label='ant colony')
    plt.plot(test_metrics_genetic[:, 2], test_metrics_genetic[:, 1], color='red', label='genetic')
    plt.legend()
    plt.xlabel('Number of nodes')
    plt.ylabel('Performance time, seconds')
    plt.show()

    # plot length of paths
    plt.plot(test_metrics_bruteforce[:, 2], test_metrics_bruteforce[:, 0], color='green', label='bruteforce')
    plt.plot(test_metrics_ant_colony[:, 2], test_metrics_ant_colony[:, 0], color='blue', label='ant colony')
    plt.plot(test_metrics_genetic[:, 2], test_metrics_genetic[:, 0], color='red', label='genetic')
    plt.legend()
    plt.xlabel('Number of nodes')
    plt.ylabel('Found length of path')
    plt.show()


    # compare ant colony alg to genetic alg on a huge number of nodes
    graphs = []
    for i in range(15, 120, 5):
        graphs.append(nx.complete_graph(i))

    for graph in graphs:
        for (u, v) in graph.edges():
            graph.edges[u, v]['weight'] = random.randint(1, 30)

    test_metrics_ant_colony = []
    test_metrics_genetic = []
    for graph in graphs:
        nodes_number = len(graph.nodes)

        start_time = time()
        length, _, _ = ant_colony(graph, 0, number_of_ants=100, alpha=0.5, beta=0.5,
                                  pheromone_evaporation_coefficient=0.1)
        finish_time = time()
        test_metrics_ant_colony.append([length, finish_time - start_time, nodes_number])

        start_time = time()
        length, _ = genetic_algorithm(graph, 0, iterations_number=100, population_size=20)
        finish_time = time()
        test_metrics_genetic.append([length, finish_time - start_time, nodes_number])
    test_metrics_ant_colony = np.array(test_metrics_ant_colony)
    test_metrics_genetic = np.array(test_metrics_genetic)

    # plot time performance
    plt.plot(test_metrics_ant_colony[:, 2], test_metrics_ant_colony[:, 1], color='blue', label='ant colony')
    plt.plot(test_metrics_genetic[:, 2], test_metrics_genetic[:, 1], color='red', label='genetic')
    plt.legend()
    plt.xlabel('Number of nodes')
    plt.ylabel('Performance time, seconds')
    plt.show()

    # plot length of paths
    plt.plot(test_metrics_ant_colony[:, 2], test_metrics_ant_colony[:, 0], color='blue', label='ant colony')
    plt.plot(test_metrics_genetic[:, 2], test_metrics_genetic[:, 0], color='red', label='genetic')
    plt.legend()
    plt.xlabel('Number of nodes')
    plt.ylabel('Found length of path')
    plt.show()


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


def ant_colony(graph: nx.Graph, start_and_end_node, number_of_ants=3, alpha=1.0, beta=1.0,
               pheromone_evaporation_coefficient=0.2, pheromone_by_ant_coefficient=1):
    """
    !One critical restriction is required for this algorithm: a graph must be complete!
    Every ant finds a Hamiltonian path and leaves pheromone on the edge after every iteration.
    """

    # copy the original graph so we can add some info to nodes and edges.
    graph = graph.copy()

    # initialize pheromone for all edges
    nx.set_edge_attributes(graph, 1, 'pheromone')

    # create ants
    ants_current_nodes = np.random.choice(graph.number_of_nodes(), number_of_ants)
    start_and_end_nodes = ants_current_nodes.copy()

    # list of lists. Each list has nodes that are left for a particular ant
    ants_nodes_left = []
    for _ in ants_current_nodes:
        nodes_left = list(graph.nodes).copy()
        ants_nodes_left.append(nodes_left)

    for epoch in range(len(graph.nodes)):
        # We need to return to our source node. So, we have to add it at the last iteration.
        if epoch == len(graph.nodes) - 1:
            for i, nodes_left in enumerate(ants_nodes_left):
                nodes_left.append(start_and_end_nodes[i])

        # if an ant moved to a new node - we remove it
        for i, nodes_left in enumerate(ants_nodes_left):
            nodes_left.remove(ants_current_nodes[i])

        ants_next_nodes = []

        for i, ant_node in enumerate(ants_current_nodes):
            ant_neighbors = ants_nodes_left[i]
            pheromone_for_all_neighbors_left = np.array(
                [graph.get_edge_data(ant_node, next_node)['pheromone'] for next_node in ant_neighbors]
            )
            weight_for_all_neighbors_left = np.array(
                [graph.get_edge_data(ant_node, next_node)['weight'] for next_node in ant_neighbors]
            )
            denominator = np.sum(np.multiply(pheromone_for_all_neighbors_left ** alpha,
                                             1 / weight_for_all_neighbors_left ** beta))

            probability_for_neighbors = []

            for neighbor in ant_neighbors:
                # Attractiveness of the next node based on the weight of the edge.
                eta = 1 / graph.get_edge_data(ant_node, neighbor)['weight']

                # Attractiveness of the next node based on the pheromone.
                tau = graph.get_edge_data(ant_node, neighbor)['pheromone']

                numerator = tau ** alpha * eta ** beta
                probability_to_move_to_this_neighbor = numerator / denominator

                probability_for_neighbors.append(probability_to_move_to_this_neighbor)

            if epoch == len(graph.nodes) - 1:
                break

            next_node = np.random.choice(ant_neighbors, 1, p=probability_for_neighbors)[0]
            ants_next_nodes.append(next_node)

        # Global update of pheromones.

        # If there's no more nodes left - we quit.
        if epoch == len(graph.nodes) - 1:
            break

        # evaporation on all edges
        for i in range(len(graph.nodes)):
            for j in range(i + 1, len(graph.nodes)):
                graph.get_edge_data(i, j)['pheromone'] *= 1 - pheromone_evaporation_coefficient

        # add pheromone to the edge which was chosen by an ant
        for i in range(len(ants_current_nodes)):
            edge = graph.get_edge_data(ants_current_nodes[i], ants_next_nodes[i])
            edge['pheromone'] += pheromone_by_ant_coefficient / edge['weight']

        ants_current_nodes = ants_next_nodes

    # Choose the best path with naive greedy traveling salesman problem algorithm using pheromone as weights.
    nodes_left = list(graph.nodes).copy()
    best_path = []
    best_path_length = 0
    pheromone_collected=0
    current_node = start_and_end_node

    for i in range(len(graph.nodes)):
        # remove node as we visit it
        nodes_left.remove(current_node)

        best_path.append(current_node)

        if len(nodes_left) == 0:
            best_path.append(start_and_end_node)
            best_path_length += graph.get_edge_data(current_node, start_and_end_node)['weight']
            pheromone_collected += graph.get_edge_data(current_node, start_and_end_node)['pheromone']
            break

        # (candidate_node, {weight:x, pheromone:y})
        candidate_nodes_and_edges = []
        for candidate_for_next_node in nodes_left:
            edge = graph.get_edge_data(current_node, candidate_for_next_node)
            candidate_nodes_and_edges.append((candidate_for_next_node, edge))

        next_node, edge = max(candidate_nodes_and_edges, key=lambda t: t[1]["pheromone"])
        best_path_length += edge['weight']
        pheromone_collected += edge['pheromone']
        current_node = next_node

    return best_path_length, best_path, pheromone_collected


def genetic_algorithm(graph: nx.Graph, start_and_end_node, population_size=5, iterations_number=5,
                      mutation_probability=0.2):
    """
    !One critical restriction is required for this algorithm: a graph must be complete!
    Inver-over genetic algorithm.
    https://www.youtube.com/watch?v=te4PYYAdTJE
    Article: Inver-over operator for the TSP   Authors: Guo Tao, Zbigniew Michalewicz
    """

    nodes = list(graph.nodes)

    # remove start node because it always needs to be at the beginning and at the end of a list and we cannot permute it
    nodes.remove(start_and_end_node)

    population = []
    for _ in range(population_size):
        population.append([start_and_end_node] + np.random.permutation(nodes).tolist() + [start_and_end_node])

    for iteration in range(iterations_number):
        for chr_idx, chromosome in enumerate(population):
            new_chromosome = chromosome.copy()
            # we cannot swap the start and end node
            first_element_to_swap = random.choice(new_chromosome[1:-1])
            seconds_element_to_swap = -1

            repeat_flag = True
            while repeat_flag:
                # mutation phase
                if random.uniform(0, 1) < mutation_probability:
                    genes_to_choose_second_element_to_swap = \
                        [x for x in new_chromosome[1:-1] if x != first_element_to_swap]
                    seconds_element_to_swap = random.choice(genes_to_choose_second_element_to_swap)
                # crossover phase
                else:
                    # randomly select another chromosome
                    another_chromosome = random.choice([x for x in population if x != chromosome])
                    # choose the closest right neighbor for the first_element_to_swap (left neighbor,
                    # if there's no right elements).
                    idx = another_chromosome.index(first_element_to_swap)
                    seconds_element_to_swap = another_chromosome[idx + 1] \
                        if idx != len(another_chromosome)-2 else another_chromosome[idx - 1]

                # check if first_element and second_element are next to each other in the list.
                first_elem_idx = new_chromosome.index(first_element_to_swap)
                second_elem_idx = new_chromosome.index(seconds_element_to_swap)
                # if they are neighbors - we stop
                if abs(first_elem_idx - second_elem_idx) == 1:
                    repeat_flag = False
                else:
                    # reverse a part of chromosome
                    if second_elem_idx > first_elem_idx:
                        new_chromosome = new_chromosome[:first_elem_idx] + \
                                         new_chromosome[first_elem_idx:second_elem_idx][::-1] + \
                                         new_chromosome[second_elem_idx:]
                    else:
                        new_chromosome = new_chromosome[:second_elem_idx] + \
                                         new_chromosome[second_elem_idx:first_elem_idx][::-1] + \
                                         new_chromosome[first_elem_idx:]

                    first_element_to_swap = seconds_element_to_swap

            # calculate fitness of chromosomes and compare them
            chromosome_fitness = 0
            new_chromosome_fitness = 0
            for i in range(len(chromosome) - 1):
                chromosome_fitness += graph.get_edge_data(chromosome[i], chromosome[i + 1])['weight']
                new_chromosome_fitness += graph.get_edge_data(new_chromosome[i], new_chromosome[i + 1])['weight']
            if new_chromosome_fitness < chromosome_fitness:
                # replace the chromosome with a new chromosome
                population[chr_idx] = new_chromosome

    # choose the best chromosome
    chromosome_fitnesses = []
    for chromosome in population:
        fitness = 0
        for i in range(len(chromosome) - 1):
            fitness += graph.get_edge_data(chromosome[i], chromosome[i + 1])['weight']
        chromosome_fitnesses.append(fitness)

    best_path_length, best_path = min(chromosome_fitnesses), \
                                  population[chromosome_fitnesses.index(min(chromosome_fitnesses))]

    return best_path_length, best_path


if __name__ == "__main__":
    main()
