import numpy as np
import random


def main():
    #equation itself: a + 2b + 3c + 4d = 30; where  a,b,c,d > 0

    brute_force_solutions = brute_force()
    print(len(brute_force_solutions), brute_force_solutions)

    genetic_algo_solutions = genetic_algo(initial_pool_size=50)
    print(len(genetic_algo_solutions), genetic_algo_solutions)

    # check whether genetic_algo solutions are correct
    print("Genetic solutions correct: ",
          len(genetic_algo_solutions.intersection(brute_force_solutions)) == len(genetic_algo_solutions)
          )


def brute_force():
    solutions = []

    for a in range(1, 22):
        for b in range(1, 13):
            for c in range(1, 10):
                for d in range(1, 7):
                    calculated = a + 2 * b + 3 * c + 4 * d
                    if calculated > 30:
                        break
                    elif calculated == 30:
                        solutions.append([a, b, c, d])

    return {tuple(i) for i in solutions}


def genetic_algo(initial_pool_size=5):
    # main source: http://masters.donntu.org/2011/fknt/gutsenko/library/article_3.htm

    initial_chromosome_pool = [initialize_chromosome() for _ in range(initial_pool_size)]
    print(initial_chromosome_pool)

    solutions_found = genetic_algo_helper(initial_chromosome_pool, 0)

    solutions_of_equation = [sol for sol in solutions_found if result_of_expression(sol) == 30]

    return {tuple(i) for i in solutions_of_equation}


def genetic_algo_helper(chromosome_pool,  steps_done, probability_to_mutate=0.2):
    if steps_done >= 5:
        return chromosome_pool

    pool_size = len(chromosome_pool)

    evaluated_results = [result_of_expression(chromosome) for chromosome in chromosome_pool]
    fitness = [abs(res - 30) for res in evaluated_results]
    likelihood_to_be_picked = [1 / res if res != 0 else 0 for res in fitness]
    probability = sum(likelihood_to_be_picked)

    number_of_zeros = likelihood_to_be_picked.count(0)
    if number_of_zeros == 0:
        likelihood_to_be_picked = [x / probability for x in likelihood_to_be_picked]
    else:
        indices_of_zeros = [i for i, x in enumerate(likelihood_to_be_picked) if x == 0]

        # if one parent is a solution, we make the probability for other candidates 0.01 so it can be picked later on
        # as another parent
        probability_when_solution_found = 0.2 / (len(likelihood_to_be_picked) - number_of_zeros)
        likelihood_to_be_picked = [probability_when_solution_found] * len(likelihood_to_be_picked)

        probability = 0.8 / number_of_zeros
        for idx in indices_of_zeros:
            likelihood_to_be_picked[idx] = probability

    def choose_parents():
        chosen_parents_indices = np.random.choice([i for i in range(pool_size)],
                                                  size=2,
                                                  p=likelihood_to_be_picked,
                                                  replace=False)
        return [chromosome_pool[chosen_parents_indices[0]],
                          chromosome_pool[chosen_parents_indices[1]]
                          ]

    parents_pool = [choose_parents() for _ in range(pool_size)]

    def cross_over(parents):
        partition_element = random.randint(1, 3)

        return parents[0][:partition_element] + parents[1][partition_element:]

    new_chromosome_pool = [cross_over(parents) for parents in parents_pool]

    def mutation(size_of_mutation=0.3):
        number_of_mutations = int(size_of_mutation * len(new_chromosome_pool))
        chromosomes_to_mutate = np.random.choice([i for i in range(pool_size)],
                                                  size=number_of_mutations,
                                                  replace=False
                                                 )
        for idx in chromosomes_to_mutate:
            new_chromosome_pool[idx] = initialize_chromosome()

    if random.uniform(0.0, 1.0) < probability_to_mutate:
        mutation(size_of_mutation=0.4)

    return genetic_algo_helper(new_chromosome_pool, steps_done=steps_done + 1)


def initialize_chromosome():
    # returns a random coefficients list
    return [random.randint(1, 22), random.randint(1, 13), random.randint(1, 10), random.randint(1, 7)]


def result_of_expression(coeffs):
    # substitutes coeffs to the expression and returns the result
    return coeffs[0] + 2 * coeffs[1] + 3 * coeffs[2] + 4 * coeffs[3]


if __name__ == "__main__":
    main()
