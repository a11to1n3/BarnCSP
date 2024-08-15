import numpy as np
from ..utils import sample_neighboring_points_3D

def find_optimal_k_points_advanced_genetic_algorithm_3D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    population_size=100,
    episodes=50,
    mutation_rate=0.1,
    crossover_rate=0.8,
    tournament_size=5,
    ## For sensitivity analysis
    sampling_budget=10000,
    neighborhood_numbers=5,
    barn_LW_ratio=2
):
    # Filtering the nodes inside the barn
    nodes_df[~barn_inside_points.flatten().astype(bool)] = 1e9

    # Reshape the data for easier access
    depth = len(nodes_df["Carbon"].values.flatten()) // (100*barn_LW_ratio*100)
    carbon_map = nodes_df["Carbon"].values.reshape(100*barn_LW_ratio, 100, depth)
    position_map = nodes_df[["X", "Y", "Z"]].values.reshape(100*barn_LW_ratio, 100, depth, -1)

    valid_indices = np.argwhere(barn_inside_points.reshape(100*barn_LW_ratio, 100, depth))

    def create_individual():
        return valid_indices[np.random.choice(len(valid_indices), size=k, replace=False)]

    def calculate_fitness(individual):
        points = carbon_map[individual[:, 0], individual[:, 1], individual[:, 2]]
        return 1 / (np.abs(np.mean(points) - in_CO2_avg) + 1e-6)

    def tournament_selection(population, fitnesses):
        selected = np.random.choice(len(population), tournament_size)
        return population[selected[np.argmax(fitnesses[selected])]]

    def uniform_crossover(parent1, parent2):
        mask = np.random.rand(k) < 0.5
        child = np.where(mask[:, np.newaxis], parent1, parent2)
        return child

    def adaptive_mutation(individual, fitness, avg_fitness, best_fitness):
        if fitness <= avg_fitness:
            adaptive_rate = mutation_rate * (best_fitness - fitness) / (best_fitness - avg_fitness)
        else:
            adaptive_rate = mutation_rate
        
        for idx in range(k):
            if np.random.rand() < adaptive_rate:
                individual[idx] = valid_indices[np.random.choice(len(valid_indices))]
        return individual

    population = np.array([create_individual() for _ in range(population_size)])
    best_individual = None
    best_fitness = float('-inf')

    for episode in range(episodes):
        fitnesses = np.array([calculate_fitness(ind) for ind in population])
        avg_fitness = np.mean(fitnesses)
        best_gen_fitness = np.max(fitnesses)
        best_gen_individual = population[np.argmax(fitnesses)]

        if best_gen_fitness > best_fitness:
            best_fitness = best_gen_fitness
            best_individual = best_gen_individual

        new_population = [best_gen_individual]  # Elitism

        while len(new_population) < population_size:
            if np.random.rand() < crossover_rate:
                parent1 = tournament_selection(population, fitnesses)
                parent2 = tournament_selection(population, fitnesses)
                child = uniform_crossover(parent1, parent2)
            else:
                child = tournament_selection(population, fitnesses)

            child = adaptive_mutation(child, calculate_fitness(child), avg_fitness, best_gen_fitness)
            new_population.append(child)

        population = np.array(new_population)

    best_points = carbon_map[best_individual[:, 0], best_individual[:, 1], best_individual[:, 2]]
    best_positions = position_map[best_individual[:, 0], best_individual[:, 1], best_individual[:, 2]]
    best_loss = np.abs(np.mean(best_points) - in_CO2_avg)

    # Do sensitivity analysis
    image_width, image_height, image_depth = 100*barn_LW_ratio, 100, depth

    best_locs = [
        [int(pos[0]), int(pos[1]), int(pos[2])] for pos in best_positions
    ]

    combinations = sample_neighboring_points_3D(
        best_locs, neighborhood_numbers, image_width, image_height, image_depth, sampling_budget
    )
    losses = []
    for combination in combinations:
        p_sum = sum(carbon_map[y, x, z] for y, x, z in combination)
        losses.append(np.abs(p_sum / k - in_CO2_avg))

    return best_loss, np.mean(losses), np.std(losses), best_positions