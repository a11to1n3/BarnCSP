import numpy as np
from ..utils import sample_neighboring_points_2D

def find_optimal_k_points_advanced_genetic_algorithm_2D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    barn_section=3.1500001,
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
    nodes_at_height_df = nodes_df[barn_inside_points.flatten().astype(bool)][
        nodes_df[barn_inside_points.flatten().astype(bool)].Y == barn_section
    ]
    nodes_at_height_df = nodes_at_height_df.reset_index()

    carbon_map = nodes_at_height_df["Carbon"].values.reshape(100*barn_LW_ratio, 100)
    position_map = nodes_at_height_df[["X", "Z"]].values.reshape(100*barn_LW_ratio, 100, -1)

    valid_indices = np.argwhere(nodes_at_height_df.index.values.reshape(100*barn_LW_ratio, 100))

    def create_individual():
        return valid_indices[np.random.choice(len(valid_indices), size=k, replace=False)]

    def calculate_fitness(individual):
        points = carbon_map[individual[:, 0], individual[:, 1]]
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

    for _ in range(episodes):
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

    best_points = carbon_map[best_individual[:, 0], best_individual[:, 1]]
    best_positions = position_map[best_individual[:, 0], best_individual[:, 1]]
    best_loss = np.abs(np.mean(best_points) - in_CO2_avg)

    image_width, image_height = 100*barn_LW_ratio, 100
    best_locs = [[int(pos[0]), int(pos[1])] for pos in best_positions]

    combinations = sample_neighboring_points_2D(
        best_locs, neighborhood_numbers, image_width, image_height, sampling_budget
    )
    losses = [np.abs(sum(carbon_map[y, x] for y, x in combination) / k - in_CO2_avg) for combination in combinations]

    return best_loss, np.mean(losses), np.std(losses), best_positions