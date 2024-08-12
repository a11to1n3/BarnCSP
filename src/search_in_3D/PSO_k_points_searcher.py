import numpy as np
from ..utils import sample_neighboring_points_3D

def find_optimal_k_points_pso_3D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    epochs=100,
    c1=1.5,
    c2=1.5,
    w=0.7,
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

    def initialize_particle():
        return valid_indices[np.random.choice(len(valid_indices), size=k, replace=False)]

    def calculate_fitness(particle):
        points = carbon_map[particle[:, 0], particle[:, 1], particle[:, 2]]
        return np.abs(np.mean(points) - in_CO2_avg)

    # Initialize particles
    particles = [initialize_particle() for _ in range(k)]
    velocities = [np.zeros_like(particle) for particle in particles]
    personal_best_positions = particles.copy()
    personal_best_fitness = [calculate_fitness(p) for p in particles]
    
    global_best_index = np.argmin(personal_best_fitness)
    global_best_position = personal_best_positions[global_best_index].copy()
    global_best_fitness = personal_best_fitness[global_best_index]

    for _ in range(epochs):
        for i in range(k):
            # Update velocity
            r1, r2 = np.random.rand(2)
            velocities[i] = (w * velocities[i] +
                             c1 * r1 * (personal_best_positions[i] - particles[i]) +
                             c2 * r2 * (global_best_position - particles[i]))
            
            # Update position
            particles[i] = particles[i] + velocities[i]
            particles[i] = np.clip(particles[i], [0, 0, 0], [100*barn_LW_ratio-1, 99, depth-1])
            particles[i] = particles[i].astype(int)

            # Evaluate fitness
            fitness = calculate_fitness(particles[i])

            # Update personal best
            if fitness < personal_best_fitness[i]:
                personal_best_fitness[i] = fitness
                personal_best_positions[i] = particles[i].copy()

            # Update global best
            if fitness < global_best_fitness:
                global_best_fitness = fitness
                global_best_position = particles[i].copy()

    best_points = carbon_map[global_best_position[:, 0], global_best_position[:, 1], global_best_position[:, 2]]
    best_positions = position_map[global_best_position[:, 0], global_best_position[:, 1], global_best_position[:, 2]]

    # Do sensitivity analysis
    image_width, image_height, image_depth = 100*barn_LW_ratio, 100, depth  # Image dimensions

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

    return global_best_fitness, np.mean(losses), np.std(losses), best_positions