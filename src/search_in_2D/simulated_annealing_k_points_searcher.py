import numpy as np
from ..utils import sample_neighboring_points_2D

def find_optimal_k_points_simulated_annealing_2D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    barn_section=3.1500001,
    initial_temperature=100,
    cooling_rate=0.995,
    epochs=1000,
    ## For sensitivity analysis
    sampling_budget=10000,
    neighborhood_numbers=5,
    barn_LW_ratio=2
):
    # Filtering the nodes at the given height
    nodes_at_height_df = nodes_df[barn_inside_points.flatten().astype(bool)][
        nodes_df[barn_inside_points.flatten().astype(bool)].Y == barn_section
    ]
    nodes_at_height_df = nodes_at_height_df.reset_index()

    # Reshape the data for easier access
    carbon_map = nodes_at_height_df["Carbon"].values.reshape(100*barn_LW_ratio, 100)
    position_map = nodes_at_height_df[["X", "Z"]].values.reshape(100*barn_LW_ratio, 100, -1)

    valid_indices = np.argwhere(nodes_at_height_df.index.values.reshape(100*barn_LW_ratio, 100))

    def get_random_solution():
        return valid_indices[np.random.choice(len(valid_indices), size=k, replace=False)]

    def get_neighbor_solution(solution):
        neighbor = solution.copy()
        index_to_change = np.random.randint(k)
        neighbor[index_to_change] = valid_indices[np.random.choice(len(valid_indices))]
        return neighbor

    def calculate_loss(solution):
        points = carbon_map[solution[:, 0], solution[:, 1]]
        return np.abs(np.mean(points) - in_CO2_avg)

    # Initialize the solution
    current_solution = get_random_solution()
    current_loss = calculate_loss(current_solution)
    best_solution = current_solution
    best_loss = current_loss

    temperature = initial_temperature

    for _ in range(epochs):
        neighbor_solution = get_neighbor_solution(current_solution)
        neighbor_loss = calculate_loss(neighbor_solution)

        # Decide whether to accept the neighbor solution
        if neighbor_loss < current_loss or np.random.random() < np.exp((current_loss - neighbor_loss) / temperature):
            current_solution = neighbor_solution
            current_loss = neighbor_loss

        # Update the best solution if necessary
        if current_loss < best_loss:
            best_solution = current_solution
            best_loss = current_loss

        # Cool down the temperature
        temperature *= cooling_rate

    best_points = carbon_map[best_solution[:, 0], best_solution[:, 1]]
    best_positions = position_map[best_solution[:, 0], best_solution[:, 1]]

    # Do sensitivity analysis
    image_width, image_height = 100*barn_LW_ratio, 100  # Image dimensions

    best_locs = [
        [int(pos[0]), int(pos[1])] for pos in best_positions
    ]

    combinations = sample_neighboring_points_2D(
        best_locs, neighborhood_numbers, image_width, image_height, sampling_budget
    )
    losses = []
    for combination in combinations:
        p_sum = sum(carbon_map[y, x] for y, x in combination)
        losses.append(np.abs(p_sum / k - in_CO2_avg))

    return best_loss, np.mean(losses), np.std(losses), best_positions