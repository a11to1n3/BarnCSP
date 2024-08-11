import numpy as np
from itertools import combinations
from ..utils import sample_neighboring_points_2D

def find_optimal_k_points_uniform_grid_search_2D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    barn_section=3.1500001,
    grid_size=10,
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

    # Create a uniform grid
    x_grid = np.linspace(0, 100*barn_LW_ratio-1, grid_size, dtype=int)
    z_grid = np.linspace(0, 99, grid_size, dtype=int)
    grid_points = np.array(np.meshgrid(x_grid, z_grid)).T.reshape(-1, 2)

    # Filter grid points to only include valid barn space
    grid_points = [grid_point for grid_point in grid_points if grid_point[0]*grid_point[1] in nodes_at_height_df.index]

    # Uniform grid search to find the best combination of k points averaging to the avg CO2
    best_loss = float('inf')
    best_points = None
    best_positions = None

    for selected_indices in combinations(grid_points, k):
        selected_indices = np.array(selected_indices)
        selected_points = carbon_map[selected_indices[:, 0], selected_indices[:, 1]]
        selected_positions = position_map[selected_indices[:, 0], selected_indices[:, 1]]

        # Calculate the average CO2 value for the selected points
        mean = np.mean(selected_points)

        # Calculate the loss
        loss = np.abs(mean - in_CO2_avg)

        # Update the best solution if necessary
        if loss < best_loss:
            best_loss = loss
            best_points = selected_points
            best_positions = selected_positions

    # Do sensitivity analysis
    image_width, image_height = 100*barn_LW_ratio, 100  # Image dimensions

    best_locs = [
        [int(pos[0]), int(pos[1])] for pos in best_positions
    ]

    n_combinations = sample_neighboring_points_2D(
        best_locs, neighborhood_numbers, image_width, image_height, sampling_budget
    )
    losses = []
    for combination in n_combinations:
        p_sum = sum(carbon_map[y, x] for y, x in combination)
        losses.append(np.abs(p_sum / k - in_CO2_avg))

    return best_loss, np.mean(losses), np.std(losses), best_positions