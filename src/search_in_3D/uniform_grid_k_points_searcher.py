import numpy as np
from itertools import combinations
from tqdm import tqdm
from ..utils import sample_neighboring_points_3D

def find_optimal_k_points_uniform_grid_search_3D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    grid_size=10,
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

    # Create a uniform grid
    x_grid = np.linspace(0, 100*barn_LW_ratio-1, grid_size, dtype=int)
    y_grid = np.linspace(0, 99, grid_size, dtype=int)
    z_grid = np.linspace(0, depth-1, grid_size, dtype=int)
    grid_points = np.array(np.meshgrid(x_grid, y_grid, z_grid)).T.reshape(-1, 3)

    # Filter grid points to only include valid barn space
    grid_points = [grid_point for grid_point in grid_points if grid_point[0]*grid_point[1]*grid_point[2] in nodes_df[barn_inside_points.flatten().astype(bool)].index]

    # Uniform grid search to find the best combination of k points averaging to the avg CO2
    best_loss = float('inf')
    best_points = None
    best_positions = None

    for selected_indices in combinations(grid_points, k):
        selected_indices = np.array(selected_indices)
        selected_points = carbon_map[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]
        selected_positions = position_map[selected_indices[:, 0], selected_indices[:, 1], selected_indices[:, 2]]

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
    image_width, image_height, image_depth = 100*barn_LW_ratio, 100, depth  # Image dimensions

    best_locs = [
        [int(pos[0]), int(pos[1]), int(pos[2])] for pos in best_positions
    ]

    n_combinations = sample_neighboring_points_3D(
        best_locs, neighborhood_numbers, image_width, image_height, image_depth, sampling_budget
    )
    losses = []
    for combination in n_combinations:
        p_sum = sum(carbon_map[y, x, z] for y, x, z in combination)
        losses.append(np.abs(p_sum / k - in_CO2_avg))

    return best_loss, np.mean(losses), np.std(losses), best_positions