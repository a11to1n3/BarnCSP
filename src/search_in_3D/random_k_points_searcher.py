import numpy as np
from ..utils import sample_neighboring_points_3D

def find_optimal_k_points_random_search_3D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    epochs=1000,
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

    # Random search to find the best combination of k points averaging to the avg CO2
    best_loss = float('inf')
    best_points = None
    best_positions = None

    valid_indices = np.argwhere(barn_inside_points.reshape(100*barn_LW_ratio, 100, depth))

    for _ in range(epochs):
        # Randomly select k points from the valid barn space
        selected_indices = valid_indices[np.random.choice(len(valid_indices), size=k, replace=False)]
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

    combinations = sample_neighboring_points_3D(
        best_locs, neighborhood_numbers, image_width, image_height, image_depth, sampling_budget
    )
    losses = []
    for combination in combinations:
        p_sum = sum(carbon_map[y, x, z] for y, x, z in combination)
        losses.append(np.abs(p_sum / k - in_CO2_avg))

    return best_loss, np.mean(losses), np.std(losses), best_positions