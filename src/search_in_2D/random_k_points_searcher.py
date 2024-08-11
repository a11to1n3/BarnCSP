import numpy as np
from ..utils import sample_neighboring_points_2D

def find_optimal_k_points_random_search_2D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    barn_section=3.1500001,
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

    # Random search to find the best combination of k points averaging to the avg CO2
    best_loss = float('inf')
    best_points = None
    best_positions = None

    valid_indices = np.argwhere(nodes_at_height_df.index.values.reshape(100*barn_LW_ratio, 100))

    for _ in range(epochs):
        # Randomly select k points from the valid barn space
        selected_indices = valid_indices[np.random.choice(len(valid_indices), size=k, replace=False)]
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

    combinations = sample_neighboring_points_2D(
        best_locs, neighborhood_numbers, image_width, image_height, sampling_budget
    )
    losses = []
    for combination in combinations:
        p_sum = sum(carbon_map[y, x] for y, x in combination)
        losses.append(np.abs(p_sum / k - in_CO2_avg))

    return best_loss, np.mean(losses), np.std(losses), best_positions