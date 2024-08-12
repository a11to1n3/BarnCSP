import numpy as np
from scipy import stats
from ..utils import sample_neighboring_points_2D

def find_optimal_k_points_monte_carlo_2D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    barn_section=3.1500001,
    convergence_threshold=1e-4,
    max_epochs=100,
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

    # Estimate the probability distribution of CO2 concentrations
    co2_values = carbon_map[valid_indices[:, 0], valid_indices[:, 1]]
    kde = stats.gaussian_kde(co2_values)

    def sample_points():
        sampled_indices = valid_indices[np.random.choice(len(valid_indices), size=k, replace=False)]
        return sampled_indices, carbon_map[sampled_indices[:, 0], sampled_indices[:, 1]]

    def calculate_loss(points):
        return np.abs(np.mean(points) - in_CO2_avg)

    best_loss = float('inf')
    best_points = None
    best_positions = None
    
    running_mean = 0
    running_var = 0
    
    for iteration in range(max_epochs):
        losses = []
        for _ in range(k):
            sampled_indices, sampled_points = sample_points()
            loss = calculate_loss(sampled_points)
            losses.append(loss)
            
            if loss < best_loss:
                best_loss = loss
                best_points = sampled_points
                best_positions = position_map[sampled_indices[:, 0], sampled_indices[:, 1]]

        # Update running statistics
        new_mean = np.mean(losses)
        new_var = np.var(losses)
        
        if iteration > 0:
            delta = new_mean - running_mean
            running_mean += delta / (iteration + 1)
            running_var += (new_var - running_var) / (iteration + 1)

            # Check for convergence
            if np.abs(delta) < convergence_threshold:
                break
        else:
            running_mean = new_mean
            running_var = new_var

    # Estimate the probability of finding a better solution
    prob_better = kde.integrate_box_1d(0, best_loss)

    # Do sensitivity analysis
    image_width, image_height = 100*barn_LW_ratio, 100  # Image dimensions

    best_locs = [
        [int(pos[0]), int(pos[1])] for pos in best_positions
    ]

    combinations = sample_neighboring_points_2D(
        best_locs, neighborhood_numbers, image_width, image_height, sampling_budget
    )
    sensitivity_losses = []
    for combination in combinations:
        p_sum = sum(carbon_map[y, x] for y, x in combination)
        sensitivity_losses.append(np.abs(p_sum / k - in_CO2_avg))

    return best_loss, np.mean(sensitivity_losses), np.std(sensitivity_losses), best_positions, prob_better, running_mean, np.sqrt(running_var)