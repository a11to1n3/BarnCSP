import numpy as np
from scipy import stats
from ..utils import sample_neighboring_points_3D

def find_optimal_k_points_monte_carlo_3D(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    convergence_threshold=1e-4,
    max_epochs=100,
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

    # Estimate the probability distribution of CO2 concentrations
    co2_values = carbon_map[valid_indices[:, 0], valid_indices[:, 1], valid_indices[:, 2]]
    kde = stats.gaussian_kde(co2_values)

    def sample_points():
        sampled_indices = valid_indices[np.random.choice(len(valid_indices), size=k, replace=False)]
        return sampled_indices, carbon_map[sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]]

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
                best_positions = position_map[sampled_indices[:, 0], sampled_indices[:, 1], sampled_indices[:, 2]]

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
    image_width, image_height, image_depth = 100*barn_LW_ratio, 100, depth  # Image dimensions

    best_locs = [
        [int(pos[0]), int(pos[1]), int(pos[2])] for pos in best_positions
    ]

    combinations = sample_neighboring_points_3D(
        best_locs, neighborhood_numbers, image_width, image_height, image_depth, sampling_budget
    )
    sensitivity_losses = []
    for combination in combinations:
        p_sum = sum(carbon_map[y, x, z] for y, x, z in combination)
        sensitivity_losses.append(np.abs(p_sum / k - in_CO2_avg))

    return best_loss, np.mean(sensitivity_losses), np.std(sensitivity_losses), best_positions, prob_better, running_mean, np.sqrt(running_var)