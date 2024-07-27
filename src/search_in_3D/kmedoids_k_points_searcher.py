import torch
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn_extra.cluster import KMedoids

from .utils import sample_neighboring_points

def find_optimal_k_points_kmedoids(
    nodes_df,
    barn_inside_points,
    k,
    in_CO2_avg,
    barn_section=3.1500001,
    lr=5e-7,
    epochs=20,
    ## For sensitivity analysis
    sampling_budget = 10000,
    neighborhood_numbers = 5,
    barn_LW_ratio=2
):

    # Filtering the nodes at the given height
    nodes_at_height_df = nodes_df[barn_inside_points.flatten().astype(bool)][
        nodes_df[barn_inside_points.flatten().astype(bool)].Y == barn_section
    ]
    nodes_at_height_df = nodes_at_height_df.reset_index()

    # Find k clusters
    kmedoids = KMedoids(n_clusters=k, random_state=0).fit(nodes_at_height_df[["X","Z","u","w","v","Carbon"]].values)

    # Initiate clusters as columns in the dataframe
    for i in range(k):
        nodes_at_height_df[f"cluster{i}"] = -2

    # Assigning number accordingly to the clusters
    for i in range(len(nodes_at_height_df)):
        nodes_at_height_df.loc[i, f"cluster{kmedoids.labels_[i]}"] = 1

    cols = [f"cluster{i}" for i in range(k)]
    cols.append("Carbon")
    cluster_image = nodes_at_height_df[cols].values.reshape(1, 100*barn_LW_ratio, 100, -1)
    position_map = nodes_at_height_df[["X", "Z"]].values.reshape(100*barn_LW_ratio, 100, -1)

    # Use DBSCAN clustering to cluster again the elements in every assigned cluster
    cluster_pools = []
    for i in range(k):
        cluster_pts = nodes_at_height_df[nodes_at_height_df[f"cluster{i}"] == 1][
            ["X", "Z", "u", "w", "v", "Carbon"]
        ].values

        hdb = DBSCAN(min_samples=9, eps=0.5)
        hdb.fit(cluster_pts)

        cluster_img = cluster_image[0, :, :, i].copy()
        cluster_img[cluster_img == 1] = hdb.labels_

        cluster_pool = cluster_image[0, :, :, k].copy()
        for i in range(cluster_pool.shape[0]):
            for j in range(cluster_pool.shape[1]):
                if cluster_img[i, j] < 0:
                    cluster_pool[i, j] = 1e9
        cluster_pools.append(cluster_pool)

    # Use gradient-based optimization to find the best combination of k points averaging to the avg CO2
    p = []
    min_index = np.random.randint(2, size=k)
    for i in range(epochs):
        if i == 0:
            # Init parameter set "p" standing for the set of k points, assign gradient to every parameter
            for j in range(k):
                p.append(
                    torch.tensor(
                        np.median(cluster_pools[j][cluster_pools[j] < 1e9]),
                        requires_grad=True,
                    )
                )
            # Init the optimizer. Here we choose RMSprop
            optimizer = torch.optim.RMSprop(p, lr=lr)
            optimizer.zero_grad()
        else:
            # Clear the gradient in the next round
            [p[j].grad.data.zero_() for j in range(k)]
            for j in range(k):
                # Find the closest point to the learning parameter
                filtered_values = [
                    (index, np.abs(value - (p[j].detach().numpy()+p[j].detach().numpy()*np.random.rand()*1e-4)))
                    for index, value in enumerate(cluster_pools[j].flatten())
                    if value < 1e9
                ]
                if len(filtered_values) == 0:
                    return None
                min_i, _ = min(
                    filtered_values, key=lambda x: x[1], default=(None, None)
                )
                min_index[j] = min_i

            # Get the location of the closest point
            min_locs = [
                [
                    min_index[j] // cluster_pools[j].shape[1],
                    min_index[j] % cluster_pools[j].shape[1],
                ]
                for j in range(k)
            ]

            # Init again with the newly found points
            p = [
                torch.tensor(
                    cluster_pools[j][min_locs[j][0], min_locs[j][1]], requires_grad=True
                )
                for j in range(k)
            ]

            # Init the optimizer again accordingly
            optimizer = torch.optim.RMSprop(p, lr=lr)
            optimizer.zero_grad()

        # Calculating the avg value of k points
        sum_p = torch.tensor(0.0, requires_grad=True)
        for j in range(k):
            sum_p = sum_p + p[j]
        mean = sum_p / len(p)

        # Optimizing k points based on the CO2 avg value
        loss = torch.nn.functional.l1_loss(mean, torch.tensor(in_CO2_avg))

        # Progress the optimizer
        loss.backward()
        optimizer.step()

    # Save the actual position of k points in the barn
    min_pos = [
                position_map[min_locs[j][0], min_locs[j][1]]
                for j in range(k)
    ]


    # Do sensitivity analysis
    image_width, image_height = 100*barn_LW_ratio, 100  # Image dimensions

    combinations = sample_neighboring_points(
        min_locs, neighborhood_numbers, image_width, image_height, sampling_budget
    )
    losses = []
    for i in range(len(combinations)):
        p_sum = 0
        for j in range(k):
            p_sum += cluster_image[0, :, :, k][combinations[i][j]]

        losses.append(np.abs(p_sum / k - in_CO2_avg))

    return loss, np.mean(losses), np.std(losses), min_pos
