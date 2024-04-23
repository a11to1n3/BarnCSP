import os
import json
from tqdm import tqdm
import argparse

import numpy as np
import pandas as pd

from src.tda_mapper_k_points_searcher import find_optimal_k_points_tda
from src.kmedoids_k_points_searcher import find_optimal_k_points_kmedoids

APP_CONFIG = {
    "results_path": "./results",
    "max_k_points": 3,
    "barn_section": 3.1500001,
}
TDA_MAPPER_CONFIG = {
    "overlapping_portion": 75,  # %
    "lr": 5e-7,
    "epochs": 20,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}
KMEDOIDS_CONFIG = {
    "lr": 5e-7,
    "epochs": 20,
    ## For sensitivity analysis
    "sampling_budget": 10000,
    "neighborhood_numbers": 5,
}


def main(args):
    # Load csv barn file
    print("[Status] Loading file ...")
    nodes_df = pd.read_csv(args.barnFilename).drop("Unnamed: 0", axis=1)
    nodes_df.rename(
        columns={
            "X [ m ]": "X",
            " Y [ m ]": "Y",
            " Z [ m ]": "Z",
            " Carbon Dioxide.Mass Fraction": "Carbon",
            " Velocity u [ m s^-1 ]": "u",
            " Velocity v [ m s^-1 ]": "v",
            " Velocity w [ m s^-1 ]": "w",
        },
        inplace=True,
    )

    # Get LW ratio from the filename
    barn_LW_ratio = [
        int(sub_str[-1]) for sub_str in args.barnFilename.split("_") if "LW" in sub_str
    ][0]

    # Filtering the barn interior and let out the outside
    print("[Status] Finding the barn-inside region ...")
    barn_inside = np.zeros((50, 100 * barn_LW_ratio, 100))
    carbon_image = nodes_df["Carbon"].values.reshape((50, 100 * barn_LW_ratio, 100))
    barn_inside[:20, :, :] = 1
    for j in range(20, 50):
        x = np.array([carbon_image[j, :, i].sum() for i in range(100)])
        for idx, val in enumerate([x[-i] - x[-i - 1] for i in range(1, 60)]):
            if val < -200:
                barn_inside[j, :, (0 + idx) : (100 - idx)] = 1
                break

    # Calculating the average CO2 concentration inside the barn
    in_CO2_avg = np.mean(nodes_df[barn_inside.flatten().astype(bool)]["Carbon"].values)

    if args.clusteringAlg.lower() == "tda-mapper":
        print("[Status] Starting tda-mapper k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "tda-mapper"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # TDA needs a space, defined by a range to operate on
        # Here: it is the boundary of the X axis of the barn-inside region
        range_at3_max = np.histogram(
            nodes_df[barn_inside.flatten().astype(bool)][
                nodes_df[barn_inside.flatten().astype(bool)].Y
                == APP_CONFIG["barn_section"]
            ]["X"].values
        )[1].max()

        range_at3_min = np.histogram(
            nodes_df[barn_inside.flatten().astype(bool)][
                nodes_df[barn_inside.flatten().astype(bool)].Y
                == APP_CONFIG["barn_section"]
            ]["X"].values
        )[1].min()

        # Search for k points
        print("[Status] Searching k points ...")
        results = [
            find_optimal_k_points_tda(
                nodes_df,
                barn_inside,
                i,
                range_at3_max,
                range_at3_min,
                in_CO2_avg,
                APP_CONFIG["barn_section"],
                overlap=TDA_MAPPER_CONFIG["overlapping_portion"],
                lr=TDA_MAPPER_CONFIG["lr"],
                epochs=TDA_MAPPER_CONFIG["epochs"],
                sampling_budget=TDA_MAPPER_CONFIG["sampling_budget"],
                neighborhood_numbers=TDA_MAPPER_CONFIG["neighborhood_numbers"],
                barn_LW_ratio=barn_LW_ratio,
            )
            for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
        ]

    elif args.clusteringAlg.lower() == "kmedoids":
        print("[Status] Starting kmedoids k-point searcher ...")
        # Update the saving path
        APP_CONFIG["results_path"] = os.path.join(
            os.path.join(APP_CONFIG["results_path"], "kmedoids"),
            args.barnFilename.split("/")[-1].split(".")[0],
        )

        # Search for k points
        print("[Status] Searching k points ...")
        results = [
            find_optimal_k_points_kmedoids(
                nodes_df,
                barn_inside,
                i,
                in_CO2_avg,
                APP_CONFIG["barn_section"],
                lr=KMEDOIDS_CONFIG["lr"],
                epochs=KMEDOIDS_CONFIG["epochs"],
                sampling_budget=KMEDOIDS_CONFIG["sampling_budget"],
                neighborhood_numbers=KMEDOIDS_CONFIG["neighborhood_numbers"],
                barn_LW_ratio=barn_LW_ratio,
            )
            for i in tqdm(range(1, APP_CONFIG["max_k_points"] + 1))
        ]

    # Prepare a dictionary for saving into a json file
    res_summary = {}
    for i in range(len(results)):
        if results[i] is not None:
            res_summary[f"{i+1}-point"] = {}
            res_summary[f"{i+1}-point"]["Min Loss"] = float(
                results[i][0].detach().numpy()
            )
            res_summary[f"{i+1}-point"]["Mean Loss"] = results[i][1]
            res_summary[f"{i+1}-point"]["Std Loss"] = results[i][2]
            res_summary[f"{i+1}-point"][f"{i+1} Points' Position"] = [
                [j[0], j[1]] for j in results[i][3]
            ]

    print("[Status] Saving ...")
    try:
        with open(APP_CONFIG["results_path"] + ".json", "w") as fp:
            json.dump(res_summary, fp)
    except OSError:
        os.mkdir("/".join(APP_CONFIG["results_path"].split("/")[:-1]))
        with open(APP_CONFIG["results_path"] + ".json", "w") as fp:
            json.dump(res_summary, fp)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Do the k-point conditional sampling.")
    parser.add_argument("barnFilename", type=str, help="the csv file of the barn")
    parser.add_argument(
        "-c",
        "--clusteringAlg",
        type=str,
        default="tda-mapper",
        help="choose among tda-mapper/kmedoids",
    )

    args = parser.parse_args()

    main(args)
