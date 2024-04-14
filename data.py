import pandas as pd
import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops
import random

destinations_info = pd.read_csv("datasets/all_pois.csv")
merged_ratings = pd.read_csv("datasets/merged_ratings.csv")
user_preferences_df = pd.read_csv("datasets/user_preferences_df.csv")
ratings_matrix = pd.read_csv("datasets/ratings_matrix.csv")
destination_chars = pd.read_csv("datasets/destination_chars.csv")

characteristics = [
    [str(ch).strip().lower() for ch in characteristic]
    for characteristic in list(destinations_info["Characteristics"].str.split(","))
    if type(characteristic) == list
    # if characteristic is list
]
characteristics = list(set([item for sublist in characteristics for item in sublist]))

destinations = destinations_info.index.unique()
users = user_preferences_df["UserID"].unique()

# as per model
users_train = list(pd.Series(users).sample(frac=0.7, random_state=0).sort_values())
users_test = list(set(users) - set(users_train))


def getRandomUsers():
    num_users = random.randint(1, 10)
    tourist_group = random.sample(users_test, num_users)
    return tourist_group


def EdgeFormation(selected_users, ratings, destinations):
    # Add edge between selected_users which have visited the same destination
    user_destination_pairs = [
        (i, j)
        for i in range(len(selected_users))
        for j in range(len(destinations))
        if j in ratings.columns and ratings.loc[selected_users[i], destinations[j]] > 0
    ]

    # Same-place edges among selected_users
    same_destinations_edges = torch.tensor(
        [
            [x[0], y[0]]
            for i, x in enumerate(user_destination_pairs)
            for j, y in enumerate(user_destination_pairs)
            if j > i and x[1] == y[1]
        ],
        dtype=torch.long,
    ).t()
    return same_destinations_edges


def getDataFor(selected_users):
    ratings_matrix_new = ratings_matrix[ratings_matrix["UserID"].isin(selected_users)]
    same_destination_edges = EdgeFormation(
        selected_users, ratings_matrix_new, destinations
    )
    data = Data(
        x=torch.tensor(ratings_matrix_new.iloc[:, 1:].values.astype("float32")),
        edge_index=same_destination_edges,
    )
    data.edge_index = add_self_loops(data.edge_index)[0]

    user_preferences_new = [
        preference.replace("\n", "")[1:-1]
        for preference in user_preferences_df[
            user_preferences_df["UserID"].isin(selected_users)
        ]["Preferences"].values
    ]

    user_preferences_new = [
        [float(v) for v in preference.strip().split(",") if v != ""]
        for preference in user_preferences_new
    ]

    data.y = torch.tensor(np.array(list(user_preferences_new)))

    return data
