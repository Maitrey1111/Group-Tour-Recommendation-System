# user selection
import math
import numpy as np

# graphs
import torch

# clustering
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import DBSCAN

# local imports
from gcn_model import GCNModel, modelParams
from data import (
    getDataFor,
    destination_chars,
    destinations_info,
    users,
    merged_ratings,
    user_preferences_df,
)


def predict(data):
    model_path = "model weights/gcn_uc_yfcc.pth"
    layers = modelParams()["layers"]
    best_model = GCNModel(layers)
    best_model.load_state_dict(torch.load(model_path))
    best_model.eval()
    predictions = best_model(data)

    # correct = (predictions == data.y).sum().item()
    # accuracy = correct / (predictions.numel())
    # print("Accuracy: ", accuracy)

    return predictions


def find_preferrable_destinations(user_preferences, destination_chars):
    destination_chars_combined = [
        chars_set.replace("\n", "")[1:-1]
        for chars_set in destination_chars["Combined_Characteristics"].values
    ]

    destination_chars_combined = [
        [int(v) for v in chars_set.strip().split(", ") if v != ""]
        for chars_set in destination_chars_combined
    ]

    cosine_similarities = torch.tensor(
        cosine_similarity(user_preferences, destination_chars_combined)
    )

    top_15_indices = torch.topk(cosine_similarities, k=15, dim=1).indices
    nearest_destinations = destination_chars.loc[
        list(top_15_indices.flatten()), "DestinationID"
    ].values

    return nearest_destinations.reshape(user_preferences.shape[0], 15)


def haversine(pos1, pos2):
    # Radius of the Earth in kilometers
    R = 6371.0
    lat1, lon1 = pos1
    lat2, lon2 = pos2

    # Convert latitude and longitude from degrees to radians
    lat1 = math.radians(float(lat1))
    lon1 = math.radians(float(lon1))
    lat2 = math.radians(float(lat2))
    lon2 = math.radians(float(lon2))

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    # Calculate the distance(s)
    distance = R * c

    return distance


def find_closest_group(preferrable_destinations, destinations_info):
    # Extract the latitude and longitude of the nearest destinations
    lats = []
    lons = []
    for destination_id in set(preferrable_destinations.flatten()):
        lat, lon = destinations_info[
            destinations_info["DestinationID"] == destination_id
        ][["Latitude", "Longitude"]].values[0]
        lats.append(lat)
        lons.append(lon)
    lat_array = np.array(lats)
    lon_array = np.array(lons)

    # Stack the latitude and longitude arrays to create a single array
    data = np.stack((lat_array, lon_array), axis=1)

    # Convert destination_ids to a list
    destination_ids_list = list(preferrable_destinations.flatten())

    # Get the latitude and longitude of the destinations
    lats = destinations_info[
        destinations_info["DestinationID"].isin(destination_ids_list)
    ]["Latitude"].values
    lons = destinations_info[
        destinations_info["DestinationID"].isin(destination_ids_list)
    ]["Longitude"].values

    # Convert latitude and longitude to numpy arrays
    lat_array = np.array(lats)
    lon_array = np.array(lons)

    # Stack the latitude and longitude arrays to create a single array
    data = np.stack((lat_array, lon_array), axis=1)

    # Perform clustering using DBSCAN - Density-Based Spatial Clustering of Applications with Noise
    clustering = DBSCAN(eps=50, metric=haversine).fit(data)
    clusters = [
        cluster_value for cluster_value in clustering.labels_ if cluster_value >= 0
    ]

    # Find the cluster with the maximum number of elements
    try:
        max_cluster_label = np.argmax(np.bincount(clusters))
    except:
        return []

    # Get the indices of the elements in the maximum cluster
    max_cluster_indices = np.where(clustering.labels_ == max_cluster_label)[0]

    # Get the destination IDs of the elements in the maximum cluster
    closest_destination_ids = [
        destination_ids_list[index] for index in max_cluster_indices
    ]

    # Return the closest destination IDs
    return closest_destination_ids


def users_list():
    return users


# Recommendation
def recommend(selected_users=["99662199@N00", "99395734@N08", "8638314@N05"]):
    data = getDataFor(selected_users)
    predictions = predict(data)

    preferrable_destinations = find_preferrable_destinations(
        user_preferences=predictions.detach().numpy(),
        destination_chars=destination_chars,
    )

    closest_destination_ids = find_closest_group(
        np.unique(preferrable_destinations), destinations_info
    )

    itinerary = []
    for destination_id in closest_destination_ids:
        name = list(
            destination_chars[destination_chars["DestinationID"] == destination_id][
                "Name"
            ]
        )
        itinerary.append(name[0])

    return selected_users, itinerary
