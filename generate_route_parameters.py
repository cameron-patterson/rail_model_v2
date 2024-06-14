import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from geopy.distance import geodesic


def generate_blocks_params(route_name):
    # Load geojson file
    f = open('data/rail_data/' + route_name + '/' + route_name + '.geojson')
    datum = json.load(f)

    # Extract lon and lat arrays from geojson file
    longitudes = np.zeros(528)
    latitudes = np.zeros(528)

    features = datum['features']
    for i in range(0, len(features)):
        feat = features[i]
        geometry = feat['geometry']
        longitudes[i] = geometry['coordinates'][0]
        latitudes[i] = geometry['coordinates'][1]

    # Combine longitude and latitude into a single array of points
    points = np.vstack((longitudes, latitudes)).T

    # Find the starting point with the highest latitude
    start_index = np.argmax(latitudes)

    # Calculate the distance matrix
    dist_matrix = cdist(points, points)

    # Start with the point having the highest latitude
    n_points = len(points)
    visited = np.zeros(n_points, dtype=bool)
    path = [start_index]
    visited[start_index] = True

    for _ in range(1, n_points):
        last_visited = path[-1]
        nearest_neighbor = np.argmin(np.where(visited, np.inf, dist_matrix[last_visited]))
        path.append(nearest_neighbor)
        visited[nearest_neighbor] = True

    # The `path` array now contains the order of points to visit
    ordered_points = points[path]

    lons = ordered_points[:, 0]
    lats = ordered_points[:, 1]

    blocks_lengths = calculate_block_lengths(lats, lons)
    angles = calculate_angles(lats, lons)

    np.savez(route_name + '_lengths_angles.npz', block_lengths=blocks_lengths, angles=angles)


def calculate_block_lengths(latitudes, longitudes):
    # Combine longitude and latitude into a single array of points
    ordered_points = np.vstack((longitudes, latitudes)).T

    # Calculate the distances between the consecutive points in the ordered path
    distances = [geodesic((ordered_points[i][1], ordered_points[i][0]),
                          (ordered_points[i + 1][1], ordered_points[i + 1][0])).km
                 for i in range(len(ordered_points) - 1)]
    distances = np.array(distances)

    return distances


def calculate_angles(latitudes, longitudes):
    # Calculate differences
    delta_lon = np.diff(longitudes)
    delta_lat = np.diff(latitudes)

    # Calculate angles in radians
    angles = np.arctan2(delta_lat, delta_lon)

    return angles


def plot_route(route_name):
    # Load geojson file
    f = open('data/rail_data/' + route_name + '/' + route_name + '.geojson')
    datum = json.load(f)

    # Extract lon and lat arrays from geojson file
    longitudes = np.zeros(528)
    latitudes = np.zeros(528)

    features = datum['features']
    for i in range(0, len(features)):
        feat = features[i]
        geometry = feat['geometry']
        longitudes[i] = geometry['coordinates'][0]
        latitudes[i] = geometry['coordinates'][1]

    # Combine longitude and latitude into a single array of points
    points = np.vstack((longitudes, latitudes)).T

    # Find the starting point with the highest latitude
    start_index = np.argmax(latitudes)

    # Calculate the distance matrix
    dist_matrix = cdist(points, points)

    # Start with the point having the highest latitude
    n_points = len(points)
    visited = np.zeros(n_points, dtype=bool)
    path = [start_index]
    visited[start_index] = True

    for _ in range(1, n_points):
        last_visited = path[-1]
        nearest_neighbor = np.argmin(np.where(visited, np.inf, dist_matrix[last_visited]))
        path.append(nearest_neighbor)
        visited[nearest_neighbor] = True

    # The `path` array now contains the order of points to visit
    ordered_points = points[path]

    lons = ordered_points[:, 0]
    lats = ordered_points[:, 1]

    fig, ax = plt.subplots(figsize=(5, 8))
    ax.plot(lons, lats)
    stations_pos = [0, 118, 172, 231, 256, 274, 288, -1]
    ax.plot(lons[stations_pos], lats[stations_pos], 'x')
    plt.show()


#generate_blocks_params('west_coast_main_line')
plot_route('west_coast_main_line')
