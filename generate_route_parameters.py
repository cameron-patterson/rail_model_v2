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
    features = datum['features']
    longitudes = np.zeros(len(features))
    latitudes = np.zeros(len(features))
    for i in range(0, len(features)):
        feat = features[i]
        geometry = feat['geometry']
        longitudes[i] = geometry['coordinates'][0]
        latitudes[i] = geometry['coordinates'][1]

    # Add start and end points (if needed)
    if route_name == 'west_coast_main_line':
        latitudes = np.insert(latitudes, 0, 55.8596993)  # Start lon
        longitudes = np.insert(longitudes, 0, -4.2576579)  # Start lat
        latitudes = np.insert(latitudes, len(latitudes), 51.5279335)  # End lon
        longitudes = np.insert(longitudes, len(longitudes), -0.1343821)  # End lat
    elif route_name == 'east_coast_main_line':
        latitudes = np.insert(latitudes, 0, 51.5310147)  # Start lon
        longitudes = np.insert(longitudes, 0, -0.1231747)  # Start lat
    elif route_name == 'glasgow_edinburgh_falkirk':
        latitudes = np.insert(latitudes, 0, 55.8621298)  # Start lon
        longitudes = np.insert(longitudes, 0, -4.2513018)  # Start lat
        latitudes = np.insert(latitudes, len(latitudes), 55.9515217)  # End lon
        longitudes = np.insert(longitudes, len(longitudes), -3.1911483)  # End lat
    else:
        print("Route starting point not specified")

    # Combine longitude and latitude into a single array of points
    points = np.vstack((longitudes, latitudes)).T

    # Find the starting point
    if route_name == 'west_coast_main_line':
        start_index = np.argmax(latitudes)
    elif route_name == 'east_coast_main_line' or route_name == 'glasgow_edinburgh_falkirk':
        start_index = np.argmin(longitudes)
    else:
        print("Route starting point not specified")

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

    start_blocks = calculate_block_lengths(lats, lons)
    start_angles = calculate_angles(lats, lons)

    # Split the longer blocks up into multiple and also appends the angles of the new blocks
    blocks_lengths = np.array([])
    angles = np.array([])
    double_locs = np.array([], dtype=int)
    j = 0
    for i in range(0, len(start_blocks)):
        if start_blocks[i] > 1.6:
            blocks_lengths = np.append(blocks_lengths, [start_blocks[i] / 2, start_blocks[i] / 2])
            angles = np.append(angles, [start_angles[i], start_angles[i]])
            double_locs = np.append(double_locs, [j, j+1])
            j += 2
        else:
            blocks_lengths = np.append(blocks_lengths, start_blocks[i])
            angles = np.append(angles, start_angles[i])
            j += 1

    np.savez(route_name + '_lengths_angles.npz', block_lengths=blocks_lengths, angles=angles)
    np.save(route_name + '_double_locs', double_locs)


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
    features = datum['features']
    longitudes = np.zeros(len(features))
    latitudes = np.zeros(len(features))
    for i in range(0, len(features)):
        feat = features[i]
        geometry = feat['geometry']
        longitudes[i] = geometry['coordinates'][0]
        latitudes[i] = geometry['coordinates'][1]

    # Add start and end points (if needed)
    if route_name == 'west_coast_main_line':
        latitudes = np.insert(latitudes, 0, 55.8596993)  # Start lon
        longitudes = np.insert(longitudes, 0, -4.2576579)  # Start lat
        latitudes = np.insert(latitudes, len(latitudes), 51.5279335)  # End lon
        longitudes = np.insert(longitudes, len(longitudes), -0.1343821)  # End lat
    elif route_name == 'east_coast_main_line':
        latitudes = np.insert(latitudes, 0, 51.5310147)  # Start lon
        longitudes = np.insert(longitudes, 0, -0.1231747)  # Start lat
    elif route_name == 'glasgow_edinburgh_falkirk':
        latitudes = np.insert(latitudes, 0, 55.8621298)  # Start lon
        longitudes = np.insert(longitudes, 0, -4.2513018)  # Start lat
        latitudes = np.insert(latitudes, len(latitudes), 55.9515217)  # End lon
        longitudes = np.insert(longitudes, len(longitudes), -3.1911483)  # End lat
    elif route_name == 'bristol_parkway_london':
        latitudes = np.insert(latitudes, len(latitudes), 51.5165159)  # End lon
        longitudes = np.insert(longitudes, len(longitudes), -0.1771918)  # End lat
    else:
        print("Route starting point not specified")

    # Combine longitude and latitude into a single array of points
    points = np.vstack((longitudes, latitudes)).T

    # Find the starting point
    if route_name == 'west_coast_main_line':
        start_index = np.argmax(latitudes)
    elif route_name == 'east_coast_main_line' or route_name == 'glasgow_edinburgh_falkirk' or route_name == 'bristol_parkway_london':
        start_index = np.argmin(longitudes)
    else:
        print("Route starting point not specified")

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
    ax.plot(lons, lats, '.-')
    ax.plot(lons[0], lats[0], 'x', color='red')
    plt.show()


def save_lon_lats(route_name):
    # Load geojson file
    f = open('data/rail_data/' + route_name + '/' + route_name + '.geojson')
    datum = json.load(f)

    # Extract lon and lat arrays from geojson file
    features = datum['features']
    longitudes = np.zeros(len(features))
    latitudes = np.zeros(len(features))
    for i in range(0, len(features)):
        feat = features[i]
        geometry = feat['geometry']
        longitudes[i] = geometry['coordinates'][0]
        latitudes[i] = geometry['coordinates'][1]

    # Add start and end points (if needed)
    if route_name == 'west_coast_main_line':
        latitudes = np.insert(latitudes, 0, 55.8596993)  # Start lon
        longitudes = np.insert(longitudes, 0, -4.2576579)  # Start lat
        latitudes = np.insert(latitudes, len(latitudes), 51.5279335)  # End lon
        longitudes = np.insert(longitudes, len(longitudes), -0.1343821)  # End lat
    elif route_name == 'east_coast_main_line':
        latitudes = np.insert(latitudes, 0, 51.5310147)  # Start lon
        longitudes = np.insert(longitudes, 0, -0.1231747)  # Start lat
    elif route_name == 'glasgow_edinburgh_falkirk':
        latitudes = np.insert(latitudes, 0, 55.8621298)  # Start lon
        longitudes = np.insert(longitudes, 0, -4.2513018)  # Start lat
        latitudes = np.insert(latitudes, len(latitudes), 55.9515217)  # End lon
        longitudes = np.insert(longitudes, len(longitudes), -3.1911483)  # End lat
    elif route_name == 'bristol_parkway_london':
        latitudes = np.insert(latitudes, len(latitudes), 51.5165159)  # End lon
        longitudes = np.insert(longitudes, len(longitudes), -0.1771918)  # End lat
    else:
        print("Route starting point not specified")

    # Combine longitude and latitude into a single array of points
    points = np.vstack((longitudes, latitudes)).T

    # Find the starting point
    if route_name == 'west_coast_main_line':
        start_index = np.argmax(latitudes)
    elif route_name == 'east_coast_main_line' or route_name == 'glasgow_edinburgh_falkirk' or route_name == 'bristol_parkway_london':
        start_index = np.argmin(longitudes)
    else:
        print("Route starting point not specified")

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

    np.savez(route_name + '_lons_lats.npz', lons=lons, lats=lats)

#generate_blocks_params('west_coast_main_line')
#generate_blocks_params('east_coast_main_line')
#generate_blocks_params('glasgow_edinburgh_falkirk')
#plot_route("east_coast_main_line")


#for r in ['west_coast_main_line', 'east_coast_main_line', 'glasgow_edinburgh_falkirk']:
#    save_lon_lats(r)
