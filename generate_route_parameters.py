import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist
from geographiclib.geodesic import Geodesic
import math
geod = Geodesic.WGS84  # define the WGS84 ellipsoid


# Generates the geographic coordinates of signalling network nodes along the route
# Saves the outputs (lons, lats) as file *route_name*_lons_lats.npz
def generate_longitudes_and_latitudes(route_name):
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

    np.savez(route_name + '_lons_lats.npz', lons=lons, lats=lats)


# Generates the length and bearing of the track circuit blocks along the route
# Bearing is defined at 0 degrees when directly north, and increasing positively clockwise
# Save the outputs (distances, bearings) as file *route_name*_distances_bearings.npz
# Also marks which track circuits are doubled for a single signal
# Save the outputs (double_locs) as file *route_name*_double_locs.npy
def generate_distances_and_bearings(route_name):
    lo_la = np.load(f"data/rail_data/{route_name}/{route_name}_lons_lats.npz")
    lons = lo_la["lons"]
    lats = lo_la["lats"]

    dists_km = []
    bears_deg = []
    new_lons = [lons[0]]
    new_lats = [lats[0]]
    for i in range(0, len(lats) - 1):
        l = geod.InverseLine(lats[i], lons[i], lats[i + 1], lons[i + 1])
        if l.s13 < 1000:
            s = l.s13
            g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            dists_km.append(g['s12'] / 1000)
            bears_deg.append(g['azi2'])
            new_lons.append(g['lon2'])
            new_lats.append(g['lat2'])
        if 1000 < l.s13 < 2000:
            ds = l.s13 / 2
            n = int(math.ceil(l.s13 / ds))
            for j in range(1, n + 1):
                s = min(ds * j, l.s13)
                g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
                dists_km.append((g['s12'] / j) / 1000)
                bears_deg.append(g['azi2'])
                new_lons.append(g['lon2'])
                new_lats.append(g['lat2'])
        if 2000 < l.s13 < 3000:
            ds = l.s13 / 3
            n = int(math.ceil(l.s13 / ds))
            for j in range(1, n + 1):
                s = min(ds * j, l.s13)
                g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
                dists_km.append((g['s12'] / j) / 1000)
                bears_deg.append(g['azi2'])
                new_lons.append(g['lon2'])
                new_lats.append(g['lat2'])
        if 3000 < l.s13 < 4000:
            ds = l.s13 / 3
            n = int(math.ceil(l.s13 / ds))
            for j in range(1, n + 1):
                s = min(ds * j, l.s13)
                g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
                dists_km.append((g['s12'] / j) / 1000)
                bears_deg.append(g['azi2'])
                new_lons.append(g['lon2'])
                new_lats.append(g['lat2'])

    #plt.plot(dists_km)
    #plt.show()
    bears_rad = np.deg2rad(bears_deg)

    np.savez(route_name + "_distances_bearings", distances=dists_km, bearings=bears_rad)
    np.savez(route_name + '_block_lons_lats.npz', lons=new_lons, lats=new_lats)


#for route in ["east_coast_main_line", "west_coast_main_line", "glasgow_edinburgh_falkirk"]:
#    generate_longitudes_and_latitudes(route)


#for route in ["east_coast_main_line", "west_coast_main_line", "glasgow_edinburgh_falkirk"]:
#    generate_distances_and_bearings(route)

