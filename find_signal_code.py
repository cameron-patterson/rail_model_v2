import numpy as np
import json
from scipy.spatial.distance import cdist
from geopy.distance import geodesic


def find_signal_code(route_name):
    # Load geojson file
    f = open('data/rail_data/' + route_name + '/' + route_name + '.geojson')
    datum = json.load(f)

    # Extract the feature list
    features = datum['features']

    # Generate the array of IDs
    ids = [feature['id'] for feature in features]
    print(ids[238])


def find_signal_ref_wcml(route_name):
    # Load geojson file
    f = open('data/rail_data/' + route_name + '/' + route_name + '.geojson')
    datum = json.load(f)

    # Extract the feature list
    features = datum['features']

    # Generate the list with ref values, indicating when ref is missing
    refs = [feature['properties'].get('ref', 'ref not found') for feature in features]

    lockerbie_ref = 'GMC826'
    penrith_ref = 'CE207'
    preston_ref = 'PN112'
    wigan_ref = 'WN44'
    warrington_ref = 'WN201'

    lockerbie_pos = refs.index(lockerbie_ref)
    penrith_pos = refs.index(penrith_ref)
    preston_pos = refs.index(preston_ref)
    lancaster_pos = preston_pos - 25
    wigan_pos = refs.index(wigan_ref)
    warrington_pos = refs.index(warrington_ref)

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

    lockerbie_lon = longitudes[lockerbie_pos]
    penrith_lon = longitudes[penrith_pos]
    preston_lon = longitudes[preston_pos]
    lancaster_lon = longitudes[lancaster_pos]
    wigan_lon = longitudes[wigan_pos]
    warrington_lon = longitudes[warrington_pos]

    #lockerbie_lat = latitudes[lockerbie_pos]
    #penrith_lat = latitudes[penrith_pos]
    #preston_lat = latitudes[preston_pos]
    #lancaster_lat = latitudes[lancaster_pos]
    #wigan_lat = latitudes[wigan_pos]
    #warrington_lat = latitudes[warrington_pos]

    lockerbie_sorted_pos = np.where(lons == lockerbie_lon)
    penrith_sorted_pos = np.where(lons == penrith_lon)
    lancaster_sorted_pos = np.where(lons == lancaster_lon)
    preston_sorted_pos = np.where(lons == preston_lon)
    wigan_sorted_pos = np.where(lons == wigan_lon)
    warrington_sorted_pos = np.where(lons == warrington_lon)

    pass

find_signal_code('west_coast_main_line')
find_signal_ref_wcml("west_coast_main_line")
