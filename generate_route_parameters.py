from geopy.distance import distance
from geopy import Point
import json
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import cdist


# Calculates the bearing
# Used as part of "generate_distances_and_bearings" function below
def calculate_bearing(start, end):
    """Calculate the initial bearing from start to end point."""
    lat1, lon1 = np.radians(start.latitude), np.radians(start.longitude)
    lat2, lon2 = np.radians(end.latitude), np.radians(end.longitude)

    dlon = lon2 - lon1
    x = np.sin(dlon) * np.cos(lat2)
    y = np.cos(lat1) * np.sin(lat2) - np.sin(lat1) * np.cos(lat2) * np.cos(dlon)

    initial_bearing = np.arctan2(x, y)
    initial_bearing = np.degrees(initial_bearing)
    compass_bearing = (initial_bearing + 360) % 360

    return compass_bearing


# Calculates the coordinates based on distance and bearing inputs
# Used as part of "plot_route" function below
def calculate_coordinates(start_lat, start_lon, distances_km, bearings_deg):
    lats = [start_lat]
    lons = [start_lon]

    for distance_km, bearing_deg in zip(distances_km, bearings_deg):
        start_point = Point(lats[-1], lons[-1])
        destination = distance(kilometers=distance_km).destination(start_point, bearing_deg)
        lats.append(destination.latitude)
        lons.append(destination.longitude)

    return np.array(lons), np.array(lats)


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
    lo_la = np.load("data/rail_data/" + route_name + "/" + route_name + "_lons_lats.npz")
    lons = lo_la["lons"]
    lats = lo_la["lats"]

    distances = []
    bearings = []
    doubles = []

    for i in range(1, len(lons)):
        # Compute distance
        start = Point(lats[i - 1], lons[i - 1])
        end = Point(lats[i], lons[i])
        dist = distance(start, end).kilometers
        # Compute bearing using a custom function
        bearing = calculate_bearing(start, end)

        if dist < 1.6:
            distances.append(dist)
            bearings.append(bearing)
        else:
            distances.append(dist/2)
            distances.append(dist/2)
            bearings.append(bearing)
            bearings.append(bearing)
            doubles.append(int(i + len(doubles)/2))
            doubles.append(int(i + 1 + ((len(doubles)-1)/2)))

    bearings = np.radians(bearings)

    np.save(route_name + "_double_locs.npy", doubles)
    np.savez(route_name + "_distances_bearings", distances=distances, bearings=bearings)


# Plots the route using both saved longitudes and latitudes and a calculation based on the distances and bearings
# Can also be used to save sub block longitudes and latitudes (by removing comment marker in code)
def plot_route(route_name):
    # Load in coordinates
    lo_la = np.load("data/rail_data/" + route_name + "/" + route_name + "_lons_lats.npz")
    lons = lo_la["lons"]
    lats = lo_la["lats"]

    # Calculate new coordinates from a starting point
    dist_bear = np.load("data/rail_data/" + route_name + "/" + route_name + "_distances_bearings.npz")
    distances = dist_bear["distances"]
    bearings = np.degrees(dist_bear["bearings"])
    start_lat = lats[0]  # Starting latitude
    start_lon = lons[0]  # Starting longitude
    new_longitudes, new_latitudes = calculate_coordinates(start_lat, start_lon, distances, bearings)
    #np.savez(route_name + '_sub_block_lons_lats.npz', lons=new_longitudes, lats=new_latitudes)

    # Plotting the result
    plt.figure(figsize=(10, 6))
    plt.plot(new_longitudes, new_latitudes, '.-')

    # Annotate index
    for i in range(len(new_latitudes)):
        plt.annotate(f'{i}', (new_longitudes[i], new_latitudes[i]), textcoords="offset points", xytext=(5, 5), ha='center')

    # Plot stations
    if route_name == "east_coast_main_line":
        plt.annotate(f'Edinburgh', (new_longitudes[0], new_latitudes[0]), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Dunbar', (-2.5134821328340253, 55.99823996612304), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Reston', (-2.1924769773550032, 55.84988437864046), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Berwick', (-2.0110546741013904, 55.77433622590785), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Alnmouth', (-1.6366665682296975, 55.39282177570046), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Morpeth', (-1.6825075373836347, 55.16251338923229), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Newcastle', (-1.6170934508323778, 54.968321202446795), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Durham', (-1.5817184004661433, 54.779373695480714), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Northallerton', (-1.4416085770254263, 54.33295761144301), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'York', (-1.0933138238589433, 53.957945702763695), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Doncaster', (-1.1396136067604175, 53.52195080663149), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Peterborough', (-0.25009682477053535, 52.57436720972351), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'London KC', (-0.12462817842099386, 51.53154478277447), textcoords="offset points", xytext=(5, 5), ha='center')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Original vs. Calculated Coordinates')
    plt.grid(True)
    plt.show()




#for route in ["east_coast_main_line", "west_coast_main_line", "glasgow_edinburgh_falkirk"]:
#    generate_longitudes_and_latitudes(route)


#for route in ["east_coast_main_line", "west_coast_main_line", "glasgow_edinburgh_falkirk"]:
#    generate_distances_and_bearings(route)


#for route in ["east_coast_main_line", "west_coast_main_line", "glasgow_edinburgh_falkirk"]:
#    plot_route(route)
