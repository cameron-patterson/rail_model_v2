import numpy as np
from scipy.io import loadmat


def find_closest_grid(section):
    lon_lats = np.load('data/rail_data/' + section + '/' + section + '_lons_lats.npz')
    lon_points = lon_lats['lons']
    lat_points = lon_lats['lats']

    data = loadmat('data/storm_e_fields/bgs_may_2024/May2024_efields_timestamp_coordinates.mat')
    lon_grid = data['longic']
    lat_grid = data['latgic']

    closest_grids = np.zeros((len(lon_points), 2))
    for i in range(0, len(lon_points)):
        lon_point = lon_points[i]
        lat_point = lat_points[i]

        # Compute the squared distance between each meshgrid point and the given coordinate
        dist_squared = (lon_grid - lon_point)**2 + (lat_grid - lat_point)**2

        # Find the indices of the minimum distance
        closest_grids[i, :] = np.unravel_index(np.argmin(dist_squared, axis=None), dist_squared.shape)

    closest_grids = closest_grids.astype(int)
    np.save(section+"_closest_grids.npy", closest_grids)


for sec in ["west_coast_main_line", "east_coast_main_line", "glasgow_edinburgh_falkirk"]:
    find_closest_grid(sec)
