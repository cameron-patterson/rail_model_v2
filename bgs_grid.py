import numpy as np
from scipy.io import loadmat


def find_closest_grid(section):
    lon_lats = np.load('data/rail_data/' + section + '/' + section + '_lons_lats.npz')
    lon_points = lon_lats['lons']
    lat_points = lon_lats['lats']

    data = loadmat('data/storm_e_fields/bgs_may2024/May2024_efields_timestamp_coordinates.mat')
    lon_grid = data['longic']
    lat_grid = data['latgic']

    closest_grids = np.zeros((len(lon_points)-1, 2))
    for i in range(0, len(lon_points)-1):
        lon_point = (lon_points[i] + lon_points[i+1])/2
        lat_point = (lat_points[i] + lat_points[i+1])/2

        # Compute the squared distance between each meshgrid point and the given coordinate
        dist_squared = (lon_grid - lon_point)**2 + (lat_grid - lat_point)**2

        # Find the indices of the minimum distance
        closest_grids[i, :] = np.unravel_index(np.argmin(dist_squared, axis=None), dist_squared.shape)

    closest_grids = closest_grids.astype(int)
    np.save(section+'_closest_grids.npy', closest_grids)


def gen_storm_e_vals(section):
    data = loadmat('data/storm_e_fields/bgs_may2024/May2024_efields_timestamp_coordinates.mat')
    exs = data['Ex']
    eys = data['Ey']

    closest_grids = np.load('data/storm_e_fields/bgs_may2024/' + section + '_closest_grids.npy')

    ex_blocks = np.zeros((len(closest_grids[:, 0]), len(exs[0, 0, :])))
    ey_blocks = np.zeros((len(closest_grids[:, 0]), len(eys[0, 0, :])))
    for i in range(0, len(closest_grids[:, 0])):
        ex_blocks[i, :] = exs[closest_grids[i, 0], closest_grids[i, 1], :]
        ey_blocks[i, :] = eys[closest_grids[i, 0], closest_grids[i, 1], :]

    np.savez(section + '_may2024_e_blocks.npz', ex_blocks=ex_blocks, ey_blocks=ey_blocks)


for sec in ["east_coast_main_line", "west_coast_main_line", "glasgow_edinburgh_falkirk"]:
    #find_closest_grid(sec)
    gen_storm_e_vals(sec)

