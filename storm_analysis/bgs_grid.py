import numpy as np
from scipy.io import loadmat


def find_closest_grid(section):
    lon_lats = np.load(f'../data/rail_data/{section}/{section}_sub_block_lons_lats.npz')
    lon_points = lon_lats['lons']
    lat_points = lon_lats['lats']

    data = loadmat(f'../data/storm_e_fields/bgs_may2024/may2024.mat')
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
    np.save(f'../data/storm_e_fields/{section}_closest_grids.npy', closest_grids)


def gen_storm_e_vals(section, storm):
    data = loadmat(f'../data/storm_e_fields/bgs_{storm}/{storm}.mat')
    exs = data['Ex']
    eys = data['Ey']

    closest_grids = np.load(f'../data/storm_e_fields/{section}_closest_grids.npy')

    ex_blocks = np.zeros((len(closest_grids[:, 0]), len(exs[0, 0, :])))
    ey_blocks = np.zeros((len(closest_grids[:, 0]), len(eys[0, 0, :])))
    for i in range(0, len(closest_grids[:, 0])):
        print(str(i/len(closest_grids[:, 0])*100) + '%')

        ex = exs[closest_grids[i, 0], closest_grids[i, 1], :]
        ey = eys[closest_grids[i, 0], closest_grids[i, 1], :]

        if np.isnan(ex).any() or np.isnan(ey).any():
            raise ValueError(f"NaN detected at index {i}, stopping execution.")

        ex_blocks[i, :] = ex
        ey_blocks[i, :] = ey

    np.savez(f'../data/storm_e_fields/bgs_{storm}/{section}_{storm}_e_blocks.npz', ex_blocks=ex_blocks, ey_blocks=ey_blocks)


for sec in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
    for storm in ['may2024', 'sep2017', 'oct2003', 'mar1989']:
        gen_storm_e_vals(sec, storm)
