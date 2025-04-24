from generate_route_parameters import calculate_bearing, calculate_coordinates
from geopy.distance import distance
from geopy import Point as Point
from shapely.geometry import Point as shapelyPoint
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D


def find_res_of_coordinates(lons, lats):
    # Load the shapefile
    shapefile_path = "../data/resistivity/BGSCivils_Resistivity_V1.shp"
    gdf = gpd.read_file(shapefile_path)

    # Convert to WGS84 (if necessary)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Import rail coordinates
    points = [shapelyPoint(lon, lat) for lon, lat in np.column_stack((lons, lats))]
    results = [gdf[gdf.contains(p)] for p in points]
    resistivities = []
    for i, res in enumerate(results):
        if not res.empty:
            resistivities.append(res["A50"].values)
        else:
            print(f"Point {i} is not inside any polygon.")
            resistivities.append(resistivities[-1])

    resistivities = np.array(resistivities)

    return resistivities


def split_path_by_distance(route_name):
    lo_la = np.load(f"../data/rail_data/{route_name}/{route_name}_split_block_lons_lats.npz")
    lons = lo_la["lons"]
    lats = lo_la["lats"]

    distances = []
    bearings = []
    block_indices = [9999]

    dist_left = 0
    for i in range(1, len(lons)):
        # Compute distance
        start = Point(lats[i - 1], lons[i - 1])
        end = Point(lats[i], lons[i])
        dist = distance(start, end).kilometers
        # Compute bearing using a custom function
        bearing = calculate_bearing(start, end)

        while dist >= 0.06:
            distances.append(0.06 - dist_left)
            bearings.append(bearing)
            block_indices.append(i-1)
            dist -= 0.06
            dist_left = 0

        if dist < 0.06:
            distances.append(dist)
            bearings.append(bearing)
            block_indices.append(9999)
            dist_left = dist

    lon_lats = np.load(f"../data/rail_data/{route_name}/{route_name}_lons_lats.npz")
    lons_0 = lon_lats['lons']
    lats_0 = lon_lats['lats']

    lons_masts_and_ends, lats_masts_and_ends = calculate_coordinates(start_lat=lats_0[0], start_lon=lons_0[0], distances_km=distances, bearings_deg=bearings)
    mast_locs = np.where(np.array(block_indices) != 9999)[0]
    end_locs = np.where(np.array(block_indices) == 9999)[0]

    resistivities_masts = find_res_of_coordinates(lons_masts_and_ends[mast_locs], lats_masts_and_ends[mast_locs])
    block_indices_masts = np.array(block_indices)[mast_locs]

    np.savez(f"{route_name}_mast_lons_lats.npz", lons=lons_masts_and_ends[mast_locs], lats=lats_masts_and_ends[mast_locs])
    np.save(f"{route_name}_resistivities_masts.npy", resistivities_masts)
    np.save(f"{route_name}_block_indices_masts.npy", block_indices_masts)

    return resistivities_masts, block_indices_masts


def split_path_by_distance_halved(route_name):
    lo_la = np.load(f"../data/rail_data/{route_name}/{route_name}_split_block_lons_lats_halved.npz")
    lons = lo_la["lons"]
    lats = lo_la["lats"]

    distances = []
    bearings = []
    block_indices = [9999]

    dist_left = 0
    for i in range(1, len(lons)):
        # Compute distance
        start = Point(lats[i - 1], lons[i - 1])
        end = Point(lats[i], lons[i])
        dist = distance(start, end).kilometers
        # Compute bearing using a custom function
        bearing = calculate_bearing(start, end)

        while dist >= 0.06:
            distances.append(0.06 - dist_left)
            bearings.append(bearing)
            block_indices.append(i-1)
            dist -= 0.06
            dist_left = 0

        if dist < 0.06:
            distances.append(dist)
            bearings.append(bearing)
            block_indices.append(9999)
            dist_left = dist

    lon_lats = np.load(f"../data/rail_data/{route_name}/{route_name}_lons_lats.npz")
    lons_0 = lon_lats['lons']
    lats_0 = lon_lats['lats']

    lons_masts_and_ends, lats_masts_and_ends = calculate_coordinates(start_lat=lats_0[0], start_lon=lons_0[0], distances_km=distances, bearings_deg=bearings)
    mast_locs = np.where(np.array(block_indices) != 9999)[0]
    end_locs = np.where(np.array(block_indices) == 9999)[0]

    resistivities_masts = find_res_of_coordinates(lons_masts_and_ends[mast_locs], lats_masts_and_ends[mast_locs])
    block_indices_masts = np.array(block_indices)[mast_locs]

    np.savez(f"{route_name}_mast_lons_lats_halved.npz", lons=lons_masts_and_ends[mast_locs], lats=lats_masts_and_ends[mast_locs])
    np.save(f"{route_name}_resistivities_masts_halved.npy", resistivities_masts)
    np.save(f"{route_name}_block_indices_masts_halved.npy", block_indices_masts)

    return resistivities_masts, block_indices_masts


def calculate_leakage(route_name):
    data = np.load(f"../data/rail_data/{route_name}/{route_name}_distances_bearings.npz")
    blocks = data["distances"]
    resistivities, block_indices = split_path_by_distance(route_name)
    block_leaks = []

    # All blocks
    for i in range(0, len(blocks)):
        locs = np.where(np.isin(block_indices, i))[0].astype(int)
        mast_resistivity = resistivities[locs]
        mast_admittance = 1 / (mast_resistivity * 0.137817)
        total_block_admittance = np.sum(mast_admittance)
        block_leak = total_block_admittance / blocks[i]
        block_leaks.append(block_leak)

    plt.plot(block_leaks)
    plt.show()

    np.save(f"{route_name}_leakage_by_block.npy", block_leaks)


def calculate_leakage_halved(route_name):
    data = np.load(f"../data/rail_data/{route_name}/{route_name}_distances_bearings_halved.npz")
    blocks = data["distances"]
    resistivities = np.load(f"../data/resistivity/{route_name}_resistivities_masts_halved.npy")
    block_indices = np.load(f"../data/resistivity/{route_name}_block_indices_masts_halved.npy")
    block_leaks = []

    # All blocks
    for i in range(0, len(blocks)):
        locs = np.where(np.isin(block_indices, i))[0].astype(int)
        mast_resistivity = resistivities[locs]
        mast_admittance = 1 / (mast_resistivity * 0.137817)
        total_block_admittance = np.sum(mast_admittance)
        block_leak = total_block_admittance / blocks[i]
        block_leaks.append(block_leak)

    plt.plot(block_leaks, '.')
    plt.show()

    np.save(f"{route_name}_leakage_by_block_halved.npy", block_leaks)


def plot_resistivity_along_line(route_name):
    resistivities = np.load(f"../data/resistivity/{route_name}_resistivities_masts.npy")

    cols = []
    for r in resistivities:
        if r <= 16:
            cols.append("blue")
        elif 16 <= r < 32:
            cols.append("cornflowerblue")
        elif 32 <= r < 64:
            cols.append("cyan")
        elif 64 <= r < 125:
            cols.append("greenyellow")
        elif 125 <= r < 250:
            cols.append("gold")
        elif 250 <= r < 500:
            cols.append("orange")
        elif r >= 500:
            cols.append("red")
        else:
            cols.append("black")

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[:, :])
    ax0.scatter(range(0, len(resistivities)), resistivities, c=cols)
    ax0.set_xlabel("Mast Index")
    ax0.set_ylabel("Resistivity (\u03A9\u22C5m)")
    if route_name == "glasgow_edinburgh_falkirk":
        ax0.set_title("Glasgow to Edinburgh via Falkirk High")
    elif route_name == "east_coast_main_line":
        ax0.set_title("East Coast Main Line")
    elif route_name == "west_coast_main_line":
        ax0.set_title("West Coast Main Line")
    else:
        ax0.set_title("{route_name}")

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='0 - 16', markerfacecolor='blue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='16 - 32', markerfacecolor='cornflowerblue', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='32 - 64', markerfacecolor='cyan', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='64 - 125', markerfacecolor='greenyellow', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='125 - 250', markerfacecolor='gold', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='250 - 500', markerfacecolor='orange', markersize=15),
                       Line2D([0], [0], marker='o', color='w', label='> 500', markerfacecolor='red', markersize=15)]
    ax0.legend(handles=legend_elements, loc='upper right')

    plt.yscale("log")
    plt.savefig(f"resistivities_masts_{route_name}.pdf")


def plot_block_leakage(route_name):
    leakage = np.load(f"../data/resistivity/{route_name}_leakage_by_block.npy")
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[:, :])
    ax0.plot(leakage, '.')
    ax0.set_xlabel("Track Circuit Block")
    ax0.set_ylabel(r"Leakage ($\mathrm{S \cdot km^{-1}}$)")
    if route_name == "glasgow_edinburgh_falkirk":
        ax0.set_title("Glasgow to Edinburgh via Falkirk High")
    elif route_name == "east_coast_main_line":
        ax0.set_title("East Coast Main Line")
    elif route_name == "west_coast_main_line":
        ax0.set_title("West Coast Main Line")
    else:
        ax0.set_title(f"{route_name}")

    plt.show()


def block_leakage_histogram(route_name):
    leakage = np.load(f"../data/resistivity/{route_name}_leakage_by_block.npy")
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[:, :])
    bins = np.linspace(0, 9, 100)
    ax0.hist(leakage, bins=bins)
    ax0.set_xlabel("Leakage (S/km)")
    ax0.set_ylabel("Number of Blocks")
    if route_name == "glasgow_edinburgh_falkirk":
        ax0.set_title("Glasgow to Edinburgh via Falkirk High")
    elif route_name == "east_coast_main_line":
        ax0.set_title("East Coast Main Line")
    elif route_name == "west_coast_main_line":
        ax0.set_title("West Coast Main Line")
    else:
        ax0.set_title(f"{route_name}")

    plt.show()


def res_leak_comb(route_name):
    resistivities = np.load(f"../data/resistivity/{route_name}_resistivities_masts.npy")
    leakage = np.load(f"../data/resistivity/{route_name}_leakage_by_block.npy")
    block_numbers = np.load(f"../data/resistivity/{route_name}_block_indices_masts.npy")

    blocks_pos = []
    for i in range(0, len(leakage)):
        block_pos = np.where(block_numbers == i)[0]
        blocks_pos.append(block_pos[int(np.floor(len(block_pos)/2))])

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[:1, :])
    ax1 = fig.add_subplot(gs[1:, :])
    ax0.plot(range(0, len(resistivities)), resistivities, '.', color='orangered')
    ax1.plot(blocks_pos, leakage, '.', color='royalblue')
    ax0.set_yscale("log")
    if route_name == "glasgow_edinburgh_falkirk":
        fig.suptitle("Glasgow to Edinburgh via Falkirk High")
        tick_spacing = 10
    elif route_name == "east_coast_main_line":
        fig.suptitle("East Coast Main Line")
        tick_spacing = 100
    elif route_name == "west_coast_main_line":
        fig.suptitle("West Coast Main Line")
        tick_spacing = 100
    else:
        fig.suptitle(f"{route_name}")
        tick_spacing = 10
    ax1.set_xticks(blocks_pos[::tick_spacing])
    ax1.set_xticklabels(range(0, len(leakage))[::tick_spacing])
    ax0.set_xlabel("Mast Index")
    ax0.set_ylabel("Resistivity (\u03A9\u22C5m)")
    ax1.set_xlabel("Block Index")
    ax1.set_ylabel(r"Leakage ($\mathrm{S \cdot km^{-1}}$)")

    plt.savefig(f"res_leak_comb_{route_name}.pdf")
    plt.show()


#for name in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
    #split_path_by_distance_halved(name)
    #calculate_leakage_halved(name)
    #calculate_leakage(name)
    #plot_resistivity_along_line(name)
    #plot_block_leakage(name)
    #block_leakage_histogram(name)
    #res_leak_comb(name)


