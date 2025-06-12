from geopy.distance import distance
from geopy import Point as Point
from shapely.geometry import Point as shapelyPoint
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from geographiclib.geodesic import Geodesic
import math
geod = Geodesic.WGS84  # Define the WGS84 ellipsoid


def split_path_by_distance(route_name):
    lo_la = np.load(f"../data/rail_data/{route_name}/{route_name}_block_lons_lats.npz")
    lons = lo_la["lons"]
    lats = lo_la["lats"]
    mast_lons = [lons[0]]
    mast_lats = [lats[0]]
    mast_block_indices = []
    leftover = 0
    for i in range(0, len(lats) - 1):
        # If part of the 60 m distance between masts runs over into the next block
        if leftover != 0:
            # Handle the part that runs over
            l = geod.InverseLine(lats[i], lons[i], lats[i + 1], lons[i + 1])
            s = leftover
            g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
            mast_lons.append(g['lon2'])
            mast_lats.append(g['lat2'])
            mast_block_indices.append(i)
            # Take the coordinate of that mast as the new start
            l = geod.InverseLine(mast_lats[-1], mast_lons[-1], lats[i + 1], lons[i + 1])
            n = int(math.ceil(l.s13 / 60))
            for j in range(1, n + 1):
                ds = 60
                s = min(ds * j, l.s13)
                g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
                # Do not add the end of block coordinate, we only need the mast coordinates
                if j != n:
                    mast_lons.append(g['lon2'])
                    mast_lats.append(g['lat2'])
                    mast_block_indices.append(i)
                else:
                    pass
                # Calcuate distance that runs over into the next block
                leftover = (ds * j) - s
        # If there is no distance runs over the start of the block. (Probably only triggers once at the start)
        else:
            l = geod.InverseLine(lats[i], lons[i], lats[i + 1], lons[i + 1])
            n = int(math.ceil(l.s13 / 60))
            for j in range(1, n + 1):
                ds = 60
                s = min(ds * j, l.s13)
                g = l.Position(s, Geodesic.STANDARD | Geodesic.LONG_UNROLL)
                if j != n:
                    mast_lons.append(g['lon2'])
                    mast_lats.append(g['lat2'])
                    mast_block_indices.append(i)
                else:
                    pass
                leftover = (ds * j) - s

    mast_lons = np.delete(mast_lons, 0)
    mast_lats = np.delete(mast_lats, 0)

    plt.rcParams['font.size'] = '15'
    fig, ax = plt.subplots(figsize=(5, 10))
    ax.plot(lons, lats, '.', zorder=6, color='orange', label='Block Ends')
    ax.plot(mast_lons, mast_lats, '.', zorder=5, color='blue', label='Masts')
    ax.legend()
    plt.xlabel("Longitude")
    plt.ylabel("Latitude")
    plt.show()

    mast_block_indices = np.array(mast_block_indices)

    np.savez(f"{route_name}_mast_lons_lats.npz", lons=mast_lons, lats=mast_lats)
    np.save(f"{route_name}_mast_block_indices.npy", mast_block_indices)


def find_res_of_coordinates(route_name, plot):
    # Load the shapefile
    shapefile_path = "../data/resistivity/BGSCivils_Resistivity_V1.shp"
    gdf = gpd.read_file(shapefile_path)

    # Load mast coordinates
    lo_la = np.load(f"../data/resistivity/{route_name}_mast_lons_lats.npz")
    lats = lo_la['lats']
    lons = lo_la['lons']

    # Convert to WGS84
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Import rail coordinates
    points = [shapelyPoint(lon, lat) for lon, lat in np.column_stack((lons, lats))]
    results = [gdf[gdf.contains(p)] for p in points]
    resistivities = []
    for i, res in enumerate(results):
        if not res.empty:
            if res["A50"].values != 0:
                resistivities.append(res["A50"].values)
            else:
                resistivities.append(resistivities[-1])
        else:
            print(f"Point {i} is not inside any polygon.")
            resistivities.append(resistivities[-1])

    resistivities = np.array(resistivities)

    # Optional plotting
    if plot:
        plt.rcParams['font.size'] = '15'
        fig, ax = plt.subplots(figsize=(6, 10))
        gdf.plot(ax=ax, color='lightgray', edgecolor='black', alpha=0.5)
        x, y = zip(*[(p.x, p.y) for p in points])
        ax.scatter(x, y, color='blue', label='Masts', s=1, zorder=5)
        if route_name == "west_coast_main_line":
            oob_points = [points[i] for i in [6, 7, 2537, 2538]]
            x, y = zip(*[(p.x, p.y) for p in oob_points])
            ax.plot(x, y, 'x', color='red', zorder=6, label='No Data')
        elif route_name == "east_coast_main_line":
            oob_points = [points[i] for i in [1448, 1515, 1516]]
            x, y = zip(*[(p.x, p.y) for p in oob_points])
            ax.plot(x, y, 'x', color='red', zorder=6, label='No Data')
        else:
            pass
        ax.legend()
        plt.xlabel("Longitude")
        plt.ylabel("Latitude")
        plt.show()

    np.save(f"{route_name}_mast_resistivities.npy", resistivities)


def calculate_leakage(route_name):
    data = np.load(f"../data/rail_data/{route_name}/{route_name}_distances_bearings.npz")
    blocks = data["distances"]

    mast_resistivities = np.load(f"../data/resistivity/{route_name}_mast_resistivities.npy")
    mast_block_indices = np.load(f"../data/resistivity/{route_name}_mast_block_indices.npy")

    plt.plot(mast_resistivities)
    plt.show()

    block_leaks = []

    # All blocks
    for i in range(0, len(blocks)):
        locs = np.where(np.isin(mast_block_indices, i))[0].astype(int)
        mast_resistivity = mast_resistivities[locs]
        mast_admittance = 1 / (mast_resistivity * 0.126)
        total_block_admittance = np.sum(mast_admittance)
        block_leak = total_block_admittance / blocks[i]
        block_leaks.append(block_leak)

    np.save(f"{route_name}_leakage_block.npy", block_leaks)


def plot_resistivity_along_line(route_name):
    resistivities = np.load(f"../data/resistivity/{route_name}_mast_resistivities.npy")

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
    #plt.show()
    plt.savefig(f"mast_resistivities_{route_name}.pdf")


def plot_block_leakage(route_name):
    leakage = np.load(f"../data/resistivity/{route_name}_leakage_block.npy")
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


def res_leak_comb(route_name):
    resistivities = np.load(f"../data/resistivity/{route_name}_mast_resistivities.npy")
    leakage = np.load(f"../data/resistivity/{route_name}_leakage_block.npy")
    block_numbers = np.load(f"../data/resistivity/{route_name}_mast_block_indices.npy")

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

    #plt.savefig(f"res_leak_comb_{route_name}.pdf")
    plt.show()


for name in ["west_coast_main_line", "glasgow_edinburgh_falkirk", "east_coast_main_line"]:
    #split_path_by_distance(name)
    #find_res_of_coordinates(name, plot=False)
    #calculate_leakage(name)
    #plot_resistivity_along_line(name)
    #plot_block_leakage(name)
    res_leak_comb(name)


