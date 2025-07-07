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
        # If part of the 70 m distance between masts runs over into the next block
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
            n = int(math.ceil(l.s13 / 70))
            for j in range(1, n + 1):
                ds = 70
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
            n = int(math.ceil(l.s13 / 70))
            for j in range(1, n + 1):
                ds = 70
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
    a20 = []
    a50 = []
    a80 = []
    for i, res in enumerate(results):
        if not res.empty:
            if res["A50"].values != 0:
                a20.append(res["A20"].values[0])
                a50.append(res["A50"].values[0])
                a80.append(res["A80"].values[0])
            else:
                a20.append(a20[-1])
                a50.append(a50[-1])
                a80.append(a80[-1])
        else:
            print(f"Point {i} is not inside any polygon.")
            a20.append(a20[-1])
            a50.append(a50[-1])
            a80.append(a80[-1])

    a20 = np.array(a20)
    a50 = np.array(a50)
    a80 = np.array(a80)

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

    np.savez(f"{route_name}_mast_resistivities.npz", a20=a20, a50=a50, a80=a80)


def calculate_leakage(route_name):
    data = np.load(f"../data/rail_data/{route_name}/{route_name}_distances_bearings.npz")
    blocks = data["distances"]

    mast_resistivities = np.load(f"../data/resistivity/{route_name}_mast_resistivities.npz")
    mast_block_indices = np.load(f"../data/resistivity/{route_name}_mast_block_indices.npy")

    mast_resistivities_a20 = mast_resistivities['a20']
    mast_resistivities_a50 = mast_resistivities['a50']
    mast_resistivities_a80 = mast_resistivities['a80']
    block_leaks_a20 = []
    block_leaks_a50 = []
    block_leaks_a80 = []
    # All blocks
    for i in range(0, len(blocks)):
        locs = np.where(np.isin(mast_block_indices, i))[0].astype(int)
        mast_resistivity_a20 = mast_resistivities_a20[locs]
        mast_admittance_a20 = 1 / (mast_resistivity_a20 * 0.126)
        total_block_admittance_a20 = np.sum(mast_admittance_a20)
        block_leak_a20 = total_block_admittance_a20 / blocks[i]
        block_leaks_a20.append(block_leak_a20)

        mast_resistivity_a50 = mast_resistivities_a50[locs]
        mast_admittance_a50 = 1 / (mast_resistivity_a50 * 0.126)
        total_block_admittance_a50 = np.sum(mast_admittance_a50)
        block_leak_a50 = total_block_admittance_a50 / blocks[i]
        block_leaks_a50.append(block_leak_a50)

        mast_resistivity_a80 = mast_resistivities_a80[locs]
        mast_admittance_a80 = 1 / (mast_resistivity_a80 * 0.126)
        total_block_admittance_a80 = np.sum(mast_admittance_a80)
        block_leak_a80 = total_block_admittance_a80 / blocks[i]
        block_leaks_a80.append(block_leak_a80)

    #np.savez(f"{route_name}_block_leakage.npz", a20=block_leaks_a20, a50=block_leaks_a50, a80=block_leaks_a80)


#for name in ["glasgow_edinburgh_falkirk", "west_coast_main_line", "east_coast_main_line"]:
    #split_path_by_distance(name)
    #find_res_of_coordinates(name, plot=False)
    #calculate_leakage(name)


