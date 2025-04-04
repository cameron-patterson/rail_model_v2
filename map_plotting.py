import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import geopandas as gpd
from shapely.geometry import LineString


def gen_shape_file(section):
    lon_lats = np.load(f"data/rail_data/{section}/{section}_split_block_lons_lats.npz")
    lon_points = lon_lats['lons']
    lat_points = lon_lats['lats']

    # Step 1: Prepare your coordinates (longitude, latitude)
    coordinates = np.array(list(zip(lon_points, lat_points)))

    # Step 2: Create a LineString from the coordinates
    line = LineString(coordinates)

    # Step 3: Create a GeoDataFrame to store the LineString
    gdf = gpd.GeoDataFrame(index=[0], crs="EPSG:4326", geometry=[line])

    # Step 4: Save the GeoDataFrame as a shapefile
    output_path = f"{section}_line_shapefile.shp"
    gdf.to_file(output_path)

    print(f"Shapefile saved to {output_path}")


def plot_map_both():
    lon_lats_gef = np.load("data/rail_data/glasgow_edinburgh_falkirk/glasgow_edinburgh_falkirk_split_block_lons_lats.npz")
    lon_points_gef = lon_lats_gef['lons']
    lat_points_gef = lon_lats_gef['lats']
    lon_lats_ecml = np.load("data/rail_data/east_coast_main_line/east_coast_main_line_split_block_lons_lats.npz")
    lon_points_ecml = lon_lats_ecml['lons']
    lat_points_ecml = lon_lats_ecml['lats']
    lon_lats_wcml = np.load("data/rail_data/west_coast_main_line/west_coast_main_line_split_block_lons_lats.npz")
    lon_points_wcml = lon_lats_wcml['lons']
    lat_points_wcml = lon_lats_wcml['lats']
    coast = np.loadtxt("data/coastline.txt")

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 10))

    gs = GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[:, :-1])
    ax1 = fig.add_subplot(gs[:, 1:])

    ax0.plot(lon_points_wcml, lat_points_wcml, linewidth=3, linestyle='-', color="lavenderblush", label="West Coast Main Line")
    ax0.plot(lon_points_gef, lat_points_gef, linewidth=3, linestyle='-', color="royalblue", label="Glasgow to Edinburgh via Falkirk High")
    ax0.plot(lon_points_wcml[249:289], lat_points_wcml[249:289], linewidth=3, linestyle='-', color="red", label="Preston to Lancaster")
    ax0.set_title("")

    ax1.plot(lon_points_wcml, lat_points_wcml, linewidth=3, linestyle='-', color="red", label="West Coast Main Line")
    ax1.plot(lon_points_gef, lat_points_gef, linewidth=3, linestyle='-', color="royalblue", label="Glasgow to Edinburgh via Falkirk High")
    ax1.plot(lon_points_ecml, lat_points_ecml, linewidth=3, linestyle='-', color="darkorange", label="East Coast Main Line")

    def common_components(ax):
        ax.plot(coast[:, 0], coast[:, 1], color='lightgrey', linewidth=1, zorder=2)
        ax.set_ylim(51, 57)
        ax.set_xlim(-6, 1)
        ax.set_xlabel("Geographic Longitude")
        ax.set_ylabel("Geographic Latitude")
        ax.legend()

    common_components(ax0)
    common_components(ax1)

    #plt.savefig("plots/map.jpg")
    plt.show()


def plot_map_new():
    lon_lats_gef = np.load("data/rail_data/glasgow_edinburgh_falkirk/glasgow_edinburgh_falkirk_split_block_lons_lats.npz")
    lon_points_gef = lon_lats_gef['lons']
    lat_points_gef = lon_lats_gef['lats']
    lon_lats_ecml = np.load("data/rail_data/east_coast_main_line/east_coast_main_line_split_block_lons_lats.npz")
    lon_points_ecml = lon_lats_ecml['lons']
    lat_points_ecml = lon_lats_ecml['lats']
    lon_lats_wcml = np.load("data/rail_data/west_coast_main_line/west_coast_main_line_split_block_lons_lats.npz")
    lon_points_wcml = lon_lats_wcml['lons']
    lat_points_wcml = lon_lats_wcml['lats']
    coast = np.loadtxt("data/coastline.txt")

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(8, 10))

    gs = GridSpec(1, 1)
    ax1 = fig.add_subplot(gs[:, :])

    ax1.plot(lon_points_wcml[0:10], lat_points_wcml[0:10], linewidth=3, linestyle='-', color="red", label="West Coast Main Line")
    ax1.plot(lon_points_gef[0:10], lat_points_gef[0:10], linewidth=3, linestyle='-', color="royalblue", label="Glasgow to Edinburgh via Falkirk High")
    ax1.plot(lon_points_ecml[0:10], lat_points_ecml[0:10], linewidth=3, linestyle='-', color="darkorange", label="East Coast Main Line")

    def common_components(ax):
        ax.plot(coast[:, 0], coast[:, 1], color='lightgrey', linewidth=1, zorder=2)
        ax.set_ylim(51, 57)
        ax.set_xlim(-6, 1)
        ax.set_xlabel("Geographic Longitude")
        ax.set_ylabel("Geographic Latitude")
        ax.legend()

    common_components(ax1)

    #plt.savefig("plots/map.jpg")
    plt.show()


def plot_route_map(route_name):
    # Load in coordinates
    lo_la = np.load("data/rail_data/" + route_name + "/" + route_name + "_sub_block_lons_lats.npz")
    lons = lo_la["lons"]
    lats = lo_la["lats"]

    # Plotting the result
    plt.figure(figsize=(10, 6))
    plt.plot(lons, lats, '.-')

    # Annotate index
    for i in range(len(lats)):
        plt.annotate(f'{i}', (lons[i], lats[i]), textcoords="offset points", xytext=(5, 5), ha='center')

    # Plot stations
    if route_name == "east_coast_main_line":
        plt.annotate(f'Edinburgh', (lons[0], lats[0]), textcoords="offset points", xytext=(5, 5), ha='center')
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

    if route_name == "west_coast_main_line":
        plt.annotate(f'Edinburgh', (lons[0], lats[0]), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Lancaster', (-2.8078097679573366, 54.048832786272634), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'Preston', (-2.7066943140909876, 53.7556202408875), textcoords="offset points", xytext=(5, 5), ha='center')
        plt.annotate(f'London Euston', (lons[-1], lats[-1]), textcoords="offset points", xytext=(5, 5), ha='center')


    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Original vs. Calculated Coordinates')
    plt.grid(True)
    plt.show()


#plot_map_new()
#plot_route_map("west_coast_main_line")
#for sec in ["west_coast_main_line", "east_coast_main_line", "glasgow_edinburgh_falkirk"]:
#    gen_shape_file(sec)
