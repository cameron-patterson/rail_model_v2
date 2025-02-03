import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import Point
import geopandas as gpd


def plot_data_centroid():
    # Load the CSV
    df = pd.read_csv("../data/resistivity/filtered_output.csv")

    # Extract longitude, latitude, and attribute values
    lon = df["longitude"].values
    lat = df["latitude"].values
    values = df["A50"].values

    plt.plot(lon, lat, '.')
    plt.show()


def find_poly_of_coordinate():
    # Load the shapefile
    shapefile_path = "../data/resistivity/BGSCivils_Resistivity_V1.shp"
    gdf = gpd.read_file(shapefile_path)

    # Convert to WGS84 (if necessary)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Check structure
    print(gdf.head())

    # Define the input coordinate (Longitude, Latitude)
    input_point = Point(-1.1276, 52.5074)

    # Find the polygon that contains the point
    polygon_match = gdf[gdf.contains(input_point)]

    # Print result
    if not polygon_match.empty:
        print("Polygon found!")
        print(polygon_match)
    else:
        print("No polygon contains this point.")

    # Find the nearest polygon (if point is outside all polygons)
    gdf["distance"] = gdf.geometry.distance(input_point)
    nearest_polygon = gdf.loc[gdf["distance"].idxmin()]

    print("Nearest polygon details:")
    print(nearest_polygon)


def find_poly_of_coordinates():
    # Load the shapefile
    shapefile_path = "../data/resistivity/BGSCivils_Resistivity_V1.shp"
    gdf = gpd.read_file(shapefile_path)

    # Convert to WGS84 (if necessary)
    if gdf.crs != "EPSG:4326":
        gdf = gdf.to_crs(epsg=4326)

    # Check structure
    print(gdf.head())

    # Import rail coordinates
    lon_lats_gef = np.load("../data/rail_data/glasgow_edinburgh_falkirk/glasgow_edinburgh_falkirk_sub_block_lons_lats.npz")
    lons_gevf = lon_lats_gef['lons']
    lats_gevf = lon_lats_gef['lats']
    points_gevf = [Point(lon, lat) for lon, lat in zip(lons_gevf, lats_gevf)]

    results_gevf = [gdf[gdf.contains(p)] for p in points_gevf]

    for i, res in enumerate(results_gevf):
        if not res.empty:
            print(f"Point {i} is inside: {res}")
        else:
            print(f"Point {i} is not inside any polygon.")

plot_data_centroid()
find_poly_of_coordinates()
