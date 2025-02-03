import geopandas as gpd


def save_datapoints():
    # Load the Shapefile
    shapefile_path = "../data/resistivity/BGSCivils_Resistivity_V1.shp"
    gdf = gpd.read_file(shapefile_path)

    # Convert from EPSG:27700 (British National Grid) to EPSG:4326 (WGS84)
    gdf = gdf.to_crs(epsg=4326)

    # Extract Longitude & Latitude Based on Geometry Type
    if gdf.geometry.iloc[0].geom_type == "Point":
        gdf["longitude"] = gdf.geometry.x
        gdf["latitude"] = gdf.geometry.y
    elif gdf.geometry.iloc[0].geom_type in ["Polygon", "MultiPolygon"]:
        gdf["longitude"] = gdf.geometry.centroid.x
        gdf["latitude"] = gdf.geometry.centroid.y
    elif gdf.geometry.iloc[0].geom_type in ["LineString", "MultiLineString"]:
        gdf["longitude"] = gdf.geometry.interpolate(0.5, normalized=True).x
        gdf["latitude"] = gdf.geometry.interpolate(0.5, normalized=True).y

    # View the first few rows (includes longitude, latitude, and attribute data)
    print(gdf.head())

    # Save to csv file
    gdf.to_csv("output_with_attributes.csv", index=False)
    print(gdf.columns)  # Lists all available fields from .dbf

    # Filter and save to second csv
    gdf[["longitude", "latitude", "A50"]].to_csv("filtered_output.csv", index=False)
