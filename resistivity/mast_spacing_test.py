import numpy as np
from geopy.distance import distance
import matplotlib.pyplot as plt


def split_path_by_distance():
    lon_lats_ecml = np.load("../data/rail_data/east_coast_main_line/east_coast_main_line_lons_lats.npz")
    lons_ecml = lon_lats_ecml['lons']
    lats_ecml = lon_lats_ecml['lats']
    lon_lats_wcml = np.load("../data/rail_data/west_coast_main_line/west_coast_main_line_lons_lats.npz")
    lons_wcml = lon_lats_wcml['lons']
    lats_wcml = lon_lats_wcml['lats']
    lon_lats_gevf = np.load("../data/rail_data/glasgow_edinburgh_falkirk/glasgow_edinburgh_falkirk_lons_lats.npz")
    lons_gevf = lon_lats_gevf['lons']
    lats_gevf = lon_lats_gevf['lats']

    points_gevf = [(lat, lon) for lat, lon in zip(lats_gevf, lons_gevf)]

    for i in range(0, len(lons_gevf)-1):
        print(distance(points_gevf[i], points_gevf[i+1]).meters)


split_path_by_distance()
