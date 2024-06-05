import json
import matplotlib.pyplot as plt
import numpy as np
import scipy as sp
import scipy.interpolate

# Load geojson file
f = open('export.geojson')
datum = json.load(f)

# Extract lon and lat arrays from geojson file
lon_all = np.zeros(522)
lat_all = np.zeros(522)

features = datum['features']
for i in range(0, len(features)):
    feat = features[i]
    geometry = feat['geometry']
    lon_all[i] = geometry['coordinates'][0]
    lat_all[i] = geometry['coordinates'][1]


