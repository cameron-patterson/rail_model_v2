import numpy as np
import pandas as pd
from datetime import datetime, time, date, timedelta


def calculate_train_speeds(train_timetable):
    train_code = train_timetable[0]
    times = train_timetable[1:]
    time_array = np.array([])
    distance_array = np.array([])
    for t in range(0, len(times)):
        if isinstance(times[t], time):
            time_array = np.append(time_array, times[t])
            distance_array = np.append(distance_array, blocks_sum[station_block_index[t]])
        else:
            pass
    date_obj = date(2024, 7, 3)
    datetime_array = [datetime.combine(date_obj, time_obj) for time_obj in time_array]

    speeds = np.zeros(len(range(1, len(datetime_array), 2)))
    n = 0
    for i in range(1, len(datetime_array), 2):
        time_seconds = (datetime_array[i] - datetime_array[i - 1]).seconds
        dist_kms = distance_array[i] - distance_array[i - 1]
        speeds[n] = dist_kms / time_seconds
        n = n + 1

    speeds_all = np.array([])
    s = 0
    for i in range(1, len(datetime_array)):
        dif = int((datetime_array[i] - datetime_array[i-1])/timedelta(seconds=60))
        if dif != 0:
            if i % 2 == 0:
                speeds_all = np.append(speeds_all, np.full(dif, 0))
            elif i % 2 != 0:
                speeds_all = np.append(speeds_all, np.full(dif, speeds[s]))
                s += 1
        else:
            pass

    train_start_pos = distance_array[0]  # Train starting position

    return train_start_pos, speeds_all


# Load the section block lengths
data = np.load("data/rail_data/" + "glasgow_edinburgh_falkirk" + "/" + "glasgow_edinburgh_falkirk" + "_lengths_angles.npz")
blocks = data["block_lengths"]
blocks_sum = np.cumsum(blocks) - 0.001  # Set positions to be 1m back from end of block

# Load the Excel file
df = pd.read_excel("data/gef_timetable.xlsx", sheet_name="g2e-3.7.24")

# Start and end times
start_time = datetime(2024, 3, 7, 6, 0)
end_time = datetime(2024, 3, 7, 9, 0)

# Generate datetime objects at 1-minute intervals using list comprehension
timeseries_day = [start_time + timedelta(minutes=i) for i in range(int((end_time - start_time).total_seconds() / 60) + 1)]

# Make 2D array for train positions in the day
train_pos_day = np.zeros((3, len(timeseries_day)))

# Convert each row to an array
rows_as_arrays = [row.to_numpy() for index, row in df.iterrows()]

station_block_index = rows_as_arrays[0][1:]

for array in rows_as_arrays[1:]:
    train_start_pos, speeds_all = calculate_train_speeds(array)

    train_pos_day[0, 0] = train_start_pos
    for t in range(1, 60):
        train_pos_day[0, t] = train_pos_day[0, t-1] + (speeds_all[t-1] * 60)

    pass

