import numpy as np
import pandas as pd
from datetime import datetime, time, date, timedelta

section = "glasgow_edinburgh_falkirk"
direction = "b"


def calculate_train_speeds(train_timetable):
    train_code = train_timetable[0]
    times = train_timetable[1:]
    time_array = np.array([])
    distance_array = np.array([])
    for t in range(0, len(times)):
        if isinstance(times[t], time):  # Checks for NaN values meaning a train does not stop at a certain station
            time_array = np.append(time_array, times[t])  # Records arrival and leaving times at stations
            distance_array = np.append(distance_array, blocks_sum[station_block_index[t]])  # Records distance at those times
        else:
            pass
    date_obj = date(2024, 8, 5)  # Date of the timetable
    datetime_array = [datetime.combine(date_obj, time_obj) for time_obj in time_array]  # Combines the times and dates to make a datetime array for the station times

    # Calculate train speed between stations
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
            if i % 2 == 0:  # If i is even and non-zero, this indicates the train is waiting at the station
                speeds_all = np.append(speeds_all, np.full(dif, 0))
            elif i % 2 != 0:
                speeds_all = np.append(speeds_all, np.full(dif, speeds[s]))
                s += 1
        else:
            pass

    train_start_pos = distance_array[0]  # Train starting position
    train_start_time = datetime_array[0]  # Train starting time

    return train_start_time, train_start_pos, speeds_all


# Load the section block lengths
data = np.load("data/rail_data/" + section + "/" + section + "_distances_bearings.npz")
if direction == "a":
    blocks = data["distances"]
    blocks_sum = np.cumsum(blocks) - 0.001  # Set positions to be 1m back from end of block
elif direction == "b":
    blocks = data["distances"]
    blocks = np.insert(blocks[0:-1], 0, 0)
    blocks_sum = np.cumsum(blocks) + 0.001  # Set positions to be 1m back from end of block
else:
    print("Error")

# Load the Excel file
df = pd.read_excel("data/axle_positions/" + section + "_timetable.xlsx", sheet_name="direction_" + direction)

# Start and end times
start_time = datetime(2024, 8, 5, 0, 0)
end_time = datetime(2024, 8, 5, 23, 59)

# Generate datetime objects at 1-minute intervals using list comprehension
timeseries_day = [start_time + timedelta(minutes=i) for i in range(int((end_time - start_time).total_seconds() / 60) + 1)]

# Convert each row of the timetable to an array
timetable_rows_as_arrays = [row.to_numpy() for index, row in df.iterrows()]

# Make 2D array for train positions in the day
train_pos_day = np.zeros((len(timetable_rows_as_arrays)-1, len(timeseries_day)))

station_block_index = timetable_rows_as_arrays[0][1:]  # block indices for the stations from first row of timetable

# Calculate train start position and speeds for each train in the timetable
for i in range(0, len(timetable_rows_as_arrays) - 1):
    timetable_array = timetable_rows_as_arrays[i+1]
    train_start_time, train_start_pos, speeds_all = calculate_train_speeds(timetable_array)

    start_index = timeseries_day.index(train_start_time)
    train_pos_day[i, start_index] = train_start_pos
    for t in range(0, len(speeds_all)):
        train_pos_day[i, t + start_index + 1] = train_pos_day[i, t + start_index] + (speeds_all[t] * 60)

train_pos_day[train_pos_day == 0] = np.nan
np.save(section + "_train_pos_day_direction_" + direction, train_pos_day)
