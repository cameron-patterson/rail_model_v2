import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Generates the axle positions given the position of the train's front axle
# Return the outputs (axle_pos_a, axle_pos_b)
def generate_axle_positions_two_track(section_name, start_positions_a, start_positions_b):
    data = np.load("data/rail_data/" + section_name + "/" + section_name + "_lengths_angles.npz")
    blocks = data["block_lengths"]
    blocks_sum = np.cumsum(blocks)

    # Input train configurations here
    #train_lengths_a = np.array([])  # The number of carriages in the trains in direction a
    train_lengths_a = np.full(len(start_positions_a), 8)  # ALT: for if all trains are the same length
    #train_lengths_b = np.array([])
    train_lengths_b = np.full(len(start_positions_b), 8)

    axle_pos_a = np.array([])
    for n_spa in range(0, len(start_positions_a)):
        train_axle_dim = np.array([0, 2, 17.5, 19.5])  # Dimensions of the axles of a single car in metres
        axle_locs = np.array([])
        for n_tla in range(0, train_lengths_a[n_spa]):
            axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))  # The next car's first axle is 24 metres from the previous car's first axle
        axle_locs = axle_locs / 1000  # Convert to kilometres
        axle_pos_a = np.append(axle_pos_a, start_positions_a[n_spa] - axle_locs)  # Obtain axle positions from first axle location and axle distribution
    axle_pos_a = axle_pos_a[axle_pos_a > 0]
    axle_pos_a = axle_pos_a[axle_pos_a < np.max(blocks_sum)]

    axle_pos_b = np.array([])
    for n_spa in range(0, len(start_positions_b)):
        train_axle_dim = np.array([0, 2, 17.5, 19.5])
        axle_locs = np.array([])
        for n_tla in range(0, train_lengths_b[n_spa]):
            axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))
        axle_locs = axle_locs / 1000
        axle_pos_b = np.append(axle_pos_b, start_positions_b[n_spa] + axle_locs)  # Note: axle_locs is added to start_positions_b since the trains are travelling in the opposite direction and axles towards the end of the train are further along the line from the start
    axle_pos_b = axle_pos_b[axle_pos_b > 0]
    axle_pos_b = axle_pos_b[axle_pos_b < np.max(blocks_sum)]

    return axle_pos_a, axle_pos_b


# Generates the axle positions for when a train's rearmost axle in at the centre of each block
# Saves the outputs (axle_pos_a_all, axle_pos_b_all) as file *route_name*_axle_positions_two_track_back_axle_block_centre.npz
def generate_axle_positions_two_track_back_axle_block_centre(section_name):
    data = np.load("data/rail_data/" + section_name + "/" + section_name + "_distances_bearings.npz")
    blocks = data["distances"]
    blocks_sum = np.cumsum(blocks)
    start_positions_all = np.array((np.insert(blocks_sum, 0, 0)[:-1] + np.insert(blocks_sum, 0, 0)[1:]) / 2)

    axle_pos_a_all = np.empty(len(start_positions_all), dtype=object)
    axle_pos_b_all = np.empty(len(start_positions_all), dtype=object)
    for i in range(0, len(start_positions_all)):
        start_positions = np.array([start_positions_all[i]])
        # Input train configurations here
        #train_lengths_a = np.array([])  # The number of carriages in the trains in direction a
        train_lengths_a = np.full(len(start_positions), 8)  # ALT: for if all trains are the same length
        #train_lengths_b = np.array([])
        train_lengths_b = np.full(len(start_positions), 8)

        axle_pos_a = np.array([])
        for n_spa in range(0, len(start_positions)):
            train_axle_dim = np.array([0, 2, 17.5, 19.5])  # Dimensions of the axles of a single car in metres
            axle_locs = np.array([])
            for n_tla in range(0, train_lengths_a[n_spa]):
                axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))  # The next car's first axle is 24 metres from the previous car's first axle
            axle_locs = axle_locs / 1000  # Convert to kilometres
            axle_pos_a = np.append(axle_pos_a, start_positions[n_spa] - axle_locs)  # Obtain axle positions from first axle location and axle distribution
            axle_pos_a = axle_pos_a + axle_locs[-1]  # Add the length of the train to the axle position so that the rear axle of the train is at the centre
        axle_pos_a = axle_pos_a[axle_pos_a > 0]
        axle_pos_a = axle_pos_a[axle_pos_a < np.max(blocks_sum)]
        axle_pos_a_all[i] = axle_pos_a

        axle_pos_b = np.array([])
        for n_spa in range(0, len(start_positions)):
            train_axle_dim = np.array([0, 2, 17.5, 19.5])
            axle_locs = np.array([])
            for n_tla in range(0, train_lengths_b[n_spa]):
                axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))
            axle_locs = axle_locs / 1000
            axle_pos_b = np.append(axle_pos_b, start_positions[n_spa] + axle_locs)  # Note: axle_locs is added to start_positions_b since the trains are travelling in the opposite direction and axles towards the end of the train are further along the line from the start
            axle_pos_b = axle_pos_b - axle_locs[-1]  # Subtract the length of the train to the axle position so that the rear axle of the train is at the centre
        axle_pos_b = axle_pos_b[axle_pos_b > 0]
        axle_pos_b = axle_pos_b[axle_pos_b < np.max(blocks_sum)]
        axle_pos_b_all[i] = axle_pos_b

    np.savez(f"data/axle_positions/block_centre/axle_positions_two_track_back_axle_at_centre_{section_name}.npz", axle_pos_a_all=axle_pos_a_all, axle_pos_b_all=axle_pos_b_all)


# Generates the axle positions for when a train's rearmost axle in at the end of each block
# Saves the outputs (axle_pos_a_all, axle_pos_b_all) as file *route_name*_axle_positions_two_track_back_axle_at_end.npz
def generate_axle_positions_two_track_back_axle_at_end(section_name):
    data = np.load("data/rail_data/" + section_name + "/" + section_name + "_distances_bearings.npz")
    blocks = data["distances"]
    blocks_sum = np.cumsum(blocks)
    start_positions_a = blocks_sum - 0.001
    start_positions_b = np.insert(blocks_sum, 0, 0)[:-1] + 0.001

    axle_pos_a_all = np.empty(len(start_positions_a), dtype=object)
    axle_pos_b_all = np.empty(len(start_positions_b), dtype=object)
    for i in range(0, len(start_positions_a)):
        start_positions = np.array([start_positions_a[i]])
        # Input train configurations here
        #train_lengths_a = np.array([])  # The number of carriages in the trains in direction a
        train_lengths_a = np.full(len(start_positions), 8)  # ALT: for if all trains are the same length

        axle_pos_a = np.array([])
        for n_spa in range(0, len(start_positions)):
            train_axle_dim = np.array([0, 2, 17.5, 19.5])  # Dimensions of the axles of a single car in metres
            axle_locs = np.array([])
            for n_tla in range(0, train_lengths_a[n_spa]):
                axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))  # The next car's first axle is 24 metres from the previous car's first axle
            axle_locs = axle_locs / 1000  # Convert to kilometres
            axle_pos_a = np.append(axle_pos_a, start_positions[n_spa] - axle_locs)  # Obtain axle positions from first axle location and axle distribution
            axle_pos_a = axle_pos_a + axle_locs[-1]  # Add the length of the train to the axle position so that the rear axle of the train is at the centre
        axle_pos_a = axle_pos_a[axle_pos_a > 0]
        axle_pos_a = axle_pos_a[axle_pos_a < np.max(blocks_sum)]
        axle_pos_a_all[i] = axle_pos_a

    for i in range(0, len(start_positions_b)):
        start_positions = np.array([start_positions_b[i]])
        # train_lengths_b = np.array([])
        train_lengths_b = np.full(len(start_positions), 8)
        axle_pos_b = np.array([])
        for n_spa in range(0, len(start_positions)):
            train_axle_dim = np.array([0, 2, 17.5, 19.5])
            axle_locs = np.array([])
            for n_tla in range(0, train_lengths_b[n_spa]):
                axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))
            axle_locs = axle_locs / 1000
            axle_pos_b = np.append(axle_pos_b, start_positions[n_spa] + axle_locs)  # Note: axle_locs is added to start_positions_b since the trains are travelling in the opposite direction and axles towards the end of the train are further along the line from the start
            axle_pos_b = axle_pos_b - axle_locs[-1]  # Subtract the length of the train to the axle position so that the rear axle of the train is at the centre
        axle_pos_b = axle_pos_b[axle_pos_b > 0]
        axle_pos_b = axle_pos_b[axle_pos_b < np.max(blocks_sum)]
        axle_pos_b_all[i] = axle_pos_b

    np.savez(f"data/axle_positions/at_end/axle_positions_two_track_back_axle_at_end_{section_name}.npz", axle_pos_a_all=axle_pos_a_all, axle_pos_b_all=axle_pos_b_all)


# Generates the axle positions based on a realistic timetable
# Saves the outputs (axle_pos_a_all, axle_pos_b_all) as file *route_name*_axle_positions_timetable.npz
def generate_axle_positions_two_track_timetable(section_name):
    pos_a = np.load('data/axle_positions/timetable/' + section_name + '_train_pos_day_direction_a.npy')
    pos_b = np.load('data/axle_positions/timetable/' + section_name + '_train_pos_day_direction_b.npy')
    data = np.load("data/rail_data/" + section_name + "/" + section_name + "_distances_bearings.npz")
    blocks = data["distances"]
    blocks_sum = np.cumsum(blocks)

    all_starts_a = np.empty((1440,), dtype=object)
    for i in range(0, len(pos_a[0, :])):
        all_starts_a[i] = pos_a[:, i][np.where(~np.isnan(pos_a[:, i]))[0]]

    all_starts_b = np.empty((1440,), dtype=object)
    for i in range(0, len(pos_b[0, :])):
        all_starts_b[i] = pos_b[:, i][np.where(~np.isnan(pos_b[:, i]))[0]]

    all_axle_pos_a = np.empty((1440,), dtype=object)
    all_axle_pos_b = np.empty((1440,), dtype=object)
    for i in range(0, len(all_starts_a)):
        start_positions_a = all_starts_a[i]
        start_positions_b = all_starts_b[i]

        # Input train configurations here
        train_lengths_a = np.full(len(start_positions_a), 8)  # ALT: for if all trains are the same length
        train_lengths_b = np.full(len(start_positions_b), 8)

        axle_pos_a = np.array([])
        for n_spa in range(0, len(start_positions_a)):
            train_axle_dim = np.array([0, 2, 17.5, 19.5])  # Dimensions of the axles of a single car in metres
            axle_locs = np.array([])
            for n_tla in range(0, train_lengths_a[n_spa]):
                axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))  # The next car's first axle is 24 metres from the previous car's first axle
            axle_locs = axle_locs / 1000  # Convert to kilometres
            axle_pos_a = np.append(axle_pos_a, start_positions_a[n_spa] - axle_locs)  # Obtain axle positions from first axle location and axle distribution
        axle_pos_a = axle_pos_a[axle_pos_a > 0]
        axle_pos_a = axle_pos_a[axle_pos_a < np.max(blocks_sum)]
        all_axle_pos_a[i] = axle_pos_a

        axle_pos_b = np.array([])
        for n_spa in range(0, len(start_positions_b)):
            train_axle_dim = np.array([0, 2, 17.5, 19.5])
            axle_locs = np.array([])
            for n_tla in range(0, train_lengths_b[n_spa]):
                axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))
            axle_locs = axle_locs / 1000
            axle_pos_b = np.append(axle_pos_b, start_positions_b[n_spa] + axle_locs)  # Note: axle_locs is added to start_positions_b since the trains are travelling in the opposite direction and axles towards the end of the train are further along the line from the start
        axle_pos_b = axle_pos_b[axle_pos_b > 0]
        axle_pos_b = axle_pos_b[axle_pos_b < np.max(blocks_sum)]
        all_axle_pos_b[i] = axle_pos_b

    np.savez(section_name + "_axle_positions_timetable", axle_pos_a_all=all_axle_pos_a, axle_pos_b_all=all_axle_pos_b)


def axle_plot_test(name):
    for name in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
        data = np.load(f"data/rail_data/{name}/{name}_distances_bearings.npz")
        blocks = data["distances"]
        blocks_sum = np.cumsum(blocks)
        blocks_sum = np.insert(blocks_sum, 0, 0)

        axles_end_a = \
        np.load(f"data/axle_positions/at_end/axle_positions_two_track_back_axle_at_end_{name}.npz", allow_pickle=True)[
            "axle_pos_a_all"]
        axles_centre_a = \
        np.load(f"data/axle_positions/block_centre/axle_positions_two_track_back_axle_block_centre_{name}.npz",
                allow_pickle=True)["axle_pos_a_all"]

        plt.rcParams['font.size'] = '15'
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 1)
        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1], sharex=ax1)

        for i in range(0, len(axles_end_a)):
            ax1.plot(axles_end_a[i], np.full(len(axles_end_a[i]), 0), 'x', color="cornflowerblue")
            ax1.axvline(blocks_sum[i], color="orangered", alpha=0.5)

        for i in range(0, len(axles_end_a)):
            ax2.plot(axles_centre_a[i], np.full(len(axles_centre_a[i]), 0), 'x', color="cornflowerblue")
            ax2.axvline(blocks_sum[i], color="orangered", alpha=0.5)
            ax2.axvline(blocks_sum[i + 1] - (blocks[i] / 2), color="orangered", alpha=0.5, linestyle='--')

        plt.show()


for sec in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
    generate_axle_positions_two_track_back_axle_block_centre(sec)
    generate_axle_positions_two_track_back_axle_at_end(sec)

#generate_axle_positions_two_track_timetable("east_coast_main_line")

