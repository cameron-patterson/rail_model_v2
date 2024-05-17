import numpy as np


def generate_axle_positions_two_track(start_positions_a, start_positions_b):
    # Input train configurations here
    #train_lengths_a = np.array([])  # The number of carriages in the trains in direction a
    train_lengths_a = np.full(len(start_positions_a), 8)  # ALT: for if all trains are the same length
    #train_lengths_b = np.array([])
    train_lengths_b = np.full(len(start_positions_b), 8)

    axle_pos_a = []
    for n_spa in range(0, len(start_positions_a)):
        train_axle_dim = np.array([0, 2, 17.5, 19.5])  # Dimensions of the axles of a single car in metres
        axle_locs = []
        for n_tla in range(0, train_lengths_a[n_spa]):
            axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))  # The next car's first axle is 24 metres from the previous car's first axle
        axle_locs = axle_locs / 1000  # Convert to kilometres
        axle_pos_a.append(start_positions_a[n_spa] - axle_locs)  # Obtain axle positions from first axle location and axle distribution

    axle_pos_b = []
    for n_spa in range(0, len(start_positions_b)):
        train_axle_dim = np.array([0, 2, 17.5, 19.5])
        axle_locs = []
        for n_tla in range(0, train_lengths_b[n_spa]):
            axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))
        axle_locs = axle_locs / 1000
        axle_pos_b.append(start_positions_b[n_spa] + axle_locs)  # Note: axle_locs is added to start_positions_b since the trains are travelling in the opposite direction and axles towards the end of the train are further along the line from the start

    np.savez("axle_positions_at_end_ge", axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b)


at_end_pos_ge = np.load("at_end_axle_pos_glasgow_edinburgh_falkirk.npz")
starts_a = at_end_pos_ge["at_end_axle_pos_a"]
starts_b = at_end_pos_ge["at_end_axle_pos_b"]

generate_axle_positions_two_track(starts_a, starts_b)
