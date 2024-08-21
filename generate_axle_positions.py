import numpy as np


def gen_front_axle_pos_block_centre(section_name):
    data = np.load("data/rail_data/" + section_name + "/" + section_name + "_lengths_angles.npz")
    blocks = data["block_lengths"]

    blocks_sum = np.insert(np.cumsum(blocks), 0, 0)
    front_axle_pos_block_centre = np.array((blocks_sum[:-1] + blocks_sum[1:])/2)

    np.save(section_name + "_front_axle_pos_block_centre", front_axle_pos_block_centre)


def generate_axle_positions_two_track_block_centre(section_name, start_positions, block):
    data = np.load("data/rail_data/" + section_name + "/" + section_name + "_lengths_angles.npz")
    blocks = data["block_lengths"]
    blocks_sum = np.cumsum(blocks)

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
    axle_pos_a = axle_pos_a[axle_pos_a > 0]
    axle_pos_a = axle_pos_a[axle_pos_a < np.max(blocks_sum)]

    axle_pos_b = np.array([])
    for n_spa in range(0, len(start_positions)):
        train_axle_dim = np.array([0, 2, 17.5, 19.5])
        axle_locs = np.array([])
        for n_tla in range(0, train_lengths_b[n_spa]):
            axle_locs = np.append(axle_locs, train_axle_dim + (n_tla * 24))
        axle_locs = axle_locs / 1000
        axle_pos_b = np.append(axle_pos_b, start_positions[n_spa] + axle_locs)  # Note: axle_locs is added to start_positions_b since the trains are travelling in the opposite direction and axles towards the end of the train are further along the line from the start
    axle_pos_b = axle_pos_b[axle_pos_b > 0]
    axle_pos_b = axle_pos_b[axle_pos_b < np.max(blocks_sum)]

    np.savez("axle_positions_block" + str(block) + "_centre_" + section_name, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b)


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

    np.savez("axle_positions_block_centre" + section_name, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b)


#for sec in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
#    starts = np.load("data/axle_positions/" + sec + "_front_axle_pos_block_centre.npy")
#    for b in range(0, len(starts)):
#        generate_axle_positions_two_track_block_centre(sec, np.array([starts[b]]), b)

