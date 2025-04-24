import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def rail_model_two_track_e_parallel(section_name, e_parallel, axle_pos_a, axle_pos_b):
    # Create dictionary of network parameters
    parameters = {"z_sig": 0.0289,  # Signalling rail series impedance (ohms/km)
                  "z_trac": 0.0289,  # Traction return rail series impedance (ohms/km)
                  "y_sig": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac": 1.6,  # Traction return rail parallel admittance in moderate conditions (siemens/km)
                  "v_power": 10,  # Track circuit power supply voltage (volts)
                  "r_power": 7.2,  # Track circuit power supply resistance (ohms)
                  "r_relay": 20,  # Track circuit relay resistance (ohms)
                  "r_cb": 1e-3,  # Cross bond resistance (ohms)
                  "r_axle": 251e-4,  # Axle resistance (ohms)
                  "i_power": 10 / 7.2,  # Track circuit power supply equivalent current source (amps)
                  "y_power": 1 / 7.2,  # Track circuit power supply admittance (siemens)
                  "y_relay": 1 / 20,  # Track circuit relay admittance (siemens)
                  "y_cb": 1 / 1e-3,  # Cross bond admittance (siemens)
                  "y_axle": 1 / 251e-4}  # Axle admittance (siemens)

    # Calculate the electrical characteristics of the rails
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig"])
    gamma_trac = np.sqrt(parameters["z_trac"] * parameters["y_trac"])
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig"])
    z0_trac = np.sqrt(parameters["z_trac"] / parameters["y_trac"])

    # Load in the lengths and bearings of the track circuit blocks
    # Note: zero degrees is directly northwards, with positive values increasing clockwise
    data = np.load(f"../data/rail_data/{section_name}/{section_name}_distances_bearings.npz")
    blocks = data["distances"]
    blocks_sum = np.cumsum(blocks)  # Cumulative sum of block lengths

    # Add cross bonds and axles which split the blocks into sub blocks
    # Note: "a" and "b" are used to identify the opposite directions of travel in this network (two-track)
    pos_cb = np.arange(0.4, np.sum(blocks), 0.4)  # Position of the cross bonds
    trac_sub_block_sum_a = np.sort(np.insert(np.concatenate((blocks_sum, pos_cb, axle_pos_a)), 0, 0))  # Traction return rail connects to axles and cross bonds
    sig_sub_block_sum_a = np.sort(np.insert(np.concatenate((blocks_sum, blocks_sum[:-1], axle_pos_a)), 0, 0))  # Signalling rail connects to axles, but need to add points on either side or IRJ
    trac_sub_block_sum_b = np.sort(np.insert(np.concatenate((blocks_sum, pos_cb, axle_pos_b)), 0, 0))
    sig_sub_block_sum_b = np.sort(np.insert(np.concatenate((blocks_sum, blocks_sum[:-1], axle_pos_b)), 0, 0))
    trac_sub_blocks_a = np.diff(trac_sub_block_sum_a)
    sig_sub_blocks_a = np.diff(sig_sub_block_sum_a)
    sig_sub_blocks_a[sig_sub_blocks_a == 0] = np.nan  # Sets a nan value to indicate the IRJ gap
    trac_sub_blocks_b = np.diff(trac_sub_block_sum_b)
    sig_sub_blocks_b = np.diff(sig_sub_block_sum_b)
    sig_sub_blocks_b[sig_sub_blocks_b == 0] = np.nan

    # Announce if cross bonds overlap with block boundaries
    if 0 in trac_sub_blocks_a or 0 in trac_sub_blocks_b:
        print("cb block overlap")
    else:
        pass

        # Set up equivalent-pi parameters
        ye_trac_a = 1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_a))  # Series admittance for traction return rail
        ye_sig_a = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))  # Series admittance for signalling rail
        ye_trac_b = 1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_b))
        ye_sig_b = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))
        yg_trac = (np.cosh(gamma_trac * blocks) - 1) * (1 / (z0_trac * np.sinh(gamma_trac * blocks)))  # Parallel admittance for traction return rail
        yg_sig = (np.cosh(gamma_sig * blocks) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * blocks)))  # Parallel admittance for signalling rail
        yg_trac_comb = np.empty(len(yg_trac) + 1)
        yg_trac_comb[0] = yg_trac[0]
        yg_trac_comb[1:-1] = yg_trac[:-1] + yg_trac[1:]
        yg_trac_comb[-1] = yg_trac[-1]

    # Calculate numbers of nodes ready to use in indexing
    n_nodes_a = len(trac_sub_block_sum_a) + len(sig_sub_block_sum_a)  # Number of nodes in direction of travel a
    n_nodes_b = len(trac_sub_block_sum_b) + len(sig_sub_block_sum_b)
    n_nodes = n_nodes_a + n_nodes_b  # Number of nodes in the whole network
    n_nodes_trac_a = len(trac_sub_block_sum_a)  # Number of nodes in the traction return rail
    n_nodes_trac_b = len(trac_sub_block_sum_b)
    n_nodes_sig_a = len(sig_sub_block_sum_a)  # Number of nodes in the signalling rail
    n_nodes_sig_b = len(sig_sub_block_sum_b)

    # Index of rail nodes in the rails
    node_locs_trac_a = np.arange(0, n_nodes_trac_a, 1).astype(int)
    node_locs_sig_a = np.arange(n_nodes_trac_a, n_nodes_a, 1).astype(int)
    node_locs_trac_b = np.arange(n_nodes_a, n_nodes_a + n_nodes_trac_b, 1).astype(int)
    node_locs_sig_b = np.arange(n_nodes_a + n_nodes_trac_b, n_nodes).astype(int)
    # Index of cross bond nodes
    node_locs_cb_a = node_locs_trac_a[np.where(np.isin(trac_sub_block_sum_a, pos_cb))[0]]
    node_locs_cb_b = node_locs_trac_b[np.where(np.isin(trac_sub_block_sum_b, pos_cb))[0]]
    # Index of axle nodes
    node_locs_axle_trac_a = node_locs_trac_a[np.where(np.isin(trac_sub_block_sum_a, axle_pos_a))[0]]
    node_locs_axle_sig_a = node_locs_sig_a[np.where(np.isin(sig_sub_block_sum_a, axle_pos_a))[0]]
    node_locs_axle_trac_b = node_locs_trac_b[np.where(np.isin(trac_sub_block_sum_b, axle_pos_b))[0]]
    node_locs_axle_sig_b = node_locs_sig_b[np.where(np.isin(sig_sub_block_sum_b, axle_pos_b))[0]]

    # Index of traction return rail power supply and relay nodes
    # Note: "a" begins with a relay and ends with a power supply, "b" begins with a power supply and ends with a relay
    node_locs_no_cb_axle_trac_a = np.delete(node_locs_trac_a, np.where(np.isin(node_locs_trac_a, np.concatenate((node_locs_cb_a, node_locs_axle_trac_a))))[0])
    node_locs_power_trac_a = node_locs_no_cb_axle_trac_a[1:]
    node_locs_relay_trac_a = node_locs_no_cb_axle_trac_a[:-1]
    node_locs_no_cb_axle_trac_b = np.delete(node_locs_trac_b, np.where(np.isin(node_locs_trac_b, np.concatenate((node_locs_cb_b, node_locs_axle_trac_b))))[0])
    node_locs_power_trac_b = node_locs_no_cb_axle_trac_b[:-1]
    node_locs_relay_trac_b = node_locs_no_cb_axle_trac_b[1:]
    node_locs_no_cb_axle_sig_a = np.delete(node_locs_sig_a, np.where(np.isin(node_locs_sig_a, node_locs_axle_sig_a))[0])
    node_locs_power_sig_a = node_locs_no_cb_axle_sig_a[1::2]
    node_locs_relay_sig_a = node_locs_no_cb_axle_sig_a[0::2]
    node_locs_no_cb_axle_sig_b = np.delete(node_locs_sig_b, np.where(np.isin(node_locs_sig_b, node_locs_axle_sig_b))[0])
    node_locs_power_sig_b = node_locs_no_cb_axle_sig_b[0::2]
    node_locs_relay_sig_b = node_locs_no_cb_axle_sig_b[1::2]

    # Calculate nodal parallel admittances and sum of admittances into the node
    # Direction "a" first
    # Traction return rail
    y_sum = np.full(n_nodes, 69).astype(float)  # Array of sum of admittances into the node
    # First node
    mask_first_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[0])
    first_trac_a = node_locs_trac_a[mask_first_trac_a]
    locs_first_trac_a = np.where(np.isin(node_locs_trac_a, first_trac_a))[0]
    y_sum[first_trac_a] = yg_trac_comb[0] + parameters["y_relay"] + ye_trac_a[locs_first_trac_a]
    # Axles
    locs_axle_trac_a = np.where(np.isin(node_locs_trac_a, node_locs_axle_trac_a))[0]
    y_sum[node_locs_axle_trac_a] = parameters["y_axle"] + ye_trac_a[locs_axle_trac_a - 1] + ye_trac_a[locs_axle_trac_a]
    # Cross bonds
    locs_cb_a = np.where(np.isin(node_locs_trac_a, node_locs_cb_a))[0]
    y_sum[node_locs_cb_a] = parameters["y_cb"] + ye_trac_a[locs_cb_a - 1] + ye_trac_a[locs_cb_a]
    # Middle nodes
    indices_other_node_trac_a = node_locs_trac_a[1:-1][~np.logical_or(np.isin(node_locs_trac_a[1:-1], node_locs_axle_trac_a), np.isin(node_locs_trac_a[1:-1], node_locs_cb_a))]
    mask_other_trac_a = np.isin(indices_other_node_trac_a, node_locs_trac_a)
    other_trac_a = indices_other_node_trac_a[mask_other_trac_a]
    locs_other_trac_a = np.where(np.isin(node_locs_trac_a, other_trac_a))[0]
    y_sum[other_trac_a] = yg_trac_comb[1:-1] + parameters["y_power"] + parameters["y_relay"] + ye_trac_a[locs_other_trac_a - 1] + ye_trac_a[locs_other_trac_a]
    # Last node
    mask_last_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[-1])
    last_trac_a = node_locs_trac_a[mask_last_trac_a]
    locs_last_trac_a = np.where(np.isin(node_locs_trac_a, last_trac_a))[0]
    y_sum[last_trac_a] = yg_trac_comb[-1] + parameters["y_power"] + ye_trac_a[locs_last_trac_a - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_relay_sig_a))[0]
    y_sum[node_locs_relay_sig_a] = yg_sig + parameters["y_relay"] + ye_sig_a[locs_relay_sig_a]
    # Power nodes
    locs_power_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]
    y_sum[node_locs_power_sig_a] = yg_sig + parameters["y_power"] + ye_sig_a[locs_power_sig_a - 1]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_a, node_locs_axle_sig_a))[0]
    y_sum[node_locs_axle_sig_a] = parameters["y_axle"] + ye_sig_a[axle_locs - 1] + ye_sig_a[axle_locs]
    # Direction "b" second
    # Traction return rail
    # First node
    mask_first_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[0])
    first_trac_b = node_locs_trac_b[mask_first_trac_b]
    locs_first_trac_b = np.where(np.isin(node_locs_trac_b, first_trac_b))[0]
    y_sum[first_trac_b] = yg_trac_comb[0] + parameters["y_relay"] + ye_trac_b[locs_first_trac_b]
    # Axles
    locs_axle_trac_b = np.where(np.isin(node_locs_trac_b, node_locs_axle_trac_b))[0]
    y_sum[node_locs_axle_trac_b] = parameters["y_axle"] + ye_trac_b[locs_axle_trac_b - 1] + ye_trac_b[locs_axle_trac_b]
    # Cross bonds
    locs_cb_b = np.where(np.isin(node_locs_trac_b, node_locs_cb_b))[0]
    y_sum[node_locs_cb_b] = parameters["y_cb"] + ye_trac_b[locs_cb_b - 1] + ye_trac_b[locs_cb_b]
    # Middle nodes
    indices_other_node_trac_b = node_locs_trac_b[1:-1][~np.logical_or(np.isin(node_locs_trac_b[1:-1], node_locs_axle_trac_b), np.isin(node_locs_trac_b[1:-1], node_locs_cb_b))]
    mask_other_trac_b = np.isin(indices_other_node_trac_b, node_locs_trac_b)
    other_trac_b = indices_other_node_trac_b[mask_other_trac_b]
    locs_other_trac_b = np.where(np.isin(node_locs_trac_b, other_trac_b))[0]
    y_sum[other_trac_b] = yg_trac_comb[1:-1] + parameters["y_power"] + parameters["y_relay"] + ye_trac_b[locs_other_trac_b - 1] + ye_trac_b[locs_other_trac_b]
    # Last node
    mask_last_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[-1])
    last_trac_b = node_locs_trac_b[mask_last_trac_b]
    locs_last_trac_b = np.where(np.isin(node_locs_trac_b, last_trac_b))[0]
    y_sum[last_trac_b] = yg_trac_comb[-1] + parameters["y_power"] + ye_trac_b[locs_last_trac_b - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]
    y_sum[node_locs_relay_sig_b] = yg_sig + parameters["y_relay"] + ye_sig_b[locs_relay_sig_b - 1]
    # Power nodes
    locs_power_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_power_sig_b))[0]
    y_sum[node_locs_power_sig_b] = yg_sig + parameters["y_power"] + ye_sig_b[locs_power_sig_b]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_b, node_locs_axle_sig_b))[0]
    y_sum[node_locs_axle_sig_b] = parameters["y_axle"] + ye_sig_b[axle_locs - 1] + ye_sig_b[axle_locs]

    # Build admittance matrix
    y_matrix = np.zeros((n_nodes, n_nodes))
    # Diagonal values
    y_matrix[range(0, n_nodes), range(0, n_nodes)] = y_sum
    # Series admittances between nodes
    y_matrix[node_locs_trac_a[:-1], node_locs_trac_a[1:]] = -ye_trac_a
    y_matrix[node_locs_trac_a[1:], node_locs_trac_a[:-1]] = -ye_trac_a
    y_matrix[node_locs_sig_a[:-1], node_locs_sig_a[1:]] = -ye_sig_a
    y_matrix[node_locs_sig_a[1:], node_locs_sig_a[:-1]] = -ye_sig_a
    y_matrix[node_locs_trac_b[:-1], node_locs_trac_b[1:]] = -ye_trac_b
    y_matrix[node_locs_trac_b[1:], node_locs_trac_b[:-1]] = -ye_trac_b
    y_matrix[node_locs_sig_b[:-1], node_locs_sig_b[1:]] = -ye_sig_b
    y_matrix[node_locs_sig_b[1:], node_locs_sig_b[:-1]] = -ye_sig_b
    # Relay admittances
    y_matrix[node_locs_relay_trac_a, node_locs_relay_sig_a] = -parameters["y_relay"]
    y_matrix[node_locs_relay_sig_a, node_locs_relay_trac_a] = -parameters["y_relay"]
    y_matrix[node_locs_relay_trac_b, node_locs_relay_sig_b] = -parameters["y_relay"]
    y_matrix[node_locs_relay_sig_b, node_locs_relay_trac_b] = -parameters["y_relay"]
    # Power admittances
    y_matrix[node_locs_power_trac_a, node_locs_power_sig_a] = -parameters["y_power"]
    y_matrix[node_locs_power_sig_a, node_locs_power_trac_a] = -parameters["y_power"]
    y_matrix[node_locs_power_trac_b, node_locs_power_sig_b] = -parameters["y_power"]
    y_matrix[node_locs_power_sig_b, node_locs_power_trac_b] = -parameters["y_power"]
    # Cross bond admittances
    y_matrix[node_locs_cb_a, node_locs_cb_b] = -parameters["y_cb"]
    y_matrix[node_locs_cb_b, node_locs_cb_a] = -parameters["y_cb"]
    # Axle admittances
    y_matrix[node_locs_axle_trac_a, node_locs_axle_sig_a] = -parameters["y_axle"]
    y_matrix[node_locs_axle_sig_a, node_locs_axle_trac_a] = -parameters["y_axle"]
    y_matrix[node_locs_axle_trac_b, node_locs_axle_sig_b] = -parameters["y_axle"]
    y_matrix[node_locs_axle_sig_b, node_locs_axle_trac_b] = -parameters["y_axle"]

    y_matrix[np.isnan(y_matrix)] = 0

    # Currents
    # Set up current matrix
    j_matrix = np.zeros([len(e_parallel), n_nodes])

    # "a" first
    e_par_trac_a = np.repeat(e_parallel[:, np.newaxis], len(trac_sub_blocks_a), axis=1)
    e_par_sig_a = np.repeat(e_parallel[:, np.newaxis], len(sig_sub_blocks_a[~np.isnan(sig_sub_blocks_a)]), axis=1)
    i_sig_a = e_par_sig_a / parameters["z_sig"]
    i_trac_a = e_par_trac_a / parameters["z_trac"]

    # "b" second
    e_par_trac_b = np.repeat(-e_parallel[:, np.newaxis], len(trac_sub_blocks_b), axis=1)
    e_par_sig_b = np.repeat(-e_parallel[:, np.newaxis], len(sig_sub_blocks_b[~np.isnan(sig_sub_blocks_b)]), axis=1)
    i_sig_b = e_par_sig_b / parameters["z_sig"]
    i_trac_b = e_par_trac_b / parameters["z_trac"]

    # "a" first
    # Traction return rail first node
    j_matrix[:, node_locs_trac_a[0]] = -i_trac_a[:, 0]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(node_locs_trac_a, node_locs_cb_a)
    indices = np.where(mask)[0]
    j_matrix[:, node_locs_cb_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Axle nodes
    mask = np.isin(node_locs_trac_a, node_locs_axle_trac_a)
    indices = np.where(mask)[0]
    j_matrix[:, node_locs_axle_trac_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Non-cross bond or axle nodes
    mask = np.isin(node_locs_trac_a, node_locs_cb_a) | np.isin(node_locs_trac_a, node_locs_axle_trac_a)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(node_locs_trac_a, node_locs_cb_a) & ~np.isin(node_locs_trac_a, node_locs_axle_trac_a)
    non_cb_axle_node_locs_centre_a = node_locs_trac_a[mask_del][1:-1]
    j_matrix[:, non_cb_axle_node_locs_centre_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices] - parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, node_locs_trac_a[-1]] = i_trac_a[:, -1] - parameters["i_power"]

    # Signalling rail nodes
    sig_relay_axle = node_locs_sig_a[np.where(~np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_relay_axle, node_locs_axle_sig_a))[0], np.where(np.isin(sig_relay_axle, node_locs_axle_sig_a))[0] - 1)))
    all_blocks = range(0, len(i_sig_a[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_relay_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_relay_axle[np.where(~np.isin(sig_relay_axle, node_locs_axle_sig_a) & ~np.isin(sig_relay_axle, whole_blocks_start))[0]]
    split_blocks_end = np.delete(node_locs_power_sig_a, np.where(np.isin(node_locs_power_sig_a, whole_blocks_end)))
    split_blocks_mid = sig_relay_axle[np.where(np.isin(sig_relay_axle, node_locs_axle_sig_a))[0]]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, whole_blocks_start))[0]]] = -i_sig_a[:, whole_blocks]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, whole_blocks_end))[0]]] = i_sig_a[:, whole_blocks] + parameters["i_power"]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_start))[0]]] = -i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_start))[0]]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_end))[0]]] = i_sig_a[:, split_blocks[np.where(~np.isin(split_blocks,np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1))[0]]] + parameters["i_power"]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_mid))[0]]] = i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1] - i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0]]

    # "b" second
    # Traction return rail first node
    j_matrix[:, node_locs_trac_b[0]] = i_trac_b[:, 0] - parameters["i_power"]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(node_locs_trac_b, node_locs_cb_b)
    indices = np.where(mask)[0]
    j_matrix[:, node_locs_cb_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Axle nodes
    mask = np.isin(node_locs_trac_b, node_locs_axle_trac_b)
    indices = np.where(mask)[0]
    j_matrix[:, node_locs_axle_trac_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Non-cross bond or axle nodes
    mask = np.isin(node_locs_trac_b, node_locs_cb_b) | np.isin(node_locs_trac_b, node_locs_axle_trac_b)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(node_locs_trac_b, node_locs_cb_b) & ~np.isin(node_locs_trac_b, node_locs_axle_trac_b)
    non_cb_axle_node_locs_centre_b = node_locs_trac_b[mask_del][1:-1]
    j_matrix[:, non_cb_axle_node_locs_centre_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1] - parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, node_locs_trac_b[-1]] = -i_trac_b[:, -1]

    # Signalling rail nodes
    sig_power_axle = node_locs_sig_b[np.where(~np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_power_axle, node_locs_axle_sig_b))[0], np.where(np.isin(sig_power_axle, node_locs_axle_sig_b))[0] - 1)))
    all_blocks = range(0, len(i_sig_b[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_power_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_power_axle[np.where(~np.isin(sig_power_axle, node_locs_axle_sig_b) & ~np.isin(sig_power_axle, whole_blocks_start))[0]]
    split_blocks_end = np.delete(node_locs_relay_sig_b, np.where(np.isin(node_locs_relay_sig_b, whole_blocks_end)))
    split_blocks_mid = sig_power_axle[np.where(np.isin(sig_power_axle, node_locs_axle_sig_b))[0]]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_start))[0]]] = i_sig_b[:, whole_blocks] + parameters["i_power"]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_end))[0]]] = -i_sig_b[:, whole_blocks]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_start))[0]]] = i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_start))[0]] + parameters["i_power"]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_end))[0]]] = -i_sig_b[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1))[0]]]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_mid))[0]]] = -i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1] + i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0]]

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    # Calculate relay voltages and currents
    # "a" first
    v_relay_top_node_a = v_matrix[node_locs_relay_sig_a]
    v_relay_bottom_node_a = v_matrix[node_locs_relay_trac_a]
    v_relay_a = v_relay_top_node_a - v_relay_bottom_node_a

    # "b" first
    v_relay_top_node_b = v_matrix[node_locs_relay_sig_b]
    v_relay_bottom_node_b = v_matrix[node_locs_relay_trac_b]
    v_relay_b = v_relay_top_node_b - v_relay_bottom_node_b

    i_relays_a = v_relay_a / parameters["r_relay"]
    i_relays_b = v_relay_b / parameters["r_relay"]

    i_relays_a = i_relays_a.T
    i_relays_b = i_relays_b.T

    return i_relays_a, i_relays_b


def rail_model_two_track_e_parallel_leakage_by_block(section_name, e_parallel, axle_pos_a, axle_pos_b):
    # Create dictionary of network parameters
    parameters = {"z_sig": 0.0289,  # Signalling rail series impedance (ohms/km)
                  "z_trac": 0.0289,  # Traction return rail series impedance (ohms/km)
                  "y_sig": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "v_power": 10,  # Track circuit power supply voltage (volts)
                  "r_power": 7.2,  # Track circuit power supply resistance (ohms)
                  "r_relay": 20,  # Track circuit relay resistance (ohms)
                  "r_cb": 1e-3,  # Cross bond resistance (ohms)
                  "r_axle": 251e-4,  # Axle resistance (ohms)
                  "i_power": 10 / 7.2,  # Track circuit power supply equivalent current source (amps)
                  "y_power": 1 / 7.2,  # Track circuit power supply admittance (siemens)
                  "y_relay": 1 / 20,  # Track circuit relay admittance (siemens)
                  "y_cb": 1 / 1e-3,  # Cross bond admittance (siemens)
                  "y_axle": 1 / 251e-4}  # Axle admittance (siemens)

    y_trac_block = np.load(f"../data/resistivity/{section_name}_leakage_by_block.npy")

    # Calculate the electrical characteristics of the rails
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig"])
    gamma_trac_block = np.sqrt(parameters["z_trac"] * y_trac_block).flatten()
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig"])
    z0_trac_block = np.sqrt(parameters["z_trac"] / y_trac_block).flatten()

    # Load in the lengths and bearings of the track circuit blocks
    # Note: zero degrees is directly northwards, with positive values increasing clockwise
    data = np.load(f"../data/rail_data/{section_name}/{section_name}_distances_bearings.npz")
    blocks = data["distances"]
    blocks_sum = np.cumsum(blocks)  # Cumulative sum of block lengths

    # Add cross bonds and axles which split the blocks into sub blocks
    # Note: "a" and "b" are used to identify the opposite directions of travel in this network (two-track)
    pos_cb = np.arange(0.4, np.sum(blocks), 0.4)  # Position of the cross bonds
    trac_sub_block_sum_a = np.sort(np.insert(np.concatenate((blocks_sum, pos_cb, axle_pos_a)), 0, 0))  # Traction return rail connects to axles and cross bonds
    sig_sub_block_sum_a = np.sort(np.insert(np.concatenate((blocks_sum, blocks_sum[:-1], axle_pos_a)), 0, 0))  # Signalling rail connects to axles, but need to add points on either side or IRJ
    trac_sub_block_sum_b = np.sort(np.insert(np.concatenate((blocks_sum, pos_cb, axle_pos_b)), 0, 0))
    sig_sub_block_sum_b = np.sort(np.insert(np.concatenate((blocks_sum, blocks_sum[:-1], axle_pos_b)), 0, 0))
    trac_sub_blocks_a = np.diff(trac_sub_block_sum_a)
    sig_sub_blocks_a = np.diff(sig_sub_block_sum_a)
    sig_sub_blocks_a[sig_sub_blocks_a == 0] = np.nan  # Sets a nan value to indicate the IRJ gap
    trac_sub_blocks_b = np.diff(trac_sub_block_sum_b)
    sig_sub_blocks_b = np.diff(sig_sub_block_sum_b)
    sig_sub_blocks_b[sig_sub_blocks_b == 0] = np.nan

    # Announce if cross bonds overlap with block boundaries
    if 0 in trac_sub_blocks_a or 0 in trac_sub_blocks_b:
        print("cb block overlap")
    else:
        pass

    # Set sub block value for z0 and gamma
    j = 0
    value = 0
    z0_trac_sub_block_a = []
    gamma_trac_sub_block_a = []
    for i in range(0, len(blocks_sum)):
        while value < blocks_sum[i]:
            value += trac_sub_blocks_a[j]
            z0_trac_sub_block_a.append(z0_trac_block[i])
            gamma_trac_sub_block_a.append(gamma_trac_block[i])
            j += 1
    gamma_trac_sub_block_a = np.array(gamma_trac_sub_block_a)
    z0_trac_sub_block_a = np.array(z0_trac_sub_block_a)

    j = 0
    value = 0
    z0_trac_sub_block_b = []
    gamma_trac_sub_block_b = []
    for i in range(0, len(blocks_sum)):
        while value < blocks_sum[i]:
            value += trac_sub_blocks_b[j]
            z0_trac_sub_block_b.append(z0_trac_block[i])
            gamma_trac_sub_block_b.append(gamma_trac_block[i])
            j += 1
    gamma_trac_sub_block_b = np.array(gamma_trac_sub_block_b)
    z0_trac_sub_block_b = np.array(z0_trac_sub_block_b)

    a = gamma_trac_sub_block_a * trac_sub_blocks_a

    # Set up equivalent-pi parameters
    ye_trac_a = 1 / (z0_trac_sub_block_a * np.sinh(gamma_trac_sub_block_a * trac_sub_blocks_a))  # Series admittance for traction return rail
    ye_trac_b = 1 / (z0_trac_sub_block_b * np.sinh(gamma_trac_sub_block_b * trac_sub_blocks_b))
    yg_trac = (np.cosh(gamma_trac_block * blocks) - 1) * (1 / (z0_trac_block * np.sinh(gamma_trac_block * blocks)))  # Parallel admittance for traction return rail

    ye_sig_a = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))  # Series admittance for signalling rail
    ye_sig_b = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))
    yg_sig = (np.cosh(gamma_sig * blocks) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * blocks)))  # Parallel admittance for signalling rail
    yg_trac_comb = np.empty(len(yg_trac) + 1)
    yg_trac_comb[0] = yg_trac[0]
    yg_trac_comb[1:-1] = yg_trac[:-1] + yg_trac[1:]
    yg_trac_comb[-1] = yg_trac[-1]

    # Calculate numbers of nodes ready to use in indexing
    n_nodes_a = len(trac_sub_block_sum_a) + len(sig_sub_block_sum_a)  # Number of nodes in direction of travel a
    n_nodes_b = len(trac_sub_block_sum_b) + len(sig_sub_block_sum_b)
    n_nodes = n_nodes_a + n_nodes_b  # Number of nodes in the whole network
    n_nodes_trac_a = len(trac_sub_block_sum_a)  # Number of nodes in the traction return rail
    n_nodes_trac_b = len(trac_sub_block_sum_b)
    n_nodes_sig_a = len(sig_sub_block_sum_a)  # Number of nodes in the signalling rail
    n_nodes_sig_b = len(sig_sub_block_sum_b)

    # Index of rail nodes in the rails
    node_locs_trac_a = np.arange(0, n_nodes_trac_a, 1).astype(int)
    node_locs_sig_a = np.arange(n_nodes_trac_a, n_nodes_a, 1).astype(int)
    node_locs_trac_b = np.arange(n_nodes_a, n_nodes_a + n_nodes_trac_b, 1).astype(int)
    node_locs_sig_b = np.arange(n_nodes_a + n_nodes_trac_b, n_nodes).astype(int)
    # Index of cross bond nodes
    node_locs_cb_a = node_locs_trac_a[np.where(np.isin(trac_sub_block_sum_a, pos_cb))[0]]
    node_locs_cb_b = node_locs_trac_b[np.where(np.isin(trac_sub_block_sum_b, pos_cb))[0]]
    # Index of axle nodes
    node_locs_axle_trac_a = node_locs_trac_a[np.where(np.isin(trac_sub_block_sum_a, axle_pos_a))[0]]
    node_locs_axle_sig_a = node_locs_sig_a[np.where(np.isin(sig_sub_block_sum_a, axle_pos_a))[0]]
    node_locs_axle_trac_b = node_locs_trac_b[np.where(np.isin(trac_sub_block_sum_b, axle_pos_b))[0]]
    node_locs_axle_sig_b = node_locs_sig_b[np.where(np.isin(sig_sub_block_sum_b, axle_pos_b))[0]]

    # Index of traction return rail power supply and relay nodes
    # Note: "a" begins with a relay and ends with a power supply, "b" begins with a power supply and ends with a relay
    node_locs_no_cb_axle_trac_a = np.delete(node_locs_trac_a, np.where(np.isin(node_locs_trac_a, np.concatenate((node_locs_cb_a, node_locs_axle_trac_a))))[0])
    node_locs_power_trac_a = node_locs_no_cb_axle_trac_a[1:]
    node_locs_relay_trac_a = node_locs_no_cb_axle_trac_a[:-1]
    node_locs_no_cb_axle_trac_b = np.delete(node_locs_trac_b, np.where(np.isin(node_locs_trac_b, np.concatenate((node_locs_cb_b, node_locs_axle_trac_b))))[0])
    node_locs_power_trac_b = node_locs_no_cb_axle_trac_b[:-1]
    node_locs_relay_trac_b = node_locs_no_cb_axle_trac_b[1:]
    node_locs_no_cb_axle_sig_a = np.delete(node_locs_sig_a, np.where(np.isin(node_locs_sig_a, node_locs_axle_sig_a))[0])
    node_locs_power_sig_a = node_locs_no_cb_axle_sig_a[1::2]
    node_locs_relay_sig_a = node_locs_no_cb_axle_sig_a[0::2]
    node_locs_no_cb_axle_sig_b = np.delete(node_locs_sig_b, np.where(np.isin(node_locs_sig_b, node_locs_axle_sig_b))[0])
    node_locs_power_sig_b = node_locs_no_cb_axle_sig_b[0::2]
    node_locs_relay_sig_b = node_locs_no_cb_axle_sig_b[1::2]

    # Calculate nodal parallel admittances and sum of admittances into the node
    # Direction "a" first
    # Traction return rail
    y_sum = np.full(n_nodes, 69).astype(float)  # Array of sum of admittances into the node
    # First node
    mask_first_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[0])
    first_trac_a = node_locs_trac_a[mask_first_trac_a]
    locs_first_trac_a = np.where(np.isin(node_locs_trac_a, first_trac_a))[0]
    y_sum[first_trac_a] = yg_trac_comb[0] + parameters["y_relay"] + ye_trac_a[locs_first_trac_a]
    # Axles
    locs_axle_trac_a = np.where(np.isin(node_locs_trac_a, node_locs_axle_trac_a))[0]
    y_sum[node_locs_axle_trac_a] = parameters["y_axle"] + ye_trac_a[locs_axle_trac_a - 1] + ye_trac_a[locs_axle_trac_a]
    # Cross bonds
    locs_cb_a = np.where(np.isin(node_locs_trac_a, node_locs_cb_a))[0]
    y_sum[node_locs_cb_a] = parameters["y_cb"] + ye_trac_a[locs_cb_a - 1] + ye_trac_a[locs_cb_a]
    # Middle nodes
    indices_other_node_trac_a = node_locs_trac_a[1:-1][~np.logical_or(np.isin(node_locs_trac_a[1:-1], node_locs_axle_trac_a), np.isin(node_locs_trac_a[1:-1], node_locs_cb_a))]
    mask_other_trac_a = np.isin(indices_other_node_trac_a, node_locs_trac_a)
    other_trac_a = indices_other_node_trac_a[mask_other_trac_a]
    locs_other_trac_a = np.where(np.isin(node_locs_trac_a, other_trac_a))[0]
    y_sum[other_trac_a] = yg_trac_comb[1:-1] + parameters["y_power"] + parameters["y_relay"] + ye_trac_a[locs_other_trac_a - 1] + ye_trac_a[locs_other_trac_a]
    # Last node
    mask_last_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[-1])
    last_trac_a = node_locs_trac_a[mask_last_trac_a]
    locs_last_trac_a = np.where(np.isin(node_locs_trac_a, last_trac_a))[0]
    y_sum[last_trac_a] = yg_trac_comb[-1] + parameters["y_power"] + ye_trac_a[locs_last_trac_a - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_relay_sig_a))[0]
    y_sum[node_locs_relay_sig_a] = yg_sig + parameters["y_relay"] + ye_sig_a[locs_relay_sig_a]
    # Power nodes
    locs_power_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]
    y_sum[node_locs_power_sig_a] = yg_sig + parameters["y_power"] + ye_sig_a[locs_power_sig_a - 1]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_a, node_locs_axle_sig_a))[0]
    y_sum[node_locs_axle_sig_a] = parameters["y_axle"] + ye_sig_a[axle_locs - 1] + ye_sig_a[axle_locs]
    # Direction "b" second
    # Traction return rail
    # First node
    mask_first_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[0])
    first_trac_b = node_locs_trac_b[mask_first_trac_b]
    locs_first_trac_b = np.where(np.isin(node_locs_trac_b, first_trac_b))[0]
    y_sum[first_trac_b] = yg_trac_comb[0] + parameters["y_relay"] + ye_trac_b[locs_first_trac_b]
    # Axles
    locs_axle_trac_b = np.where(np.isin(node_locs_trac_b, node_locs_axle_trac_b))[0]
    y_sum[node_locs_axle_trac_b] = parameters["y_axle"] + ye_trac_b[locs_axle_trac_b - 1] + ye_trac_b[locs_axle_trac_b]
    # Cross bonds
    locs_cb_b = np.where(np.isin(node_locs_trac_b, node_locs_cb_b))[0]
    y_sum[node_locs_cb_b] = parameters["y_cb"] + ye_trac_b[locs_cb_b - 1] + ye_trac_b[locs_cb_b]
    # Middle nodes
    indices_other_node_trac_b = node_locs_trac_b[1:-1][~np.logical_or(np.isin(node_locs_trac_b[1:-1], node_locs_axle_trac_b), np.isin(node_locs_trac_b[1:-1], node_locs_cb_b))]
    mask_other_trac_b = np.isin(indices_other_node_trac_b, node_locs_trac_b)
    other_trac_b = indices_other_node_trac_b[mask_other_trac_b]
    locs_other_trac_b = np.where(np.isin(node_locs_trac_b, other_trac_b))[0]
    y_sum[other_trac_b] = yg_trac_comb[1:-1] + parameters["y_power"] + parameters["y_relay"] + ye_trac_b[locs_other_trac_b - 1] + ye_trac_b[locs_other_trac_b]
    # Last node
    mask_last_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[-1])
    last_trac_b = node_locs_trac_b[mask_last_trac_b]
    locs_last_trac_b = np.where(np.isin(node_locs_trac_b, last_trac_b))[0]
    y_sum[last_trac_b] = yg_trac_comb[-1] + parameters["y_power"] + ye_trac_b[locs_last_trac_b - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]
    y_sum[node_locs_relay_sig_b] = yg_sig + parameters["y_relay"] + ye_sig_b[locs_relay_sig_b - 1]
    # Power nodes
    locs_power_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_power_sig_b))[0]
    y_sum[node_locs_power_sig_b] = yg_sig + parameters["y_power"] + ye_sig_b[locs_power_sig_b]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_b, node_locs_axle_sig_b))[0]
    y_sum[node_locs_axle_sig_b] = parameters["y_axle"] + ye_sig_b[axle_locs - 1] + ye_sig_b[axle_locs]

    # Build admittance matrix
    y_matrix = np.zeros((n_nodes, n_nodes))
    # Diagonal values
    y_matrix[range(0, n_nodes), range(0, n_nodes)] = y_sum
    # Series admittances between nodes
    y_matrix[node_locs_trac_a[:-1], node_locs_trac_a[1:]] = -ye_trac_a
    y_matrix[node_locs_trac_a[1:], node_locs_trac_a[:-1]] = -ye_trac_a
    y_matrix[node_locs_sig_a[:-1], node_locs_sig_a[1:]] = -ye_sig_a
    y_matrix[node_locs_sig_a[1:], node_locs_sig_a[:-1]] = -ye_sig_a
    y_matrix[node_locs_trac_b[:-1], node_locs_trac_b[1:]] = -ye_trac_b
    y_matrix[node_locs_trac_b[1:], node_locs_trac_b[:-1]] = -ye_trac_b
    y_matrix[node_locs_sig_b[:-1], node_locs_sig_b[1:]] = -ye_sig_b
    y_matrix[node_locs_sig_b[1:], node_locs_sig_b[:-1]] = -ye_sig_b
    # Relay admittances
    y_matrix[node_locs_relay_trac_a, node_locs_relay_sig_a] = -parameters["y_relay"]
    y_matrix[node_locs_relay_sig_a, node_locs_relay_trac_a] = -parameters["y_relay"]
    y_matrix[node_locs_relay_trac_b, node_locs_relay_sig_b] = -parameters["y_relay"]
    y_matrix[node_locs_relay_sig_b, node_locs_relay_trac_b] = -parameters["y_relay"]
    # Power admittances
    y_matrix[node_locs_power_trac_a, node_locs_power_sig_a] = -parameters["y_power"]
    y_matrix[node_locs_power_sig_a, node_locs_power_trac_a] = -parameters["y_power"]
    y_matrix[node_locs_power_trac_b, node_locs_power_sig_b] = -parameters["y_power"]
    y_matrix[node_locs_power_sig_b, node_locs_power_trac_b] = -parameters["y_power"]
    # Cross bond admittances
    y_matrix[node_locs_cb_a, node_locs_cb_b] = -parameters["y_cb"]
    y_matrix[node_locs_cb_b, node_locs_cb_a] = -parameters["y_cb"]
    # Axle admittances
    y_matrix[node_locs_axle_trac_a, node_locs_axle_sig_a] = -parameters["y_axle"]
    y_matrix[node_locs_axle_sig_a, node_locs_axle_trac_a] = -parameters["y_axle"]
    y_matrix[node_locs_axle_trac_b, node_locs_axle_sig_b] = -parameters["y_axle"]
    y_matrix[node_locs_axle_sig_b, node_locs_axle_trac_b] = -parameters["y_axle"]

    y_matrix[np.isnan(y_matrix)] = 0

    # Currents
    # Set up current matrix
    j_matrix = np.zeros([len(e_parallel), n_nodes])

    # "a" first
    e_par_trac_a = np.repeat(e_parallel[:, np.newaxis], len(trac_sub_blocks_a), axis=1)
    e_par_sig_a = np.repeat(e_parallel[:, np.newaxis], len(sig_sub_blocks_a[~np.isnan(sig_sub_blocks_a)]), axis=1)
    i_sig_a = e_par_sig_a / parameters["z_sig"]
    i_trac_a = e_par_trac_a / parameters["z_trac"]

    # "b" second
    e_par_trac_b = np.repeat(-e_parallel[:, np.newaxis], len(trac_sub_blocks_b), axis=1)
    e_par_sig_b = np.repeat(-e_parallel[:, np.newaxis], len(sig_sub_blocks_b[~np.isnan(sig_sub_blocks_b)]), axis=1)
    i_sig_b = e_par_sig_b / parameters["z_sig"]
    i_trac_b = e_par_trac_b / parameters["z_trac"]

    # "a" first
    # Traction return rail first node
    j_matrix[:, node_locs_trac_a[0]] = -i_trac_a[:, 0]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(node_locs_trac_a, node_locs_cb_a)
    indices = np.where(mask)[0]
    j_matrix[:, node_locs_cb_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Axle nodes
    mask = np.isin(node_locs_trac_a, node_locs_axle_trac_a)
    indices = np.where(mask)[0]
    j_matrix[:, node_locs_axle_trac_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Non-cross bond or axle nodes
    mask = np.isin(node_locs_trac_a, node_locs_cb_a) | np.isin(node_locs_trac_a, node_locs_axle_trac_a)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(node_locs_trac_a, node_locs_cb_a) & ~np.isin(node_locs_trac_a, node_locs_axle_trac_a)
    non_cb_axle_node_locs_centre_a = node_locs_trac_a[mask_del][1:-1]
    j_matrix[:, non_cb_axle_node_locs_centre_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices] - parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, node_locs_trac_a[-1]] = i_trac_a[:, -1] - parameters["i_power"]

    # Signalling rail nodes
    sig_relay_axle = node_locs_sig_a[np.where(~np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_relay_axle, node_locs_axle_sig_a))[0], np.where(np.isin(sig_relay_axle, node_locs_axle_sig_a))[0] - 1)))
    all_blocks = range(0, len(i_sig_a[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_relay_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_relay_axle[np.where(~np.isin(sig_relay_axle, node_locs_axle_sig_a) & ~np.isin(sig_relay_axle, whole_blocks_start))[0]]
    split_blocks_end = np.delete(node_locs_power_sig_a, np.where(np.isin(node_locs_power_sig_a, whole_blocks_end)))
    split_blocks_mid = sig_relay_axle[np.where(np.isin(sig_relay_axle, node_locs_axle_sig_a))[0]]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, whole_blocks_start))[0]]] = -i_sig_a[:, whole_blocks]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, whole_blocks_end))[0]]] = i_sig_a[:, whole_blocks] + parameters["i_power"]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_start))[0]]] = -i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_start))[0]]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_end))[0]]] = i_sig_a[:, split_blocks[np.where(~np.isin(split_blocks,np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1))[0]]] + parameters["i_power"]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_mid))[0]]] = i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1] - i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0]]

    # "b" second
    # Traction return rail first node
    j_matrix[:, node_locs_trac_b[0]] = i_trac_b[:, 0] - parameters["i_power"]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(node_locs_trac_b, node_locs_cb_b)
    indices = np.where(mask)[0]
    j_matrix[:, node_locs_cb_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Axle nodes
    mask = np.isin(node_locs_trac_b, node_locs_axle_trac_b)
    indices = np.where(mask)[0]
    j_matrix[:, node_locs_axle_trac_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Non-cross bond or axle nodes
    mask = np.isin(node_locs_trac_b, node_locs_cb_b) | np.isin(node_locs_trac_b, node_locs_axle_trac_b)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(node_locs_trac_b, node_locs_cb_b) & ~np.isin(node_locs_trac_b, node_locs_axle_trac_b)
    non_cb_axle_node_locs_centre_b = node_locs_trac_b[mask_del][1:-1]
    j_matrix[:, non_cb_axle_node_locs_centre_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1] - parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, node_locs_trac_b[-1]] = -i_trac_b[:, -1]

    # Signalling rail nodes
    sig_power_axle = node_locs_sig_b[np.where(~np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_power_axle, node_locs_axle_sig_b))[0], np.where(np.isin(sig_power_axle, node_locs_axle_sig_b))[0] - 1)))
    all_blocks = range(0, len(i_sig_b[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_power_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_power_axle[np.where(~np.isin(sig_power_axle, node_locs_axle_sig_b) & ~np.isin(sig_power_axle, whole_blocks_start))[0]]
    split_blocks_end = np.delete(node_locs_relay_sig_b, np.where(np.isin(node_locs_relay_sig_b, whole_blocks_end)))
    split_blocks_mid = sig_power_axle[np.where(np.isin(sig_power_axle, node_locs_axle_sig_b))[0]]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_start))[0]]] = i_sig_b[:, whole_blocks] + parameters["i_power"]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_end))[0]]] = -i_sig_b[:, whole_blocks]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_start))[0]]] = i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_start))[0]] + parameters["i_power"]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_end))[0]]] = -i_sig_b[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1))[0]]]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_mid))[0]]] = -i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1] + i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0]]

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    # Calculate relay voltages and currents
    # "a" first
    v_relay_top_node_a = v_matrix[node_locs_relay_sig_a]
    v_relay_bottom_node_a = v_matrix[node_locs_relay_trac_a]
    v_relay_a = v_relay_top_node_a - v_relay_bottom_node_a

    # "b" first
    v_relay_top_node_b = v_matrix[node_locs_relay_sig_b]
    v_relay_bottom_node_b = v_matrix[node_locs_relay_trac_b]
    v_relay_b = v_relay_top_node_b - v_relay_bottom_node_b

    i_relays_a = v_relay_a / parameters["r_relay"]
    i_relays_b = v_relay_b / parameters["r_relay"]

    i_relays_a = i_relays_a.T
    i_relays_b = i_relays_b.T

    return i_relays_a, i_relays_b


def save_relay_currents_rs(route_name):
    e_par = np.linspace(-100, 100, 2001)
    print(f"{route_name}_first_start")
    ia0, ib0 = rail_model_two_track_e_parallel(section_name=route_name, e_parallel=e_par, axle_pos_a=[], axle_pos_b=[])
    print(f"{route_name}_first_halfway")
    ia, ib = rail_model_two_track_e_parallel_leakage_by_block(section_name=route_name, e_parallel=e_par, axle_pos_a=[], axle_pos_b=[])

    np.save(f"clear_ia_unileak_{route_name}", ia0)
    np.save(f"clear_ia_varleak_{route_name}", ia)


def plot_relay_currents_rs(route_name, e_par):
    ia0, ib0 = rail_model_two_track_e_parallel(section_name=route_name, e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
    ia, ib = rail_model_two_track_e_parallel_leakage_by_block(section_name=route_name, e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[:, :])
    ax0.plot(ia0[0], '.', label="Uniform Leakage", color="royalblue")
    ax0.plot(ia[0], 'x', label="Leakage by block", color="orangered")
    ax0.set_title(f"E = {e_par[0]} " + r"$\mathrm{V \cdot km^{-1}}$")
    ax0.legend(loc="upper center")
    ax0.set_xlabel("Block Index")
    ax0.set_ylabel("Current Through Relay (A)")
    if route_name == "glasgow_edinburgh_falkirk":
        fig.suptitle("Glasgow to Edinburgh via Falkirk High")
    elif route_name == "east_coast_main_line":
        fig.suptitle("East Coast Main Line")
    elif route_name == "west_coast_main_line":
        fig.suptitle("West Coast Main Line")
    else:
        fig.suptitle(f"{route_name}")

    plt.show()
    #plt.savefig(f"leakage_relay_currents_E{e_par}_{route_name}.pdf")


def plot_relay_currents_dif_rs(route_name, e_par):
    ia0, ib0 = rail_model_two_track_e_parallel(section_name=route_name, e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
    ia, ib = rail_model_two_track_e_parallel_leakage_by_block(section_name=route_name, e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
    leakage = np.load(f"../data/resistivity/{route_name}_leakage_by_block.npy")

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1)
    gs.update(hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1], sharex=ax0)
    ax0.plot(leakage, '.', color="royalblue")
    ax1.plot(ia[0] - ia0[0], '.', color="cornflowerblue")
    ax0.axhline(1.6, linestyle='--', color="royalblue", alpha=0.75)
    ax1.axhline(0, linestyle='--', color="cornflowerblue", alpha=0.75)
    ax0.set_title(f"E = {e_par[0]} " + r"$\mathrm{V \cdot km^{-1}}$")
    ax0.label_outer()
    ax1.set_xlabel("Block Index")
    ax0.set_ylabel(r"Leakage ($\mathrm{S \cdot km^{-1}}$)")
    ax1.set_ylabel("Current Difference (A)")
    if route_name == "glasgow_edinburgh_falkirk":
        fig.suptitle("Glasgow to Edinburgh via Falkirk High")
    elif route_name == "east_coast_main_line":
        fig.suptitle("East Coast Main Line")
        #ax1.set_ylim(-0.01, 0.03)
        ax1.set_xlim(-1, 914)
        boundaries = np.array([24.5, 66.5, 107.5, 128.5, 323.5, 427.5, 593.5, 658.5, 707.5, 739.5])
        for b in boundaries:
            ax0.axvline(b, color="gray", alpha=0.25)
            ax1.axvline(b, color="gray", alpha=0.25)
    elif route_name == "west_coast_main_line":
        fig.suptitle("West Coast Main Line")
        ax1.set_xlim(-1, 936)
        #boundaries = np.array([71.5, 124.5, 170.5, 668.5, 708.5, 744])
        #for b in boundaries:
        #    ax0.axvline(b, color="gray", alpha=0.25)
        #    ax1.axvline(b, color="gray", alpha=0.25)
    else:
        fig.suptitle(f"{route_name}")

    fig.align_ylabels([ax0, ax1])
    #plt.show()
    plt.savefig(f"leakage_relay_currents_dif_E{e_par}_{route_name}_rs.pdf")


def find_thresholds_rs(route_name):
    leakage = np.load(f"../data/resistivity/{route_name}_leakage_by_block.npy")
    e_par = np.linspace(-100, 100, 10001)
    ia_og, ib_og = rail_model_two_track_e_parallel(section_name=route_name, e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
    ia, ib = rail_model_two_track_e_parallel_leakage_by_block(section_name=route_name, e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
    e_thresholds = []
    e_thresholds_og = []
    for i in range(0, len(ia[0, :])):
        currents = ia[:, i]
        current_threshold_dif = np.abs(currents - 0.055)
        e_threshold = e_par[np.where(current_threshold_dif == np.min(current_threshold_dif))]
        e_thresholds.append(e_threshold)
        currents_og = ia_og[:, i]
        current_threshold_dif_og = np.abs(currents_og - 0.055)
        e_threshold_og = e_par[np.where(current_threshold_dif_og == np.min(current_threshold_dif_og))]
        e_thresholds_og.append(e_threshold_og)
    e_thresholds = np.array(e_thresholds)
    e_thresholds_og = np.array(e_thresholds_og)
    e_thresholds = np.where((e_thresholds == 100) | (e_thresholds == -100), np.nan, e_thresholds)
    e_thresholds_og = np.where((e_thresholds_og == 100) | (e_thresholds_og == -100), np.nan, e_thresholds_og)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 4)
    gs.update(hspace=0)
    gs.update(wspace=0.28)
    ax0 = fig.add_subplot(gs[:1, :1])
    ax1 = fig.add_subplot(gs[1:, :1])
    ax2 = fig.add_subplot(gs[:1, 1:])
    ax3 = fig.add_subplot(gs[1:, 1:])
    ax0.set_xticks([])
    ax2.set_xticks([])

    ax0.axhline(1.6, linestyle='--', color="royalblue", alpha=0.75)
    ax0.set_ylabel(r"Leakage ($\mathrm{S \cdot km^{-1}}$)")
    ax1.set_xlabel("Block Index")
    ax1.set_ylabel("\u0394 Threshold ($\mathrm{V \cdot km^{-1}}$)")
    ax1.axhline(0, linestyle='--', color="limegreen", alpha=0.75)
    ax2.axhline(1.6, linestyle='--', color="royalblue", alpha=0.75)
    ax3.set_xlabel("Block Index")
    ax3.axhline(0, linestyle='--', color="limegreen", alpha=0.75)

    if route_name == "glasgow_edinburgh_falkirk":
        fig.suptitle("Glasgow to Edinburgh via Falkirk High")
        ax0.plot(range(0, 27), leakage[:27], '.', color="royalblue")
        ax1.plot(range(0, 27), e_thresholds[:27] - e_thresholds_og[:27], '.', color="limegreen")
        ax2.plot(range(27, 119), leakage[27:], '.', color="royalblue")
        ax3.plot(range(27, 119), e_thresholds[27:] - e_thresholds_og[27:], '.', color="limegreen")
        ax0.set_ylim(-0.2, 6)
        ax2.set_ylim(-0.2, 6)
        ax1.set_ylim(-40, 110)
        ax3.set_ylim(-1.5, 3.5)
        ax0.set_xlim(-1, 27)
        ax1.set_xlim(-1, 27)
        ax2.set_xlim(27, 119)
        ax3.set_xlim(27, 119)
        ax0.set_xticks([])
        ax2.set_xticks([])
        ax1.set_xticks([0, 10, 20, 27])
        ax3.set_xticks([27, 40, 60, 80, 100])
        boundaries = np.array([82.5, 94.5])
        for b in boundaries:
            ax2.axvline(b, color="gray", alpha=0.25)
            ax3.axvline(b, color="gray", alpha=0.25)

    elif route_name == "east_coast_main_line":
        fig.suptitle("East Coast Main Line")
        ax0.plot(range(0, 34), leakage[:34], '.', color="royalblue")
        ax1.plot(range(0, 34), e_thresholds[:34] - e_thresholds_og[:34], '.', color="limegreen")
        ax2.plot(range(34, 914), leakage[34:], '.', color="royalblue")
        ax3.plot(range(34, 914), e_thresholds[34:] - e_thresholds_og[34:], '.', color="limegreen")

        ax0.set_ylim(-0.2, 6.2)
        ax2.set_ylim(-0.2, 6.2)
        ax1.set_ylim(-160, 30)
        ax3.set_ylim(-0.5, 2.5)
        ax0.set_xlim(-1, 34)
        ax1.set_xlim(-1, 34)
        ax2.set_xlim(34, 914)
        ax3.set_xlim(34, 914)
        ax0.set_xticks([])
        ax2.set_xticks([])
        ax1.set_xticks([0, 10, 20, 33])
        ax3.set_xticks([33, 100, 200, 300, 400, 500, 600, 700, 800, 900])
        ax0.set_yticks([0, 2, 4, 6])
        ax2.set_yticks([0, 2, 4, 6])
        ax1.set_yticks([-150, -100, -50, 0])
        ax3.set_yticks([-0.5, 0, 0.5, 1, 1.5, 2])

        boundaries = np.array([66.5, 107.5, 128.5, 323.5, 427.5, 593.5, 658.5, 707.5, 739.5])
        for b in boundaries:
            ax2.axvline(b, color="gray", alpha=0.25)
            ax3.axvline(b, color="gray", alpha=0.25)

    elif route_name == "west_coast_main_line":
        fig.suptitle("West Coast Main Line")
        ax0.plot(range(0, 34), leakage[:34], '.', color="royalblue")
        ax1.plot(range(0, 34), e_thresholds[:34] - e_thresholds_og[:34], '.', color="limegreen")
        ax2.plot(range(34, 936), leakage[34:], '.', color="royalblue")
        ax3.plot(range(34, 936), e_thresholds[34:] - e_thresholds_og[34:], '.', color="limegreen")
        ax0.set_ylim(-0.2, 8.2)
        ax2.set_ylim(-0.2, 8.2)
        ax0.set_xlim(-1, 34)
        ax1.set_xlim(-1, 34)
        ax2.set_xlim(33, 764)
        ax3.set_xlim(33, 764)
        ax3.set_ylim(-0.6, 0.3)
        ax1.set_xticks([0, 10, 20, 30])
        ax3.set_xticks([30, 100, 200, 300, 400, 500, 600, 700, 800, 900])
        ax2.set_yticks([0, 2, 4, 6, 8])
        ax3.set_yticks([-0.6, -0.4, -0.2, 0, 0.2])

    else:
        fig.suptitle(f"{route_name}")

    fig.align_ylabels([ax0, ax1])
    plt.savefig(f"leakage_threshold_dif_{route_name}.pdf")
    #plt.show()


def crossover_rs(route_name):
    e_par = np.array([-5, -2.5, 0, 2.5, 5])

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 1)
    gs.update(hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])

    ax0.plot([], label="Uniform Leakage", color="royalblue")
    ax1.plot([], label="Leakage by block", color="orangered")

    ia0, ib0 = rail_model_two_track_e_parallel(section_name=route_name, e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
    ia, ib = rail_model_two_track_e_parallel_leakage_by_block(section_name=route_name, e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
    for i in range(0, len(e_par)):
        ax0.plot(ia0[i, :], color="royalblue")
        ax1.plot(ia[i, :], color="orangered")

        ax0.legend()
        ax0.set_ylabel("Current Through Relay (A)")
        ax1.legend()
        ax1.set_xlabel("Block Index")
        ax1.set_ylabel("Current Through Relay (A)")

    if route_name == "west_coast_main_line":
        fig.suptitle(f"West Coast Main Line")
        ax0.axvline(13.43, color="gray", alpha=0.25)
        ax1.axvline(10.64, color="gray", alpha=0.25)
        ax0.set_xlim(0, 935)
        ax1.set_xlim(0, 935)
        ax0.set_xticks([])
        ax1.set_xticks([0, 5, 10, 15, 20])
        ax0.set_xlim(0, 20)
        ax1.set_xlim(0, 20)

    elif route_name == "east_coast_main_line":
        fig.suptitle(f"East Coast Main Line")
        ax0.axvline(9.655, color="gray", alpha=0.25)
        ax1.axvline(10.98, color="gray", alpha=0.25)
        ax0.set_xlim(0, 913)
        ax1.set_xlim(0, 913)
        ax0.set_xticks([])
        ax1.set_xticks([0, 5, 10, 15, 20])
        ax0.set_xlim(0, 20)
        ax1.set_xlim(0, 20)

    elif route_name == "glasgow_edinburgh_falkirk":
        fig.suptitle(f"Glasgow to Edinburgh via Falkirk")
        ax0.axvline(9.156, color="gray", alpha=0.25)
        ax1.axvline(5.806, color="gray", alpha=0.25)
        ax0.set_xlim(0, 118)
        ax1.set_xlim(0, 118)
        ax0.set_xticks([])
        ax1.set_xticks([0, 5, 10, 15, 20])
        ax0.set_xlim(0, 20)
        ax1.set_xlim(0, 20)

    else:
        fig.suptitle(f"{route_name}")

    plt.savefig(f"crossover_{route_name}.pdf")
    #plt.show()


def save_relay_currents_ws(route_name):
    e_par = np.linspace(-100, 100, 2001)
    axle_data = np.load(f"../data/axle_positions/block_centre/axle_positions_two_track_back_axle_at_centre_{route_name}.npz", allow_pickle=True)
    axle_pos_a = axle_data["axle_pos_a_all"]
    axles_a_first_half = [val for arr in axle_pos_a[0::2] for val in arr]
    axles_a_second_half = [val for arr in axle_pos_a[1::2] for val in arr]
    print(f"{route_name}_first_start")
    ia0_first, ib0_first = rail_model_two_track_e_parallel(section_name=route_name, e_parallel=e_par, axle_pos_a=axles_a_first_half, axle_pos_b=[])
    print(f"{route_name}_first_halfway")
    ia_first, ib_first = rail_model_two_track_e_parallel_leakage_by_block(section_name=route_name, e_parallel=e_par, axle_pos_a=axles_a_first_half, axle_pos_b=[])
    print(f"{route_name}_second_start")
    ia0_second, ib0_second = rail_model_two_track_e_parallel(section_name=route_name, e_parallel=e_par, axle_pos_a=axles_a_second_half, axle_pos_b=[])
    print(f"{route_name}_second_halfway")
    ia_second, ib_second = rail_model_two_track_e_parallel_leakage_by_block(section_name=route_name, e_parallel=e_par, axle_pos_a=axles_a_second_half, axle_pos_b=[])

    ia0 = ia0_first
    ia = ia_first
    ia0[:, 1::2] = ia0_second[:, 1::2]
    ia[:, 1::2] = ia_second[:, 1::2]

    np.save(f"occupied_ia_unileak_{route_name}", ia0)
    np.save(f"occupied_ia_varleak_{route_name}", ia)


def plot_relay_currents_ws(route_name):
    ia0s = np.load(f"../data/resistivity/currents/occupied_ia_unileak_{route_name}.npy")
    ias = np.load(f"../data/resistivity/currents/occupied_ia_varleak_{route_name}.npy")

    index = 1001
    e = np.linspace(-100, 100, 2001)
    e_par = e[index]
    ia0 = ia0s[index, :]
    ia = ias[index, :]

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1)
    gs.update(hspace=0)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax0.plot(ia0, '.', label="Uniform Leakage", color="royalblue")
    ax0.plot(ia, '.', label="Leakage by block", color="orangered")
    ax0.set_title(f"E = {e_par} " + r"$\mathrm{V \cdot km^{-1}}$")
    ax0.legend()
    ax1.set_xlabel("Block Index")
    ax0.set_ylabel("Current Through Relay (A)")
    ax1.set_ylabel("Current Difference (A)")
    ax0.set_xticks([])
    ax1.plot(np.array(ia) - np.array(ia0), '.', color='darkorchid')
    boundaries = np.array([66.5, 107.5, 128.5, 323.5, 427.5, 593.5, 658.5, 707.5, 739.5])
    for b in boundaries:
        ax0.axvline(b, color="gray", alpha=0.25)
        ax1.axvline(b, color="gray", alpha=0.25)
    if route_name == "glasgow_edinburgh_falkirk":
        fig.suptitle("Glasgow to Edinburgh via Falkirk High")
        ax0.set_xlim(-1, 119)
        ax1.set_xlim(-1, 119)
    elif route_name == "east_coast_main_line":
        fig.suptitle("East Coast Main Line")
        ax0.set_xlim(-1, 914)
        ax1.set_xlim(-1, 914)
        if e_par == 0:
            ax0.set_ylim(-0.0035, 0.0032)
            ax1.set_ylim(-0.004, 0.0032)
            ax0.set_yticks([-0.003, -0.002, -0.001, 0, 0.001, 0.002, 0.003])
            ax1.set_yticks([-0.004, -0.003, -0.002, -0.001, 0, 0.001, 0.002, 0.003])
        elif e_par == 5:
            ax0.set_ylim(-0.16, 0)
            ax1.set_ylim(-0.005, 0.011)
            ax0.set_yticks([-0.15, -0.125, -0.10, -0.075, -0.05, -0.025, 0])
            ax1.set_yticks([-0.005, -0.0025, 0, 0.0025, 0.005, 0.0075, 0.01])
        elif e_par == -5:
            ax0.set_ylim(-0.01, 0.15)
            ax1.set_ylim(-0.01, 0.006)
            ax0.set_yticks([0, 0.025, 0.05, 0.075, 0.1, 0.125, 0.15])
            ax1.set_yticks([-0.01, -0.0075, -0.005, -0.0025, 0, 0.0025, 0.005])
        else:
            pass
    elif route_name == "west_coast_main_line":
        fig.suptitle("West Coast Main Line")
        ax0.set_xlim(-1, 936)
        ax1.set_xlim(-1, 936)
    else:
        fig.suptitle(f"{route_name}")

    fig.align_ylabels([ax0, ax1])
    #plt.savefig(f"leakage_relay_currents_E{e_par}_{route_name}_ws.pdf")
    plt.show()


def find_thresholds_ws(route_name):
    leakage = np.load(f"../data/resistivity/{route_name}_leakage_by_block.npy")
    e_par = np.linspace(-100, 100, 2001)
    ia = np.load(f"../data/resistivity/currents/occupied_ia_varleak_{route_name}.npy")
    ia_og = np.load(f"../data/resistivity/currents/occupied_ia_unileak_{route_name}.npy")
    e_thresholds = []
    e_thresholds_og = []
    for i in range(0, len(ia[0, :])):
        currents = ia[:, i]
        current_threshold_dif = np.abs(currents - 0.081)
        e_threshold = e_par[np.where(current_threshold_dif == np.min(current_threshold_dif))]
        e_thresholds.append(e_threshold)
        currents_og = ia_og[:, i]
        current_threshold_dif_og = np.abs(currents_og - 0.081)
        e_threshold_og = e_par[np.where(current_threshold_dif_og == np.min(current_threshold_dif_og))]
        e_thresholds_og.append(e_threshold_og)
    e_thresholds = np.array(e_thresholds)
    e_thresholds_og = np.array(e_thresholds_og)
    e_thresholds = np.where((e_thresholds == 100) | (e_thresholds == -100), np.nan, e_thresholds)
    e_thresholds_og = np.where((e_thresholds_og == 100) | (e_thresholds_og == -100), np.nan, e_thresholds_og)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 7))
    gs = GridSpec(2, 1)
    gs.update(hspace=0)
    gs.update(wspace=0.28)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax0.set_xticks([])

    ax0.axhline(1.6, linestyle='--', color="royalblue", alpha=0.75)
    ax0.set_ylabel(r"Leakage ($\mathrm{S \cdot km^{-1}}$)")
    ax1.set_xlabel("Block Index")
    ax1.set_ylabel("\u0394 Threshold ($\mathrm{V \cdot km^{-1}}$)")
    ax1.axhline(0, linestyle='--', color="limegreen", alpha=0.75)

    if route_name == "glasgow_edinburgh_falkirk":
        fig.suptitle("Glasgow to Edinburgh via Falkirk High")
        ax0.plot(range(0, 119), leakage, '.', color="royalblue")
        ax1.plot(range(0, 119), e_thresholds - e_thresholds_og, '.', color="limegreen")
        #ax0.set_ylim(-0.2, 6)
        #ax1.set_ylim(-40, 110)
        ax0.set_xlim(-1, 119)
        ax1.set_xlim(-1, 119)
        ax0.set_xticks([])
        #ax1.set_xticks([0, 10, 20, 27])
        boundaries = np.array([82.5, 94.5])
        for b in boundaries:
            ax0.axvline(b, color="gray", alpha=0.25)
            ax1.axvline(b, color="gray", alpha=0.25)

    elif route_name == "east_coast_main_line":
        fig.suptitle("East Coast Main Line")
        ax0.plot(range(0, 914), leakage, '.', color="royalblue")
        ax1.plot(range(0, 914), e_thresholds - e_thresholds_og, '.', color="limegreen")
        #ax0.set_ylim(-0.2, 6.2)
        #ax1.set_ylim(-160, 30)
        ax0.set_xlim(-1, 914)
        ax1.set_xlim(-1, 914)
        ax0.set_xticks([])
        #ax1.set_xticks([0, 10, 20, 33])
        #ax0.set_yticks([0, 2, 4, 6])
        #ax1.set_yticks([-150, -100, -50, 0])

        boundaries = np.array([66.5, 107.5, 128.5, 323.5, 427.5, 593.5, 658.5, 707.5, 739.5])
        for b in boundaries:
            ax0.axvline(b, color="gray", alpha=0.25)
            ax1.axvline(b, color="gray", alpha=0.25)

    elif route_name == "west_coast_main_line":
        fig.suptitle("West Coast Main Line")
        ax0.plot(range(0, 936), leakage, '.', color="royalblue")
        ax1.plot(range(0, 936), e_thresholds - e_thresholds_og, '.', color="limegreen")
        #ax0.set_ylim(-0.2, 8.2)
        ax0.set_xlim(-1, 936)
        ax1.set_xlim(-1, 936)
        #ax1.set_xticks([0, 10, 20, 30])

    else:
        fig.suptitle(f"{route_name}")

    fig.align_ylabels([ax0, ax1])
    plt.savefig(f"leakage_threshold_dif_ws_{route_name}.pdf")
    #plt.show()


#for name in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
#    for e in [0, 5, -5]:
#        plot_relay_currents_rs(name, np.array([e]))

#for name in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
#    for e in [0, 5, -5]:
#        plot_relay_currents_dif_rs(name, np.array([e]))

#for name in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
#    find_thresholds_rs(name)

#for name in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
#    crossover_rs(name)

for name in ["east_coast_main_line", "west_coast_main_line", "glasgow_edinburgh_falkirk"]:
    save_relay_currents_rs(name)
    #plot_relay_currents_rs(name, e_par=np.array([0]))
    #save_relay_currents_ws(name)
    #plot_relay_currents_ws(name)
    #find_thresholds_ws(name)
