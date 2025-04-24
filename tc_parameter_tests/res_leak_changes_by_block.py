import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def rail_model_two_track_e_parallel_leak_939_block(leakage, resistance, e_parallel, axle_pos_a, axle_pos_b):
    # Create dictionary of network parameters
    parameters = {"z_sig": resistance,  # Signalling rail series impedance (ohms/km)
                  "z_trac": resistance,  # Traction return rail series impedance (ohms/km)
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

    # Load in the lengths and bearings of the track circuit blocks
    # Note: zero degrees is directly northwards, with positive values increasing clockwise
    blocks = np.full(201, 0.68)
    blocks_sum = np.cumsum(blocks)  # Cumulative sum of block lengths

    y_trac_block = np.full(len(blocks), 1.6)
    y_trac_block[100] = leakage

    # Calculate the electrical characteristics of the rails
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig"])
    gamma_trac_block = np.sqrt(parameters["z_trac"] * y_trac_block).flatten()
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig"])
    z0_trac_block = np.sqrt(parameters["z_trac"] / y_trac_block).flatten()

    # Add cross bonds and axles which split the blocks into sub blocks
    # Note: "a" and "b" are used to identify the opposite directions of travel in this network (two-track)
    pos_cb = np.arange(0.40001, np.sum(blocks), 0.4)  # Position of the cross bonds
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


def rail_model_two_track_e_parallel_leak_966_block(leakage, resistance, e_parallel, axle_pos_a, axle_pos_b):
    # Create dictionary of network parameters
    parameters = {"z_sig": resistance,  # Signalling rail series impedance (ohms/km)
                  "z_trac": resistance,  # Traction return rail series impedance (ohms/km)
                  "y_sig": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "v_power": 10,  # Track circuit power supply voltage (volts)
                  "r_power": 7.2,  # Track circuit power supply resistance (ohms)
                  "r_relay": 9,  # Track circuit relay resistance (ohms)
                  "r_cb": 1e-3,  # Cross bond resistance (ohms)
                  "r_axle": 251e-4,  # Axle resistance (ohms)
                  "i_power": 10 / 7.2,  # Track circuit power supply equivalent current source (amps)
                  "y_power": 1 / 7.2,  # Track circuit power supply admittance (siemens)
                  "y_relay": 1 / 9,  # Track circuit relay admittance (siemens)
                  "y_cb": 1 / 1e-3,  # Cross bond admittance (siemens)
                  "y_axle": 1 / 251e-4}  # Axle admittance (siemens)

    # Load in the lengths and bearings of the track circuit blocks
    # Note: zero degrees is directly northwards, with positive values increasing clockwise
    blocks = np.full(201, 0.68)
    blocks_sum = np.cumsum(blocks)  # Cumulative sum of block lengths

    y_trac_block = np.full(len(blocks), 1.6)
    y_trac_block[100] = leakage

    # Calculate the electrical characteristics of the rails
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig"])
    gamma_trac_block = np.sqrt(parameters["z_trac"] * y_trac_block).flatten()
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig"])
    z0_trac_block = np.sqrt(parameters["z_trac"] / y_trac_block).flatten()

    # Add cross bonds and axles which split the blocks into sub blocks
    # Note: "a" and "b" are used to identify the opposite directions of travel in this network (two-track)
    pos_cb = np.arange(0.40001, np.sum(blocks), 0.4)  # Position of the cross bonds
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


def plot_leak_dif_rs():
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])

    res = 0.0289
    leak = [8.29, 4.14, 2.46, 1.06, 0.53, 0.265]
    e_all = np.arange(-20, 20.1, 0.01)
    thresh939 = np.empty(len(leak))
    thresh966 = np.empty(len(leak))
    lines = []
    labels = []
    for i in range(0, len(leak)):
        ia939, ib939 = rail_model_two_track_e_parallel_leak_939_block(leakage=leak[i], resistance=res, e_parallel=e_all, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
        ia966, ib966 = rail_model_two_track_e_parallel_leak_966_block(leakage=leak[i], resistance=res, e_parallel=e_all, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
        line, = ax0.plot(e_all, ia939[:, 100])
        ax1.plot(e_all, ia966[:, 100])
        lines.append(line)
        labels.append(f'Leakage = {leak[i]} S/km')

    ax0.set_xlabel("Electric Field Strength (V/km)")
    ax0.set_ylabel("Current Through Relay (A)")
    ax0.axhline(0.055, color="red", linestyle="-")
    ax0.set_xlim(9.52, 9.64)
    ax0.set_ylim(0.053, 0.057)
    ax0.set_title(f"BR939A Relay")

    ax1.set_xlabel("Electric Field Strength (V/km)")
    #ax1.set_ylabel("Current Through Relay (A)")
    ax1.axhline(0.081, color="red", linestyle="-")
    min_thresh966 = np.min(thresh966)
    max_thresh966 = np.max(thresh966)
    ax1.axvline(min_thresh966, color='gray', linestyle='-')
    ax1.axvline(max_thresh966, color='gray', linestyle='--')
    ax1.set_xlim(9.94, 10.02)
    ax1.set_ylim(0.079, 0.083)
    ax1.set_title(f"BR966 F2 Relay")

    #gs.update(wspace=0.01)
    ax0.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)

    fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=4)

    #plt.savefig("leakage_rs.pdf")
    plt.show()


def plot_leak_rs_current():
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    leak = np.linspace(0.265, 8.29, 50)

    def save():
        leaks9390 = np.empty(len(leak))
        leaks9660 = np.empty(len(leak))
        leaks939 = np.empty(len(leak))
        leaks966 = np.empty(len(leak))
        for i in range(0, len(leak)):
            print(i)
            ia9390, ib9390 = rail_model_two_track_e_parallel_leak_939_block(leakage=leak[i], resistance=0.0289, e_parallel=np.array([0]), axle_pos_a=np.array([]), axle_pos_b=np.array([]))
            ia939, ib939 = rail_model_two_track_e_parallel_leak_939_block(leakage=leak[i], resistance=0.0289,
                                                                    e_parallel=np.array([10]), axle_pos_a=np.array([]),
                                                                    axle_pos_b=np.array([]))
            leaks9390[i] = ia9390[0, 100]
            leaks939[i] = ia939[0, 100]
            ia9660, ib9660 = rail_model_two_track_e_parallel_leak_966_block(leakage=leak[i], resistance=0.0289, e_parallel=np.array([0]), axle_pos_a=np.array([]), axle_pos_b=np.array([]))
            ia966, ib966 = rail_model_two_track_e_parallel_leak_966_block(leakage=leak[i], resistance=0.0289,
                                                                    e_parallel=np.array([10]), axle_pos_a=np.array([]),
                                                                    axle_pos_b=np.array([]))
            leaks9660[i] = ia9660[0, 100]
            leaks966[i] = ia966[0, 100]

        np.save("leaks9390_block.npy", leaks9390)
        np.save("leaks939_block.npy", leaks939)
        np.save("leaks9660_block.npy", leaks9660)
        np.save("leaks966_block.npy", leaks966)

    #save()

    leaks9390 = np.load("leaks9390_block.npy")
    leaks939 = np.load("leaks939_block.npy")
    leaks9660 = np.load("leaks9660_block.npy")
    leaks966 = np.load("leaks966_block.npy")

    ax0.plot(leak, leaks9390, label="BR939A Relay", color='tomato')
    ax0.set_ylabel("Current Through Relay (A)")
    ax0.set_title(f"E = 0 V/km")
    ax0.set_ylim(0.2733, 0.2747)
    ax0.set_xlim(0, 8.34)
    ax0.set_yticks([0.2735, 0.2740, 0.2745])
    ax0.set_xticks([2, 4, 6, 8])
    ax0.legend()
    ax0.grid(color="grey", alpha=0.5)

    ax1.plot(leak, leaks939, label="BR939A Relay", color='tomato')
    ax1.set_title(f"E = -10 V/km")
    ax1.set_ylim(0.0447, 0.0461)
    ax1.set_yticks([0.0450, 0.0455, 0.0460])
    ax1.set_xlim(0, 8.34)
    ax1.set_xticks([2, 4, 6, 8])
    ax1.legend()
    ax1.grid(color="grey", alpha=0.5)

    ax2.plot(leak, leaks9660, label="BR966 F2 Relay", color='steelblue')
    ax2.set_xlabel("Leakage (S/km)")
    ax2.set_ylabel("Current Through Relay (A)")
    ax2.set_ylim(0.4893, 0.4913)
    ax2.set_yticks([0.4895, 0.4900, 0.4905, 0.4910])
    ax2.set_xlim(0, 8.34)
    ax2.set_xticks([0, 2, 4, 6, 8])
    ax2.legend()
    ax2.grid(color="grey", alpha=0.5)

    ax3.plot(leak, leaks966, label="BR966 F2 Relay", color='steelblue')
    ax3.set_xlabel("Leakage (S/km)")
    ax3.set_ylim(0.0789, 0.0809)
    ax3.set_yticks([0.0790, 0.0795, 0.0800, 0.0805])
    ax3.set_xlim(0, 8.34)
    ax3.set_xticks([0, 2, 4, 6, 8])
    ax3.legend()
    ax3.grid(color="grey", alpha=0.5)

    gs.update(hspace=0)

    ax0.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)

    # plt.savefig("leakage_rs.pdf")
    plt.show()


def plot_leak_dif_ws():
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    leak = [8.29, 4.14, 2.46, 1.06, 0.53, 0.265]
    e_all = np.arange(-20, 20.1, 0.01)
    lines = []
    labels = []
    for i in range(0, len(leak)):
        ia939, ib939 = rail_model_two_track_e_parallel_leak_939_block(leakage=leak[i], resistance=0.0289, e_parallel=e_all, axle_pos_a=np.array([68.67]), axle_pos_b=np.array([]))
        ia966, ib966 = rail_model_two_track_e_parallel_leak_966_block(leakage=leak[i], resistance=0.0289, e_parallel=e_all, axle_pos_a=np.array([68.67]), axle_pos_b=np.array([]))
        line, = ax0.plot(e_all, ia939[:, 100])
        ax1.plot(e_all, ia966[:, 100])
        ax2.plot(e_all, ia939[:, 100])
        ax3.plot(e_all, ia966[:, 100])
        labels.append(f'Leakage = {leak[i]} S/km')
        lines.append(line)

    ax0.set_ylabel("Current Through Relay (A)")
    ax0.axhline(0.081, color="limegreen", linestyle="--")
    ax0.set_xlim(-2.366885, -2.366240)
    ax0.set_ylim(0.08098193, 0.08101807)
    ax0.set_title(f"BR939A Relay")

    ax1.axhline(0.12, color="limegreen", linestyle="--")
    ax1.set_xlim(-1.560374, -1.559674)
    ax1.set_ylim(0.1199583, 0.1200417)
    ax1.set_title(f"BR966 F2 Relay")

    ax2.set_xlabel("Electric Field Strength (V/km)")
    ax2.set_ylabel("Current Through Relay (A)")
    ax2.axhline(-0.081, color="limegreen", linestyle="--")
    #x2.set_xlim(-2.366885, -2.366240)
    #ax2.set_ylim(0.08098193, 0.08101807)

    ax3.set_xlabel("Electric Field Strength (V/km)")
    ax3.axhline(-0.12, color="limegreen", linestyle="--")
    #ax3.set_xlim(-1.560374, -1.559674)
    #ax3.set_ylim(0.1199583, 0.1200417)

    ax0.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax0.get_xaxis().get_major_formatter().set_useOffset(False)
    ax1.get_xaxis().get_major_formatter().set_useOffset(False)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.get_xaxis().get_major_formatter().set_useOffset(False)
    ax3.get_xaxis().get_major_formatter().set_useOffset(False)

    fig.legend(lines, labels, loc='lower center', fancybox=True, shadow=True, ncol=4)

    #plt.savefig("leakage_rs.pdf")
    plt.show()


def plot_leak_ws_current():
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 8))
    gs = GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[0, 1])
    ax2 = fig.add_subplot(gs[1, 0])
    ax3 = fig.add_subplot(gs[1, 1])

    leak = np.linspace(0.265, 8.29, 50)

    def save():
        leaks9390 = np.empty(len(leak))
        leaks9660 = np.empty(len(leak))
        leaks939 = np.empty(len(leak))
        leaks966 = np.empty(len(leak))
        for i in range(0, len(leak)):
            print(i)
            ia9390, ib9390 = rail_model_two_track_e_parallel_leak_939_block(leakage=leak[i], resistance=0.0289, e_parallel=np.array([0]), axle_pos_a=np.array([68.34]), axle_pos_b=np.array([]))
            ia939, ib939 = rail_model_two_track_e_parallel_leak_939_block(leakage=leak[i], resistance=0.0289, e_parallel=np.array([-10]), axle_pos_a=np.array([68.34]), axle_pos_b=np.array([]))
            leaks9390[i] = ia9390[0, 100]
            leaks939[i] = ia939[0, 100]
            ia9660, ib9660 = rail_model_two_track_e_parallel_leak_966_block(leakage=leak[i], resistance=0.0289, e_parallel=np.array([0]), axle_pos_a=np.array([68.34]), axle_pos_b=np.array([]))
            ia966, ib966 = rail_model_two_track_e_parallel_leak_966_block(leakage=leak[i], resistance=0.0289, e_parallel=np.array([-10]), axle_pos_a=np.array([68.34]), axle_pos_b=np.array([]))
            leaks9660[i] = ia9660[0, 100]
            leaks966[i] = ia966[0, 100]

        np.save("leaks9390ws_block.npy", leaks9390)
        np.save("leaks939ws_block.npy", leaks939)
        np.save("leaks9660ws_block.npy", leaks9660)
        np.save("leaks966ws_block.npy", leaks966)

    #save()

    leaks9390 = np.load("leaks9390ws_block.npy")
    leaks939 = np.load("leaks939ws_block.npy")
    leaks9660 = np.load("leaks9660ws_block.npy")
    leaks966 = np.load("leaks966ws_block.npy")

    ax0.plot(leak, leaks9390, label="BR939A Relay", color='tomato')
    ax0.set_ylabel("Current Through Relay (A)")
    ax0.set_title(f"E = 0 V/km")
    #ax0.set_ylim(0.269, 0.295)
    #ax0.set_xlim(0, 8.34)
    #ax0.set_yticks([0.27, 0.28, 0.29])
    #ax0.set_xticks([2, 4, 6, 8])
    ax0.legend()
    ax0.grid(color="grey", alpha=0.5)

    ax1.plot(leak, leaks939, label="BR939A Relay", color='tomato')
    ax1.set_title(f"E = -10 V/km")
    #ax1.set_ylim(0.039, 0.065)
    #ax1.set_yticks([0.04, 0.05, 0.06])
    #ax1.set_xlim(0, 8.34)
    #ax1.set_xticks([2, 4, 6, 8])
    ax1.legend()
    ax1.grid(color="grey", alpha=0.5)

    ax2.plot(leak, leaks9660, label="BR966 F2 Relay", color='steelblue')
    ax2.set_xlabel("Leakage (S/km)")
    ax2.set_ylabel("Current Through Relay (A)")
    #ax2.set_ylim(0.48, 0.518)
    #ax2.set_yticks([0.48, 0.49, 0.50, 0.51])
    #ax2.set_xlim(0, 8.34)
    #ax2.set_xticks([0, 2, 4, 6, 8])
    ax2.legend()
    ax2.grid(color="grey", alpha=0.5)

    ax3.plot(leak, leaks966, label="BR966 F2 Relay", color='steelblue')
    ax3.set_xlabel("Leakage (S/km)")
    #ax3.set_ylim(0.075, 0.107)
    #ax3.set_yticks([0.07, 0.08, 0.09, 0.1])
    #ax3.set_xlim(0, 8.34)
    #ax3.set_xticks([0, 2, 4, 6, 8])
    ax3.legend()
    ax3.grid(color="grey", alpha=0.5)

    gs.update(hspace=0)

    ax0.get_yaxis().get_major_formatter().set_useOffset(False)
    ax1.get_yaxis().get_major_formatter().set_useOffset(False)
    ax2.get_yaxis().get_major_formatter().set_useOffset(False)
    ax3.get_yaxis().get_major_formatter().set_useOffset(False)

    # plt.savefig("leakage_ws.pdf")

    plt.show()


#plot_leak_dif_rs()
#plot_leak_rs_current()
plot_leak_dif_ws()
#plot_leak_ws_current()
