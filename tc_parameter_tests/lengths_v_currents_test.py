import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


# Track circuits using 20 ohm relays to BR 939A, are to be a maximum of 1000m in length.
# Track circuits using 9 ohm relays to BR 966 F2, are to be a maximum of 680m in length.
# Track circuits using 60 ohm relays to BR 966 F9, are to be a maximum of 350m in length.

def rail_model_tc_type(tc_type, block_length, conditions, e_parallel, axle_pos_a, axle_pos_b):
    # Create dictionary of network parameters
    parameters = {"z_sig": 0.0289,  # Signalling rail series impedance (ohms/km)
                  "z_trac": 0.0289,  # Traction return rail series impedance (ohms/km)
                  "y_sig_moderate": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_moderate": 1.6,  # Traction return rail parallel admittance in moderate conditions (siemens/km)
                  "y_sig_dry": 0.025,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_dry": 1.53,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "y_sig_wet": 0.4,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_wet": 2,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "r_cb": 1e-3,  # Cross bond resistance (ohms)
                  "r_axle": 251e-4,  # Axle resistance (ohms)
                  "y_cb": 1 / 1e-3,  # Cross bond admittance (siemens)
                  "y_axle": 1 / 251e-4}  # Axle admittance (siemens)
    tc_parameter_dicts = {
        "br867_br939a": {"v_power": 10, "r_power": 7.2, "r_relay": 20, "i_power": 10 / 7.2, "y_power": 1 / 7.2, "y_relay": 1 / 20},
        "br867_br966f2": {"v_power": 10, "r_power": 7.2, "r_relay": 9, "i_power": 10 / 7.2, "y_power": 1 / 7.2, "y_relay": 1 / 9},
        "br867_br966f9": {"v_power": 10, "r_power": 7.2, "r_relay": 60, "i_power": 10 / 7.2, "y_power": 1 / 7.2, "y_relay": 1 / 60}
    }
    tc_parameters = tc_parameter_dicts.get(tc_type)

    if tc_parameters is None:
        raise ValueError(f"Unknown type: {type}")

    # Calculate the electrical characteristics of the rails
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig_"+conditions])
    gamma_trac = np.sqrt(parameters["z_trac"] * parameters["y_trac_"+conditions])
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig_"+conditions])
    z0_trac = np.sqrt(parameters["z_trac"] / parameters["y_trac_"+conditions])

    # Load in the lengths and bearings of the track circuit blocks
    # Note: zero degrees is directly northwards, with positive values increasing clockwise
    n_blocks = int(np.round(100/block_length))
    blocks = np.full(n_blocks, block_length)
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
    yg_trac_a = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_a) - 1) * (1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_a))))  # Parallel admittance for traction return rail
    yg_sig_a = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_a) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))))  # Parallel admittance for signalling rail
    yg_trac_b = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_b) - 1) * (1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_b))))
    yg_sig_b = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_b) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))))

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
    yg = np.full(n_nodes, 69).astype(float)  # Array of parallel admittances
    y_sum = np.full(n_nodes, 69).astype(float)  # Array of sum of admittances into the node
    # First node
    mask_first_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[0])
    first_trac_a = node_locs_trac_a[mask_first_trac_a]
    locs_first_trac_a = np.where(np.isin(node_locs_trac_a, first_trac_a))[0]
    yg[first_trac_a] = 0.5 * yg_trac_a[locs_first_trac_a]
    y_sum[first_trac_a] = yg[first_trac_a] + tc_parameters["y_relay"] + ye_trac_a[locs_first_trac_a]
    # Axles
    locs_axle_trac_a = np.where(np.isin(node_locs_trac_a, node_locs_axle_trac_a))[0]
    yg[node_locs_axle_trac_a] = (0.5 * yg_trac_a[locs_axle_trac_a - 1]) + (0.5 * yg_trac_a[locs_axle_trac_a])
    y_sum[node_locs_axle_trac_a] = yg[node_locs_axle_trac_a] + parameters["y_axle"] + ye_trac_a[locs_axle_trac_a - 1] + ye_trac_a[locs_axle_trac_a]
    # Cross bonds
    locs_cb_a = np.where(np.isin(node_locs_trac_a, node_locs_cb_a))[0]
    yg[node_locs_cb_a] = (0.5 * yg_trac_a[locs_cb_a - 1]) + (0.5 * yg_trac_a[locs_cb_a])
    y_sum[node_locs_cb_a] = yg[node_locs_cb_a] + parameters["y_cb"] + ye_trac_a[locs_cb_a - 1] + ye_trac_a[locs_cb_a]
    # Middle nodes
    indices_other_node_trac_a = node_locs_trac_a[1:-1][~np.logical_or(np.isin(node_locs_trac_a[1:-1], node_locs_axle_trac_a), np.isin(node_locs_trac_a[1:-1], node_locs_cb_a))]
    mask_other_trac_a = np.isin(indices_other_node_trac_a, node_locs_trac_a)
    other_trac_a = indices_other_node_trac_a[mask_other_trac_a]
    locs_other_trac_a = np.where(np.isin(node_locs_trac_a, other_trac_a))[0]
    yg[other_trac_a] = (0.5 * yg_trac_a[locs_other_trac_a - 1]) + (0.5 * yg_trac_a[locs_other_trac_a])
    y_sum[other_trac_a] = yg[other_trac_a] + tc_parameters["y_power"] + tc_parameters["y_relay"] + ye_trac_a[locs_other_trac_a - 1] + ye_trac_a[locs_other_trac_a]
    # Last node
    mask_last_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[-1])
    last_trac_a = node_locs_trac_a[mask_last_trac_a]
    locs_last_trac_a = np.where(np.isin(node_locs_trac_a, last_trac_a))[0]
    yg[last_trac_a] = 0.5 * yg_trac_a[locs_last_trac_a - 1]
    y_sum[last_trac_a] = yg[last_trac_a] + tc_parameters["y_power"] + ye_trac_a[locs_last_trac_a - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_relay_sig_a))[0]
    yg[node_locs_relay_sig_a] = 0.5 * yg_sig_a[locs_relay_sig_a]
    y_sum[node_locs_relay_sig_a] = yg[node_locs_relay_sig_a] + tc_parameters["y_relay"] + ye_sig_a[locs_relay_sig_a]
    # Power nodes
    locs_power_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]
    yg[node_locs_power_sig_a] = 0.5 * yg_sig_a[locs_power_sig_a - 1]
    y_sum[node_locs_power_sig_a] = yg[node_locs_power_sig_a] + tc_parameters["y_power"] + ye_sig_a[locs_power_sig_a - 1]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_a, node_locs_axle_sig_a))[0]
    yg[node_locs_axle_sig_a] = (0.5 * yg_sig_a[axle_locs - 1]) + (0.5 * yg_sig_a[axle_locs])
    y_sum[node_locs_axle_sig_a] = yg[node_locs_axle_sig_a] + parameters["y_axle"] + ye_sig_a[axle_locs - 1] + ye_sig_a[axle_locs]
    # Direction "b" second
    # Traction return rail
    # First node
    mask_first_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[0])
    first_trac_b = node_locs_trac_b[mask_first_trac_b]
    locs_first_trac_b = np.where(np.isin(node_locs_trac_b, first_trac_b))[0]
    yg[first_trac_b] = 0.5 * yg_trac_b[locs_first_trac_b]
    y_sum[first_trac_b] = yg[first_trac_b] + tc_parameters["y_relay"] + ye_trac_b[locs_first_trac_b]
    # Axles
    locs_axle_trac_b = np.where(np.isin(node_locs_trac_b, node_locs_axle_trac_b))[0]
    yg[node_locs_axle_trac_b] = (0.5 * yg_trac_b[locs_axle_trac_b - 1]) + (0.5 * yg_trac_b[locs_axle_trac_b])
    y_sum[node_locs_axle_trac_b] = yg[node_locs_axle_trac_b] + parameters["y_axle"] + ye_trac_b[locs_axle_trac_b - 1] + ye_trac_b[locs_axle_trac_b]
    # Cross bonds
    locs_cb_b = np.where(np.isin(node_locs_trac_b, node_locs_cb_b))[0]
    yg[node_locs_cb_b] = (0.5 * yg_trac_b[locs_cb_b - 1]) + (0.5 * yg_trac_b[locs_cb_b])
    y_sum[node_locs_cb_b] = yg[node_locs_cb_b] + parameters["y_cb"] + ye_trac_b[locs_cb_b - 1] + ye_trac_b[locs_cb_b]
    # Middle nodes
    indices_other_node_trac_b = node_locs_trac_b[1:-1][~np.logical_or(np.isin(node_locs_trac_b[1:-1], node_locs_axle_trac_b),np.isin(node_locs_trac_b[1:-1], node_locs_cb_b))]
    mask_other_trac_b = np.isin(indices_other_node_trac_b, node_locs_trac_b)
    other_trac_b = indices_other_node_trac_b[mask_other_trac_b]
    locs_other_trac_b = np.where(np.isin(node_locs_trac_b, other_trac_b))[0]
    yg[other_trac_b] = (0.5 * yg_trac_b[locs_other_trac_b - 1]) + (0.5 * yg_trac_b[locs_other_trac_b])
    y_sum[other_trac_b] = yg[other_trac_b] + tc_parameters["y_power"] + tc_parameters["y_relay"] + ye_trac_b[locs_other_trac_b - 1] + ye_trac_b[locs_other_trac_b]
    # Last node
    mask_last_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[-1])
    last_trac_b = node_locs_trac_b[mask_last_trac_b]
    locs_last_trac_b = np.where(np.isin(node_locs_trac_b, last_trac_b))[0]
    yg[last_trac_b] = 0.5 * yg_trac_b[locs_last_trac_b - 1]
    y_sum[last_trac_b] = yg[last_trac_b] + tc_parameters["y_power"] + ye_trac_b[locs_last_trac_b - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]
    yg[node_locs_relay_sig_b] = 0.5 * yg_sig_b[locs_relay_sig_b - 1]
    y_sum[node_locs_relay_sig_b] = yg[node_locs_relay_sig_b] + tc_parameters["y_relay"] + ye_sig_b[locs_relay_sig_b - 1]
    # Power nodes
    locs_power_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_power_sig_b))[0]
    yg[node_locs_power_sig_b] = 0.5 * yg_sig_b[locs_power_sig_b]
    y_sum[node_locs_power_sig_b] = yg[node_locs_power_sig_b] + tc_parameters["y_power"] + ye_sig_b[locs_power_sig_b]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_b, node_locs_axle_sig_b))[0]
    yg[node_locs_axle_sig_b] = (0.5 * yg_sig_b[axle_locs - 1]) + (0.5 * yg_sig_b[axle_locs])
    y_sum[node_locs_axle_sig_b] = yg[node_locs_axle_sig_b] + parameters["y_axle"] + ye_sig_b[axle_locs - 1] + ye_sig_b[axle_locs]

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
    y_matrix[node_locs_relay_trac_a, node_locs_relay_sig_a] = -tc_parameters["y_relay"]
    y_matrix[node_locs_relay_sig_a, node_locs_relay_trac_a] = -tc_parameters["y_relay"]
    y_matrix[node_locs_relay_trac_b, node_locs_relay_sig_b] = -tc_parameters["y_relay"]
    y_matrix[node_locs_relay_sig_b, node_locs_relay_trac_b] = -tc_parameters["y_relay"]
    # Power admittances
    y_matrix[node_locs_power_trac_a, node_locs_power_sig_a] = -tc_parameters["y_power"]
    y_matrix[node_locs_power_sig_a, node_locs_power_trac_a] = -tc_parameters["y_power"]
    y_matrix[node_locs_power_trac_b, node_locs_power_sig_b] = -tc_parameters["y_power"]
    y_matrix[node_locs_power_sig_b, node_locs_power_trac_b] = -tc_parameters["y_power"]
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
    j_matrix[:, non_cb_axle_node_locs_centre_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices] - tc_parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, node_locs_trac_a[-1]] = i_trac_a[:, -1] - tc_parameters["i_power"]

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
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, whole_blocks_end))[0]]] = i_sig_a[:, whole_blocks] + tc_parameters["i_power"]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_start))[0]]] = -i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_start))[0]]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_end))[0]]] = i_sig_a[:, split_blocks[np.where(~np.isin(split_blocks,np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1))[0]]] + tc_parameters["i_power"]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_mid))[0]]] = i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1] - i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0]]

    # "b" second
    # Traction return rail first node
    j_matrix[:, node_locs_trac_b[0]] = i_trac_b[:, 0] - tc_parameters["i_power"]
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
    j_matrix[:, non_cb_axle_node_locs_centre_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1] - tc_parameters["i_power"]
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
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_start))[0]]] = i_sig_b[:, whole_blocks] + tc_parameters["i_power"]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_end))[0]]] = -i_sig_b[:, whole_blocks]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_start))[0]]] = i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_start))[0]] + tc_parameters["i_power"]
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

    i_relays_a = v_relay_a / tc_parameters["r_relay"]
    i_relays_b = v_relay_b / tc_parameters["r_relay"]

    i_relays_a = i_relays_a.T
    i_relays_b = i_relays_b.T

    return i_relays_a, i_relays_b


def rail_model_tc_type_only_middle(tc_type, block_length, conditions, e_parallel, axle_pos_a, axle_pos_b):
    # Create dictionary of network parameters
    parameters = {"z_sig": 0.0289,  # Signalling rail series impedance (ohms/km)
                  "z_trac": 0.0289,  # Traction return rail series impedance (ohms/km)
                  "y_sig_moderate": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_moderate": 1.6,  # Traction return rail parallel admittance in moderate conditions (siemens/km)
                  "y_sig_dry": 0.025,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_dry": 1.53,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "y_sig_wet": 0.4,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_wet": 2,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "r_cb": 1e-3,  # Cross bond resistance (ohms)
                  "r_axle": 251e-4,  # Axle resistance (ohms)
                  "y_cb": 1 / 1e-3,  # Cross bond admittance (siemens)
                  "y_axle": 1 / 251e-4}  # Axle admittance (siemens)
    tc_parameter_dicts = {
        "br867_br939a": {"v_power": 10, "r_power": 7.2, "r_relay": 20, "i_power": 10 / 7.2, "y_power": 1 / 7.2, "y_relay": 1 / 20},
        "br867_br966f2": {"v_power": 10, "r_power": 7.2, "r_relay": 9, "i_power": 10 / 7.2, "y_power": 1 / 7.2, "y_relay": 1 / 9},
        "br867_br966f9": {"v_power": 10, "r_power": 7.2, "r_relay": 60, "i_power": 10 / 7.2, "y_power": 1 / 7.2, "y_relay": 1 / 60}
    }
    tc_parameters = tc_parameter_dicts.get(tc_type)

    if tc_parameters is None:
        raise ValueError(f"Unknown type: {type}")

    # Calculate the electrical characteristics of the rails
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig_"+conditions])
    gamma_trac = np.sqrt(parameters["z_trac"] * parameters["y_trac_"+conditions])
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig_"+conditions])
    z0_trac = np.sqrt(parameters["z_trac"] / parameters["y_trac_"+conditions])

    # Load in the lengths and bearings of the track circuit blocks
    # Note: zero degrees is directly northwards, with positive values increasing clockwise
    blocks = np.full(100, 1.00001)
    blocks[50] = block_length
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
    yg_trac_a = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_a) - 1) * (1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_a))))  # Parallel admittance for traction return rail
    yg_sig_a = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_a) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))))  # Parallel admittance for signalling rail
    yg_trac_b = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_b) - 1) * (1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_b))))
    yg_sig_b = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_b) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))))

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
    yg = np.full(n_nodes, 69).astype(float)  # Array of parallel admittances
    y_sum = np.full(n_nodes, 69).astype(float)  # Array of sum of admittances into the node
    # First node
    mask_first_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[0])
    first_trac_a = node_locs_trac_a[mask_first_trac_a]
    locs_first_trac_a = np.where(np.isin(node_locs_trac_a, first_trac_a))[0]
    yg[first_trac_a] = 0.5 * yg_trac_a[locs_first_trac_a]
    y_sum[first_trac_a] = yg[first_trac_a] + tc_parameters["y_relay"] + ye_trac_a[locs_first_trac_a]
    # Axles
    locs_axle_trac_a = np.where(np.isin(node_locs_trac_a, node_locs_axle_trac_a))[0]
    yg[node_locs_axle_trac_a] = (0.5 * yg_trac_a[locs_axle_trac_a - 1]) + (0.5 * yg_trac_a[locs_axle_trac_a])
    y_sum[node_locs_axle_trac_a] = yg[node_locs_axle_trac_a] + parameters["y_axle"] + ye_trac_a[locs_axle_trac_a - 1] + ye_trac_a[locs_axle_trac_a]
    # Cross bonds
    locs_cb_a = np.where(np.isin(node_locs_trac_a, node_locs_cb_a))[0]
    yg[node_locs_cb_a] = (0.5 * yg_trac_a[locs_cb_a - 1]) + (0.5 * yg_trac_a[locs_cb_a])
    y_sum[node_locs_cb_a] = yg[node_locs_cb_a] + parameters["y_cb"] + ye_trac_a[locs_cb_a - 1] + ye_trac_a[locs_cb_a]
    # Middle nodes
    indices_other_node_trac_a = node_locs_trac_a[1:-1][~np.logical_or(np.isin(node_locs_trac_a[1:-1], node_locs_axle_trac_a), np.isin(node_locs_trac_a[1:-1], node_locs_cb_a))]
    mask_other_trac_a = np.isin(indices_other_node_trac_a, node_locs_trac_a)
    other_trac_a = indices_other_node_trac_a[mask_other_trac_a]
    locs_other_trac_a = np.where(np.isin(node_locs_trac_a, other_trac_a))[0]
    yg[other_trac_a] = (0.5 * yg_trac_a[locs_other_trac_a - 1]) + (0.5 * yg_trac_a[locs_other_trac_a])
    y_sum[other_trac_a] = yg[other_trac_a] + tc_parameters["y_power"] + tc_parameters["y_relay"] + ye_trac_a[locs_other_trac_a - 1] + ye_trac_a[locs_other_trac_a]
    # Last node
    mask_last_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[-1])
    last_trac_a = node_locs_trac_a[mask_last_trac_a]
    locs_last_trac_a = np.where(np.isin(node_locs_trac_a, last_trac_a))[0]
    yg[last_trac_a] = 0.5 * yg_trac_a[locs_last_trac_a - 1]
    y_sum[last_trac_a] = yg[last_trac_a] + tc_parameters["y_power"] + ye_trac_a[locs_last_trac_a - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_relay_sig_a))[0]
    yg[node_locs_relay_sig_a] = 0.5 * yg_sig_a[locs_relay_sig_a]
    y_sum[node_locs_relay_sig_a] = yg[node_locs_relay_sig_a] + tc_parameters["y_relay"] + ye_sig_a[locs_relay_sig_a]
    # Power nodes
    locs_power_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]
    yg[node_locs_power_sig_a] = 0.5 * yg_sig_a[locs_power_sig_a - 1]
    y_sum[node_locs_power_sig_a] = yg[node_locs_power_sig_a] + tc_parameters["y_power"] + ye_sig_a[locs_power_sig_a - 1]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_a, node_locs_axle_sig_a))[0]
    yg[node_locs_axle_sig_a] = (0.5 * yg_sig_a[axle_locs - 1]) + (0.5 * yg_sig_a[axle_locs])
    y_sum[node_locs_axle_sig_a] = yg[node_locs_axle_sig_a] + parameters["y_axle"] + ye_sig_a[axle_locs - 1] + ye_sig_a[axle_locs]
    # Direction "b" second
    # Traction return rail
    # First node
    mask_first_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[0])
    first_trac_b = node_locs_trac_b[mask_first_trac_b]
    locs_first_trac_b = np.where(np.isin(node_locs_trac_b, first_trac_b))[0]
    yg[first_trac_b] = 0.5 * yg_trac_b[locs_first_trac_b]
    y_sum[first_trac_b] = yg[first_trac_b] + tc_parameters["y_relay"] + ye_trac_b[locs_first_trac_b]
    # Axles
    locs_axle_trac_b = np.where(np.isin(node_locs_trac_b, node_locs_axle_trac_b))[0]
    yg[node_locs_axle_trac_b] = (0.5 * yg_trac_b[locs_axle_trac_b - 1]) + (0.5 * yg_trac_b[locs_axle_trac_b])
    y_sum[node_locs_axle_trac_b] = yg[node_locs_axle_trac_b] + parameters["y_axle"] + ye_trac_b[locs_axle_trac_b - 1] + ye_trac_b[locs_axle_trac_b]
    # Cross bonds
    locs_cb_b = np.where(np.isin(node_locs_trac_b, node_locs_cb_b))[0]
    yg[node_locs_cb_b] = (0.5 * yg_trac_b[locs_cb_b - 1]) + (0.5 * yg_trac_b[locs_cb_b])
    y_sum[node_locs_cb_b] = yg[node_locs_cb_b] + parameters["y_cb"] + ye_trac_b[locs_cb_b - 1] + ye_trac_b[locs_cb_b]
    # Middle nodes
    indices_other_node_trac_b = node_locs_trac_b[1:-1][~np.logical_or(np.isin(node_locs_trac_b[1:-1], node_locs_axle_trac_b),np.isin(node_locs_trac_b[1:-1], node_locs_cb_b))]
    mask_other_trac_b = np.isin(indices_other_node_trac_b, node_locs_trac_b)
    other_trac_b = indices_other_node_trac_b[mask_other_trac_b]
    locs_other_trac_b = np.where(np.isin(node_locs_trac_b, other_trac_b))[0]
    yg[other_trac_b] = (0.5 * yg_trac_b[locs_other_trac_b - 1]) + (0.5 * yg_trac_b[locs_other_trac_b])
    y_sum[other_trac_b] = yg[other_trac_b] + tc_parameters["y_power"] + tc_parameters["y_relay"] + ye_trac_b[locs_other_trac_b - 1] + ye_trac_b[locs_other_trac_b]
    # Last node
    mask_last_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[-1])
    last_trac_b = node_locs_trac_b[mask_last_trac_b]
    locs_last_trac_b = np.where(np.isin(node_locs_trac_b, last_trac_b))[0]
    yg[last_trac_b] = 0.5 * yg_trac_b[locs_last_trac_b - 1]
    y_sum[last_trac_b] = yg[last_trac_b] + tc_parameters["y_power"] + ye_trac_b[locs_last_trac_b - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]
    yg[node_locs_relay_sig_b] = 0.5 * yg_sig_b[locs_relay_sig_b - 1]
    y_sum[node_locs_relay_sig_b] = yg[node_locs_relay_sig_b] + tc_parameters["y_relay"] + ye_sig_b[locs_relay_sig_b - 1]
    # Power nodes
    locs_power_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_power_sig_b))[0]
    yg[node_locs_power_sig_b] = 0.5 * yg_sig_b[locs_power_sig_b]
    y_sum[node_locs_power_sig_b] = yg[node_locs_power_sig_b] + tc_parameters["y_power"] + ye_sig_b[locs_power_sig_b]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_b, node_locs_axle_sig_b))[0]
    yg[node_locs_axle_sig_b] = (0.5 * yg_sig_b[axle_locs - 1]) + (0.5 * yg_sig_b[axle_locs])
    y_sum[node_locs_axle_sig_b] = yg[node_locs_axle_sig_b] + parameters["y_axle"] + ye_sig_b[axle_locs - 1] + ye_sig_b[axle_locs]

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
    y_matrix[node_locs_relay_trac_a, node_locs_relay_sig_a] = -tc_parameters["y_relay"]
    y_matrix[node_locs_relay_sig_a, node_locs_relay_trac_a] = -tc_parameters["y_relay"]
    y_matrix[node_locs_relay_trac_b, node_locs_relay_sig_b] = -tc_parameters["y_relay"]
    y_matrix[node_locs_relay_sig_b, node_locs_relay_trac_b] = -tc_parameters["y_relay"]
    # Power admittances
    y_matrix[node_locs_power_trac_a, node_locs_power_sig_a] = -tc_parameters["y_power"]
    y_matrix[node_locs_power_sig_a, node_locs_power_trac_a] = -tc_parameters["y_power"]
    y_matrix[node_locs_power_trac_b, node_locs_power_sig_b] = -tc_parameters["y_power"]
    y_matrix[node_locs_power_sig_b, node_locs_power_trac_b] = -tc_parameters["y_power"]
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
    j_matrix[:, non_cb_axle_node_locs_centre_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices] - tc_parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, node_locs_trac_a[-1]] = i_trac_a[:, -1] - tc_parameters["i_power"]

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
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, whole_blocks_end))[0]]] = i_sig_a[:, whole_blocks] + tc_parameters["i_power"]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_start))[0]]] = -i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_start))[0]]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_end))[0]]] = i_sig_a[:, split_blocks[np.where(~np.isin(split_blocks,np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1))[0]]] + tc_parameters["i_power"]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_mid))[0]]] = i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1] - i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0]]

    # "b" second
    # Traction return rail first node
    j_matrix[:, node_locs_trac_b[0]] = i_trac_b[:, 0] - tc_parameters["i_power"]
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
    j_matrix[:, non_cb_axle_node_locs_centre_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1] - tc_parameters["i_power"]
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
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_start))[0]]] = i_sig_b[:, whole_blocks] + tc_parameters["i_power"]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_end))[0]]] = -i_sig_b[:, whole_blocks]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_start))[0]]] = i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_start))[0]] + tc_parameters["i_power"]
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

    i_relays_a = v_relay_a / tc_parameters["r_relay"]
    i_relays_b = v_relay_b / tc_parameters["r_relay"]

    i_relays_a = i_relays_a.T
    i_relays_b = i_relays_b.T

    return i_relays_a, i_relays_b


def save_lengths_v_currents_rs(tc_type):
    bl_range = np.arange(0.100001, 2.000001, 0.01)
    e_range = np.arange(-40, 40.1, 0.1)

    lengths_v_currents = np.empty((len(bl_range), len(e_range)))
    for i in range(0, len(bl_range)):
        bl = bl_range[i]
        print(f"{np.round(i/len(bl_range) * 100, 2)}%")
        ia, ib = rail_model_tc_type(tc_type=tc_type, block_length=bl, conditions="moderate", e_parallel=e_range, axle_pos_a=np.array([]), axle_pos_b=np.array([]))
        n_halfway = int(np.round(100 / (bl*2)))
        lengths_v_currents[i] = ia[:, n_halfway]
    np.save(f"C:/Users/patters9/PycharmProjects/rail_model_v2/data/tc_type_lengths_v_currents/lengths_v_currents_{tc_type}_rs", lengths_v_currents)


def save_lengths_v_currents_ws(tc_type):
    bl_range = np.arange(0.100001, 2.000001, 0.01)
    e_range = np.arange(-40, 40.1, 0.1)

    lengths_v_currents = np.empty((len(bl_range), len(e_range)))
    for i in range(0, len(bl_range)):
        bl = bl_range[i]
        print(f"{np.round(i/len(bl_range) * 100, 2)}%")

        n_blocks = int(np.round(100 / bl))
        blocks = np.full(n_blocks, bl)
        blocks_sum = np.cumsum(blocks)
        n_halfway = int(np.round(100 / (bl * 2)))
        axle_pos = blocks_sum[n_halfway] - 0.0000001
        ia, ib = rail_model_tc_type(tc_type=tc_type, block_length=bl, conditions="moderate", e_parallel=e_range, axle_pos_a=np.array([axle_pos]), axle_pos_b=np.array([]))
        lengths_v_currents[i] = ia[:, n_halfway]
    np.save(f"C:/Users/patters9/PycharmProjects/rail_model_v2/data/tc_type_lengths_v_currents/lengths_v_currents_{tc_type}_ws", lengths_v_currents)


def save_lengths_v_currents_thresholds_rs(tc_type):
    bl_range = np.arange(0.100001, 2.000001, 0.01)
    e_range = np.arange(-40, 40.1, 0.1)

    if tc_type.endswith("br939a"):
        drop_out = 0.055
        pick_up = 0.081
    elif tc_type.endswith("br966f2"):
        drop_out = 0.081
        pick_up = 0.120
    elif tc_type.endswith("br966f9"):
        drop_out = 0.032
        pick_up = 0.047
    else:
        raise ValueError(f"Unknown type: {tc_type}")

    lengths_v_currents = np.load(f"C:/Users/patters9/PycharmProjects/rail_model_v2/data/tc_type_lengths_v_currents/lengths_v_currents_{tc_type}_rs.npy")
    thresholds = np.empty(len(lengths_v_currents[:, 0]))
    for i in range(0, len(lengths_v_currents[:, 0])):
        currents = lengths_v_currents[i, :]
        misoperation_currents = currents[currents < drop_out]
        if len(misoperation_currents) > 0:
            threshold_current = np.max(misoperation_currents)
            tc_loc = np.where(currents == threshold_current)[0]
            threshold_e_field = e_range[tc_loc]
            thresholds[i] = threshold_e_field
        else:
            thresholds[i] = np.nan

    np.save(f"C:/Users/patters9/PycharmProjects/rail_model_v2/data/tc_type_lengths_v_currents/lengths_v_currents_thresholds_{tc_type}_rs", thresholds)


def save_lengths_v_currents_thresholds_ws(tc_type):
    bl_range = np.arange(0.100001, 2.000001, 0.01)
    e_range = np.arange(-40, 40.1, 0.1)

    if tc_type.endswith("br939a"):
        drop_out = 0.055
        pick_up = 0.081
    elif tc_type.endswith("br966f2"):
        drop_out = 0.081
        pick_up = 0.120
    elif tc_type.endswith("br966f9"):
        drop_out = 0.032
        pick_up = 0.047
    else:
        raise ValueError(f"Unknown type: {tc_type}")

    lengths_v_currents = np.load(f"../data/tc_type_lengths_v_currents/lengths_v_currents_{tc_type}_ws.npy")
    thresholds = np.empty(len(lengths_v_currents[:, 0]))
    for i in range(0, len(lengths_v_currents[:, 0])):
        currents = lengths_v_currents[i, :]
        misoperation_currents = currents[currents > pick_up]
        if len(misoperation_currents) > 0:
            threshold_current = np.min(misoperation_currents)
            tc_loc = np.where(currents == threshold_current)[0]
            threshold_e_field = e_range[tc_loc]
            thresholds[i] = threshold_e_field
        else:
            thresholds[i] = np.nan

    np.save(f"../data/tc_type_lengths_v_currents/lengths_v_currents_thresholds_{tc_type}_ws", thresholds)


def plot_lengths_v_currents_thresholds_rs():
    bl_range = np.arange(0.100001, 2.000001, 0.01)

    thresholds_br939a = np.load("../data/tc_type_lengths_v_currents/lengths_v_currents_thresholds_br867_br939a_rs.npy")
    thresholds_br966f2 = np.load(
        "../data/tc_type_lengths_v_currents/lengths_v_currents_thresholds_br867_br966f2_rs.npy")
    thresholds_br966f9 = np.load(
        "../data/tc_type_lengths_v_currents/lengths_v_currents_thresholds_br867_br966f9_rs.npy")

    index_939_under = np.where(bl_range <= 1)[0]
    index_939_over = np.where(bl_range > 1)[0]
    index_966f2_under = np.where(bl_range <= 0.68)[0]
    index_966f2_over = np.where(bl_range > 0.68)[0]
    index_966f9_under = np.where(bl_range <= 0.35)[0]
    index_966f9_over = np.where(bl_range > 0.35)[0]

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 8))

    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[:, :])

    ax0.plot(bl_range[index_939_under], thresholds_br939a[index_939_under], label="BR867 - BR939A", color="royalblue")
    ax0.plot(bl_range[index_966f2_under], thresholds_br966f2[index_966f2_under], label="BR867 - BR966 F2", color="tomato")
    ax0.plot(bl_range[index_966f9_under], thresholds_br966f9[index_966f9_under], label="BR867 - BR966 F9", color="orange")

    ax0.plot(bl_range[index_939_over], thresholds_br939a[index_939_over], linestyle="--", color="royalblue")
    ax0.plot(bl_range[index_966f2_over], thresholds_br966f2[index_966f2_over], linestyle="--", color="tomato")
    ax0.plot(bl_range[index_966f9_over], thresholds_br966f9[index_966f9_over], linestyle="--", color="orange")

    ax0.set_xlim(0, 2)
    ax0.set_ylim(0, 40)
    ax0.grid(color="grey", alpha=0.25)
    ax0.legend()
    ax0.set_ylabel("Electric Field Threshold for Right Side Failure (V/km)")
    ax0.set_xlabel("Length of Track Circuit Block (km)")
    #plt.savefig("tc_relay_types_lengths_v_currents_test_rs.pdf")
    plt.show()


def plot_lengths_v_currents_thresholds_ws():
    bl_range = np.arange(0.100001, 2.000001, 0.01)

    thresholds_br939a = np.load("../data/tc_type_lengths_v_currents/lengths_v_currents_thresholds_br867_br939a_ws.npy")
    thresholds_br966f2 = np.load(
        "../data/tc_type_lengths_v_currents/lengths_v_currents_thresholds_br867_br966f2_ws.npy")
    thresholds_br966f9 = np.load(
        "../data/tc_type_lengths_v_currents/lengths_v_currents_thresholds_br867_br966f9_ws.npy")

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(12, 8))

    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[:, :])

    index_939_under = np.where(bl_range <= 1)[0]
    index_939_over = np.where(bl_range > 1)[0]
    index_966f2_under = np.where(bl_range <= 0.68)[0]
    index_966f2_over = np.where(bl_range > 0.68)[0]
    index_966f9_under = np.where(bl_range <= 0.35)[0]
    index_966f9_over = np.where(bl_range > 0.35)[0]

    ax0.plot(bl_range[index_939_under], thresholds_br939a[index_939_under], label="BR867 - BR939A", color="royalblue")
    ax0.plot(bl_range[index_966f2_under], thresholds_br966f2[index_966f2_under], label="BR867 - BR966 F2", color="tomato")
    ax0.plot(bl_range[index_966f9_under], thresholds_br966f9[index_966f9_under], label="BR867 - BR966 F9", color="orange")

    ax0.plot(bl_range[index_939_over], thresholds_br939a[index_939_over], linestyle="--", color="royalblue")
    ax0.plot(bl_range[index_966f2_over], thresholds_br966f2[index_966f2_over], linestyle="--", color="tomato")
    ax0.plot(bl_range[index_966f9_over], thresholds_br966f9[index_966f9_over], linestyle="--", color="orange")

    ax0.set_xlim(0, 2)
    ax0.set_ylim(-30, 0)
    ax0.grid(color="grey", alpha=0.25)
    ax0.legend()
    ax0.set_ylabel("Electric Field Threshold for Wrong Side Failure (V/km)")
    ax0.set_xlabel("Length of Track Circuit Block (km)")
    #plt.savefig("tc_relay_types_lengths_v_currents_test_ws.pdf")
    plt.show()


def plot_only_middle_compare():
    ia, ib = rail_model_tc_type(tc_type="br867_br939a", block_length=0.50001, conditions="moderate", e_parallel=np.array([1]), axle_pos_a=np.array([]), axle_pos_b=np.array([]))
    ia2, ib2 = rail_model_tc_type_only_middle(tc_type="br867_br939a", block_length=0.50001, conditions="moderate", e_parallel=np.array([1]), axle_pos_a=np.array([]), axle_pos_b=np.array([]))

    n_blocks = int(np.round(100 / 0.50001))
    blocks = np.full(n_blocks, 0.50001)
    bl_range = np.cumsum(blocks)

    blocks = np.full(100, 1.00001)
    blocks[50] = 0.50001
    bl_range2 = np.cumsum(blocks)

    plt.plot(bl_range, ia[0], '.', label="All blocks are 0.5km long")
    plt.plot(bl_range2, ia2[0], '.', label="Only centre block is 0.5km long")
    plt.axhline(0.055, color="red")
    plt.axhline(0.081, color="green", linestyle="--")
    plt.ylim(0, 0.35)
    plt.xlabel("Distance along the track (km)")
    plt.ylabel("Current through relay (A)")
    plt.legend()
    plt.show()


plot_only_middle_compare()
plot_lengths_v_currents_thresholds_rs()
plot_lengths_v_currents_thresholds_ws()
