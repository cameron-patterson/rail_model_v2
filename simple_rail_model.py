import numpy as np
import matplotlib.pyplot as plt


# Inputs
# section_name - Name of the section to run the model for. Can be "glasgow_edinburgh_falkirk", "east_coast_main_line", or "west_coast_main_line".
# ex - The array of ex values for the chosen MT site. Measured in V/km.
# ey - The array of ey values for the chosen MT site. Measured in V/km.
# axle_pos_a - The locations of any trains in one direction of travel. Measured in km along the line. Leave as empty, i.e., "axle_pos_a = []" for no trains.
# axle_pos_b - The locations of any trains in the opposite direction of travel. Measured in km along the line. Leave as empty, i.e., "axle_pos_b = []" for no trains.

# Returns
# ia - The current across each track circuit relay in one direction of travel for each electric field value. Returns an array that is of size [number of electric field values X number of blocks].
# ib - The current across each track circuit relay in the opposite direction of travel for each electric field value. Returns an array that is of size [number of electric field values X number of blocks].
def rail_model_two_track(section_name, ex, ey, axle_pos_a, axle_pos_b):
    # Create dictionary of network parameters
    parameters = {"z_sig": 0.0289,  # Signalling rail series impedance (ohms/km)
                  "z_trac": 0.0289,  # Traction return rail series impedance (ohms/km)
                  "y_sig_moderate": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_moderate": 1.6,  # Traction return rail parallel admittance in moderate conditions (siemens/km)
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
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig_moderate"])
    gamma_trac = np.sqrt(parameters["z_trac"] * parameters["y_trac_moderate"])
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig_moderate"])
    z0_trac = np.sqrt(parameters["z_trac"] / parameters["y_trac_moderate"])

    # Load in the lengths and bearings of the track circuit blocks
    # Note: zero degrees is directly northwards, with positive values increasing clockwise
    data = np.load(f"data/rail_data/{section_name}/{section_name}_distances_bearings.npz")
    blocks = data["distances"]
    bearings = data["bearings"]
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
    y_sum[first_trac_a] = yg[first_trac_a] + parameters["y_relay"] + ye_trac_a[locs_first_trac_a]
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
    y_sum[other_trac_a] = yg[other_trac_a] + parameters["y_power"] + parameters["y_relay"] + ye_trac_a[locs_other_trac_a - 1] + ye_trac_a[locs_other_trac_a]
    # Last node
    mask_last_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[-1])
    last_trac_a = node_locs_trac_a[mask_last_trac_a]
    locs_last_trac_a = np.where(np.isin(node_locs_trac_a, last_trac_a))[0]
    yg[last_trac_a] = 0.5 * yg_trac_a[locs_last_trac_a - 1]
    y_sum[last_trac_a] = yg[last_trac_a] + parameters["y_power"] + ye_trac_a[locs_last_trac_a - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_relay_sig_a))[0]
    yg[node_locs_relay_sig_a] = 0.5 * yg_sig_a[locs_relay_sig_a]
    y_sum[node_locs_relay_sig_a] = yg[node_locs_relay_sig_a] + parameters["y_relay"] + ye_sig_a[locs_relay_sig_a]
    # Power nodes
    locs_power_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]
    yg[node_locs_power_sig_a] = 0.5 * yg_sig_a[locs_power_sig_a - 1]
    y_sum[node_locs_power_sig_a] = yg[node_locs_power_sig_a] + parameters["y_power"] + ye_sig_a[locs_power_sig_a - 1]
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
    y_sum[first_trac_b] = yg[first_trac_b] + parameters["y_relay"] + ye_trac_b[locs_first_trac_b]
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
    y_sum[other_trac_b] = yg[other_trac_b] + parameters["y_power"] + parameters["y_relay"] + ye_trac_b[locs_other_trac_b - 1] + ye_trac_b[locs_other_trac_b]
    # Last node
    mask_last_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[-1])
    last_trac_b = node_locs_trac_b[mask_last_trac_b]
    locs_last_trac_b = np.where(np.isin(node_locs_trac_b, last_trac_b))[0]
    yg[last_trac_b] = 0.5 * yg_trac_b[locs_last_trac_b - 1]
    y_sum[last_trac_b] = yg[last_trac_b] + parameters["y_power"] + ye_trac_b[locs_last_trac_b - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]
    yg[node_locs_relay_sig_b] = 0.5 * yg_sig_b[locs_relay_sig_b - 1]
    y_sum[node_locs_relay_sig_b] = yg[node_locs_relay_sig_b] + parameters["y_relay"] + ye_sig_b[locs_relay_sig_b - 1]
    # Power nodes
    locs_power_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_power_sig_b))[0]
    yg[node_locs_power_sig_b] = 0.5 * yg_sig_b[locs_power_sig_b]
    y_sum[node_locs_power_sig_b] = yg[node_locs_power_sig_b] + parameters["y_power"] + ye_sig_b[locs_power_sig_b]
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

    # Define ex_blocks and ey_blocks
    ex_blocks = np.empty((len(trac_sub_blocks_a), len(ex)))
    ey_blocks = np.empty((len(trac_sub_blocks_a), len(ey)))
    for i in range(0, len(trac_sub_blocks_a)):
        ex_blocks[i] = ex
        ey_blocks[i] = ey

    # Restructure angles array based on the new sub blocks
    bearings_a = bearings
    bearings_b = (bearings + np.pi) % (2*np.pi)
    block_indices_trac_a = np.searchsorted(blocks_sum, np.delete(trac_sub_block_sum_a, 0, 0))
    trac_sub_blocks_angles_a = bearings_a[block_indices_trac_a]
    trac_sub_blocks_ex_a = ex_blocks[block_indices_trac_a]
    trac_sub_blocks_ey_a = ey_blocks[block_indices_trac_a]
    block_indices_sig_a = np.searchsorted(blocks_sum, np.delete(sig_sub_block_sum_a, locs_relay_sig_a))
    sig_sub_block_angles_a = bearings_a[block_indices_sig_a]
    sig_sub_blocks_ex_a = ex_blocks[block_indices_sig_a]
    sig_sub_blocks_ey_a = ey_blocks[block_indices_sig_a]
    block_indices_trac_b = np.searchsorted(blocks_sum, np.delete(trac_sub_block_sum_b, 0, 0))
    trac_sub_blocks_angles_b = bearings_b[block_indices_trac_b]
    trac_sub_blocks_ex_b = ex_blocks[block_indices_trac_b]
    trac_sub_blocks_ey_b = ey_blocks[block_indices_trac_b]
    block_indices_sig_b = np.searchsorted(blocks_sum, np.delete(sig_sub_block_sum_b, locs_power_sig_b))
    sig_sub_block_angles_b = bearings_b[block_indices_sig_b]
    sig_sub_blocks_ex_b = ex_blocks[block_indices_sig_b]
    sig_sub_blocks_ey_b = ey_blocks[block_indices_sig_b]

    # Currents
    # Set up current matrix
    j_matrix = np.zeros([len(ex_blocks[0, :]), n_nodes])

    # "a" first
    trac_sb_angles_a_broadcasted = trac_sub_blocks_angles_a[:, np.newaxis]
    e_x_par_trac_a = trac_sub_blocks_ex_a * np.cos(trac_sb_angles_a_broadcasted)
    e_x_par_trac_a = e_x_par_trac_a.T
    e_y_par_trac_a = trac_sub_blocks_ey_a * np.sin(trac_sb_angles_a_broadcasted)
    e_y_par_trac_a = e_y_par_trac_a.T
    e_par_trac_a = e_x_par_trac_a + e_y_par_trac_a
    sig_sb_angles_a_broadcasted = sig_sub_block_angles_a[:, np.newaxis]
    e_x_par_sig_a = sig_sub_blocks_ex_a * np.cos(sig_sb_angles_a_broadcasted)
    e_x_par_sig_a = e_x_par_sig_a.T
    e_y_par_sig_a = sig_sub_blocks_ey_a * np.sin(sig_sb_angles_a_broadcasted)
    e_y_par_sig_a = e_y_par_sig_a.T
    e_par_sig_a = e_x_par_sig_a + e_y_par_sig_a
    i_sig_a = e_par_sig_a / parameters["z_sig"]
    i_trac_a = e_par_trac_a / parameters["z_trac"]

    # "b" second
    trac_sb_angles_b_broadcasted = trac_sub_blocks_angles_b[:, np.newaxis]
    e_x_par_trac_b = trac_sub_blocks_ex_b * np.cos(trac_sb_angles_b_broadcasted)
    e_x_par_trac_b = e_x_par_trac_b.T
    e_y_par_trac_b = trac_sub_blocks_ey_b * np.sin(trac_sb_angles_b_broadcasted)
    e_y_par_trac_b = e_y_par_trac_b.T
    e_par_trac_b = e_x_par_trac_b + e_y_par_trac_b
    sig_sb_angles_b_broadcasted = sig_sub_block_angles_b[:, np.newaxis]
    e_x_par_sig_b = sig_sub_blocks_ex_b * np.cos(sig_sb_angles_b_broadcasted)
    e_x_par_sig_b = e_x_par_sig_b.T
    e_y_par_sig_b = sig_sub_blocks_ey_b * np.sin(sig_sb_angles_b_broadcasted)
    e_y_par_sig_b = e_y_par_sig_b.T
    e_par_sig_b = e_x_par_sig_b + e_y_par_sig_b
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


# Example 1
# Run the model for the Glasgow to Edinburgh via Falkirk High line, an electric field of ex = -0.65 and ey = 1.4 at time index 0, and ex = 1.2 and ey = -2.1 at time index 1. No trains.
ia, ib = rail_model_two_track(section_name="glasgow_edinburgh_falkirk", ex=[-0.65, 1.2], ey=[1.4, -2.1], axle_pos_a=[], axle_pos_b=[])

# Plot the currents across the relays in a single direction of travel for time index 0 as points
plt.plot(ia[0], '.')

# Plot horizontal lines to indicate the relay operational current thresholds.
# 0.073 is the pick up current, the value the current needs to be above to re-energise the relay from a de-energised state.
# Plot as a dashed green line.
plt.axhline(0.073, linestyle='--', color='green')

# 0.050 is the drop out current, the value the currents needs to drop below to de-energise the relay from an energised state.
# Plot as a solid red line.
# If any of the currents in blocks that contain no trains drop below this line, it means the signal is misoperating.
plt.axhline(0.050, color='red')

# Set axis labels
plt.xlabel("Track circuit block")
plt.ylabel("Current through relay (A)")

plt.show()

# Example 2
# Run the model for the Glasgow to Edinburgh via Falkirk High line, an electric field of ex = -0.65 and ey = 1.4 at time index 0, and ex = 1.2 and ey = -2.1 at time index 1. No trains.
ia, ib = rail_model_two_track(section_name="glasgow_edinburgh_falkirk", ex=[-0.65, 1.2, 2.4], ey=[1.4, -2.1, -1], axle_pos_a=[], axle_pos_b=[])

# Create a loot to plot the currents across the relays in a single direction of travel for each time index as points
for i in range(0, len(ia[:,0])):
    plt.plot(ia[i], '.')

    # Plot horizontal lines to indicate the relay operational current thresholds.
    # 0.073 is the pick up current, the value the current needs to be above to re-energise the relay from a de-energised state.
    # Plot as a dashed green line.
    plt.axhline(0.073, linestyle='--', color='green')

    # 0.050 is the drop out current, the value the currents needs to drop below to de-energise the relay from an energised state.
    # Plot as a solid red line.
    # If any of the currents in blocks that contain no trains drop below this line, it means the signal is misoperating.
    plt.axhline(0.050, color='red')

    # Set axis labels
    plt.xlabel("Track circuit block")
    plt.ylabel("Current through relay (A)")

    plt.show()

# Example 3
# Run the model for the Glasgow to Edinburgh via Falkirk High line, an electric field of ex = -0.65 and ey = 1.4 at time index 0, and ex = 1.2 and ey = -2.1 at time index 1. No trains.
ia, ib = rail_model_two_track(section_name="glasgow_edinburgh_falkirk", ex=[-0.65, 1.2, 2.4], ey=[1.4, -2.1, -1], axle_pos_a=[], axle_pos_b=[])

# Set up a subplot with 2 rows and 1 column
# "fig" will control the whole figure
# "ax" will allow us to plot in separate panels
fig, ax = plt.subplots(2, 1)

# Create a loot to plot the currents across the relays in both directions of travel for each time index as points in different panels
for i in range(0, len(ia[:, 0])):
    ax[0].plot(ia[i], '.')
    ax[1].plot(ib[i], '.')

    # Plot horizontal lines to indicate the relay operational current thresholds.
    # 0.073 is the pick up current, the value the current needs to be above to re-energise the relay from a de-energised state.
    # Plot as a dashed green line.
    ax[0].axhline(0.073, linestyle='--', color='green')
    ax[1].axhline(0.073, linestyle='--', color='green')

    # 0.050 is the drop out current, the value the currents needs to drop below to de-energise the relay from an energised state.
    # Plot as a solid red line.
    # If any of the currents in blocks that contain no trains drop below this line, it means the signal is misoperating.
    ax[0].axhline(0.050, color='red')
    ax[1].axhline(0.050, color='red')

    # Set axis labels
    # Notice that now we are using subplots, it's ".set_xlabel", rather than just ".xlabel"
    ax[0].set_xlabel("Track circuit block")
    ax[0].set_ylabel("Current through relay (A)")
    ax[1].set_xlabel("Track circuit block")
    ax[1].set_ylabel("Current through relay (A)")

    plt.show()
