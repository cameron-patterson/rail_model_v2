import matplotlib.pyplot as plt
import numpy as np
from matplotlib.gridspec import GridSpec


def rail_model_two_track(length, e_par, axle_pos_a, axle_pos_b, relay_type, electrical_staggering, cross_bonds):
    # Define network parameters
    z_sig = 0.0289  # Signalling rail series impedance (ohms/km)
    z_trac = 0.0289  # Traction return rail series impedance (ohms/km)
    y_sig = 0.1  # Signalling rail parallel admittance for moderate conditions (siemens/km)
    i_power = 10 / 7.2  # Track circuit power supply equivalent current source (amps)
    y_power = 1 / 7.2  # Track circuit power supply admittance (siemens)
    if relay_type == "BR939A":
        y_relay = 1 / 20  # Track circuit relay admittance (siemens)
    elif relay_type == "BR966F2":
        y_relay = 1 / 9  # Track circuit relay admittance (siemens)
    elif relay_type == "BR966F9":
        y_relay = 1 / 60  # Track circuit relay admittance (siemens)
    else:
        print("Relay not recognised")
        return
    y_cb = 1 / 1e-3  # Cross bond admittance (siemens)
    y_axle = 1 / 251e-4  # Axle admittance (siemens)

    # Load in the lengths and bearings of the track circuit blocks
    # Note: zero degrees is directly northwards, with positive values increasing clockwise
    blocks = np.full(201, 0.68)
    blocks[100] = length
    bearings = np.full(201, 0)
    blocks_sum = np.cumsum(blocks)  # Cumulative sum of block lengths

    # Override E blocks to assume parallel
    print("Overriding ex_blocks and ey_blocks to assume parallel")
    ex_blocks = np.full((len(blocks), len(e_par)), e_par)
    ey_blocks = np.full((len(blocks), len(e_par)), e_par)

    # Load traction rail block leakages
    # y_trac_block = np.load(f"y_trac_block_{section_name}.npy")
    print("Overriding y_trac_block for constant traction rail leakage with only changes in the middle block")
    y_trac_block = np.full(len(blocks), 0.783)

    # Calculate the electrical characteristics of the rails
    gamma_sig = np.sqrt(z_sig * y_sig)
    gamma_trac_block = np.sqrt(z_trac * y_trac_block).flatten()
    z0_sig = np.sqrt(z_sig / y_sig)
    z0_trac_block = np.sqrt(z_trac / y_trac_block).flatten()

    # Add cross bonds and axles which split the blocks into sub blocks
    # Note: "a" and "b" are used to identify the opposite directions of travel in this network (two-track)
    if cross_bonds:
        pos_cb = np.arange(0.40001, np.sum(blocks), 0.4)  # Position of the cross bonds
    else:
        pos_cb = []
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
    ye_trac_a = 1 / (z0_trac_sub_block_a * np.sinh(
        gamma_trac_sub_block_a * trac_sub_blocks_a))  # Series admittance for traction return rail
    ye_trac_b = 1 / (z0_trac_sub_block_b * np.sinh(gamma_trac_sub_block_b * trac_sub_blocks_b))
    yg_trac = (np.cosh(gamma_trac_block * blocks) - 1) * (1 / (
                z0_trac_block * np.sinh(gamma_trac_block * blocks)))  # Parallel admittance for traction return rail

    ye_sig_a = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))  # Series admittance for signalling rail
    ye_sig_b = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))
    yg_sig = (np.cosh(gamma_sig * blocks) - 1) * (
                1 / (z0_sig * np.sinh(gamma_sig * blocks)))  # Parallel admittance for signalling rail
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
    # n_nodes_sig_a = len(sig_sub_block_sum_a)  # Number of nodes in the signalling rail
    # n_nodes_sig_b = len(sig_sub_block_sum_b)

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
    y_sum[first_trac_a] = yg_trac_comb[0] + y_relay + ye_trac_a[locs_first_trac_a]
    # Axles
    locs_axle_trac_a = np.where(np.isin(node_locs_trac_a, node_locs_axle_trac_a))[0]
    y_sum[node_locs_axle_trac_a] = y_axle + ye_trac_a[locs_axle_trac_a - 1] + ye_trac_a[locs_axle_trac_a]
    # Cross bonds
    locs_cb_a = np.where(np.isin(node_locs_trac_a, node_locs_cb_a))[0]
    y_sum[node_locs_cb_a] = y_cb + ye_trac_a[locs_cb_a - 1] + ye_trac_a[locs_cb_a]
    # Middle nodes
    indices_other_node_trac_a = node_locs_trac_a[1:-1][~np.logical_or(np.isin(node_locs_trac_a[1:-1], node_locs_axle_trac_a), np.isin(node_locs_trac_a[1:-1], node_locs_cb_a))]
    mask_other_trac_a = np.isin(indices_other_node_trac_a, node_locs_trac_a)
    other_trac_a = indices_other_node_trac_a[mask_other_trac_a]
    locs_other_trac_a = np.where(np.isin(node_locs_trac_a, other_trac_a))[0]
    y_sum[other_trac_a] = yg_trac_comb[1:-1] + y_power + y_relay + ye_trac_a[locs_other_trac_a - 1] + ye_trac_a[locs_other_trac_a]
    # Last node
    mask_last_trac_a = np.isin(node_locs_trac_a, node_locs_trac_a[-1])
    last_trac_a = node_locs_trac_a[mask_last_trac_a]
    locs_last_trac_a = np.where(np.isin(node_locs_trac_a, last_trac_a))[0]
    y_sum[last_trac_a] = yg_trac_comb[-1] + y_power + ye_trac_a[locs_last_trac_a - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_relay_sig_a))[0]
    y_sum[node_locs_relay_sig_a] = yg_sig + y_relay + ye_sig_a[locs_relay_sig_a]
    # Power nodes
    locs_power_sig_a = np.where(np.isin(node_locs_sig_a, node_locs_power_sig_a))[0]
    y_sum[node_locs_power_sig_a] = yg_sig + y_power + ye_sig_a[locs_power_sig_a - 1]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_a, node_locs_axle_sig_a))[0]
    y_sum[node_locs_axle_sig_a] = y_axle + ye_sig_a[axle_locs - 1] + ye_sig_a[axle_locs]
    # Direction "b" second
    # Traction return rail
    # First node
    mask_first_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[0])
    first_trac_b = node_locs_trac_b[mask_first_trac_b]
    locs_first_trac_b = np.where(np.isin(node_locs_trac_b, first_trac_b))[0]
    y_sum[first_trac_b] = yg_trac_comb[0] + y_relay + ye_trac_b[locs_first_trac_b]
    # Axles
    locs_axle_trac_b = np.where(np.isin(node_locs_trac_b, node_locs_axle_trac_b))[0]
    y_sum[node_locs_axle_trac_b] = y_axle + ye_trac_b[locs_axle_trac_b - 1] + ye_trac_b[locs_axle_trac_b]
    # Cross bonds
    locs_cb_b = np.where(np.isin(node_locs_trac_b, node_locs_cb_b))[0]
    y_sum[node_locs_cb_b] = y_cb + ye_trac_b[locs_cb_b - 1] + ye_trac_b[locs_cb_b]
    # Middle nodes
    indices_other_node_trac_b = node_locs_trac_b[1:-1][~np.logical_or(np.isin(node_locs_trac_b[1:-1], node_locs_axle_trac_b), np.isin(node_locs_trac_b[1:-1], node_locs_cb_b))]
    mask_other_trac_b = np.isin(indices_other_node_trac_b, node_locs_trac_b)
    other_trac_b = indices_other_node_trac_b[mask_other_trac_b]
    locs_other_trac_b = np.where(np.isin(node_locs_trac_b, other_trac_b))[0]
    y_sum[other_trac_b] = yg_trac_comb[1:-1] + y_power + y_relay + ye_trac_b[locs_other_trac_b - 1] + ye_trac_b[locs_other_trac_b]
    # Last node
    mask_last_trac_b = np.isin(node_locs_trac_b, node_locs_trac_b[-1])
    last_trac_b = node_locs_trac_b[mask_last_trac_b]
    locs_last_trac_b = np.where(np.isin(node_locs_trac_b, last_trac_b))[0]
    y_sum[last_trac_b] = yg_trac_comb[-1] + y_power + ye_trac_b[locs_last_trac_b - 1]
    # Signalling rail
    # Relay nodes
    locs_relay_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_relay_sig_b))[0]
    y_sum[node_locs_relay_sig_b] = yg_sig + y_relay + ye_sig_b[locs_relay_sig_b - 1]
    # Power nodes
    locs_power_sig_b = np.where(np.isin(node_locs_sig_b, node_locs_power_sig_b))[0]
    y_sum[node_locs_power_sig_b] = yg_sig + y_power + ye_sig_b[locs_power_sig_b]
    # Axle nodes
    axle_locs = np.where(np.isin(node_locs_sig_b, node_locs_axle_sig_b))[0]
    y_sum[node_locs_axle_sig_b] = y_axle + ye_sig_b[axle_locs - 1] + ye_sig_b[axle_locs]

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
    y_matrix[node_locs_relay_trac_a, node_locs_relay_sig_a] = -y_relay
    y_matrix[node_locs_relay_sig_a, node_locs_relay_trac_a] = -y_relay
    y_matrix[node_locs_relay_trac_b, node_locs_relay_sig_b] = -y_relay
    y_matrix[node_locs_relay_sig_b, node_locs_relay_trac_b] = -y_relay
    # Power admittances
    y_matrix[node_locs_power_trac_a, node_locs_power_sig_a] = -y_power
    y_matrix[node_locs_power_sig_a, node_locs_power_trac_a] = -y_power
    y_matrix[node_locs_power_trac_b, node_locs_power_sig_b] = -y_power
    y_matrix[node_locs_power_sig_b, node_locs_power_trac_b] = -y_power
    # Cross bond admittances
    y_matrix[node_locs_cb_a, node_locs_cb_b] = -y_cb
    y_matrix[node_locs_cb_b, node_locs_cb_a] = -y_cb
    # Axle admittances
    y_matrix[node_locs_axle_trac_a, node_locs_axle_sig_a] = -y_axle
    y_matrix[node_locs_axle_sig_a, node_locs_axle_trac_a] = -y_axle
    y_matrix[node_locs_axle_trac_b, node_locs_axle_sig_b] = -y_axle
    y_matrix[node_locs_axle_sig_b, node_locs_axle_trac_b] = -y_axle

    y_matrix[np.isnan(y_matrix)] = 0

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
    i_sig_a = e_par_sig_a / z_sig
    i_trac_a = e_par_trac_a / z_trac

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
    i_sig_b = e_par_sig_b / z_sig
    i_trac_b = e_par_trac_b / z_trac

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
    j_matrix[:, non_cb_axle_node_locs_centre_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices] - i_power
    # Traction return rail last node
    j_matrix[:, node_locs_trac_a[-1]] = i_trac_a[:, -1] - i_power

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
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, whole_blocks_end))[0]]] = i_sig_a[:, whole_blocks] + i_power
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_start))[0]]] = -i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_start))[0]]
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_end))[0]]] = i_sig_a[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1))[0]]] + i_power
    j_matrix[:, node_locs_sig_a[np.where(np.isin(node_locs_sig_a, split_blocks_mid))[0]]] = i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1] - i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0]]
    # "b" second
    # Traction return rail first node
    j_matrix[:, node_locs_trac_b[0]] = i_trac_b[:, 0] - i_power
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
    j_matrix[:, non_cb_axle_node_locs_centre_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1] - i_power
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
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_start))[0]]] = i_sig_b[:, whole_blocks] + i_power
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, whole_blocks_end))[0]]] = -i_sig_b[:, whole_blocks]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_start))[0]]] = i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_start))[0]] + i_power
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_end))[0]]] = -i_sig_b[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1))[0]]]
    j_matrix[:, node_locs_sig_b[np.where(np.isin(node_locs_sig_b, split_blocks_mid))[0]]] = -i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1] + i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0]]

    # Electrically stagger the power supplies
    if electrical_staggering:
        node_locs_power_trac_a_down = node_locs_power_trac_a[1::2]
        node_locs_power_sig_a_down = node_locs_power_sig_a[1::2]
        node_locs_power_trac_b_down = np.flip(np.flip(node_locs_power_trac_b)[1::2])
        node_locs_power_sig_b_down = np.flip(np.flip(node_locs_power_sig_b)[1::2])

        j_matrix[:, node_locs_power_trac_a_down] = j_matrix[:, node_locs_power_trac_a_down] + (2 * i_power)
        j_matrix[:, node_locs_power_sig_a_down] = j_matrix[:, node_locs_power_sig_a_down] - (2 * i_power)
        j_matrix[:, node_locs_power_trac_b_down] = j_matrix[:, node_locs_power_trac_b_down] + (2 * i_power)
        j_matrix[:, node_locs_power_sig_b_down] = j_matrix[:, node_locs_power_sig_b_down] - (2 * i_power)
    else:
        pass

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

    i_relays_a = v_relay_a * y_relay
    i_relays_b = v_relay_b * y_relay

    i_relays_a = i_relays_a.T
    i_relays_b = i_relays_b.T

    return i_relays_a, i_relays_b


def plot_length_changes(section_name):
    data = np.load(f"../data/rail_data/{section_name}/{section_name}_distances_bearings.npz")
    blocks = data["distances"]

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(10, 5))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[0])
    ax0.plot(blocks, '.')
    ax0.set_xlim(-5, len(blocks)+4)

    if section_name == "glasgow_edinburgh_falkirk":
        ax0.set_title("Glasgow to Edinburgh via Falkirk High")
    elif section_name == "west_coast_main_line":
        ax0.set_title("West Coast Main Line")
    elif section_name == "east_coast_main_line":
        ax0.set_title("East Coast Main Line")

    ax0.set_ylabel("Block Length (km)")
    ax0.set_xlabel("Block Index")

    plt.show()
    #plt.savefig(f"block_lengths_{section_name}")


def length_currents_0():
    lengths = np.linspace(0.1, 1, 10)
    ia_mid = np.empty(len(lengths))
    for i in range(0, len(lengths)):
        ia, ib = rail_model_two_track(length=lengths[i], e_par=np.array([0]), axle_pos_a=np.array([]), axle_pos_b=np.array([]), relay_type="BR939A", electrical_staggering=False, cross_bonds=True)
        ia_mid[i] = ia[:, 100]
        print(i)

    plt.plot(lengths, ia_mid)
    plt.show()


def length_e_field_rs():
    e_fields = np.linspace(-20, 20, 401)
    lengths939 = np.array([0.25, 0.5, 1, 2])
    lengths966 = np.array([0.17, 0.34, 0.68, 1.36])

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])

    for i in range(0, len(lengths939)):
        ia939, ib939 = rail_model_two_track(length=lengths939[i], e_par=e_fields, axle_pos_a=np.array([]), axle_pos_b=np.array([]), relay_type="BR939A", electrical_staggering=False, cross_bonds=True)
        ax0.plot(e_fields, ia939[:, 100], label=f'{lengths939[i]} km')
        ax0.axhline(0.055, color='tomato', linestyle='-')
        ax0.set_title('BR939A Relay')

        ia966, ib966 = rail_model_two_track(length=lengths966[i], e_par=e_fields, axle_pos_a=np.array([]), axle_pos_b=np.array([]), relay_type="BR966F2", electrical_staggering=False, cross_bonds=True)
        ax1.plot(e_fields, ia966[:, 100], label=f'{lengths966[i]} km')
        ax1.axhline(0.081, color='tomato', linestyle='-')
        ax1.set_title('BR966 F2 Relay')

    def ax_add(ax):
        ax.set_xlabel('Electric Field Strength (V/km)')
        ax.set_ylabel('Current Through Relay (A)')
        ax.set_xlim(-20, 20)
        ax.legend()

    ax_add(ax0)
    ax_add(ax1)

    plt.show()


def length_e_field_ws():
    e_fields = np.linspace(-10, 10, 201)
    lengths939 = np.array([0.25, 0.5, 1, 2])
    lengths966 = np.array([0.17, 0.34, 0.68, 1.36])
    axle_spacing = np.array([0, 2, 17.5, 19.5, 24, 26, 41.5, 43.5, 48, 50, 65.5, 67.5, 72, 74, 89.5, 91.5]) / 1000

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 7))
    gs = GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[:, 0])
    ax1 = fig.add_subplot(gs[:, 1])

    for i in range(0, len(lengths939)):
        axle_pos_939 = axle_spacing + 68 + (lengths939[i] / 2)
        ia939, ib939 = rail_model_two_track(length=lengths939[i], e_par=e_fields, axle_pos_a=axle_pos_939, axle_pos_b=np.array([]), relay_type="BR939A", electrical_staggering=False, cross_bonds=True)
        ax0.plot(e_fields, ia939[:, 100], label=f'{lengths939[i]} km')
        ax0.axhline(0.081, color='lime', linestyle='--')
        ax0.axhline(-0.081, color='lime', linestyle='--')
        ax0.set_title('BR939A Relay')

        axle_pos_966 = axle_spacing + 68 + (lengths966[i] / 2)
        ia966, ib966 = rail_model_two_track(length=lengths966[i], e_par=e_fields, axle_pos_a=axle_pos_966, axle_pos_b=np.array([]), relay_type="BR966F2", electrical_staggering=False, cross_bonds=True)
        ax1.plot(e_fields, ia966[:, 100], label=f'{lengths966[i]} km')
        ax1.axhline(0.12, color='lime', linestyle='--')
        ax1.axhline(-0.12, color='lime', linestyle='--')
        ax1.set_title('BR966 F2 Relay')

    def ax_add(ax):
        ax.set_xlabel('Electric Field Strength (V/km)')
        ax.set_ylabel('Current Through Relay (A)')
        ax.set_xlim(-10, 10)
        ax.legend()

    ax_add(ax0)
    ax_add(ax1)

    plt.show()


length_e_field_ws()

