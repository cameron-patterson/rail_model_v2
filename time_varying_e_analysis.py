import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def wrong_side_two_track_currents_e_blocks(section_name, conditions, storm, axle_pos_a, axle_pos_b):
    # Create dictionary of network parameters
    parameters = {"z_sig": 0.0289,  # Signalling rail series impedance (ohms/km)
                  "z_trac": 0.0289,  # Traction return rail series impedance (ohms/km)
                  "y_sig_moderate": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_moderate": 1.6,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "y_sig_dry": 0.025,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_dry": 1.53,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "y_sig_wet": 0.4,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_wet": 2,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
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

    # Load in network sub block lengths and angles for the chosen section
    sub_block_lengths = np.load("data\\network_parameters\\" + section_name + "\\sub_blocks_" + section_name + ".npz")
    trac_node_positions = sub_block_lengths["blocks_sum_cb"]  # Positions of nodes along the traction return rail (km)
    sig_block_lengths = sub_block_lengths["sig_sub_blocks"]  # Lengths of the signalling rail for each block (km)
    block_angles = np.load("data\\network_parameters\\" + section_name + "\\angles_" + section_name + ".npz")
    block_angles_a = block_angles["sig_angles_a"]  # Angles of each block in the "a" direction
    block_angles_b = block_angles["sig_angles_b"]  # Angles of each block in the "b" direction (180 degrees from "a")

    # Load in network nodal indices for the chosen section
    network_nodes = np.load("data\\network_parameters\\" + section_name + "\\nodes_" + section_name + ".npz")
    n_nodes = network_nodes["n_nodes"]  # Total number of nodes in the section
    trac_node_indices_a = network_nodes["trac_node_locs_a"]  # Traction return rail node indices for "a" direction
    trac_node_indices_b = network_nodes["trac_node_locs_b"]  # Traction return rail node indices for "b" direction
    sig_node_indices_a = network_nodes["sig_node_locs_a"]  # Signalling rail node indices for "a" direction
    sig_node_indices_b = network_nodes["sig_node_locs_b"]  # Signalling rail node indices for "b" direction
    cb_node_indices_a = network_nodes["cb_node_locs_a"]  # Cross bond node indices for "a" direction
    cb_node_indices_b = network_nodes["cb_node_locs_b"]  # Cross bond node indices for "b" direction
    # trac_node_indices_power_a = network_nodes["trac_node_locs_power_a"]  # Traction return rail relay node indices for "a" direction
    # trac_node_indices_power_b = network_nodes["trac_node_locs_power_b"]  # Traction return rail relay node indices for "b" direction
    trac_node_indices_relay_a = network_nodes["trac_node_locs_relay_a"]  # Traction return rail relay node indices for "a" direction
    trac_node_indices_relay_b = network_nodes["trac_node_locs_relay_b"]  # Traction return rail relay node indices for "b" direction
    sig_node_indices_power_a = network_nodes["sig_node_locs_power_a"]  # Traction return rail power supply node indices for "a" direction
    sig_node_indices_power_b = network_nodes["sig_node_locs_power_b"]  # Traction return rail power supply node indices for "b" direction
    sig_node_indices_relay_a = network_nodes["sig_node_locs_relay_a"]  # Signalling rail relay node indices for "a" direction
    sig_node_indices_relay_b = network_nodes["sig_node_locs_relay_b"]  # Signalling rail relay node indices for "b" direction

    # Calculate the electrical characteristics of the rails
    gamma_trac = np.sqrt(parameters["z_trac"] * parameters["y_trac_" + conditions])  # Propagation constant of the traction return rail (ohms)
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig_" + conditions])  # Propagation constant of the signalling rail (ohms)
    z0_trac = np.sqrt(parameters["z_trac"] / parameters["y_trac_" + conditions])  # Characteristic impedance of the traction return rail (km^-1)
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig_" + conditions])  # Characteristic impedance of the signalling rail (km^-1)

    # Calculate the distance along the rail of nodes on the signalling rail
    sig_sub_blocks_sums = np.cumsum(sig_block_lengths)  # Cumulative sum of signalling blocks lengths
    sig_sub_blocks_sums_zero = np.hstack((0, sig_sub_blocks_sums))  # Cumulative sum of signalling blocks lengths starting with 0
    sig_node_positions = np.hstack((sig_sub_blocks_sums_zero[0], np.repeat(sig_sub_blocks_sums_zero[1:-1], 2), sig_sub_blocks_sums_zero[-1]))  # Distance along the rail for the signalling rail nodes

    # Calculate axle node indices for trains in the "a" direction
    if len(axle_pos_a) != 0:
        axle_node_positions_a = np.array(
            sorted(axle_pos_a.flatten()))  # Sort axle node positions of "a" and convert to numpy array
        starting_node = sig_node_indices_b[-1] + 1  # Starting node index for the traction return rail of "a"
        trac_node_indices_axle_a = np.arange(starting_node, starting_node + len(
            axle_node_positions_a))  # Nodal indices for axles on the traction return rail of "a"
        starting_node = trac_node_indices_axle_a[-1] + 1  # Starting node index for the signalling rail of "a"
        sig_axle_node_indices_a = np.arange(starting_node, starting_node + len(
            axle_node_positions_a))  # Nodal indices for axles on the signalling rail of "a"
    else:
        pass

    # Calculate axle node indices for trains in the "b" direction
    if len(axle_pos_b) != 0:
        axle_node_positions_b = np.array(
            sorted(axle_pos_b.flatten()))  # Sort axle node positions of "b" and convert to numpy array
        if len(axle_pos_a) != 0:
            starting_node = sig_axle_node_indices_a[-1] + 1  # Starting node index if there are trains on "a"
        else:
            starting_node = sig_node_indices_b[-1] + 1  # Starting node index if there are no trains on "a"
        trac_node_indices_axle_b = np.arange(starting_node, starting_node + len(
            axle_node_positions_b))  # Nodal indices for axles on the traction return rail of "b"
        starting_node = trac_node_indices_axle_b[-1] + 1  # Starting node index for the signalling rail of "b"
        sig_axle_node_indices_b = np.arange(starting_node, starting_node + len(
            axle_node_positions_b))  # Nodal indices for axles on the signalling rail of "b"
    else:
        pass

    # Combine the nodal positions and indices for the rails and the axles into single arrays
    if len(axle_pos_a) != 0 and len(axle_pos_b) != 0:  # If there are trains in both "a" and "b"
        all_trac_node_positions_a = np.hstack((trac_node_positions, axle_node_positions_a))
        all_sig_node_positions_a = np.hstack((sig_node_positions, axle_node_positions_a))
        all_trac_node_positions_b = np.hstack((trac_node_positions, axle_node_positions_b))
        all_sig_node_positions_b = np.hstack((sig_node_positions, axle_node_positions_b))
        all_trac_node_indices_a = np.hstack((trac_node_indices_a, trac_node_indices_axle_a))
        all_sig_node_indices_a = np.hstack((sig_node_indices_a, sig_axle_node_indices_a))
        all_trac_node_indices_b = np.hstack((trac_node_indices_b, trac_node_indices_axle_b))
        all_sig_node_indices_b = np.hstack((sig_node_indices_b, sig_axle_node_indices_b))

    elif len(axle_pos_a) != 0 and len(axle_pos_b) == 0:  # If there are trains in "a", but not in "b"
        all_trac_node_positions_a = np.hstack((trac_node_positions, axle_node_positions_a))
        all_sig_node_positions_a = np.hstack((sig_node_positions, axle_node_positions_a))
        all_trac_node_positions_b = np.copy(trac_node_positions)
        all_sig_node_positions_b = np.copy(sig_node_positions)
        all_trac_node_indices_a = np.hstack((trac_node_indices_a, trac_node_indices_axle_a))
        all_sig_node_indices_a = np.hstack((sig_node_indices_a, sig_axle_node_indices_a))
        all_trac_node_indices_b = np.copy(trac_node_indices_b)
        all_sig_node_indices_b = np.copy(sig_node_indices_b)

    else:  # If there are trains in "b", but not in "a"
        all_trac_node_positions_a = np.copy(trac_node_positions)
        all_sig_node_positions_a = np.copy(sig_node_positions)
        all_trac_node_positions_b = np.hstack((trac_node_positions, axle_node_positions_b))
        all_sig_node_positions_b = np.hstack((sig_node_positions, axle_node_positions_b))
        all_trac_node_indices_a = np.copy(trac_node_indices_a)
        all_sig_node_indices_a = np.copy(sig_node_indices_a)
        all_trac_node_indices_b = np.hstack((trac_node_indices_b, trac_node_indices_axle_b))
        all_sig_node_indices_b = np.hstack((sig_node_indices_b, sig_axle_node_indices_b))

    # Create dictionaries of nodal indices and positions
    trac_a_dict = dict(zip(all_trac_node_indices_a, all_trac_node_positions_a))
    sig_a_dict = dict(zip(all_sig_node_indices_a, all_sig_node_positions_a))
    trac_b_dict = dict(zip(all_trac_node_indices_b, all_trac_node_positions_b))
    sig_b_dict = dict(zip(all_sig_node_indices_b, all_sig_node_positions_b))

    # Sort the dictionaries based on positions
    sorted_trac_a_dict = dict(sorted(trac_a_dict.items(), key=lambda item: item[1]))
    sorted_sig_a_dict = dict(sorted(sig_a_dict.items(), key=lambda item: item[1]))
    sorted_trac_b_dict = dict(sorted(trac_b_dict.items(), key=lambda item: item[1]))
    sorted_sig_b_dict = dict(sorted(sig_b_dict.items(), key=lambda item: item[1]))

    # Save the sorted indices as new arrays
    all_trac_node_indices_a = np.array(list(sorted_trac_a_dict.keys()))
    all_sig_node_indices_a = np.array(list(sorted_sig_a_dict.keys()))
    all_trac_node_indices_b = np.array(list(sorted_trac_b_dict.keys()))
    all_sig_node_indices_b = np.array(list(sorted_sig_b_dict.keys()))

    # Make a new zeroed admittance matrix for the new restructured network
    n_nodes_restructured = len(all_trac_node_indices_a) + len(all_sig_node_indices_a) + len(
        all_trac_node_indices_b) + len(all_sig_node_indices_b)
    y_matrix_restructured = np.zeros((n_nodes_restructured, n_nodes_restructured))

    # Load in the nodal admittance matrix of the original network
    y_matrix = np.load(
        "data\\network_parameters\\" + section_name + "\\nodal_admittance_matrix_" + section_name + "_" + conditions + ".npy")

    # Place values from the original network into the restructured network
    y_matrix_restructured[0:n_nodes, 0:n_nodes] = y_matrix

    # Find which nodes need to be calculated or recalculated
    # "a" first
    # Traction return rail
    # Find the indices of the subset elements in the original array
    sub_set_indices = np.where(np.isin(all_trac_node_indices_a, trac_node_indices_axle_a))[0]
    # Create a mask for the neighbours
    mask = np.zeros(len(all_trac_node_indices_a), dtype=bool)
    mask[sub_set_indices] = True
    # Add neighbours to the mask
    mask[np.maximum(sub_set_indices - 1, 0)] = True
    mask[np.minimum(sub_set_indices + 1, len(all_trac_node_indices_a) - 1)] = True
    # Extract the values from the original array using the mask
    recalculate_trac_node_a = all_trac_node_indices_a[mask]
    # Signalling rail
    # Find the indices of the subset elements in the original array
    sub_set_indices = np.where(np.isin(all_sig_node_indices_a, sig_axle_node_indices_a))[0]
    # Create a mask for the neighbours
    mask = np.zeros(len(all_sig_node_indices_a), dtype=bool)
    mask[sub_set_indices] = True
    # Add neighbours to the mask
    mask[np.maximum(sub_set_indices - 1, 0)] = True
    mask[np.minimum(sub_set_indices + 1, len(all_sig_node_indices_a) - 1)] = True
    # Extract the values from the original array using the mask
    recalculate_sig_node_a = all_sig_node_indices_a[mask]
    # "b" second
    # Traction return rail
    # Find the indices of the subset elements in the original array
    sub_set_indices = np.where(np.isin(all_trac_node_indices_b, trac_node_indices_axle_b))[0]
    # Create a mask for the neighbours
    mask = np.zeros(len(all_trac_node_indices_b), dtype=bool)
    mask[sub_set_indices] = True
    # Add neighbours to the mask
    mask[np.maximum(sub_set_indices - 1, 0)] = True
    mask[np.minimum(sub_set_indices + 1, len(all_trac_node_indices_b) - 1)] = True
    # Extract the values from the original array using the mask
    recalculate_trac_node_b = all_trac_node_indices_b[mask]
    # Signalling rail
    # Find the indices of the subset elements in the original array
    sub_set_indices = np.where(np.isin(all_sig_node_indices_b, sig_axle_node_indices_b))[0]
    # Create a mask for the neighbours
    mask = np.zeros(len(all_sig_node_indices_b), dtype=bool)
    mask[sub_set_indices] = True
    # Add neighbours to the mask
    mask[np.maximum(sub_set_indices - 1, 0)] = True
    mask[np.minimum(sub_set_indices + 1, len(all_sig_node_indices_b) - 1)] = True
    # Extract the values from the original array using the mask
    recalculate_sig_node_b = all_sig_node_indices_b[mask]

    # Recalculate the equivalent pi-circuit parameters for the new network
    # Get the sorted nodal positions
    all_trac_node_positions_sorted_a = list(sorted_trac_a_dict.values())
    all_sig_node_positions_sorted_a = list(sorted_sig_a_dict.values())
    all_trac_node_positions_sorted_b = list(sorted_trac_b_dict.values())
    all_sig_node_positions_sorted_b = list(sorted_sig_b_dict.values())

    # Calculate the length of the sub blocks
    trac_sub_blocks_a = np.diff(all_trac_node_positions_sorted_a)
    sig_sub_blocks_a = np.diff(all_sig_node_positions_sorted_a)
    sig_sub_blocks_a[
        sig_sub_blocks_a == 0] = np.nan  # Sub blocks with length zero on the signalling rail indicate calculating rail joints, these need to be nans
    trac_sub_blocks_b = np.diff(all_trac_node_positions_sorted_b)
    sig_sub_blocks_b = np.diff(all_sig_node_positions_sorted_b)
    sig_sub_blocks_b[sig_sub_blocks_b == 0] = np.nan

    # Set up equivalent pi-parameters
    # "a" first
    ye_sig_a = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))  # Series admittance for signalling rail
    ye_trac_a = 1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_a))  # Series admittance for traction return rail
    yg_sig_a = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_a) - 1) * (
                1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))))  # Parallel admittance for signalling rail
    yg_trac_a = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_a) - 1) * (1 / (
                z0_trac * np.sinh(gamma_trac * trac_sub_blocks_a))))  # Parallel admittance for traction return rail
    # "b" second
    ye_sig_b = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))  # Series admittance for signalling rail
    ye_trac_b = 1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_b))  # Series admittance for traction return rail
    yg_sig_b = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_b) - 1) * (
                1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))))  # Parallel admittance for signalling rail
    yg_trac_b = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_b) - 1) * (1 / (
                z0_trac * np.sinh(gamma_trac * trac_sub_blocks_b))))  # Parallel admittance for traction return rail

    # Recalculate nodal parallel admittances and compute new sums
    # "a" first
    # Traction return rail
    recalculate_yg_trac_a = np.full(len(all_trac_node_indices_a), -1).astype(
        float)  # Array of parallel admittances to place recalculated values in, with negative values to filter out unused cells later
    recalculate_y_sum_trac_a = np.full(len(all_trac_node_indices_a), -1).astype(
        float)  # Array of sum of admittances into the node to place recalculated values in, with negative values to filter out unused cells later
    mask_first = np.isin(recalculate_trac_node_a, all_trac_node_indices_a[
        0])  # Mask to determine if the first traction rail node needs to be recalculated
    if np.any(mask_first):
        first = recalculate_trac_node_a[mask_first]  # Index of the nodes to be recalculated
        first_locs = np.where(np.isin(all_trac_node_indices_a, first))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[first_locs] = 0.5 * yg_trac_a[first_locs]  # New nodal parallel admittance
        recalculate_y_sum_trac_a[first_locs] = recalculate_yg_trac_a[first_locs] + parameters["y_relay"] + ye_trac_a[
            first_locs]
    else:
        pass
    mask_axle = np.isin(trac_node_indices_axle_a,
                        recalculate_trac_node_a)  # Mask to determine if any traction return rail axle nodes needs to be recalculated
    if np.any(mask_axle):
        axle = trac_node_indices_axle_a[mask_axle]  # Index of the nodes to be recalculated
        axle_locs = np.where(np.isin(all_trac_node_indices_a, axle))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[axle_locs] = (0.5 * yg_trac_a[axle_locs - 1]) + (
                    0.5 * yg_trac_a[axle_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_a[axle_locs] = recalculate_yg_trac_a[axle_locs] + parameters["y_axle"] + ye_trac_a[
            axle_locs - 1] + ye_trac_a[axle_locs]
    else:
        pass
    mask_cb = np.isin(cb_node_indices_a,
                      recalculate_trac_node_a)  # Mask to determine if any traction return rail cb nodes needs to be recalculated
    if np.any(mask_cb):
        cb = cb_node_indices_a[mask_cb]  # Index of the nodes to be recalculated
        cb_locs = np.where(np.isin(all_trac_node_indices_a, cb))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[cb_locs] = (0.5 * yg_trac_a[cb_locs - 1]) + (
                    0.5 * yg_trac_a[cb_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_a[cb_locs] = recalculate_yg_trac_a[cb_locs] + parameters["y_cb"] + ye_trac_a[
            cb_locs - 1] + ye_trac_a[cb_locs]
    else:
        pass
    other_node_indices = all_trac_node_indices_a[1:-2][
        ~np.logical_or(np.isin(all_trac_node_indices_a[1:-2], trac_node_indices_axle_a),
                       np.isin(all_trac_node_indices_a[1:-2], cb_node_indices_a))]
    mask_other = np.isin(other_node_indices,
                         recalculate_trac_node_a)  # Mask to determine if any other traction return rail nodes needs to be recalculated
    if np.any(mask_other):
        other = other_node_indices[mask_other]  # Index of the nodes to be recalculated
        other_locs = np.where(np.isin(all_trac_node_indices_a, other))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[other_locs] = (0.5 * yg_trac_a[other_locs - 1]) + (
                    0.5 * yg_trac_a[other_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_a[other_locs] = recalculate_yg_trac_a[other_locs] + parameters["y_power"] + parameters[
            "y_relay"] + ye_trac_a[other_locs - 1] + ye_trac_a[other_locs]
    else:
        pass
    mask_last = np.isin(recalculate_trac_node_a, all_trac_node_indices_a[
        -1])  # Mask to determine if the last traction rail node needs to be recalculated
    if np.any(mask_last):
        last = recalculate_trac_node_a[mask_last]  # Index of the nodes to be recalculated
        last_locs = np.where(np.isin(all_trac_node_indices_a, last))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[last_locs] = 0.5 * yg_trac_a[last_locs - 1]  # New nodal parallel admittance
        recalculate_y_sum_trac_a[last_locs] = recalculate_yg_trac_a[last_locs] + parameters["y_power"] + ye_trac_a[
            last_locs - 1]
    else:
        pass
    recalculate_yg_trac_a = recalculate_yg_trac_a[recalculate_yg_trac_a != -1]  # Unused cells removed
    recalculate_y_sum_trac_a = recalculate_y_sum_trac_a[recalculate_y_sum_trac_a != -1]  # Unused cells removed

    # Signalling rail
    recalculate_yg_sig_a = np.full(len(all_sig_node_indices_a), -1).astype(
        float)  # Array of parallel admittances to place recalculated values in, with negative values to filter out unused cells later
    recalculate_y_sum_sig_a = np.full(len(all_sig_node_indices_a), -1).astype(
        float)  # Array of sum of admittances into the node to place recalculated values in, with negative values to filter out unused cells later
    mask_relay = np.isin(sig_node_indices_relay_a,
                         recalculate_sig_node_a)  # Mask to determine if any signalling rail relay nodes needs to be recalculated
    if np.any(mask_relay):
        relay = sig_node_indices_relay_a[mask_relay]  # Index of the nodes to be recalculated
        relay_locs = np.where(np.isin(all_sig_node_indices_a, relay))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_a[relay_locs] = 0.5 * yg_sig_a[relay_locs]  # New nodal parallel admittance
        recalculate_y_sum_sig_a[relay_locs] = recalculate_yg_sig_a[relay_locs] + parameters["y_relay"] + ye_sig_a[
            relay_locs]
    else:
        pass
    mask_power = np.isin(sig_node_indices_power_a,
                         recalculate_sig_node_a)  # Mask to determine if any signalling rail power nodes needs to be recalculated
    if np.any(mask_power):
        power = sig_node_indices_power_a[mask_power]  # Index of the nodes to be recalculated
        power_locs = np.where(np.isin(all_sig_node_indices_a, power))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_a[power_locs] = 0.5 * yg_sig_a[power_locs - 1]  # New nodal parallel admittance
        recalculate_y_sum_sig_a[power_locs] = recalculate_yg_sig_a[power_locs] + parameters["y_power"] + ye_sig_a[
            power_locs - 1]
    else:
        pass
    other_node_indices = all_sig_node_indices_a[
        ~np.logical_or(np.isin(all_sig_node_indices_a, sig_node_indices_relay_a),
                       np.isin(all_sig_node_indices_a, sig_node_indices_power_a))]
    mask_other = np.isin(other_node_indices,
                         recalculate_sig_node_a)  # Mask to determine if any other signalling rail nodes needs to be recalculated
    if np.any(mask_other):
        other = other_node_indices[mask_other]  # Index of the nodes to be recalculated
        other_locs = np.where(np.isin(all_sig_node_indices_a, other))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_a[other_locs] = (0.5 * yg_sig_a[other_locs - 1]) + (
                    0.5 * yg_sig_a[other_locs])  # New nodal parallel admittance
        recalculate_y_sum_sig_a[other_locs] = recalculate_yg_sig_a[other_locs] + parameters["y_axle"] + ye_sig_a[
            other_locs - 1] + ye_sig_a[other_locs]
    else:
        pass
    recalculate_yg_sig_a = recalculate_yg_sig_a[recalculate_yg_sig_a != -1]  # Unused cells removed
    recalculate_y_sum_sig_a = recalculate_y_sum_sig_a[recalculate_y_sum_sig_a != -1]  # Unused cells removed

    # "b" second
    # Traction return rail
    recalculate_yg_trac_b = np.full(len(all_trac_node_indices_b), -1).astype(
        float)  # Array of parallel admittances to place recalculated values in, with negative values to filter out unused cells later
    recalculate_y_sum_trac_b = np.full(len(all_trac_node_indices_b), -1).astype(
        float)  # Array of sum of admittances into the node to place recalculated values in, with negative values to filter out unused cells later
    mask_first = np.isin(recalculate_trac_node_b, all_trac_node_indices_b[
        0])  # Mask to determine if the first traction rail node needs to be recalculated
    if np.any(mask_first):
        first = recalculate_trac_node_b[mask_first]  # Index of the nodes to be recalculated
        first_locs = np.where(np.isin(all_trac_node_indices_b, first))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[first_locs] = 0.5 * yg_trac_b[first_locs]  # New nodal parallel admittance
        recalculate_y_sum_trac_b[first_locs] = recalculate_yg_trac_b[first_locs] + parameters["y_power"] + ye_trac_b[
            first_locs]
    else:
        pass
    mask_axle = np.isin(trac_node_indices_axle_b,
                        recalculate_trac_node_b)  # Mask to determine if any traction return rail axle nodes needs to be recalculated
    if np.any(mask_axle):
        axle = trac_node_indices_axle_b[mask_axle]  # Index of the nodes to be recalculated
        axle_locs = np.where(np.isin(all_trac_node_indices_b, axle))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[axle_locs] = (0.5 * yg_trac_b[axle_locs - 1]) + (
                    0.5 * yg_trac_b[axle_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_b[axle_locs] = recalculate_yg_trac_b[axle_locs] + parameters["y_axle"] + ye_trac_b[
            axle_locs - 1] + ye_trac_b[axle_locs]
    else:
        pass
    mask_cb = np.isin(cb_node_indices_b,
                      recalculate_trac_node_b)  # Mask to determine if any traction return rail cb nodes needs to be recalculated
    if np.any(mask_cb):
        cb = cb_node_indices_b[mask_cb]  # Index of the nodes to be recalculated
        cb_locs = np.where(np.isin(all_trac_node_indices_b, cb))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[cb_locs] = (0.5 * yg_trac_b[cb_locs - 1]) + (
                    0.5 * yg_trac_b[cb_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_b[cb_locs] = recalculate_yg_trac_b[cb_locs] + parameters["y_cb"] + ye_trac_b[
            cb_locs - 1] + ye_trac_b[cb_locs]
    else:
        pass
    other_node_indices = all_trac_node_indices_b[1:-2][
        ~np.logical_or(np.isin(all_trac_node_indices_b[1:-2], trac_node_indices_axle_b),
                       np.isin(all_trac_node_indices_b[1:-2], cb_node_indices_b))]
    mask_other = np.isin(other_node_indices,
                         recalculate_trac_node_b)  # Mask to determine if any other traction return rail nodes needs to be recalculated
    if np.any(mask_other):
        other = other_node_indices[mask_other]  # Index of the nodes to be recalculated
        other_locs = np.where(np.isin(all_trac_node_indices_b, other))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[other_locs] = (0.5 * yg_trac_b[other_locs - 1]) + (
                    0.5 * yg_trac_b[other_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_b[other_locs] = recalculate_yg_trac_b[other_locs] + parameters["y_power"] + parameters[
            "y_relay"] + ye_trac_b[other_locs - 1] + ye_trac_b[other_locs]
    else:
        pass
    mask_last = np.isin(recalculate_trac_node_b, all_trac_node_indices_b[
        -1])  # Mask to determine if the last traction rail node needs to be recalculated
    if np.any(mask_last):
        last = recalculate_trac_node_b[mask_last]  # Index of the nodes to be recalculated
        last_locs = np.where(np.isin(all_trac_node_indices_b, last))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[last_locs] = 0.5 * yg_trac_b[last_locs - 1]  # New nodal parallel admittance
        recalculate_y_sum_trac_b[last_locs] = recalculate_yg_trac_b[last_locs] + parameters["y_relay"] + ye_trac_b[
            last_locs - 1]
    else:
        pass
    recalculate_yg_trac_b = recalculate_yg_trac_b[recalculate_yg_trac_b != -1]  # Unused cells removed
    recalculate_y_sum_trac_b = recalculate_y_sum_trac_b[recalculate_y_sum_trac_b != -1]  # Unused cells removed
    # Signalling rail
    recalculate_yg_sig_b = np.full(len(all_sig_node_indices_b), -1).astype(
        float)  # Array of parallel admittances to place recalculated values in, with negative values to filter out unused cells later
    recalculate_y_sum_sig_b = np.full(len(all_sig_node_indices_b), -1).astype(
        float)  # Array of sum of admittances into the node to place recalculated values in, with negative values to filter out unused cells later
    mask_relay = np.isin(sig_node_indices_relay_b,
                         recalculate_sig_node_b)  # Mask to determine if any signalling rail relay nodes needs to be recalculated
    if np.any(mask_relay):
        relay = sig_node_indices_relay_b[mask_relay]  # Index of the nodes to be recalculated
        relay_locs = np.where(np.isin(all_sig_node_indices_b, relay))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_b[relay_locs] = 0.5 * yg_sig_b[relay_locs - 1]  # New nodal parallel admittance
        recalculate_y_sum_sig_b[relay_locs] = recalculate_yg_sig_b[relay_locs] + parameters["y_relay"] + ye_sig_b[
            relay_locs - 1]
    else:
        pass
    mask_power = np.isin(sig_node_indices_power_b,
                         recalculate_sig_node_b)  # Mask to determine if any signalling rail power nodes needs to be recalculated
    if np.any(mask_power):
        power = sig_node_indices_power_b[mask_power]  # Index of the nodes to be recalculated
        power_locs = np.where(np.isin(all_sig_node_indices_b, power))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_b[power_locs] = 0.5 * yg_sig_b[power_locs]  # New nodal parallel admittance
        recalculate_y_sum_sig_b[power_locs] = recalculate_yg_sig_b[power_locs] + parameters["y_power"] + ye_sig_b[
            power_locs]
    else:
        pass
    other_node_indices = all_sig_node_indices_b[
        ~np.logical_or(np.isin(all_sig_node_indices_b, sig_node_indices_relay_b),
                       np.isin(all_sig_node_indices_b, sig_node_indices_power_b))]
    mask_other = np.isin(other_node_indices,
                         recalculate_sig_node_b)  # Mask to determine if any other signalling rail nodes needs to be recalculated
    if np.any(mask_other):
        other = other_node_indices[mask_other]  # Index of the nodes to be recalculated
        other_locs = np.where(np.isin(all_sig_node_indices_b, other))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_b[other_locs] = (0.5 * yg_sig_b[other_locs - 1]) + (
                    0.5 * yg_sig_b[other_locs])  # New nodal parallel admittance
        recalculate_y_sum_sig_b[other_locs] = recalculate_yg_sig_b[other_locs] + parameters["y_axle"] + ye_sig_b[
            other_locs - 1] + ye_sig_b[other_locs]
    else:
        pass
    recalculate_yg_sig_b = recalculate_yg_sig_b[recalculate_yg_sig_b != -1]  # Unused cells removed
    recalculate_y_sum_sig_b = recalculate_y_sum_sig_b[recalculate_y_sum_sig_b != -1]  # Unused cells removed

    # Update the reshaped nodal admittance matrix
    # Replace the diagonal values
    # "a" first
    y_matrix_restructured[recalculate_trac_node_a, recalculate_trac_node_a] = recalculate_y_sum_trac_a
    y_matrix_restructured[recalculate_sig_node_a, recalculate_sig_node_a] = recalculate_y_sum_sig_a
    # "b" second
    y_matrix_restructured[recalculate_trac_node_b, recalculate_trac_node_b] = recalculate_y_sum_trac_b
    y_matrix_restructured[recalculate_sig_node_b, recalculate_sig_node_b] = recalculate_y_sum_sig_b

    # Series admittances between nodes
    # "a" first
    i = np.where(np.isin(all_trac_node_indices_a, recalculate_trac_node_a[:-1]))[0]
    y_matrix_restructured[np.array(all_trac_node_indices_a)[i], np.array(all_trac_node_indices_a)[i + 1]] = \
    y_matrix_restructured[np.array(all_trac_node_indices_a)[i + 1], np.array(all_trac_node_indices_a)[i]] = -ye_trac_a[
        i]
    i = np.where(np.isin(all_sig_node_indices_a, recalculate_sig_node_a[:-1]))[0]
    y_matrix_restructured[np.array(all_sig_node_indices_a)[i], np.array(all_sig_node_indices_a)[i + 1]] = \
    y_matrix_restructured[np.array(all_sig_node_indices_a)[i + 1], np.array(all_sig_node_indices_a)[i]] = -ye_sig_a[i]
    # "b" second
    i = np.where(np.isin(all_trac_node_indices_b, recalculate_trac_node_b[:-1]))[0]
    y_matrix_restructured[np.array(all_trac_node_indices_b)[i], np.array(all_trac_node_indices_b)[i + 1]] = \
    y_matrix_restructured[np.array(all_trac_node_indices_b)[i + 1], np.array(all_trac_node_indices_b)[i]] = -ye_trac_b[
        i]
    i = np.where(np.isin(all_sig_node_indices_b, recalculate_sig_node_b[:-1]))[0]
    y_matrix_restructured[np.array(all_sig_node_indices_b)[i], np.array(all_sig_node_indices_b)[i + 1]] = \
    y_matrix_restructured[np.array(all_sig_node_indices_b)[i + 1], np.array(all_sig_node_indices_b)[i]] = -ye_sig_b[i]
    y_matrix_restructured[np.isnan(y_matrix_restructured)] = 0  # Set any NaN values from the IRJs to 0

    # Axle admittances
    # "a" first
    y_matrix_restructured[trac_node_indices_axle_a, sig_axle_node_indices_a] = -parameters["y_axle"]
    y_matrix_restructured[sig_axle_node_indices_a, trac_node_indices_axle_a] = -parameters["y_axle"]
    # "b" second
    y_matrix_restructured[trac_node_indices_axle_b, sig_axle_node_indices_b] = -parameters["y_axle"]
    y_matrix_restructured[sig_axle_node_indices_b, trac_node_indices_axle_b] = -parameters["y_axle"]

    # Restructure angles array based on the new sub blocks
    # "a" first
    cumsum_trac_sb_a = np.cumsum(trac_sub_blocks_a)
    block_indices_trac_a = np.searchsorted(sig_sub_blocks_sums, cumsum_trac_sb_a)
    trac_sb_angles_a = block_angles_a[block_indices_trac_a]
    cumsum_sig_sb_a = np.cumsum(sig_sub_blocks_a[~np.isnan(sig_sub_blocks_a)])
    block_indices_sig_a = np.searchsorted(sig_sub_blocks_sums, cumsum_sig_sb_a)
    sig_sb_angles_a = block_angles_a[block_indices_sig_a]
    # "b" second
    cumsum_trac_sb_b = np.cumsum(trac_sub_blocks_b)
    block_indices_trac_b = np.searchsorted(sig_sub_blocks_sums, cumsum_trac_sb_b)
    trac_sb_angles_b = block_angles_b[block_indices_trac_b]
    cumsum_sig_sb_b = np.cumsum(sig_sub_blocks_b[~np.isnan(sig_sub_blocks_b)])
    block_indices_sig_b = np.searchsorted(sig_sub_blocks_sums, cumsum_sig_sb_b)
    sig_sb_angles_b = block_angles_b[block_indices_sig_b]

    # Load in storm e_field data
    storm_es = np.load('data/storm_e_fields/bgs_' + storm + '/' + section_name + '_' + storm + '_e_blocks.npz')
    ex_blocks = storm_es['ex_blocks']/1000
    ey_blocks = storm_es['ey_blocks']/1000

    _, counts = np.unique(trac_sb_angles_a, return_counts=True)
    trac_sb_ex_a = np.repeat(ex_blocks, counts, axis=0)
    trac_sb_ey_a = np.repeat(ey_blocks, counts, axis=0)
    _, counts = np.unique(sig_sb_angles_a, return_counts=True)
    sig_sb_ex_a = np.repeat(ex_blocks, counts, axis=0)
    sig_sb_ey_a = np.repeat(ey_blocks, counts, axis=0)
    _, counts = np.unique(trac_sb_angles_b, return_counts=True)
    trac_sb_ex_b = np.repeat(ex_blocks, counts, axis=0)
    trac_sb_ey_b = np.repeat(ey_blocks, counts, axis=0)
    _, counts = np.unique(sig_sb_angles_b, return_counts=True)
    sig_sb_ex_b = np.repeat(ex_blocks, counts, axis=0)
    sig_sb_ey_b = np.repeat(ey_blocks, counts, axis=0)

    # Currents
    # Set up current matrix
    j_matrix = np.zeros([len(ex_blocks[0, :]), n_nodes_restructured])

    # "a" first
    trac_sb_angles_a_broadcasted = trac_sb_angles_a[:, np.newaxis]
    e_x_par_trac_a = trac_sb_ex_a * np.cos(trac_sb_angles_a_broadcasted)
    e_x_par_trac_a = e_x_par_trac_a.T
    e_y_par_trac_a = trac_sb_ey_a * np.sin(trac_sb_angles_a_broadcasted)
    e_y_par_trac_a = e_y_par_trac_a.T
    e_par_trac_a = e_x_par_trac_a + e_y_par_trac_a
    sig_sb_angles_a_broadcasted = sig_sb_angles_a[:, np.newaxis]
    e_x_par_sig_a = sig_sb_ex_a * np.cos(sig_sb_angles_a_broadcasted)
    e_x_par_sig_a = e_x_par_sig_a.T
    e_y_par_sig_a = sig_sb_ey_a * np.sin(sig_sb_angles_a_broadcasted)
    e_y_par_sig_a = e_y_par_sig_a.T
    e_par_sig_a = e_x_par_sig_a + e_y_par_sig_a
    i_sig_a = e_par_sig_a / parameters["z_sig"]
    i_trac_a = e_par_trac_a / parameters["z_trac"]

    # "b" second
    trac_sb_angles_b_broadcasted = trac_sb_angles_b[:, np.newaxis]
    e_x_par_trac_b = trac_sb_ex_b * np.cos(trac_sb_angles_b_broadcasted)
    e_x_par_trac_b = e_x_par_trac_b.T
    e_y_par_trac_b = trac_sb_ey_b * np.sin(trac_sb_angles_b_broadcasted)
    e_y_par_trac_b = e_y_par_trac_b.T
    e_par_trac_b = e_x_par_trac_b + e_y_par_trac_b
    sig_sb_angles_b_broadcasted = sig_sb_angles_b[:, np.newaxis]
    e_x_par_sig_b = sig_sb_ex_b * np.cos(sig_sb_angles_b_broadcasted)
    e_x_par_sig_b = e_x_par_sig_b.T
    e_y_par_sig_b = sig_sb_ey_b * np.sin(sig_sb_angles_b_broadcasted)
    e_y_par_sig_b = e_y_par_sig_b.T
    e_par_sig_b = e_x_par_sig_b + e_y_par_sig_b
    i_sig_b = e_par_sig_b / parameters["z_sig"]
    i_trac_b = e_par_trac_b / parameters["z_trac"]

    # "a" first
    # Traction return rail first node
    j_matrix[:, all_trac_node_indices_a[0]] = -i_trac_a[:, 0]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(all_trac_node_indices_a, cb_node_indices_a)
    indices = np.where(mask)[0]
    j_matrix[:, cb_node_indices_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Axle nodes
    mask = np.isin(all_trac_node_indices_a, trac_node_indices_axle_a)
    indices = np.where(mask)[0]
    j_matrix[:, trac_node_indices_axle_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Non-cross bond or axle nodes
    mask = np.isin(all_trac_node_indices_a, cb_node_indices_a) | np.isin(all_trac_node_indices_a, trac_node_indices_axle_a)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(all_trac_node_indices_a, cb_node_indices_a) & ~np.isin(all_trac_node_indices_a, trac_node_indices_axle_a)
    non_cb_axle_node_locs_centre_a = all_trac_node_indices_a[mask_del][1:-1]
    j_matrix[:, non_cb_axle_node_locs_centre_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices] - parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, all_trac_node_indices_a[-1]] = i_trac_a[:, -1] - parameters["i_power"]
    # Signalling rail nodes
    sig_relay_axle = all_sig_node_indices_a[np.where(~np.isin(all_sig_node_indices_a, sig_node_indices_power_a))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_relay_axle, sig_axle_node_indices_a))[0], np.where(np.isin(sig_relay_axle, sig_axle_node_indices_a))[0] - 1)))
    all_blocks = range(0, len(i_sig_a[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_relay_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_relay_axle[np.where(~np.isin(sig_relay_axle, sig_axle_node_indices_a) & ~np.isin(sig_relay_axle, whole_blocks_start))[0]]
    split_blocks_end = split_blocks_start + 1
    split_blocks_mid = sig_relay_axle[np.where(np.isin(sig_relay_axle, sig_axle_node_indices_a))[0]]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, whole_blocks_start))[0]]] = -i_sig_a[:, whole_blocks]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, whole_blocks_end))[0]]] = i_sig_a[:, whole_blocks] + parameters["i_power"]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, split_blocks_start))[0]]] = -i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_start))[0]]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, split_blocks_end))[0]]] = i_sig_a[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1))[0]]] + parameters["i_power"]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, split_blocks_mid))[0]]] = i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1] - i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0]]

    # "b" second
    # Traction return rail first node
    j_matrix[:, all_trac_node_indices_b[0]] = i_trac_b[:, 0] - parameters["i_power"]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(all_trac_node_indices_b, cb_node_indices_b)
    indices = np.where(mask)[0]
    j_matrix[:, cb_node_indices_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Axle nodes
    mask = np.isin(all_trac_node_indices_b, trac_node_indices_axle_b)
    indices = np.where(mask)[0]
    j_matrix[:, trac_node_indices_axle_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Non-cross bond or axle nodes
    mask = np.isin(all_trac_node_indices_b, cb_node_indices_b) | np.isin(all_trac_node_indices_b,
                                                                         trac_node_indices_axle_b)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(all_trac_node_indices_b, cb_node_indices_b) & ~np.isin(all_trac_node_indices_b,
                                                                               trac_node_indices_axle_b)
    non_cb_axle_node_locs_centre_b = all_trac_node_indices_b[mask_del][1:-1]
    j_matrix[:, non_cb_axle_node_locs_centre_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1] - parameters[
        "i_power"]
    # Traction return rail last node
    j_matrix[:, all_trac_node_indices_b[-1]] = -i_trac_b[:, -1]
    # Signalling rail nodes
    sig_power_axle = all_sig_node_indices_b[np.where(~np.isin(all_sig_node_indices_b, sig_node_indices_relay_b))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_power_axle, sig_axle_node_indices_b))[0],
                                               np.where(np.isin(sig_power_axle, sig_axle_node_indices_b))[0] - 1)))
    all_blocks = range(0, len(i_sig_b[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_power_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_power_axle[
        np.where(~np.isin(sig_power_axle, sig_axle_node_indices_b) & ~np.isin(sig_power_axle, whole_blocks_start))[0]]
    split_blocks_end = split_blocks_start + 1
    split_blocks_mid = sig_power_axle[np.where(np.isin(sig_power_axle, sig_axle_node_indices_b))[0]]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, whole_blocks_start))[0]]] = i_sig_b[:, whole_blocks] + parameters["i_power"]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, whole_blocks_end))[0]]] = -i_sig_b[:, whole_blocks]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, split_blocks_start))[0]]] = i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_start))[0]] + parameters["i_power"]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, split_blocks_end))[0]]] = i_sig_b[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1))[0]]]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, split_blocks_mid))[0]]] = -i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1] + i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0]]

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix_restructured)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    # Calculate relay voltages and currents
    # "a" first
    v_relay_top_node_a = v_matrix[sig_node_indices_relay_a]
    v_relay_bottom_node_a = v_matrix[trac_node_indices_relay_a]
    v_relay_a = v_relay_top_node_a - v_relay_bottom_node_a

    # "b" first
    v_relay_top_node_b = v_matrix[sig_node_indices_relay_b]
    v_relay_bottom_node_b = v_matrix[trac_node_indices_relay_b]
    v_relay_b = v_relay_top_node_b - v_relay_bottom_node_b

    i_relays_a = v_relay_a / parameters["r_relay"]
    i_relays_b = v_relay_b / parameters["r_relay"]

    i_relays_a = i_relays_a.T
    i_relays_b = i_relays_b.T

    return i_relays_a, i_relays_b


def right_side_two_track_currents_e_blocks_timetable(section_name, conditions, ex_blocks, ey_blocks):
    # Create dictionary of network parameters
    parameters = {"z_sig": 0.0289,  # Signalling rail series impedance (ohms/km)
                  "z_trac": 0.0289,  # Traction return rail series impedance (ohms/km)
                  "y_sig_moderate": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_moderate": 1.6,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "y_sig_dry": 0.025,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_dry": 1.53,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "y_sig_wet": 0.4,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_wet": 2,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "v_power": 10,  # Track circuit power supply voltage (volts)
                  "r_power": 7.2,  # Track circuit power supply resistance (ohms)
                  "r_relay": 20,  # Track circuit relay resistance (ohms)
                  "r_cb": 1e-3,  # Cross bond resistance (ohms)
                  "r_shunt": 251e-4,  # Shunt (axle) resistance (ohms)
                  "i_power": 10 / 7.2,  # Track circuit power supply equivalent current source (amps)
                  "y_power": 1 / 7.2,  # Track circuit power supply admittance (siemens)
                  "y_relay": 1 / 20,  # Track circuit relay admittance (siemens)
                  "y_cb": 1 / 1e-3,  # Cross bond admittance (siemens)
                  "y_shunt": 1 / 251e-4}  # Shunt (axle) admittance (siemens)

    # Load in section network sub block angles
    sub_block_angles = np.load("data\\network_parameters\\" + section_name + "\\angles_" + section_name + ".npz")
    trac_angles_a = sub_block_angles["trac_angles_a"]
    trac_angles_b = sub_block_angles["trac_angles_b"]
    sig_angles_a = sub_block_angles["sig_angles_a"]
    sig_angles_b = sub_block_angles["sig_angles_b"]

    # Load in section network node indices
    network_nodes = np.load("data\\network_parameters\\" + section_name + "\\nodes_" + section_name + ".npz")
    n_nodes = network_nodes["n_nodes"]
    # n_nodes_trac = network_nodes["n_nodes_trac"]
    trac_node_locs_a = network_nodes["trac_node_locs_a"]
    trac_node_locs_b = network_nodes["trac_node_locs_b"]
    # sig_node_locs_a = network_nodes["sig_node_locs_a"]
    # sig_node_locs_b = network_nodes["sig_node_locs_b"]
    cb_node_locs_a = network_nodes["cb_node_locs_a"]
    cb_node_locs_b = network_nodes["cb_node_locs_b"]
    trac_node_locs_relay_a = network_nodes["trac_node_locs_relay_a"]
    trac_node_locs_relay_b = network_nodes["trac_node_locs_relay_b"]
    sig_node_locs_power_a = network_nodes["sig_node_locs_power_a"]
    sig_node_locs_power_b = network_nodes["sig_node_locs_power_b"]
    sig_node_locs_relay_a = network_nodes["sig_node_locs_relay_a"]
    sig_node_locs_relay_b = network_nodes["sig_node_locs_relay_b"]

    _, counts = np.unique(trac_angles_a, return_counts=True)
    trac_sb_ex_a = np.repeat(ex_blocks, counts, axis=0)
    trac_sb_ey_a = np.repeat(ey_blocks, counts, axis=0)
    _, counts = np.unique(sig_angles_a, return_counts=True)
    sig_sb_ex_a = np.repeat(ex_blocks, counts, axis=0)
    sig_sb_ey_a = np.repeat(ey_blocks, counts, axis=0)
    _, counts = np.unique(trac_angles_b, return_counts=True)
    trac_sb_ex_b = np.repeat(ex_blocks, counts, axis=0)
    trac_sb_ey_b = np.repeat(ey_blocks, counts, axis=0)
    _, counts = np.unique(sig_angles_b, return_counts=True)
    sig_sb_ex_b = np.repeat(ex_blocks, counts, axis=0)
    sig_sb_ey_b = np.repeat(ey_blocks, counts, axis=0)

    # Currents
    # Set up current matrix
    j_matrix = np.zeros([len(ex_blocks[0, :]), n_nodes])

    # "a" first
    trac_sb_angles_a_broadcasted = trac_angles_a[:, np.newaxis]
    e_x_par_trac_a = trac_sb_ex_a * np.cos(trac_sb_angles_a_broadcasted)
    e_x_par_trac_a = e_x_par_trac_a.T
    e_y_par_trac_a = trac_sb_ey_a * np.sin(trac_sb_angles_a_broadcasted)
    e_y_par_trac_a = e_y_par_trac_a.T
    e_par_trac_a = e_x_par_trac_a + e_y_par_trac_a
    sig_sb_angles_a_broadcasted = sig_angles_a[:, np.newaxis]
    e_x_par_sig_a = sig_sb_ex_a * np.cos(sig_sb_angles_a_broadcasted)
    e_x_par_sig_a = e_x_par_sig_a.T
    e_y_par_sig_a = sig_sb_ey_a * np.sin(sig_sb_angles_a_broadcasted)
    e_y_par_sig_a = e_y_par_sig_a.T
    e_par_sig_a = e_x_par_sig_a + e_y_par_sig_a
    i_sig_a = e_par_sig_a / parameters["z_sig"]
    i_trac_a = e_par_trac_a / parameters["z_trac"]

    # "b" second
    trac_sb_angles_b_broadcasted = trac_angles_b[:, np.newaxis]
    e_x_par_trac_b = trac_sb_ex_b * np.cos(trac_sb_angles_b_broadcasted)
    e_x_par_trac_b = e_x_par_trac_b.T
    e_y_par_trac_b = trac_sb_ey_b * np.sin(trac_sb_angles_b_broadcasted)
    e_y_par_trac_b = e_y_par_trac_b.T
    e_par_trac_b = e_x_par_trac_b + e_y_par_trac_b
    sig_sb_angles_b_broadcasted = sig_angles_b[:, np.newaxis]
    e_x_par_sig_b = sig_sb_ex_b * np.cos(sig_sb_angles_b_broadcasted)
    e_x_par_sig_b = e_x_par_sig_b.T
    e_y_par_sig_b = sig_sb_ey_b * np.sin(sig_sb_angles_b_broadcasted)
    e_y_par_sig_b = e_y_par_sig_b.T
    e_par_sig_b = e_x_par_sig_b + e_y_par_sig_b
    i_sig_b = e_par_sig_b / parameters["z_sig"]
    i_trac_b = e_par_trac_b / parameters["z_trac"]

    # "a" first
    # Traction return rail first node
    j_matrix[:, trac_node_locs_a[0]] = -i_trac_a[:, 0]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(trac_node_locs_a, cb_node_locs_a)
    indices = np.where(mask)[0]
    j_matrix[:, cb_node_locs_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Non-cross bond nodes
    mask = np.isin(trac_node_locs_a, cb_node_locs_a)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(trac_node_locs_a, cb_node_locs_a)
    non_cb_node_locs_centre_a = trac_node_locs_a[mask_del][1:-1]
    j_matrix[:, non_cb_node_locs_centre_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices] - parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, trac_node_locs_a[-1]] = i_trac_a[:, -1] - parameters["i_power"]
    # Signalling rail nodes
    # Relay nodes
    j_matrix[:, sig_node_locs_relay_a] = -i_sig_a
    # Power nodes
    j_matrix[:, sig_node_locs_power_a] = i_sig_a + parameters["i_power"]

    # "b" second
    # Traction return rail first node
    j_matrix[:, trac_node_locs_b[0]] = i_trac_b[:, 0] - parameters["i_power"]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(trac_node_locs_b, cb_node_locs_b)
    indices = np.where(mask)[0]
    j_matrix[:, cb_node_locs_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Non-cross bond nodes
    mask = np.isin(trac_node_locs_b, cb_node_locs_b)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(trac_node_locs_b, cb_node_locs_b)
    non_cb_node_locs_centre_b = trac_node_locs_b[mask_del][1:-1]
    j_matrix[:, non_cb_node_locs_centre_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1] - parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, trac_node_locs_b[-1]] = -i_trac_b[:, -1]
    # Signalling rail nodes
    # Power nodes
    j_matrix[:, sig_node_locs_power_b] = parameters["i_power"] + i_sig_b
    # Relay nodes
    j_matrix[:, sig_node_locs_relay_b] = -i_sig_b

    # Calculate voltage matrix
    # Load network admittance matrix
    y_matrix = np.load("data\\network_parameters\\" + section_name + "\\nodal_admittance_matrix_" + section_name + "_" + conditions + ".npy")

    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    # Calculate relay voltages and currents
    # "a" first
    v_relay_top_node_a = v_matrix[sig_node_locs_relay_a]
    v_relay_bottom_node_a = v_matrix[trac_node_locs_relay_a]
    v_relay_a = v_relay_top_node_a - v_relay_bottom_node_a

    # "b" first
    v_relay_top_node_b = v_matrix[sig_node_locs_relay_b]
    v_relay_bottom_node_b = v_matrix[trac_node_locs_relay_b]
    v_relay_b = v_relay_top_node_b - v_relay_bottom_node_b

    i_relays_a = v_relay_a / parameters["r_relay"]
    i_relays_b = v_relay_b / parameters["r_relay"]

    i_relays_a = i_relays_a.T
    i_relays_b = i_relays_b.T

    return i_relays_a, i_relays_b


def wrong_side_two_track_currents_e_blocks_timetable(section_name, conditions, ex_blocks, ey_blocks, axle_pos_a, axle_pos_b):
    # Create dictionary of network parameters
    parameters = {"z_sig": 0.0289,  # Signalling rail series impedance (ohms/km)
                  "z_trac": 0.0289,  # Traction return rail series impedance (ohms/km)
                  "y_sig_moderate": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_moderate": 1.6,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "y_sig_dry": 0.025,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_dry": 1.53,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
                  "y_sig_wet": 0.4,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_wet": 2,  # Traction return rail parallel admittance for moderate conditions (siemens/km)
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

    # Load in network sub block lengths and angles for the chosen section
    sub_block_lengths = np.load("data\\network_parameters\\" + section_name + "\\sub_blocks_" + section_name + ".npz")
    trac_node_positions = sub_block_lengths["blocks_sum_cb"]  # Positions of nodes along the traction return rail (km)
    sig_block_lengths = sub_block_lengths["sig_sub_blocks"]  # Lengths of the signalling rail for each block (km)
    block_angles = np.load("data\\network_parameters\\" + section_name + "\\angles_" + section_name + ".npz")
    block_angles_a = block_angles["sig_angles_a"]  # Angles of each block in the "a" direction
    block_angles_b = block_angles["sig_angles_b"]  # Angles of each block in the "b" direction (180 degrees from "a")

    # Load in network nodal indices for the chosen section
    network_nodes = np.load("data\\network_parameters\\" + section_name + "\\nodes_" + section_name + ".npz")
    n_nodes = network_nodes["n_nodes"]  # Total number of nodes in the section
    trac_node_indices_a = network_nodes["trac_node_locs_a"]  # Traction return rail node indices for "a" direction
    trac_node_indices_b = network_nodes["trac_node_locs_b"]  # Traction return rail node indices for "b" direction
    sig_node_indices_a = network_nodes["sig_node_locs_a"]  # Signalling rail node indices for "a" direction
    sig_node_indices_b = network_nodes["sig_node_locs_b"]  # Signalling rail node indices for "b" direction
    cb_node_indices_a = network_nodes["cb_node_locs_a"]  # Cross bond node indices for "a" direction
    cb_node_indices_b = network_nodes["cb_node_locs_b"]  # Cross bond node indices for "b" direction
    # trac_node_indices_power_a = network_nodes["trac_node_locs_power_a"]  # Traction return rail relay node indices for "a" direction
    # trac_node_indices_power_b = network_nodes["trac_node_locs_power_b"]  # Traction return rail relay node indices for "b" direction
    trac_node_indices_relay_a = network_nodes["trac_node_locs_relay_a"]  # Traction return rail relay node indices for "a" direction
    trac_node_indices_relay_b = network_nodes["trac_node_locs_relay_b"]  # Traction return rail relay node indices for "b" direction
    sig_node_indices_power_a = network_nodes["sig_node_locs_power_a"]  # Traction return rail power supply node indices for "a" direction
    sig_node_indices_power_b = network_nodes["sig_node_locs_power_b"]  # Traction return rail power supply node indices for "b" direction
    sig_node_indices_relay_a = network_nodes["sig_node_locs_relay_a"]  # Signalling rail relay node indices for "a" direction
    sig_node_indices_relay_b = network_nodes["sig_node_locs_relay_b"]  # Signalling rail relay node indices for "b" direction

    # Calculate the electrical characteristics of the rails
    gamma_trac = np.sqrt(parameters["z_trac"] * parameters["y_trac_" + conditions])  # Propagation constant of the traction return rail (ohms)
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig_" + conditions])  # Propagation constant of the signalling rail (ohms)
    z0_trac = np.sqrt(parameters["z_trac"] / parameters["y_trac_" + conditions])  # Characteristic impedance of the traction return rail (km^-1)
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig_" + conditions])  # Characteristic impedance of the signalling rail (km^-1)

    # Calculate the distance along the rail of nodes on the signalling rail
    sig_sub_blocks_sums = np.cumsum(sig_block_lengths)  # Cumulative sum of signalling blocks lengths
    sig_sub_blocks_sums_zero = np.hstack((0, sig_sub_blocks_sums))  # Cumulative sum of signalling blocks lengths starting with 0
    sig_node_positions = np.hstack((sig_sub_blocks_sums_zero[0], np.repeat(sig_sub_blocks_sums_zero[1:-1], 2), sig_sub_blocks_sums_zero[-1]))  # Distance along the rail for the signalling rail nodes

    # Calculate axle node indices for trains in the "a" direction
    if len(axle_pos_a) != 0:
        axle_node_positions_a = np.array(
            sorted(axle_pos_a.flatten()))  # Sort axle node positions of "a" and convert to numpy array
        starting_node = sig_node_indices_b[-1] + 1  # Starting node index for the traction return rail of "a"
        trac_node_indices_axle_a = np.arange(starting_node, starting_node + len(
            axle_node_positions_a))  # Nodal indices for axles on the traction return rail of "a"
        starting_node = trac_node_indices_axle_a[-1] + 1  # Starting node index for the signalling rail of "a"
        sig_axle_node_indices_a = np.arange(starting_node, starting_node + len(
            axle_node_positions_a))  # Nodal indices for axles on the signalling rail of "a"
    else:
        pass

    # Calculate axle node indices for trains in the "b" direction
    if len(axle_pos_b) != 0:
        axle_node_positions_b = np.array(
            sorted(axle_pos_b.flatten()))  # Sort axle node positions of "b" and convert to numpy array
        if len(axle_pos_a) != 0:
            starting_node = sig_axle_node_indices_a[-1] + 1  # Starting node index if there are trains on "a"
        else:
            starting_node = sig_node_indices_b[-1] + 1  # Starting node index if there are no trains on "a"
        trac_node_indices_axle_b = np.arange(starting_node, starting_node + len(
            axle_node_positions_b))  # Nodal indices for axles on the traction return rail of "b"
        starting_node = trac_node_indices_axle_b[-1] + 1  # Starting node index for the signalling rail of "b"
        sig_axle_node_indices_b = np.arange(starting_node, starting_node + len(
            axle_node_positions_b))  # Nodal indices for axles on the signalling rail of "b"
    else:
        pass

    # Combine the nodal positions and indices for the rails and the axles into single arrays
    if len(axle_pos_a) != 0 and len(axle_pos_b) != 0:  # If there are trains in both "a" and "b"
        all_trac_node_positions_a = np.hstack((trac_node_positions, axle_node_positions_a))
        all_sig_node_positions_a = np.hstack((sig_node_positions, axle_node_positions_a))
        all_trac_node_positions_b = np.hstack((trac_node_positions, axle_node_positions_b))
        all_sig_node_positions_b = np.hstack((sig_node_positions, axle_node_positions_b))
        all_trac_node_indices_a = np.hstack((trac_node_indices_a, trac_node_indices_axle_a))
        all_sig_node_indices_a = np.hstack((sig_node_indices_a, sig_axle_node_indices_a))
        all_trac_node_indices_b = np.hstack((trac_node_indices_b, trac_node_indices_axle_b))
        all_sig_node_indices_b = np.hstack((sig_node_indices_b, sig_axle_node_indices_b))

    elif len(axle_pos_a) != 0 and len(axle_pos_b) == 0:  # If there are trains in "a", but not in "b"
        all_trac_node_positions_a = np.hstack((trac_node_positions, axle_node_positions_a))
        all_sig_node_positions_a = np.hstack((sig_node_positions, axle_node_positions_a))
        all_trac_node_positions_b = np.copy(trac_node_positions)
        all_sig_node_positions_b = np.copy(sig_node_positions)
        all_trac_node_indices_a = np.hstack((trac_node_indices_a, trac_node_indices_axle_a))
        all_sig_node_indices_a = np.hstack((sig_node_indices_a, sig_axle_node_indices_a))
        all_trac_node_indices_b = np.copy(trac_node_indices_b)
        all_sig_node_indices_b = np.copy(sig_node_indices_b)

    else:  # If there are trains in "b", but not in "a"
        all_trac_node_positions_a = np.copy(trac_node_positions)
        all_sig_node_positions_a = np.copy(sig_node_positions)
        all_trac_node_positions_b = np.hstack((trac_node_positions, axle_node_positions_b))
        all_sig_node_positions_b = np.hstack((sig_node_positions, axle_node_positions_b))
        all_trac_node_indices_a = np.copy(trac_node_indices_a)
        all_sig_node_indices_a = np.copy(sig_node_indices_a)
        all_trac_node_indices_b = np.hstack((trac_node_indices_b, trac_node_indices_axle_b))
        all_sig_node_indices_b = np.hstack((sig_node_indices_b, sig_axle_node_indices_b))

    # Create dictionaries of nodal indices and positions
    trac_a_dict = dict(zip(all_trac_node_indices_a, all_trac_node_positions_a))
    sig_a_dict = dict(zip(all_sig_node_indices_a, all_sig_node_positions_a))
    trac_b_dict = dict(zip(all_trac_node_indices_b, all_trac_node_positions_b))
    sig_b_dict = dict(zip(all_sig_node_indices_b, all_sig_node_positions_b))

    # Sort the dictionaries based on positions
    sorted_trac_a_dict = dict(sorted(trac_a_dict.items(), key=lambda item: item[1]))
    sorted_sig_a_dict = dict(sorted(sig_a_dict.items(), key=lambda item: item[1]))
    sorted_trac_b_dict = dict(sorted(trac_b_dict.items(), key=lambda item: item[1]))
    sorted_sig_b_dict = dict(sorted(sig_b_dict.items(), key=lambda item: item[1]))

    # Save the sorted indices as new arrays
    all_trac_node_indices_a = np.array(list(sorted_trac_a_dict.keys()))
    all_sig_node_indices_a = np.array(list(sorted_sig_a_dict.keys()))
    all_trac_node_indices_b = np.array(list(sorted_trac_b_dict.keys()))
    all_sig_node_indices_b = np.array(list(sorted_sig_b_dict.keys()))

    # Make a new zeroed admittance matrix for the new restructured network
    n_nodes_restructured = len(all_trac_node_indices_a) + len(all_sig_node_indices_a) + len(
        all_trac_node_indices_b) + len(all_sig_node_indices_b)
    y_matrix_restructured = np.zeros((n_nodes_restructured, n_nodes_restructured))

    # Load in the nodal admittance matrix of the original network
    y_matrix = np.load(
        "data\\network_parameters\\" + section_name + "\\nodal_admittance_matrix_" + section_name + "_" + conditions + ".npy")

    # Place values from the original network into the restructured network
    y_matrix_restructured[0:n_nodes, 0:n_nodes] = y_matrix

    # Find which nodes need to be calculated or recalculated
    # "a" first
    # Traction return rail
    # Find the indices of the subset elements in the original array
    sub_set_indices = np.where(np.isin(all_trac_node_indices_a, trac_node_indices_axle_a))[0]
    # Create a mask for the neighbours
    mask = np.zeros(len(all_trac_node_indices_a), dtype=bool)
    mask[sub_set_indices] = True
    # Add neighbours to the mask
    mask[np.maximum(sub_set_indices - 1, 0)] = True
    mask[np.minimum(sub_set_indices + 1, len(all_trac_node_indices_a) - 1)] = True
    # Extract the values from the original array using the mask
    recalculate_trac_node_a = all_trac_node_indices_a[mask]
    # Signalling rail
    # Find the indices of the subset elements in the original array
    sub_set_indices = np.where(np.isin(all_sig_node_indices_a, sig_axle_node_indices_a))[0]
    # Create a mask for the neighbours
    mask = np.zeros(len(all_sig_node_indices_a), dtype=bool)
    mask[sub_set_indices] = True
    # Add neighbours to the mask
    mask[np.maximum(sub_set_indices - 1, 0)] = True
    mask[np.minimum(sub_set_indices + 1, len(all_sig_node_indices_a) - 1)] = True
    # Extract the values from the original array using the mask
    recalculate_sig_node_a = all_sig_node_indices_a[mask]
    # "b" second
    # Traction return rail
    # Find the indices of the subset elements in the original array
    sub_set_indices = np.where(np.isin(all_trac_node_indices_b, trac_node_indices_axle_b))[0]
    # Create a mask for the neighbours
    mask = np.zeros(len(all_trac_node_indices_b), dtype=bool)
    mask[sub_set_indices] = True
    # Add neighbours to the mask
    mask[np.maximum(sub_set_indices - 1, 0)] = True
    mask[np.minimum(sub_set_indices + 1, len(all_trac_node_indices_b) - 1)] = True
    # Extract the values from the original array using the mask
    recalculate_trac_node_b = all_trac_node_indices_b[mask]
    # Signalling rail
    # Find the indices of the subset elements in the original array
    sub_set_indices = np.where(np.isin(all_sig_node_indices_b, sig_axle_node_indices_b))[0]
    # Create a mask for the neighbours
    mask = np.zeros(len(all_sig_node_indices_b), dtype=bool)
    mask[sub_set_indices] = True
    # Add neighbours to the mask
    mask[np.maximum(sub_set_indices - 1, 0)] = True
    mask[np.minimum(sub_set_indices + 1, len(all_sig_node_indices_b) - 1)] = True
    # Extract the values from the original array using the mask
    recalculate_sig_node_b = all_sig_node_indices_b[mask]

    # Recalculate the equivalent pi-circuit parameters for the new network
    # Get the sorted nodal positions
    all_trac_node_positions_sorted_a = list(sorted_trac_a_dict.values())
    all_sig_node_positions_sorted_a = list(sorted_sig_a_dict.values())
    all_trac_node_positions_sorted_b = list(sorted_trac_b_dict.values())
    all_sig_node_positions_sorted_b = list(sorted_sig_b_dict.values())

    # Calculate the length of the sub blocks
    trac_sub_blocks_a = np.diff(all_trac_node_positions_sorted_a)
    sig_sub_blocks_a = np.diff(all_sig_node_positions_sorted_a)
    sig_sub_blocks_a[
        sig_sub_blocks_a == 0] = np.nan  # Sub blocks with length zero on the signalling rail indicate calculating rail joints, these need to be nans
    trac_sub_blocks_b = np.diff(all_trac_node_positions_sorted_b)
    sig_sub_blocks_b = np.diff(all_sig_node_positions_sorted_b)
    sig_sub_blocks_b[sig_sub_blocks_b == 0] = np.nan

    # Set up equivalent pi-parameters
    # "a" first
    ye_sig_a = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))  # Series admittance for signalling rail
    ye_trac_a = 1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_a))  # Series admittance for traction return rail
    yg_sig_a = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_a) - 1) * (
                1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))))  # Parallel admittance for signalling rail
    yg_trac_a = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_a) - 1) * (1 / (
                z0_trac * np.sinh(gamma_trac * trac_sub_blocks_a))))  # Parallel admittance for traction return rail
    # "b" second
    ye_sig_b = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))  # Series admittance for signalling rail
    ye_trac_b = 1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_b))  # Series admittance for traction return rail
    yg_sig_b = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_b) - 1) * (
                1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))))  # Parallel admittance for signalling rail
    yg_trac_b = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_b) - 1) * (1 / (
                z0_trac * np.sinh(gamma_trac * trac_sub_blocks_b))))  # Parallel admittance for traction return rail

    # Recalculate nodal parallel admittances and compute new sums
    # "a" first
    # Traction return rail
    recalculate_yg_trac_a = np.full(len(all_trac_node_indices_a), -1).astype(
        float)  # Array of parallel admittances to place recalculated values in, with negative values to filter out unused cells later
    recalculate_y_sum_trac_a = np.full(len(all_trac_node_indices_a), -1).astype(
        float)  # Array of sum of admittances into the node to place recalculated values in, with negative values to filter out unused cells later
    mask_first = np.isin(recalculate_trac_node_a, all_trac_node_indices_a[
        0])  # Mask to determine if the first traction rail node needs to be recalculated
    if np.any(mask_first):
        first = recalculate_trac_node_a[mask_first]  # Index of the nodes to be recalculated
        first_locs = np.where(np.isin(all_trac_node_indices_a, first))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[first_locs] = 0.5 * yg_trac_a[first_locs]  # New nodal parallel admittance
        recalculate_y_sum_trac_a[first_locs] = recalculate_yg_trac_a[first_locs] + parameters["y_relay"] + ye_trac_a[
            first_locs]
    else:
        pass
    mask_axle = np.isin(trac_node_indices_axle_a,
                        recalculate_trac_node_a)  # Mask to determine if any traction return rail axle nodes needs to be recalculated
    if np.any(mask_axle):
        axle = trac_node_indices_axle_a[mask_axle]  # Index of the nodes to be recalculated
        axle_locs = np.where(np.isin(all_trac_node_indices_a, axle))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[axle_locs] = (0.5 * yg_trac_a[axle_locs - 1]) + (
                    0.5 * yg_trac_a[axle_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_a[axle_locs] = recalculate_yg_trac_a[axle_locs] + parameters["y_axle"] + ye_trac_a[
            axle_locs - 1] + ye_trac_a[axle_locs]
    else:
        pass
    mask_cb = np.isin(cb_node_indices_a,
                      recalculate_trac_node_a)  # Mask to determine if any traction return rail cb nodes needs to be recalculated
    if np.any(mask_cb):
        cb = cb_node_indices_a[mask_cb]  # Index of the nodes to be recalculated
        cb_locs = np.where(np.isin(all_trac_node_indices_a, cb))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[cb_locs] = (0.5 * yg_trac_a[cb_locs - 1]) + (
                    0.5 * yg_trac_a[cb_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_a[cb_locs] = recalculate_yg_trac_a[cb_locs] + parameters["y_cb"] + ye_trac_a[
            cb_locs - 1] + ye_trac_a[cb_locs]
    else:
        pass
    other_node_indices = all_trac_node_indices_a[1:-2][
        ~np.logical_or(np.isin(all_trac_node_indices_a[1:-2], trac_node_indices_axle_a),
                       np.isin(all_trac_node_indices_a[1:-2], cb_node_indices_a))]
    mask_other = np.isin(other_node_indices,
                         recalculate_trac_node_a)  # Mask to determine if any other traction return rail nodes needs to be recalculated
    if np.any(mask_other):
        other = other_node_indices[mask_other]  # Index of the nodes to be recalculated
        other_locs = np.where(np.isin(all_trac_node_indices_a, other))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[other_locs] = (0.5 * yg_trac_a[other_locs - 1]) + (
                    0.5 * yg_trac_a[other_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_a[other_locs] = recalculate_yg_trac_a[other_locs] + parameters["y_power"] + parameters[
            "y_relay"] + ye_trac_a[other_locs - 1] + ye_trac_a[other_locs]
    else:
        pass
    mask_last = np.isin(recalculate_trac_node_a, all_trac_node_indices_a[
        -1])  # Mask to determine if the last traction rail node needs to be recalculated
    if np.any(mask_last):
        last = recalculate_trac_node_a[mask_last]  # Index of the nodes to be recalculated
        last_locs = np.where(np.isin(all_trac_node_indices_a, last))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_a[last_locs] = 0.5 * yg_trac_a[last_locs - 1]  # New nodal parallel admittance
        recalculate_y_sum_trac_a[last_locs] = recalculate_yg_trac_a[last_locs] + parameters["y_power"] + ye_trac_a[
            last_locs - 1]
    else:
        pass
    recalculate_yg_trac_a = recalculate_yg_trac_a[recalculate_yg_trac_a != -1]  # Unused cells removed
    recalculate_y_sum_trac_a = recalculate_y_sum_trac_a[recalculate_y_sum_trac_a != -1]  # Unused cells removed

    # Signalling rail
    recalculate_yg_sig_a = np.full(len(all_sig_node_indices_a), -1).astype(
        float)  # Array of parallel admittances to place recalculated values in, with negative values to filter out unused cells later
    recalculate_y_sum_sig_a = np.full(len(all_sig_node_indices_a), -1).astype(
        float)  # Array of sum of admittances into the node to place recalculated values in, with negative values to filter out unused cells later
    mask_relay = np.isin(sig_node_indices_relay_a,
                         recalculate_sig_node_a)  # Mask to determine if any signalling rail relay nodes needs to be recalculated
    if np.any(mask_relay):
        relay = sig_node_indices_relay_a[mask_relay]  # Index of the nodes to be recalculated
        relay_locs = np.where(np.isin(all_sig_node_indices_a, relay))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_a[relay_locs] = 0.5 * yg_sig_a[relay_locs]  # New nodal parallel admittance
        recalculate_y_sum_sig_a[relay_locs] = recalculate_yg_sig_a[relay_locs] + parameters["y_relay"] + ye_sig_a[
            relay_locs]
    else:
        pass
    mask_power = np.isin(sig_node_indices_power_a,
                         recalculate_sig_node_a)  # Mask to determine if any signalling rail power nodes needs to be recalculated
    if np.any(mask_power):
        power = sig_node_indices_power_a[mask_power]  # Index of the nodes to be recalculated
        power_locs = np.where(np.isin(all_sig_node_indices_a, power))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_a[power_locs] = 0.5 * yg_sig_a[power_locs - 1]  # New nodal parallel admittance
        recalculate_y_sum_sig_a[power_locs] = recalculate_yg_sig_a[power_locs] + parameters["y_power"] + ye_sig_a[
            power_locs - 1]
    else:
        pass
    other_node_indices = all_sig_node_indices_a[
        ~np.logical_or(np.isin(all_sig_node_indices_a, sig_node_indices_relay_a),
                       np.isin(all_sig_node_indices_a, sig_node_indices_power_a))]
    mask_other = np.isin(other_node_indices,
                         recalculate_sig_node_a)  # Mask to determine if any other signalling rail nodes needs to be recalculated
    if np.any(mask_other):
        other = other_node_indices[mask_other]  # Index of the nodes to be recalculated
        other_locs = np.where(np.isin(all_sig_node_indices_a, other))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_a[other_locs] = (0.5 * yg_sig_a[other_locs - 1]) + (
                    0.5 * yg_sig_a[other_locs])  # New nodal parallel admittance
        recalculate_y_sum_sig_a[other_locs] = recalculate_yg_sig_a[other_locs] + parameters["y_axle"] + ye_sig_a[
            other_locs - 1] + ye_sig_a[other_locs]
    else:
        pass
    recalculate_yg_sig_a = recalculate_yg_sig_a[recalculate_yg_sig_a != -1]  # Unused cells removed
    recalculate_y_sum_sig_a = recalculate_y_sum_sig_a[recalculate_y_sum_sig_a != -1]  # Unused cells removed

    # "b" second
    # Traction return rail
    recalculate_yg_trac_b = np.full(len(all_trac_node_indices_b), -1).astype(
        float)  # Array of parallel admittances to place recalculated values in, with negative values to filter out unused cells later
    recalculate_y_sum_trac_b = np.full(len(all_trac_node_indices_b), -1).astype(
        float)  # Array of sum of admittances into the node to place recalculated values in, with negative values to filter out unused cells later
    mask_first = np.isin(recalculate_trac_node_b, all_trac_node_indices_b[
        0])  # Mask to determine if the first traction rail node needs to be recalculated
    if np.any(mask_first):
        first = recalculate_trac_node_b[mask_first]  # Index of the nodes to be recalculated
        first_locs = np.where(np.isin(all_trac_node_indices_b, first))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[first_locs] = 0.5 * yg_trac_b[first_locs]  # New nodal parallel admittance
        recalculate_y_sum_trac_b[first_locs] = recalculate_yg_trac_b[first_locs] + parameters["y_power"] + ye_trac_b[
            first_locs]
    else:
        pass
    mask_axle = np.isin(trac_node_indices_axle_b,
                        recalculate_trac_node_b)  # Mask to determine if any traction return rail axle nodes needs to be recalculated
    if np.any(mask_axle):
        axle = trac_node_indices_axle_b[mask_axle]  # Index of the nodes to be recalculated
        axle_locs = np.where(np.isin(all_trac_node_indices_b, axle))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[axle_locs] = (0.5 * yg_trac_b[axle_locs - 1]) + (
                    0.5 * yg_trac_b[axle_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_b[axle_locs] = recalculate_yg_trac_b[axle_locs] + parameters["y_axle"] + ye_trac_b[
            axle_locs - 1] + ye_trac_b[axle_locs]
    else:
        pass
    mask_cb = np.isin(cb_node_indices_b,
                      recalculate_trac_node_b)  # Mask to determine if any traction return rail cb nodes needs to be recalculated
    if np.any(mask_cb):
        cb = cb_node_indices_b[mask_cb]  # Index of the nodes to be recalculated
        cb_locs = np.where(np.isin(all_trac_node_indices_b, cb))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[cb_locs] = (0.5 * yg_trac_b[cb_locs - 1]) + (
                    0.5 * yg_trac_b[cb_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_b[cb_locs] = recalculate_yg_trac_b[cb_locs] + parameters["y_cb"] + ye_trac_b[
            cb_locs - 1] + ye_trac_b[cb_locs]
    else:
        pass
    other_node_indices = all_trac_node_indices_b[1:-2][
        ~np.logical_or(np.isin(all_trac_node_indices_b[1:-2], trac_node_indices_axle_b),
                       np.isin(all_trac_node_indices_b[1:-2], cb_node_indices_b))]
    mask_other = np.isin(other_node_indices,
                         recalculate_trac_node_b)  # Mask to determine if any other traction return rail nodes needs to be recalculated
    if np.any(mask_other):
        other = other_node_indices[mask_other]  # Index of the nodes to be recalculated
        other_locs = np.where(np.isin(all_trac_node_indices_b, other))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[other_locs] = (0.5 * yg_trac_b[other_locs - 1]) + (
                    0.5 * yg_trac_b[other_locs])  # New nodal parallel admittance
        recalculate_y_sum_trac_b[other_locs] = recalculate_yg_trac_b[other_locs] + parameters["y_power"] + parameters[
            "y_relay"] + ye_trac_b[other_locs - 1] + ye_trac_b[other_locs]
    else:
        pass
    mask_last = np.isin(recalculate_trac_node_b, all_trac_node_indices_b[
        -1])  # Mask to determine if the last traction rail node needs to be recalculated
    if np.any(mask_last):
        last = recalculate_trac_node_b[mask_last]  # Index of the nodes to be recalculated
        last_locs = np.where(np.isin(all_trac_node_indices_b, last))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_trac_b[last_locs] = 0.5 * yg_trac_b[last_locs - 1]  # New nodal parallel admittance
        recalculate_y_sum_trac_b[last_locs] = recalculate_yg_trac_b[last_locs] + parameters["y_relay"] + ye_trac_b[
            last_locs - 1]
    else:
        pass
    recalculate_yg_trac_b = recalculate_yg_trac_b[recalculate_yg_trac_b != -1]  # Unused cells removed
    recalculate_y_sum_trac_b = recalculate_y_sum_trac_b[recalculate_y_sum_trac_b != -1]  # Unused cells removed
    # Signalling rail
    recalculate_yg_sig_b = np.full(len(all_sig_node_indices_b), -1).astype(
        float)  # Array of parallel admittances to place recalculated values in, with negative values to filter out unused cells later
    recalculate_y_sum_sig_b = np.full(len(all_sig_node_indices_b), -1).astype(
        float)  # Array of sum of admittances into the node to place recalculated values in, with negative values to filter out unused cells later
    mask_relay = np.isin(sig_node_indices_relay_b,
                         recalculate_sig_node_b)  # Mask to determine if any signalling rail relay nodes needs to be recalculated
    if np.any(mask_relay):
        relay = sig_node_indices_relay_b[mask_relay]  # Index of the nodes to be recalculated
        relay_locs = np.where(np.isin(all_sig_node_indices_b, relay))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_b[relay_locs] = 0.5 * yg_sig_b[relay_locs - 1]  # New nodal parallel admittance
        recalculate_y_sum_sig_b[relay_locs] = recalculate_yg_sig_b[relay_locs] + parameters["y_relay"] + ye_sig_b[
            relay_locs - 1]
    else:
        pass
    mask_power = np.isin(sig_node_indices_power_b,
                         recalculate_sig_node_b)  # Mask to determine if any signalling rail power nodes needs to be recalculated
    if np.any(mask_power):
        power = sig_node_indices_power_b[mask_power]  # Index of the nodes to be recalculated
        power_locs = np.where(np.isin(all_sig_node_indices_b, power))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_b[power_locs] = 0.5 * yg_sig_b[power_locs]  # New nodal parallel admittance
        recalculate_y_sum_sig_b[power_locs] = recalculate_yg_sig_b[power_locs] + parameters["y_power"] + ye_sig_b[
            power_locs]
    else:
        pass
    other_node_indices = all_sig_node_indices_b[
        ~np.logical_or(np.isin(all_sig_node_indices_b, sig_node_indices_relay_b),
                       np.isin(all_sig_node_indices_b, sig_node_indices_power_b))]
    mask_other = np.isin(other_node_indices,
                         recalculate_sig_node_b)  # Mask to determine if any other signalling rail nodes needs to be recalculated
    if np.any(mask_other):
        other = other_node_indices[mask_other]  # Index of the nodes to be recalculated
        other_locs = np.where(np.isin(all_sig_node_indices_b, other))[
            0]  # Position along the rail of the nodes to be recalculated
        recalculate_yg_sig_b[other_locs] = (0.5 * yg_sig_b[other_locs - 1]) + (
                    0.5 * yg_sig_b[other_locs])  # New nodal parallel admittance
        recalculate_y_sum_sig_b[other_locs] = recalculate_yg_sig_b[other_locs] + parameters["y_axle"] + ye_sig_b[
            other_locs - 1] + ye_sig_b[other_locs]
    else:
        pass
    recalculate_yg_sig_b = recalculate_yg_sig_b[recalculate_yg_sig_b != -1]  # Unused cells removed
    recalculate_y_sum_sig_b = recalculate_y_sum_sig_b[recalculate_y_sum_sig_b != -1]  # Unused cells removed

    # Update the reshaped nodal admittance matrix
    # Replace the diagonal values
    # "a" first
    y_matrix_restructured[recalculate_trac_node_a, recalculate_trac_node_a] = recalculate_y_sum_trac_a
    y_matrix_restructured[recalculate_sig_node_a, recalculate_sig_node_a] = recalculate_y_sum_sig_a
    # "b" second
    y_matrix_restructured[recalculate_trac_node_b, recalculate_trac_node_b] = recalculate_y_sum_trac_b
    y_matrix_restructured[recalculate_sig_node_b, recalculate_sig_node_b] = recalculate_y_sum_sig_b

    # Series admittances between nodes
    # "a" first
    i = np.where(np.isin(all_trac_node_indices_a, recalculate_trac_node_a[:-1]))[0]
    y_matrix_restructured[np.array(all_trac_node_indices_a)[i], np.array(all_trac_node_indices_a)[i + 1]] = \
    y_matrix_restructured[np.array(all_trac_node_indices_a)[i + 1], np.array(all_trac_node_indices_a)[i]] = -ye_trac_a[
        i]
    i = np.where(np.isin(all_sig_node_indices_a, recalculate_sig_node_a[:-1]))[0]
    y_matrix_restructured[np.array(all_sig_node_indices_a)[i], np.array(all_sig_node_indices_a)[i + 1]] = \
    y_matrix_restructured[np.array(all_sig_node_indices_a)[i + 1], np.array(all_sig_node_indices_a)[i]] = -ye_sig_a[i]
    # "b" second
    i = np.where(np.isin(all_trac_node_indices_b, recalculate_trac_node_b[:-1]))[0]
    y_matrix_restructured[np.array(all_trac_node_indices_b)[i], np.array(all_trac_node_indices_b)[i + 1]] = \
    y_matrix_restructured[np.array(all_trac_node_indices_b)[i + 1], np.array(all_trac_node_indices_b)[i]] = -ye_trac_b[
        i]
    i = np.where(np.isin(all_sig_node_indices_b, recalculate_sig_node_b[:-1]))[0]
    y_matrix_restructured[np.array(all_sig_node_indices_b)[i], np.array(all_sig_node_indices_b)[i + 1]] = \
    y_matrix_restructured[np.array(all_sig_node_indices_b)[i + 1], np.array(all_sig_node_indices_b)[i]] = -ye_sig_b[i]
    y_matrix_restructured[np.isnan(y_matrix_restructured)] = 0  # Set any NaN values from the IRJs to 0

    # Axle admittances
    # "a" first
    y_matrix_restructured[trac_node_indices_axle_a, sig_axle_node_indices_a] = -parameters["y_axle"]
    y_matrix_restructured[sig_axle_node_indices_a, trac_node_indices_axle_a] = -parameters["y_axle"]
    # "b" second
    y_matrix_restructured[trac_node_indices_axle_b, sig_axle_node_indices_b] = -parameters["y_axle"]
    y_matrix_restructured[sig_axle_node_indices_b, trac_node_indices_axle_b] = -parameters["y_axle"]

    # Restructure angles array based on the new sub blocks
    # "a" first
    cumsum_trac_sb_a = np.cumsum(trac_sub_blocks_a)
    block_indices_trac_a = np.searchsorted(sig_sub_blocks_sums, cumsum_trac_sb_a)
    trac_sb_angles_a = block_angles_a[block_indices_trac_a]
    cumsum_sig_sb_a = np.cumsum(sig_sub_blocks_a[~np.isnan(sig_sub_blocks_a)])
    block_indices_sig_a = np.searchsorted(sig_sub_blocks_sums, cumsum_sig_sb_a)
    sig_sb_angles_a = block_angles_a[block_indices_sig_a]
    # "b" second
    cumsum_trac_sb_b = np.cumsum(trac_sub_blocks_b)
    block_indices_trac_b = np.searchsorted(sig_sub_blocks_sums, cumsum_trac_sb_b)
    trac_sb_angles_b = block_angles_b[block_indices_trac_b]
    cumsum_sig_sb_b = np.cumsum(sig_sub_blocks_b[~np.isnan(sig_sub_blocks_b)])
    block_indices_sig_b = np.searchsorted(sig_sub_blocks_sums, cumsum_sig_sb_b)
    sig_sb_angles_b = block_angles_b[block_indices_sig_b]

    _, counts = np.unique(trac_sb_angles_a, return_counts=True)
    trac_sb_ex_a = np.repeat(ex_blocks, counts, axis=0)
    trac_sb_ey_a = np.repeat(ey_blocks, counts, axis=0)
    _, counts = np.unique(sig_sb_angles_a, return_counts=True)
    sig_sb_ex_a = np.repeat(ex_blocks, counts, axis=0)
    sig_sb_ey_a = np.repeat(ey_blocks, counts, axis=0)
    _, counts = np.unique(trac_sb_angles_b, return_counts=True)
    trac_sb_ex_b = np.repeat(ex_blocks, counts, axis=0)
    trac_sb_ey_b = np.repeat(ey_blocks, counts, axis=0)
    _, counts = np.unique(sig_sb_angles_b, return_counts=True)
    sig_sb_ex_b = np.repeat(ex_blocks, counts, axis=0)
    sig_sb_ey_b = np.repeat(ey_blocks, counts, axis=0)

    # Currents
    # Set up current matrix
    j_matrix = np.zeros([len(ex_blocks[0, :]), n_nodes_restructured])

    # "a" first
    trac_sb_angles_a_broadcasted = trac_sb_angles_a[:, np.newaxis]
    e_x_par_trac_a = trac_sb_ex_a * np.cos(trac_sb_angles_a_broadcasted)
    e_x_par_trac_a = e_x_par_trac_a.T
    e_y_par_trac_a = trac_sb_ey_a * np.sin(trac_sb_angles_a_broadcasted)
    e_y_par_trac_a = e_y_par_trac_a.T
    e_par_trac_a = e_x_par_trac_a + e_y_par_trac_a
    sig_sb_angles_a_broadcasted = sig_sb_angles_a[:, np.newaxis]
    e_x_par_sig_a = sig_sb_ex_a * np.cos(sig_sb_angles_a_broadcasted)
    e_x_par_sig_a = e_x_par_sig_a.T
    e_y_par_sig_a = sig_sb_ey_a * np.sin(sig_sb_angles_a_broadcasted)
    e_y_par_sig_a = e_y_par_sig_a.T
    e_par_sig_a = e_x_par_sig_a + e_y_par_sig_a
    i_sig_a = e_par_sig_a / parameters["z_sig"]
    i_trac_a = e_par_trac_a / parameters["z_trac"]

    # "b" second
    trac_sb_angles_b_broadcasted = trac_sb_angles_b[:, np.newaxis]
    e_x_par_trac_b = trac_sb_ex_b * np.cos(trac_sb_angles_b_broadcasted)
    e_x_par_trac_b = e_x_par_trac_b.T
    e_y_par_trac_b = trac_sb_ey_b * np.sin(trac_sb_angles_b_broadcasted)
    e_y_par_trac_b = e_y_par_trac_b.T
    e_par_trac_b = e_x_par_trac_b + e_y_par_trac_b
    sig_sb_angles_b_broadcasted = sig_sb_angles_b[:, np.newaxis]
    e_x_par_sig_b = sig_sb_ex_b * np.cos(sig_sb_angles_b_broadcasted)
    e_x_par_sig_b = e_x_par_sig_b.T
    e_y_par_sig_b = sig_sb_ey_b * np.sin(sig_sb_angles_b_broadcasted)
    e_y_par_sig_b = e_y_par_sig_b.T
    e_par_sig_b = e_x_par_sig_b + e_y_par_sig_b
    i_sig_b = e_par_sig_b / parameters["z_sig"]
    i_trac_b = e_par_trac_b / parameters["z_trac"]

    # "a" first
    # Traction return rail first node
    j_matrix[:, all_trac_node_indices_a[0]] = -i_trac_a[:, 0]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(all_trac_node_indices_a, cb_node_indices_a)
    indices = np.where(mask)[0]
    j_matrix[:, cb_node_indices_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Axle nodes
    mask = np.isin(all_trac_node_indices_a, trac_node_indices_axle_a)
    indices = np.where(mask)[0]
    j_matrix[:, trac_node_indices_axle_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices]
    # Non-cross bond or axle nodes
    mask = np.isin(all_trac_node_indices_a, cb_node_indices_a) | np.isin(all_trac_node_indices_a, trac_node_indices_axle_a)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(all_trac_node_indices_a, cb_node_indices_a) & ~np.isin(all_trac_node_indices_a, trac_node_indices_axle_a)
    non_cb_axle_node_locs_centre_a = all_trac_node_indices_a[mask_del][1:-1]
    j_matrix[:, non_cb_axle_node_locs_centre_a] = i_trac_a[:, indices - 1] - i_trac_a[:, indices] - parameters["i_power"]
    # Traction return rail last node
    j_matrix[:, all_trac_node_indices_a[-1]] = i_trac_a[:, -1] - parameters["i_power"]
    # Signalling rail nodes
    sig_relay_axle = all_sig_node_indices_a[np.where(~np.isin(all_sig_node_indices_a, sig_node_indices_power_a))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_relay_axle, sig_axle_node_indices_a))[0], np.where(np.isin(sig_relay_axle, sig_axle_node_indices_a))[0] - 1)))
    all_blocks = range(0, len(i_sig_a[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_relay_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_relay_axle[np.where(~np.isin(sig_relay_axle, sig_axle_node_indices_a) & ~np.isin(sig_relay_axle, whole_blocks_start))[0]]
    split_blocks_end = split_blocks_start + 1
    split_blocks_mid = sig_relay_axle[np.where(np.isin(sig_relay_axle, sig_axle_node_indices_a))[0]]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, whole_blocks_start))[0]]] = -i_sig_a[:, whole_blocks]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, whole_blocks_end))[0]]] = i_sig_a[:, whole_blocks] + parameters["i_power"]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, split_blocks_start))[0]]] = -i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_start))[0]]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, split_blocks_end))[0]]] = i_sig_a[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1))[0]]] + parameters["i_power"]
    j_matrix[:, all_sig_node_indices_a[np.where(np.isin(all_sig_node_indices_a, split_blocks_mid))[0]]] = i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0] - 1] - i_sig_a[:, np.where(np.isin(sig_relay_axle, split_blocks_mid))[0]]

    # "b" second
    # Traction return rail first node
    j_matrix[:, all_trac_node_indices_b[0]] = i_trac_b[:, 0] - parameters["i_power"]
    # Traction return rail centre nodes
    # Cross bond nodes
    mask = np.isin(all_trac_node_indices_b, cb_node_indices_b)
    indices = np.where(mask)[0]
    j_matrix[:, cb_node_indices_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Axle nodes
    mask = np.isin(all_trac_node_indices_b, trac_node_indices_axle_b)
    indices = np.where(mask)[0]
    j_matrix[:, trac_node_indices_axle_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1]
    # Non-cross bond or axle nodes
    mask = np.isin(all_trac_node_indices_b, cb_node_indices_b) | np.isin(all_trac_node_indices_b,
                                                                         trac_node_indices_axle_b)
    indices = np.where(~mask)[0][1:-1]
    mask_del = ~np.isin(all_trac_node_indices_b, cb_node_indices_b) & ~np.isin(all_trac_node_indices_b,
                                                                               trac_node_indices_axle_b)
    non_cb_axle_node_locs_centre_b = all_trac_node_indices_b[mask_del][1:-1]
    j_matrix[:, non_cb_axle_node_locs_centre_b] = i_trac_b[:, indices] - i_trac_b[:, indices - 1] - parameters[
        "i_power"]
    # Traction return rail last node
    j_matrix[:, all_trac_node_indices_b[-1]] = -i_trac_b[:, -1]
    # Signalling rail nodes
    sig_power_axle = all_sig_node_indices_b[np.where(~np.isin(all_sig_node_indices_b, sig_node_indices_relay_b))[0]]
    split_blocks = np.unique(np.sort(np.append(np.where(np.isin(sig_power_axle, sig_axle_node_indices_b))[0],
                                               np.where(np.isin(sig_power_axle, sig_axle_node_indices_b))[0] - 1)))
    all_blocks = range(0, len(i_sig_b[0]))
    whole_blocks = np.where(~np.isin(all_blocks, split_blocks))[0]
    whole_blocks_start = sig_power_axle[whole_blocks]
    whole_blocks_end = whole_blocks_start + 1
    split_blocks_start = sig_power_axle[
        np.where(~np.isin(sig_power_axle, sig_axle_node_indices_b) & ~np.isin(sig_power_axle, whole_blocks_start))[0]]
    split_blocks_end = split_blocks_start + 1
    split_blocks_mid = sig_power_axle[np.where(np.isin(sig_power_axle, sig_axle_node_indices_b))[0]]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, whole_blocks_start))[0]]] = i_sig_b[:, whole_blocks] + parameters["i_power"]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, whole_blocks_end))[0]]] = -i_sig_b[:, whole_blocks]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, split_blocks_start))[0]]] = i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_start))[0]] + parameters["i_power"]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, split_blocks_end))[0]]] = i_sig_b[:, split_blocks[np.where(~np.isin(split_blocks, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1))[0]]]
    j_matrix[:, all_sig_node_indices_b[np.where(np.isin(all_sig_node_indices_b, split_blocks_mid))[0]]] = -i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0] - 1] + i_sig_b[:, np.where(np.isin(sig_power_axle, split_blocks_mid))[0]]

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix_restructured)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    # Calculate relay voltages and currents
    # "a" first
    v_relay_top_node_a = v_matrix[sig_node_indices_relay_a]
    v_relay_bottom_node_a = v_matrix[trac_node_indices_relay_a]
    v_relay_a = v_relay_top_node_a - v_relay_bottom_node_a

    # "b" first
    v_relay_top_node_b = v_matrix[sig_node_indices_relay_b]
    v_relay_bottom_node_b = v_matrix[trac_node_indices_relay_b]
    v_relay_b = v_relay_top_node_b - v_relay_bottom_node_b

    i_relays_a = v_relay_a / parameters["r_relay"]
    i_relays_b = v_relay_b / parameters["r_relay"]

    i_relays_a = i_relays_a.T
    i_relays_b = i_relays_b.T

    return i_relays_a, i_relays_b


def generate_timetable_currents(section_name, storm):
    timetable_axles = np.load("data/axle_positions/timetable/" + section_name + "_axle_positions_timetable.npz", allow_pickle=True)
    axle_positions_a_all = timetable_axles["axle_pos_a_all"]
    axle_positions_b_all = timetable_axles["axle_pos_b_all"]

    axle_positions_a_all = np.concatenate((axle_positions_a_all, axle_positions_a_all))
    axle_positions_b_all = np.concatenate((axle_positions_b_all, axle_positions_b_all))

    # Load in storm e_field data
    storm_es = np.load('data/storm_e_fields/bgs_' + storm + '/' + section_name + '_' + storm + '_e_blocks.npz')
    ex_blocks_all = storm_es['ex_blocks'] / 1000
    ey_blocks_all = storm_es['ey_blocks'] / 1000

    for i in range(0, len(axle_positions_a_all)):
        axle_pos_a = axle_positions_a_all[i]
        axle_pos_b = axle_positions_b_all[i]
        ex_blocks = ex_blocks_all[:, i:i+1]
        ey_blocks = ey_blocks_all[:, i:i+1]

        if len(axle_pos_a) > 0:
            ia, ib = wrong_side_two_track_currents_e_blocks_timetable(section_name, "moderate", ex_blocks, ey_blocks, axle_pos_a, axle_pos_b)
            print(i)
        else:
            ia, ib = right_side_two_track_currents_e_blocks_timetable(section_name, "moderate", ex_blocks, ey_blocks)
            print(i)







generate_timetable_currents("east_coast_main_line", "may2024")
