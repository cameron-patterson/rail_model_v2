import numpy as np


def wrong_side_two_track(section_name, conditions, axle_pos_a, axle_pos_b, exs, eys):
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

    # Load in section network sub block lengths and angles
    sub_block_lengths = np.load("data\\network_parameters\\" + section_name + "\\sub_blocks_" + section_name + ".npz")
    trac_node_positions = sub_block_lengths["blocks_sum_cb"]
    #trac_sub_blocks = sub_block_lengths["trac_sub_blocks"]
    sig_sub_blocks = sub_block_lengths["sig_sub_blocks"]
    sub_block_angles = np.load("data\\network_parameters\\" + section_name + "\\angles_" + section_name + ".npz")
    trac_angles_a = sub_block_angles["trac_angles_a"]
    trac_angles_b = sub_block_angles["trac_angles_b"]
    sig_angles_a = sub_block_angles["sig_angles_a"]
    sig_angles_b = sub_block_angles["sig_angles_b"]

    # Load in section network node indices
    network_nodes = np.load("data\\network_parameters\\" + section_name + "\\nodes_" + section_name + ".npz")
    n_nodes = network_nodes["n_nodes"]
    #n_nodes_trac = network_nodes["n_nodes_trac"]
    trac_node_locs_a = network_nodes["trac_node_locs_a"]
    trac_node_locs_b = network_nodes["trac_node_locs_b"]
    sig_node_locs_a = network_nodes["sig_node_locs_a"]
    sig_node_locs_b = network_nodes["sig_node_locs_b"]
    cb_node_locs_a = network_nodes["cb_node_locs_a"]
    cb_node_locs_b = network_nodes["cb_node_locs_b"]
    trac_node_locs_power_a = network_nodes["trac_node_locs_power_a"]
    trac_node_locs_power_b = network_nodes["trac_node_locs_power_b"]
    trac_node_locs_relay_a = network_nodes["trac_node_locs_relay_a"]
    trac_node_locs_relay_b = network_nodes["trac_node_locs_relay_b"]
    sig_node_locs_power_a = network_nodes["sig_node_locs_power_a"]
    sig_node_locs_power_b = network_nodes["sig_node_locs_power_b"]
    sig_node_locs_relay_a = network_nodes["sig_node_locs_relay_a"]
    sig_node_locs_relay_b = network_nodes["sig_node_locs_relay_b"]

    # Calculate the electrical characteristics of the rails
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig_" + conditions])
    gamma_trac = np.sqrt(parameters["z_trac"] * parameters["y_trac_" + conditions])
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig_" + conditions])
    z0_trac = np.sqrt(parameters["z_trac"] / parameters["y_trac_" + conditions])

    # Create dictionary containing each node's index and its position along the rails
    trac_node_indices_a = np.copy(trac_node_locs_a)
    trac_node_positions_a = np.copy(trac_node_positions)

    sig_sub_blocks_sums = np.cumsum(sig_sub_blocks)
    sig_node_indices_a = np.copy(sig_node_locs_a)
    sig_node_positions_a = np.zeros(len(sig_node_locs_a))
    sig_node_positions_a[0] = 0
    n_sb = 0
    for i in range(1, len(sig_node_locs_a)-1, 2):
        sig_node_positions_a[i] = sig_sub_blocks_sums[n_sb]
        sig_node_positions_a[i+1] = sig_sub_blocks_sums[n_sb]
        n_sb += 1
    sig_node_positions_a[-1] = sig_sub_blocks_sums[-1]

    trac_node_indices_b = np.copy(trac_node_locs_b)
    trac_node_positions_b = np.copy(trac_node_positions)

    sig_sub_blocks_sums = np.cumsum(sig_sub_blocks)
    sig_node_indices_b = np.copy(sig_node_locs_b)
    sig_node_positions_b = np.zeros(len(sig_node_locs_b))
    sig_node_positions_b[0] = 0
    n_sb = 0
    for i in range(1, len(sig_node_locs_b) - 1, 2):
        sig_node_positions_b[i] = sig_sub_blocks_sums[n_sb]
        sig_node_positions_b[i + 1] = sig_sub_blocks_sums[n_sb]
        n_sb += 1
    sig_node_positions_b[-1] = sig_sub_blocks_sums[-1]

    if len(axle_pos_a) != 0:
        trac_axle_node_positions_a = np.array(sorted(axle_pos_a.flatten()))
        starting_node = sig_node_indices_b[-1] + 1
        trac_axle_node_indices_a = np.arange(starting_node, starting_node + len(trac_axle_node_positions_a))

        sig_axle_node_positions_a = np.array(sorted(axle_pos_a.flatten()))
        starting_node = trac_axle_node_indices_a[-1] + 1
        sig_axle_node_indices_a = np.arange(starting_node, starting_node + len(sig_axle_node_positions_a))
    else:
        pass

    if len(axle_pos_b) != 0:
        trac_axle_node_positions_b = np.array(sorted(axle_pos_b.flatten()))
        if len(axle_pos_a) != 0:
            starting_node = sig_axle_node_indices_a[-1] + 1
        else:
            starting_node = sig_node_indices_b[-1] + 1
        trac_axle_node_indices_b = np.arange(starting_node, starting_node + len(trac_axle_node_positions_b))

        sig_axle_node_positions_b = np.array(sorted(axle_pos_b.flatten()))
        starting_node = trac_axle_node_indices_b[-1] + 1
        sig_axle_node_indices_b = np.arange(starting_node, starting_node + len(sig_axle_node_positions_b))
    else:
        pass

    if len(axle_pos_a) != 0 and len(axle_pos_b) != 0:
        all_trac_node_positions_a = np.hstack((trac_node_positions_a, trac_axle_node_positions_a))
        all_sig_node_positions_a = np.hstack((sig_node_positions_a, sig_axle_node_positions_a))
        all_trac_node_positions_b = np.hstack((trac_node_positions_b, trac_axle_node_positions_b))
        all_sig_node_positions_b = np.hstack((sig_node_positions_b, sig_axle_node_positions_b))
        all_trac_node_indices_a = np.hstack((trac_node_indices_a, trac_axle_node_indices_a))
        all_sig_node_indices_a = np.hstack((sig_node_indices_a, sig_axle_node_indices_a))
        all_trac_node_indices_b = np.hstack((trac_node_indices_b, trac_axle_node_indices_b))
        all_sig_node_indices_b = np.hstack((sig_node_indices_b, sig_axle_node_indices_b))

    elif len(axle_pos_a) != 0 and len(axle_pos_b) == 0:
        all_trac_node_positions_a = np.hstack((trac_node_positions_a, trac_axle_node_positions_a))
        all_sig_node_positions_a = np.hstack((sig_node_positions_a, sig_axle_node_positions_a))
        all_trac_node_positions_b = np.copy(trac_node_positions_b)
        all_sig_node_positions_b = np.copy(sig_node_positions_b)
        all_trac_node_indices_a = np.hstack((trac_node_indices_a, trac_axle_node_indices_a))
        all_sig_node_indices_a = np.hstack((sig_node_indices_a, sig_axle_node_indices_a))
        all_trac_node_indices_b = np.copy(trac_node_indices_b)
        all_sig_node_indices_b = np.copy(sig_node_indices_b)

    else:
        all_trac_node_positions_a = np.copy(trac_node_positions_a)
        all_sig_node_positions_a = np.copy(sig_node_positions_a)
        all_trac_node_positions_b = np.hstack((trac_node_positions_b, trac_axle_node_positions_b))
        all_sig_node_positions_b = np.hstack((sig_node_positions_b, sig_axle_node_positions_b))
        all_trac_node_indices_a = np.copy(trac_node_indices_a)
        all_sig_node_indices_a = np.copy(sig_node_indices_a)
        all_trac_node_indices_b = np.hstack((trac_node_indices_b, trac_axle_node_indices_b))
        all_sig_node_indices_b = np.hstack((sig_node_indices_b, sig_axle_node_indices_b))

    trac_a_dict = dict(zip(all_trac_node_indices_a, all_trac_node_positions_a))
    sig_a_dict = dict(zip(all_sig_node_indices_a, all_sig_node_positions_a))
    trac_b_dict = dict(zip(all_trac_node_indices_b, all_trac_node_positions_b))
    sig_b_dict = dict(zip(all_sig_node_indices_b, all_sig_node_positions_b))

    sorted_trac_a_dict = dict(sorted(trac_a_dict.items(), key=lambda item: item[1]))
    sorted_sig_a_dict = dict(sorted(sig_a_dict.items(), key=lambda item: item[1]))
    sorted_trac_b_dict = dict(sorted(trac_b_dict.items(), key=lambda item: item[1]))
    sorted_sig_b_dict = dict(sorted(sig_b_dict.items(), key=lambda item: item[1]))

    all_trac_node_indices_a = list(sorted_trac_a_dict.keys())
    all_sig_node_indices_a = list(sorted_sig_a_dict.keys())
    all_trac_node_indices_b = list(sorted_trac_b_dict.keys())
    all_sig_node_indices_b = list(sorted_sig_b_dict.keys())

    # Make a new zeroed admittance matrix for the new restructured network
    n_nodes_restructured = len(all_trac_node_indices_a) + len(all_sig_node_indices_a) + len(all_trac_node_indices_b) + len(all_sig_node_indices_b)
    y_matrix_restructured = np.zeros((n_nodes_restructured, n_nodes_restructured))

    # Load in the nodal admittance matrix of the original network
    y_matrix = np.load("data\\network_parameters\\" + section_name + "\\nodal_admittance_matrix_" + section_name + "_" + conditions + ".npy")

    # Place values from the original network into the restructured network
    y_matrix_restructured[0:n_nodes, 0:n_nodes] = y_matrix

    # Find which nodes need to be calculated or recalculated
    # "a" first
    recalculate_trac_node_a = []  # Traction rail nodes first
    # First node
    if all_trac_node_indices_a[1] in trac_axle_node_indices_a:
        recalculate_trac_node_a.append(all_trac_node_indices_a[0])
    else:
        pass
    # Middle nodes
    for i in range(1, len(all_trac_node_indices_a) - 1):
        if (all_trac_node_indices_a[i - 1] in trac_axle_node_indices_a) or (all_trac_node_indices_a[i] in trac_axle_node_indices_a) or (all_trac_node_indices_a[i + 1] in trac_axle_node_indices_a):
            recalculate_trac_node_a.append(all_trac_node_indices_a[i])
        else:
            pass
    # Last nodes
    if all_trac_node_indices_a[-2] in trac_axle_node_indices_a:
        recalculate_trac_node_a.append(all_trac_node_indices_a[-1])
    else:
        pass
    recalculate_sig_node_a = []  # Signalling rail nodes second
    # First node
    if all_sig_node_indices_a[1] in sig_axle_node_indices_a:
        recalculate_sig_node_a.append(all_sig_node_indices_a[0])
    else:
        pass
    # Middle nodes
    for i in range(1, len(all_sig_node_indices_a) - 1):
        if (all_sig_node_indices_a[i - 1] in sig_axle_node_indices_a) or (all_sig_node_indices_a[i] in sig_axle_node_indices_a) or (all_sig_node_indices_a[i + 1] in sig_axle_node_indices_a):
            recalculate_sig_node_a.append(all_sig_node_indices_a[i])
        else:
            pass
    # Last nodes
    if all_sig_node_indices_a[-2] in sig_axle_node_indices_a:
        recalculate_sig_node_a.append(all_sig_node_indices_a[-1])
    else:
        pass
    # "b" second
    recalculate_trac_node_b = []  # Traction rail nodes first
    # First node
    if all_trac_node_indices_b[1] in trac_axle_node_indices_b:
        recalculate_trac_node_b.append(all_trac_node_indices_b[0])
    else:
        pass
    # Middle nodes
    for i in range(1, len(all_trac_node_indices_b) - 1):
        if (all_trac_node_indices_b[i - 1] in trac_axle_node_indices_b) or (all_trac_node_indices_b[i] in trac_axle_node_indices_b) or (all_trac_node_indices_b[i + 1] in trac_axle_node_indices_b):
            recalculate_trac_node_b.append(all_trac_node_indices_b[i])
        else:
            pass
    # Last nodes
    if all_trac_node_indices_b[-2] in trac_axle_node_indices_b:
        recalculate_trac_node_b.append(all_trac_node_indices_b[-1])
    else:
        pass
    recalculate_sig_node_b = []  # Signalling rail nodes second
    # First node
    if all_sig_node_indices_b[1] in sig_axle_node_indices_b:
        recalculate_sig_node_b.append(all_sig_node_indices_b[0])
    else:
        pass
    # Middle nodes
    for i in range(1, len(all_sig_node_indices_b) - 1):
        if (all_sig_node_indices_b[i - 1] in sig_axle_node_indices_b) or (all_sig_node_indices_b[i] in sig_axle_node_indices_b) or (all_sig_node_indices_b[i + 1] in sig_axle_node_indices_b):
            recalculate_sig_node_b.append(all_sig_node_indices_b[i])
        else:
            pass
    # Last nodes
    if all_sig_node_indices_b[-2] in sig_axle_node_indices_b:
        recalculate_sig_node_b.append(all_sig_node_indices_b[-1])
    else:
        pass

    # Recalculate the equivalent pi-circuit parameters for the new network
    # Get the sorted nodal positions
    all_trac_node_positions_sorted_a = list(sorted_trac_a_dict.values())
    all_sig_node_positions_sorted_a = list(sorted_sig_a_dict.values())
    all_trac_node_positions_sorted_b = list(sorted_trac_b_dict.values())
    all_sig_node_positions_sorted_b = list(sorted_sig_b_dict.values())

    # Calculate the length of the sub blocks
    trac_sub_blocks_a = np.diff(all_trac_node_positions_sorted_a)
    sig_sub_blocks_a = np.diff(all_sig_node_positions_sorted_a)
    sig_sub_blocks_a[sig_sub_blocks_a == 0] = np.nan  # Sub blocks with length zero on the signalling rail indicate insulating rail joints, these need to be nans
    trac_sub_blocks_b = np.diff(all_trac_node_positions_sorted_b)
    sig_sub_blocks_b = np.diff(all_sig_node_positions_sorted_b)
    sig_sub_blocks_b[sig_sub_blocks_b == 0] = np.nan

    # Set up equivalent pi-parameters
    # "a" first
    ye_sig_a = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))  # Series admittance for signalling rail
    ye_trac_a = 1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_a))  # Series admittance for traction return rail
    yg_sig_a = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_a) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_a))))  # Parallel admittance for signalling rail
    yg_trac_a = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_a) - 1) * (1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_a))))  # Parallel admittance for traction return rail
    # "b" second
    ye_sig_b = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))  # Series admittance for signalling rail
    ye_trac_b = 1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_b))  # Series admittance for traction return rail
    yg_sig_b = 2 * ((np.cosh(gamma_sig * sig_sub_blocks_b) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks_b))))  # Parallel admittance for signalling rail
    yg_trac_b = 2 * ((np.cosh(gamma_trac * trac_sub_blocks_b) - 1) * (1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks_b))))  # Parallel admittance for traction return rail

    # Recalculate nodal parallel admittances
    # "a" first
    recalculate_yg_trac_a = np.zeros(len(recalculate_trac_node_a))  # Traction return rail
    n = 0
    for i in recalculate_trac_node_a:
        j = np.argwhere(all_trac_node_indices_a == i)
        # If it's the first node
        if i == trac_node_locs_a[0]:
            recalculate_yg_trac_a[n] = 0.5 * yg_trac_a[j]
            n += 1
        # If it's the last node
        elif i == trac_node_locs_a[-1]:
            recalculate_yg_trac_a[n] = 0.5 * yg_trac_a[j - 1]
        # Otherwise
        else:
            recalculate_yg_trac_a[n] = (0.5 * yg_trac_a[j - 1]) + (0.5 * yg_trac_a[j])
            n += 1
    recalculate_yg_sig_a = np.zeros(len(recalculate_sig_node_a))  # Signalling rail
    n = 0
    for i in recalculate_sig_node_a:
        j = np.argwhere(all_sig_node_indices_a == i)
        if i in sig_node_locs_relay_a:
            recalculate_yg_sig_a[n] = 0.5 * yg_sig_a[j]
            n += 1
        elif i in sig_node_locs_power_a:
            recalculate_yg_sig_a[n] = 0.5 * yg_sig_a[j - 1]
            n += 1
        else:
            recalculate_yg_sig_a[n] = (0.5 * yg_sig_a[j - 1]) + (0.5 * yg_sig_a[j])
            n += 1
    # "b" second
    recalculate_yg_trac_b = np.zeros(len(recalculate_trac_node_b))  # Traction return rail
    n = 0
    for i in recalculate_trac_node_b:
        j = np.argwhere(all_trac_node_indices_b == i)
        # If it's the first node
        if i == trac_node_locs_b[0]:
            recalculate_yg_trac_b[n] = 0.5 * yg_trac_b[j]
            n += 1
        # If it's the last node
        elif i == trac_node_locs_b[-1]:
            recalculate_yg_trac_b[n] = 0.5 * yg_trac_b[j - 1]
        # Otherwise
        else:
            recalculate_yg_trac_b[n] = (0.5 * yg_trac_b[j - 1]) + (0.5 * yg_trac_b[j])
            n += 1
    recalculate_yg_sig_b = np.zeros(len(recalculate_sig_node_b))  # Signalling rail
    n = 0
    for i in recalculate_sig_node_b:
        j = np.argwhere(all_sig_node_indices_b == i)
        if i in sig_node_locs_power_b:
            recalculate_yg_sig_b[n] = 0.5 * yg_sig_b[j]
            n += 1
        elif i in sig_node_locs_relay_b:
            recalculate_yg_sig_b[n] = 0.5 * yg_sig_b[j - 1]
            n += 1
        else:
            recalculate_yg_sig_b[n] = (0.5 * yg_sig_b[j - 1]) + (0.5 * yg_sig_b[j])
            n += 1

    # Recalculate sum of parallel admittances into nodes
    # "a" first
    recalculate_y_sum_trac_a = np.zeros(len(recalculate_trac_node_a))  # Traction return rail
    n = 0
    for i in recalculate_trac_node_a:
        j = np.argwhere(all_trac_node_indices_a == i)
        # If it's the first node
        if i == all_trac_node_indices_a[0]:
            recalculate_y_sum_trac_a[n] = recalculate_yg_trac_a[n] + parameters["y_relay"] + ye_trac_a[j]
            n += 1
        # If it's an axle node
        elif i in trac_axle_node_indices_a:
            recalculate_y_sum_trac_a[n] = recalculate_yg_trac_a[n] + parameters["y_axle"] + ye_trac_a[j - 1] + ye_trac_a[j]
            n += 1
        # If it's a cross bond node
        elif i in cb_node_locs_a:
            recalculate_y_sum_trac_a[n] = recalculate_yg_trac_a[n] + parameters["y_cb"] + ye_trac_a[j - 1] + ye_trac_a[j]
            n += 1
        # If it's the last node
        elif i == all_trac_node_indices_a[-1]:
            recalculate_y_sum_trac_a[n] = recalculate_yg_trac_a[n] + parameters["y_power"] + ye_trac_a[j - 1]
            n += 1
        # If it's a normal middle node
        else:
            recalculate_y_sum_trac_a[n] = recalculate_yg_trac_a[n] + parameters["y_power"] + parameters["y_relay"] + ye_trac_a[j - 1] + ye_trac_a[j]
            n += 1
    recalculate_y_sum_sig_a = np.zeros(len(recalculate_sig_node_a))  # Signalling rail
    n = 0
    for i in recalculate_sig_node_a:
        j = np.argwhere(all_sig_node_indices_a == i)
        # If it's a relay node
        if i in sig_node_locs_relay_a:
            recalculate_y_sum_sig_a[n] = recalculate_yg_sig_a[n] + parameters["y_relay"] + ye_sig_a[j]
            n += 1
        # If it's a power supply node
        elif i in sig_node_locs_power_a:
            recalculate_y_sum_sig_a[n] = recalculate_yg_sig_a[n] + parameters["y_power"] + ye_sig_a[j - 1]
            n += 1
        # If it's an axle node
        elif i in sig_axle_node_indices_a:
            recalculate_y_sum_sig_a[n] = recalculate_yg_sig_a[n] + parameters["y_axle"] + ye_sig_a[j - 1] + ye_sig_a[j]
            n += 1
        else:
            print("Error")
    # "b" second
    recalculate_y_sum_trac_b = np.zeros(len(recalculate_trac_node_b))  # Traction return rail
    n = 0
    for i in recalculate_trac_node_b:
        j = np.argwhere(all_trac_node_indices_b == i)
        # If it's the first node
        if i == all_trac_node_indices_b[0]:
            recalculate_y_sum_trac_b[n] = recalculate_yg_trac_b[n] + parameters["y_power"] + ye_trac_b[j]
            n += 1
        # If it's an axle node
        elif i in trac_axle_node_indices_b:
            recalculate_y_sum_trac_b[n] = recalculate_yg_trac_b[n] + parameters["y_axle"] + ye_trac_b[j - 1] + ye_trac_b[j]
            n += 1
        # If it's a cross bond node
        elif i in cb_node_locs_b:
            recalculate_y_sum_trac_b[n] = recalculate_yg_trac_b[n] + parameters["y_cb"] + ye_trac_b[j - 1] + ye_trac_b[j]
            n += 1
        # If it's the last node
        elif i == all_trac_node_indices_b[-1]:
            recalculate_y_sum_trac_b[n] = recalculate_yg_trac_b[n] + parameters["y_relay"] + ye_trac_b[j - 1]
            n += 1
        # If it's a normal middle node
        else:
            recalculate_y_sum_trac_b[n] = recalculate_yg_trac_b[n] + parameters["y_power"] + parameters["y_relay"] + ye_trac_b[j - 1] + ye_trac_b[j]
            n += 1
    recalculate_y_sum_sig_b = np.zeros(len(recalculate_sig_node_b))  # Signalling rail
    n = 0
    for i in recalculate_sig_node_b:
        j = np.argwhere(all_sig_node_indices_b == i)
        # If it's a relay node
        if i in sig_node_locs_relay_b:
            recalculate_y_sum_sig_b[n] = recalculate_yg_sig_b[n] + parameters["y_relay"] + ye_sig_b[j - 1]
            n += 1
        # If it's a power supply node
        elif i in sig_node_locs_power_b:
            recalculate_y_sum_sig_b[n] = recalculate_yg_sig_b[n] + parameters["y_power"] + ye_sig_b[j]
            n += 1
        # If it's an axle node
        elif i in sig_axle_node_indices_b:
            recalculate_y_sum_sig_b[n] = recalculate_yg_sig_b[n] + parameters["y_axle"] + ye_sig_b[j - 1] + ye_sig_b[j]
            n += 1
        else:
            print("Error")

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
    y_matrix_restructured[np.array(all_trac_node_indices_a)[i], np.array(all_trac_node_indices_a)[i + 1]] = y_matrix_restructured[np.array(all_trac_node_indices_a)[i + 1], np.array(all_trac_node_indices_a)[i]] = -ye_trac_a[i]
    i = np.where(np.isin(all_sig_node_indices_a, recalculate_sig_node_a[:-1]))[0]
    y_matrix_restructured[np.array(all_sig_node_indices_a)[i], np.array(all_sig_node_indices_a)[i + 1]] = y_matrix_restructured[np.array(all_sig_node_indices_a)[i + 1], np.array(all_sig_node_indices_a)[i]] = -ye_sig_a[i]
    # "b" second
    i = np.where(np.isin(all_trac_node_indices_b, recalculate_trac_node_b[:-1]))[0]
    y_matrix_restructured[np.array(all_trac_node_indices_b)[i], np.array(all_trac_node_indices_b)[i + 1]] = y_matrix_restructured[np.array(all_trac_node_indices_b)[i + 1], np.array(all_trac_node_indices_b)[i]] = -ye_trac_b[i]
    i = np.where(np.isin(all_sig_node_indices_b, recalculate_sig_node_b[:-1]))[0]
    y_matrix_restructured[np.array(all_sig_node_indices_b)[i], np.array(all_sig_node_indices_b)[i + 1]] = y_matrix_restructured[np.array(all_sig_node_indices_b)[i + 1], np.array(all_sig_node_indices_b)[i]] = -ye_sig_b[i]
    y_matrix_restructured[np.isnan(y_matrix_restructured)] = 0  # Set any NaN values from the IRJs to 0

    # Axle admittances
    # "a" first
    y_matrix_restructured[trac_axle_node_indices_a, sig_axle_node_indices_a] = -parameters["y_axle"]
    y_matrix_restructured[sig_axle_node_indices_a, trac_axle_node_indices_a] = -parameters["y_axle"]
    # "b" second
    y_matrix_restructured[trac_axle_node_indices_b, sig_axle_node_indices_b] = -parameters["y_axle"]
    y_matrix_restructured[sig_axle_node_indices_b, trac_axle_node_indices_b] = -parameters["y_axle"]

    # Restructure angles array based on the new sub blocks
    # "a" first
    cumsum_trac_sb_a = np.cumsum(trac_sub_blocks_a)
    block_indices_trac_a = np.searchsorted(sig_sub_blocks_sums, cumsum_trac_sb_a)
    trac_sb_angles_a = sig_angles_a[block_indices_trac_a]
    cumsum_sig_sb_a = np.cumsum(sig_sub_blocks_a[~np.isnan(sig_sub_blocks_a)])
    block_indices_sig_a = np.searchsorted(sig_sub_blocks_sums, cumsum_sig_sb_a)
    sig_sb_angles_a = sig_angles_a[block_indices_sig_a]
    # "b" second
    cumsum_trac_sb_b = np.cumsum(trac_sub_blocks_b)
    block_indices_trac_b = np.searchsorted(sig_sub_blocks_sums, cumsum_trac_sb_b)
    trac_sb_angles_b = sig_angles_b[block_indices_trac_b]
    cumsum_sig_sb_b = np.cumsum(sig_sub_blocks_b[~np.isnan(sig_sub_blocks_b)])
    block_indices_sig_b = np.searchsorted(sig_sub_blocks_sums, cumsum_sig_sb_b)
    sig_sb_angles_b = sig_angles_b[block_indices_sig_b]

    # Currents
    i_relays_all_a = np.zeros([len(exs), len(sig_angles_a)])
    i_relays_all_b = np.zeros([len(exs), len(sig_angles_a)])
    for es in range(0, len(exs)):
        # Set up current matrix
        j_matrix = np.zeros(n_nodes)
        ex = exs[es]
        ey = eys[es]

        # "a" first
        e_x_par_trac_a = ex * np.cos(0.5 * np.pi - trac_angles_a)
        e_y_par_trac_a = ey * np.cos(trac_angles_a)
        e_par_trac_a = e_x_par_trac_a + e_y_par_trac_a
        e_x_par_sig_a = ex * np.cos(0.5 * np.pi - sig_angles_a)
        e_y_par_sig_a = ey * np.cos(sig_angles_a)
        e_par_sig_a = e_x_par_sig_a + e_y_par_sig_a
        i_sig_a = e_par_sig_a / parameters["z_sig"]
        i_trac_a = e_par_trac_a / parameters["z_trac"]

        # "b" second
        e_x_par_trac_b = ex * np.cos(0.5 * np.pi - trac_angles_b)
        e_y_par_trac_b = ey * np.cos(trac_angles_b)
        e_par_trac_b = e_x_par_trac_b + e_y_par_trac_b
        e_x_par_sig_b = ex * np.cos(0.5 * np.pi - sig_angles_b)
        e_y_par_sig_b = ey * np.cos(sig_angles_b)
        e_par_sig_b = e_x_par_sig_b + e_y_par_sig_b
        i_sig_b = e_par_sig_b / parameters["z_sig"]
        i_trac_b = e_par_trac_b / parameters["z_trac"]

        # "a" first
        index_sb = np.arange(0, n_nodes_trac, 1)
        # Traction return rail first node
        j_matrix[trac_node_locs_a[0]] = -i_trac_a[0]
        # Traction return rail centre nodes
        for i in index_sb[1:-1]:
            pos = trac_node_locs_a[i]
            if pos in cb_node_locs_a:
                j_matrix[pos] = i_trac_a[i - 1] - i_trac_a[i]
            elif pos not in cb_node_locs_a:
                j_matrix[pos] = i_trac_a[i - 1] - i_trac_a[i] - parameters["i_power"]
            else:
                print("Error")
        # Traction return rail last node
        j_matrix[trac_node_locs_a[-1]] = i_trac_a[-1] - parameters["i_power"]
        # Signalling rail nodes
        n_sb = 0
        for i in sig_node_locs_a:
            if i in sig_node_locs_relay_a:
                j_matrix[i] = -i_sig_a[n_sb]
            elif i in sig_node_locs_power_a:
                j_matrix[i] = i_sig_a[n_sb] + parameters["i_power"]
                n_sb = n_sb + 1
            else:
                print("Error")

        # "b" second
        index_sb = np.arange(0, n_nodes_trac, 1)
        # Traction return rail first node
        j_matrix[trac_node_locs_b[0]] = i_trac_b[0] - parameters["i_power"]
        # Traction return rail centre nodes
        for i in index_sb[1:-1]:
            pos = trac_node_locs_b[i]
            if pos in cb_node_locs_b:
                j_matrix[pos] = i_trac_b[i] - i_trac_b[i - 1]
            elif pos not in cb_node_locs_b:
                j_matrix[pos] = i_trac_b[i] - i_trac_b[i - 1] - parameters["i_power"]
            else:
                print("Error")
        # Traction return rail last node
        j_matrix[trac_node_locs_b[-1]] = -i_trac_b[-1]

        # Signalling rail nodes
        n_sb = 0
        for i in sig_node_locs_b:
            if i in sig_node_locs_power_b:
                j_matrix[i] = parameters["i_power"] + i_sig_b[n_sb]
            elif i in sig_node_locs_relay_b:
                j_matrix[i] = -i_sig_b[n_sb]
                n_sb = n_sb + 1
            else:
                print("Error")

        # Calculate voltage matrix
        # Calculate inverse of admittance matrix
        y_matrix_inv = np.linalg.inv(y_matrix_restructured)

        # Calculate nodal voltages
        v_matrix = np.matmul(y_matrix_inv, j_matrix)

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

        i_relays_all_a[es, :] = i_relays_a
        i_relays_all_b[es, :] = i_relays_b

    return i_relays_all_a, i_relays_all_b


axle_positions = np.load("axle_positions.npz")
axle_positions_a = axle_positions["axle_pos_a"]
axle_positions_b = axle_positions["axle_pos_b"]
ex = np.array([0])
ey = np.array([0])
wrong_side_two_track("test", "moderate", axle_positions_a, axle_positions_b, ex, ey)
