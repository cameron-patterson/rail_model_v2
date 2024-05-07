import numpy as np


def axle_pos_dist_both_sides(start_pos_a, start_pos_b, n_carriages):
    train_axle_dim = np.array([0, 2, 17.5, 19.5])  # with next carriage starting at 24m along
    axle_locs = []
    for i in range(0, n_carriages):
        axle_locs = np.append(axle_locs, train_axle_dim + (i * 24))
    axle_locs = axle_locs / 1000
    axle_locs = 0 - axle_locs

    axle_pos_a = []
    axle_pos_b = []
    for st_a in start_pos_a:
        axle_pos_a.append(st_a + axle_locs)
    for st_b in start_pos_b:
        axle_pos_b.append(st_b - axle_locs)

    return axle_pos_a, axle_pos_b


def reconfigure_network_two_track(section_name, conditions, train_locs_a, train_locs_b):
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
    blocks_sum_cb = sub_block_lengths["blocks_sum_cb"]
    trac_sub_blocks = sub_block_lengths["trac_sub_blocks"]
    sig_sub_blocks = sub_block_lengths["sig_sub_blocks"]
    sub_block_angles = np.load("data\\network_parameters\\" + section_name + "\\angles_" + section_name + ".npz")
    trac_angles_a = sub_block_angles["trac_angles_a"]
    trac_angles_b = sub_block_angles["trac_angles_b"]
    sig_angles_a = sub_block_angles["sig_angles_a"]
    sig_angles_b = sub_block_angles["sig_angles_b"]

    # Load in section network node indices
    network_nodes = np.load("data\\network_parameters\\" + section_name + "\\nodes_" + section_name + ".npz")
    n_nodes = network_nodes["n_nodes"]
    n_nodes_trac = network_nodes["n_nodes_trac"]
    trac_node_locs_a = network_nodes["trac_node_locs_a"]
    trac_node_locs_b = network_nodes["trac_node_locs_b"]
    sig_node_locs_a = network_nodes["sig_node_locs_a"]
    sig_node_locs_b = network_nodes["sig_node_locs_b"]
    cb_node_locs_a = network_nodes["cb_node_locs_a"]
    cb_node_locs_b = network_nodes["cb_node_locs_b"]
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

    # Calculate train axle positions
    trains_a, trains_b = axle_pos_dist_both_sides(train_locs_a, train_locs_b, n_carriages=8)

    # Calculate the new number of nodes in the network and set axle node indices
    n_nodes_restructured = 0 + n_nodes
    # "a" first
    axle_pos_a = []
    for i in range(0, len(trains_a)):
        train_a = trains_a[i]
        n_nodes_restructured += len(train_a) * 2
        axle_pos_a.append(train_a)
        axle_pos_a = np.flip(axle_pos_a[0])
        trac_node_locs_axle_a = np.arange(n_nodes, n_nodes + ((n_nodes_restructured - n_nodes) / 2)).astype(int)  # Index of traction return rail axle nodes for "a"
        sig_node_locs_axle_a = np.arange(trac_node_locs_axle_a[-1] + 1, n_nodes_restructured).astype(int)  # Index of signalling rail axle nodes for "a"

    # "b" second
    axle_pos_b = []
    for i in range(0, len(trains_b)):
        train_b = trains_b[i]
        n_nodes_restructured += len(train_b) * 2
        axle_pos_b.append(train_b)
    axle_pos_b = axle_pos_b[0]
    trac_node_locs_axle_b = np.arange(sig_node_locs_axle_a[-1] + 1, sig_node_locs_axle_a[-1] + ((n_nodes_restructured - sig_node_locs_axle_a[-1]) / 2))  # Index of traction return rail axle nodes for "b"
    sig_node_locs_axle_b = np.arange(trac_node_locs_axle_b[-1] + 1, n_nodes_restructured)    # Index of signalling rail axle nodes for "b"

    # Make a new zeroed admittance matrix for the new restructured network
    y_matrix_restructured = np.zeros((n_nodes_restructured, n_nodes_restructured))

    # Load in the nodal admittance matrix of the original network
    y_matrix = np.load("data\\network_parameters\\" + section_name + "\\nodal_admittance_matrix_test_" + conditions + ".npy")

    # Place values from the original network into the restructured network
    y_matrix_restructured[0:n_nodes, 0:n_nodes] = y_matrix

    # Rebuild admittance matrix
    # Axles (new)
    # "a" first
    for i, j in zip(trac_node_locs_axle_a, sig_node_locs_axle_a):
        y_matrix_restructured[int(i), int(j)] = -parameters["y_axle"]
        y_matrix_restructured[int(j), int(i)] = -parameters["y_axle"]

    # "b" second
    for i, j in zip(trac_node_locs_axle_b, sig_node_locs_axle_b):
        y_matrix_restructured[int(i), int(j)] = -parameters["y_axle"]
        y_matrix_restructured[int(j), int(i)] = -parameters["y_axle"]

    # New sub block lengths and length sums
    trac_sub_blocks_sum = np.cumsum(trac_sub_blocks)
    sig_sub_blocks_sum = np.cumsum(sig_sub_blocks)
    for train_a in trains_a:
        trac_sub_blocks_sum_a = np.sort(np.concatenate((trac_sub_blocks_sum, train_a)))
        trac_sub_blocks_sum_a = np.insert(trac_sub_blocks_sum_a, 0, 0)
        trac_sub_blocks_a = np.diff(trac_sub_blocks_sum_a)  # Traction return rail sub block lengths
        sig_sub_blocks_sum_a = np.sort(np.concatenate((sig_sub_blocks_sum, train_a)))
        sig_sub_blocks_sum_a = np.insert(sig_sub_blocks_sum_a, 0, 0)
        sig_sub_blocks_a = np.diff(sig_sub_blocks_sum_a)  # Signalling rail sub block lengths
    for train_b in trains_b:
        trac_sub_blocks_sum_b = np.sort(np.concatenate((trac_sub_blocks_sum, train_b)))
        trac_sub_blocks_sum_b = np.insert(trac_sub_blocks_sum_b, 0, 0)
        trac_sub_blocks_b = np.diff(trac_sub_blocks_sum_b)  # Traction return rail sub block lengths
        sig_sub_blocks_sum_b = np.sort(np.concatenate((sig_sub_blocks_sum, train_b)))
        sig_sub_blocks_sum_b = np.insert(sig_sub_blocks_sum_b, 0, 0)
        sig_sub_blocks_b = np.diff(sig_sub_blocks_sum_b)  # Signalling rail sub block lengths

    # Set sub block angles for the traction return and signalling rails
    # "a" first
    trac_angles_a = np.zeros(len(trac_sub_blocks_a))
    cumsum_sb_a = np.cumsum(trac_sub_blocks_a)
    n_b = 0
    for n_sb in range(0, len(cumsum_sb_a)):
        if cumsum_sb_a[n_sb] < sig_sub_blocks_sum[n_b]:
            trac_angles_a[n_sb] = sig_angles_a[n_b]
        elif cumsum_sb_a[n_sb] == sig_sub_blocks_sum[n_b]:
            trac_angles_a[n_sb] = sig_angles_a[n_b]
            n_b += 1
        else:
            print("Error")

    sig_angles_a = np.zeros(len(sig_sub_blocks_a))
    cumsum_sb_a = np.cumsum(sig_sub_blocks_a)
    n_b = 0
    for n_sb in range(0, len(cumsum_sb_a)):
        if cumsum_sb_a[n_sb] < sig_sub_blocks_sum[n_b]:
            sig_angles_a[n_sb] = sig_angles_a[n_b]
        elif cumsum_sb_a[n_sb] == sig_sub_blocks_sum[n_b]:
            sig_angles_a[n_sb] = sig_angles_a[n_b]
            n_b += 1
        else:
            print("Error")
    # "b" second
    trac_angles_b = np.zeros(len(trac_sub_blocks_b))
    cumsum_sb_b = np.cumsum(trac_sub_blocks_b)
    n_b = 0
    for n_sb in range(0, len(cumsum_sb_b)):
        if cumsum_sb_b[n_sb] < sig_sub_blocks_sum[n_b]:
            trac_angles_b[n_sb] = sig_angles_b[n_b]
        elif cumsum_sb_b[n_sb] == sig_sub_blocks_sum[n_b]:
            trac_angles_b[n_sb] = sig_angles_b[n_b]
            n_b += 1
        else:
            print("Error")
    trac_angles_b = trac_angles_b - np.pi

    sig_angles_b = np.zeros(len(sig_sub_blocks_b))
    cumsum_sb_b = np.cumsum(sig_sub_blocks_b)
    n_b = 0
    for n_sb in range(0, len(cumsum_sb_b)):
        if cumsum_sb_b[n_sb] < sig_sub_blocks_sum[n_b]:
            sig_angles_b[n_sb] = sig_angles_b[n_b]
        elif cumsum_sb_b[n_sb] == sig_sub_blocks_sum[n_b]:
            sig_angles_b[n_sb] = sig_angles_b[n_b]
            n_b += 1
        else:
            print("Error")
    sig_angles_b = sig_angles_b - np.pi

    # Set up equivalent-pi parameters for the new sub blocks
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

    # Calculate indices for restructured network
    # "Traction return rail
    # "a" first
    trac_sub_blocks_index_a = np.zeros(len(trac_sub_blocks_sum_a)).astype(int)
    for i in range(0, len(blocks_sum_cb)):
        trac_sub_blocks_index_a[np.argwhere(trac_sub_blocks_sum_a == blocks_sum_cb[i])] = trac_node_locs_a[i]
    for i in range(0, len(axle_pos_a)):
        trac_sub_blocks_index_a[np.argwhere(trac_sub_blocks_sum_a == axle_pos_a[i])] = trac_node_locs_axle_a[i]
    # "b" second
    trac_sub_blocks_index_b = np.zeros(len(trac_sub_blocks_sum_b)).astype(int)
    for i in range(0, len(blocks_sum_cb)):
        trac_sub_blocks_index_b[np.argwhere(trac_sub_blocks_sum_b == blocks_sum_cb[i])] = trac_node_locs_b[i]
    for i in range(0, len(axle_pos_b)):
        trac_sub_blocks_index_b[np.argwhere(trac_sub_blocks_sum_b == axle_pos_b[i])] = trac_node_locs_axle_b[i]
    # "Signalling rail
    # "a" first
    print(sig_sub_blocks_sum)
    print(sig_sub_blocks_sum_a)


    # Find the index of nodes that need to be recalculated
    # "Traction return rail
    # "a" first
    trac_nodes_to_recalculate_a = []
    if trac_sub_blocks_index_a[0] in trac_node_locs_axle_a or trac_sub_blocks_index_a[1] in trac_node_locs_axle_a:
        trac_nodes_to_recalculate_a.append(trac_sub_blocks_index_a[0])
    else:
        pass
    for i in range(1, len(trac_sub_blocks_index_a)-1):
        if trac_sub_blocks_index_a[i-1] in trac_node_locs_axle_a or trac_sub_blocks_index_a[i] in trac_node_locs_axle_a or trac_sub_blocks_index_a[i+1] in trac_node_locs_axle_a:
            trac_nodes_to_recalculate_a.append(trac_sub_blocks_index_a[i])
        else:
            pass
    if trac_sub_blocks_index_a[-1] in trac_node_locs_axle_a or trac_sub_blocks_index_a[-1] in trac_node_locs_axle_a:
        trac_nodes_to_recalculate_a.append(trac_sub_blocks_index_a[-1])
    else:
        pass
    # "b" second
    trac_nodes_to_recalculate_b = []
    if trac_sub_blocks_index_b[0] in trac_node_locs_axle_b or trac_sub_blocks_index_b[1] in trac_node_locs_axle_b:
        trac_nodes_to_recalculate_b.append(trac_sub_blocks_index_b[0])
    else:
        pass
    for i in range(1, len(trac_sub_blocks_index_b) - 1):
        if trac_sub_blocks_index_b[i - 1] in trac_node_locs_axle_b or trac_sub_blocks_index_b[i] in trac_node_locs_axle_b or trac_sub_blocks_index_b[i + 1] in trac_node_locs_axle_b:
            trac_nodes_to_recalculate_b.append(trac_sub_blocks_index_b[i])
        else:
            pass
    if trac_sub_blocks_index_b[-1] in trac_node_locs_axle_b or trac_sub_blocks_index_b[-1] in trac_node_locs_axle_b:
        trac_nodes_to_recalculate_b.append(trac_sub_blocks_index_b[-1])
    else:
        pass

    #print(np.asarray(trac_nodes_to_recalculate_a))
    #print(np.asarray(trac_nodes_to_recalculate_b))


reconfigure_network_two_track("test", "moderate", [1.21], [1])
