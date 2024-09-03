import numpy as np


def build_network_two_track(section_name, conditions):
    # Create dictionary of network parameters
    parameters = {"z_sig": 0.0289,  # Signalling rail series impedance (ohms/km)
                  "z_trac": 0.0289,  # Traction return rail series impedance (ohms/km)
                  "y_sig_moderate": 0.1,  # Signalling rail parallel admittance for moderate conditions (siemens/km)
                  "y_trac_moderate": 1.6,  # Traction return rail parallel admittance in moderate conditions (siemens/km)
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

    # Calculate the electrical characteristics of the rails
    gamma_sig = np.sqrt(parameters["z_sig"] * parameters["y_sig_"+conditions])
    gamma_trac = np.sqrt(parameters["z_trac"] * parameters["y_trac_"+conditions])
    z0_sig = np.sqrt(parameters["z_sig"] / parameters["y_sig_"+conditions])
    z0_trac = np.sqrt(parameters["z_trac"] / parameters["y_trac_"+conditions])

    # Load in the lengths and angles of the track circuit blocks
    # Note: zero degrees is directly eastwards, with positive values counter-clockwise and negative values clockwise
    data = np.load("data/rail_data/" + section_name + "/" + section_name + "_distances_bearings.npz")
    blocks = data["distances"]
    bearings = data["bearings"]
    n_blocks = int(len(blocks))  # Number of blocks in this section
    blocks_sum = np.cumsum(blocks)  # Cumulative sum of block lengths

    # Add cross bonds, which split up the traction return rail blocks into sub blocks
    pos_cb = np.arange(0.4, np.sum(blocks), 0.4)  # Position of the cross bonds
    blocks_sum_cb = np.sort(np.concatenate((blocks_sum, pos_cb)))  # Cumulative sum of sub block lengths with cross bonds added

    # Since this is the base network with no trains added, the signalling rail sub blocks are equal to the blocks of
    # the route, and the traction return rail sub blocks are defined by the cross bond locations. Both directions of
    # travel have identical length sub block lengths, the differences in component configurations are handled later
    sig_sub_blocks = blocks  # Signalling rail sub block lengths
    blocks_sum_cb = np.insert(blocks_sum_cb, 0, 0)
    trac_sub_blocks = np.diff(blocks_sum_cb)  # Traction return rail sub block lengths
    # If cross bonds overlap with block boundaries, set the length of the sub block to be non-zero to avoid errors
    if 0 in trac_sub_blocks:
        trac_sub_blocks[np.argwhere(trac_sub_blocks == 0)] = 1e-10
    else:
        pass

    # Save in a zip file to be used in the analysis
    np.savez("sub_blocks_" + section_name, blocks_sum_cb=blocks_sum_cb, trac_sub_blocks=trac_sub_blocks, sig_sub_blocks=sig_sub_blocks)

    # Set up equivalent-pi parameters
    ye_sig = 1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks))  # Series admittance for signalling rail
    ye_trac = 1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks))  # Series admittance for traction return rail
    yg_sig = 2 * ((np.cosh(gamma_sig * sig_sub_blocks) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * sig_sub_blocks))))  # Parallel admittance for signalling rail
    yg_trac = 2 * ((np.cosh(gamma_trac * trac_sub_blocks) - 1) * (1 / (z0_trac * np.sinh(gamma_trac * trac_sub_blocks))))  # Parallel admittance for traction return rail

    # Calculate numbers of nodes ready to use in indexing
    n_nodes_single = int(n_blocks * 3 + 1 + len(pos_cb))  # Number of nodes in a single direction of travel
    n_nodes = n_nodes_single * 2  # Number of nodes in the whole network (*2 for two-track)
    n_nodes_trac = len(trac_sub_blocks) + 1  # Number of nodes in the traction return rail

    # Index of rail nodes in the traction return rail
    # Note: "a" and "b" are used to identify the opposite directions of travel in this network (two-track)
    trac_node_locs_a = np.arange(0, n_nodes_trac, 1).astype(int)
    trac_node_locs_b = np.arange(n_nodes_single, n_nodes_single + n_nodes_trac, 1).astype(int)

    # Index of rail nodes in the signalling rail
    sig_node_locs_a = np.arange(n_nodes_trac, n_nodes_single, 1).astype(int)
    sig_node_locs_b = np.arange(n_nodes_single + n_nodes_trac, n_nodes_single * 2, 1).astype(int)

    # Index of cross bond nodes
    # "a" first
    cb_node_locs = np.zeros(len(pos_cb))
    for i in range(0, len(pos_cb)):
        cb_node_locs[i] = np.argwhere(np.cumsum(trac_sub_blocks) == pos_cb[i])[0] + 1
    cb_node_locs = cb_node_locs.astype(int)
    cb_node_locs_a = trac_node_locs_a[cb_node_locs]
    # "b" second
    cb_node_locs = np.zeros(len(pos_cb))
    for i in range(0, len(pos_cb)):
        cb_node_locs[i] = np.argwhere(np.cumsum(trac_sub_blocks) == pos_cb[i])[0] + 1
    cb_node_locs = cb_node_locs.astype(int)
    cb_node_locs_b = trac_node_locs_b[cb_node_locs]

    # Index of traction return rail power supply and relay nodes
    # Note: "a" begins with a relay and ends with a power supply, "b" begins with a power supply and ends with a relay
    # "a" first
    trac_node_locs_no_cb = np.delete(trac_node_locs_a, cb_node_locs_a)
    trac_node_locs_power_a = trac_node_locs_no_cb[1:]
    trac_node_locs_relay_a = trac_node_locs_no_cb[:-1]
    # "b" second
    trac_node_locs_no_cb = np.delete(trac_node_locs_b, cb_node_locs_b - n_nodes_single)
    trac_node_locs_power_b = trac_node_locs_no_cb[:-1]
    trac_node_locs_relay_b = trac_node_locs_no_cb[1:]

    # Index of signalling rail power supply and relay nodes
    # "a" first
    sig_node_locs_power_a = sig_node_locs_a[1::2]
    sig_node_locs_relay_a = sig_node_locs_a[0::2]
    # "b" second
    sig_node_locs_power_b = sig_node_locs_b[0::2]
    sig_node_locs_relay_b = sig_node_locs_b[1::2]

    # Save network node locations for later analysis
    np.savez("nodes_"+section_name, n_nodes=n_nodes, n_nodes_trac=n_nodes_trac, trac_node_locs_a=trac_node_locs_a,
             trac_node_locs_b=trac_node_locs_b, sig_node_locs_a=sig_node_locs_a, sig_node_locs_b=sig_node_locs_b,
             cb_node_locs_a=cb_node_locs_a, cb_node_locs_b=cb_node_locs_b,
             trac_node_locs_power_a=trac_node_locs_power_a, trac_node_locs_power_b=trac_node_locs_power_b,
             trac_node_locs_relay_a=trac_node_locs_relay_a, trac_node_locs_relay_b=trac_node_locs_relay_b,
             sig_node_locs_power_a=sig_node_locs_power_a, sig_node_locs_power_b=sig_node_locs_power_b,
             sig_node_locs_relay_a=sig_node_locs_relay_a, sig_node_locs_relay_b=sig_node_locs_relay_b)

    # Calculate nodal parallel admittances
    yg = np.zeros(n_nodes)

    # "a" first
    # Traction return rail first node
    yg[trac_node_locs_a[0]] = 0.5 * yg_trac[0]
    # Traction return rail centre nodes
    for i in np.arange(0, len(yg_trac) - 1):
        yg[trac_node_locs_a[i + 1]] = (0.5 * yg_trac[i]) + (0.5 * yg_trac[i + 1])
    # Traction return rail last node
    yg[trac_node_locs_a[-1]] = 0.5 * yg_trac[-1]
    # Signalling rail nodes
    n_sb = 0
    for i in np.arange(0, len(sig_node_locs_a)):
        if sig_node_locs_a[i] in sig_node_locs_relay_a:
            yg[sig_node_locs_a[i]] = 0.5 * yg_sig[n_sb]
        elif sig_node_locs_a[i] in sig_node_locs_power_a:
            yg[sig_node_locs_a[i]] = 0.5 * yg_sig[n_sb]
            n_sb += 1
        else:
            print("Error")

    # "b" second
    # Traction return rail first node
    yg[trac_node_locs_b[0]] = 0.5 * yg_trac[0]
    # Traction return rail centre nodes
    for i in np.arange(0, len(yg_trac) - 1):
        yg[trac_node_locs_b[i + 1]] = (0.5 * yg_trac[i]) + (0.5 * yg_trac[i + 1])
    # Traction return rail last node
    yg[trac_node_locs_b[-1]] = 0.5 * yg_trac[-1]
    # Signalling rail nodes
    n_sb = 0
    for i in np.arange(0, len(sig_node_locs_b)):
        if sig_node_locs_b[i] in sig_node_locs_power_b:
            yg[sig_node_locs_b[i]] = 0.5 * yg_sig[n_sb]
        elif sig_node_locs_b[i] in sig_node_locs_relay_b:
            yg[sig_node_locs_b[i]] = 0.5 * yg_sig[n_sb]
            n_sb += 1
        else:
            print("Error")

    # Calculate sum of parallel admittances into nodes
    y_sums = np.zeros(n_nodes)

    # "a" first
    index_sb = np.arange(0, n_nodes_trac, 1)
    # Traction return rail first node
    y_sums[trac_node_locs_a[0]] = yg[trac_node_locs_a[0]] + parameters["y_relay"] + ye_trac[0]
    # Traction return rail centre nodes
    for i in index_sb[1:-1]:
        if trac_node_locs_a[i] in cb_node_locs_a:
            y_sums[trac_node_locs_a[i]] = yg[trac_node_locs_a[i]] + ye_trac[i - 1] + ye_trac[i] + parameters["y_cb"]
        elif trac_node_locs_a[i] not in cb_node_locs_a:
            y_sums[trac_node_locs_a[i]] = yg[trac_node_locs_a[i]] + ye_trac[i - 1] + ye_trac[i] + parameters["y_relay"] + parameters["y_power"]
        else:
            print("Error")
    # Traction return rail last node
    y_sums[trac_node_locs_a[-1]] = yg[trac_node_locs_a[-1]] + ye_trac[-1] + parameters["y_power"]

    # Signalling rail nodes
    n_sb = 0
    for i in sig_node_locs_a:
        if i in sig_node_locs_relay_a:
            y_sums[i] = yg[i] + ye_sig[n_sb] + parameters["y_relay"]
        elif i in sig_node_locs_power_a:
            y_sums[i] = yg[i] + ye_sig[n_sb] + parameters["y_power"]
            n_sb += 1
        else:
            print("Error")

    # "b" second
    index_sb = np.arange(0, n_nodes_trac, 1)
    # Traction return rail first node
    y_sums[trac_node_locs_b[0]] = yg[trac_node_locs_b[0]] + parameters["y_power"] + ye_trac[0]
    # Traction return rail centre nodes
    for i in index_sb[1:-1]:
        if trac_node_locs_b[i] in cb_node_locs_b:
            y_sums[trac_node_locs_b[i]] = yg[trac_node_locs_b[i]] + ye_trac[i - 1] + ye_trac[i] + parameters["y_cb"]
        elif trac_node_locs_b[i] not in cb_node_locs_b:
            y_sums[trac_node_locs_b[i]] = yg[trac_node_locs_b[i]] + ye_trac[i - 1] + ye_trac[i] + parameters["y_relay"] + parameters["y_power"]
        else:
            print("Error")
    # Traction return rail last node
    y_sums[trac_node_locs_b[-1]] = yg[trac_node_locs_b[-1]] + ye_trac[-1] + parameters["y_relay"]

    # Signalling rail nodes
    n_sb = 0
    for i in sig_node_locs_b:
        if i in sig_node_locs_power_b:
            y_sums[i] = yg[i] + ye_sig[n_sb] + parameters["y_power"]
        elif i in sig_node_locs_relay_b:
            y_sums[i] = yg[i] + ye_sig[n_sb] + parameters["y_relay"]
            n_sb += 1
        else:
            print("Error")

    # Build admittance matrix
    y_matrix = np.zeros((n_nodes, n_nodes))

    # Diagonal elements
    for i in np.arange(0, n_nodes, 1):
        y_matrix[i, i] = y_sums[i]

    # Cross bonds
    for i, j in zip(cb_node_locs_a, cb_node_locs_b):
        y_matrix[int(i), int(j)] = -parameters["y_cb"]
        y_matrix[int(j), int(i)] = -parameters["y_cb"]

    # "a" first
    index_sb = np.arange(0, n_nodes_trac, 1)
    # Traction return rail elements
    for i in index_sb[0:-1]:
        y_matrix[trac_node_locs_a[i], trac_node_locs_a[i + 1]] = -ye_trac[i]
        y_matrix[trac_node_locs_a[i + 1], trac_node_locs_a[i]] = -ye_trac[i]
    # Signalling rail elements
    n_sb = 0
    for i in sig_node_locs_a:
        if (i in sig_node_locs_relay_a) and (i + 1 in sig_node_locs_power_a):
            y_matrix[i, i + 1] = -ye_sig[n_sb]
        elif (i in sig_node_locs_power_a) and (i - 1 in sig_node_locs_relay_a):
            y_matrix[i, i - 1] = -ye_sig[n_sb]
            n_sb += 1
        else:
            print("Error")
    # Track circuit relay elements
    for i, j in zip(trac_node_locs_relay_a, sig_node_locs_relay_a):
        y_matrix[int(i), int(j)] = -parameters["y_relay"]
        y_matrix[int(j), int(i)] = -parameters["y_relay"]
    # Track circuit power supply elements
    for i, j in zip(trac_node_locs_power_a, sig_node_locs_power_a):
        y_matrix[int(i), int(j)] = -parameters["y_power"]
        y_matrix[int(j), int(i)] = -parameters["y_power"]

    # "b" second
    index_sb = np.arange(0, n_nodes_trac, 1)
    # Traction return rail elements
    for i in index_sb[0:-1]:
        y_matrix[trac_node_locs_b[i], trac_node_locs_b[i + 1]] = -ye_trac[i]
        y_matrix[trac_node_locs_b[i + 1], trac_node_locs_b[i]] = -ye_trac[i]
    # Signalling rail elements
    n_sb = 0
    for i in sig_node_locs_b:
        if (i in sig_node_locs_power_b) and (i + 1 in sig_node_locs_relay_b):
            y_matrix[i, i + 1] = -ye_sig[n_sb]
        elif (i in sig_node_locs_relay_b) and (i - 1 in sig_node_locs_power_b):
            y_matrix[i, i - 1] = -ye_sig[n_sb]
            n_sb += 1
        else:
            print("Error")
    # Track circuit power elements
    for i, j in zip(trac_node_locs_power_b, sig_node_locs_power_b):
        y_matrix[int(i), int(j)] = -parameters["y_power"]
        y_matrix[int(j), int(i)] = -parameters["y_power"]
    # Track circuit relay elements
    for i, j in zip(trac_node_locs_relay_b, sig_node_locs_relay_b):
        y_matrix[int(i), int(j)] = -parameters["y_relay"]
        y_matrix[int(j), int(i)] = -parameters["y_relay"]

    # Save network nodal admittance matrix in a file to be used in the analysis
    np.save("nodal_admittance_matrix_"+section_name+"_"+conditions, y_matrix)

    # Set angles for sub blocks
    # Note: Only the traction return rail has been split into sub blocks, so the signalling rail angles are equal to the
    # original block angles
    # "a" first
    trac_angles_a = np.zeros(len(trac_sub_blocks))
    cumsum_sb_a = np.cumsum(trac_sub_blocks)
    n_b = 0
    for n_sb in range(0, len(cumsum_sb_a)):
        if cumsum_sb_a[n_sb] < blocks_sum[n_b]:
            trac_angles_a[n_sb] = bearings[n_b]
        elif cumsum_sb_a[n_sb] == blocks_sum[n_b]:
            trac_angles_a[n_sb] = bearings[n_b]
            n_b = n_b + 1
        else:
            print("Error")
    sig_angles_a = bearings

    # "b" second
    trac_angles_b = np.zeros(len(trac_sub_blocks))
    cumsum_sb_b = np.cumsum(trac_sub_blocks)
    n_b = 0
    for n_sb in range(0, len(cumsum_sb_b)):
        if cumsum_sb_b[n_sb] < blocks_sum[n_b]:
            trac_angles_b[n_sb] = bearings[n_b]
        elif cumsum_sb_b[n_sb] == blocks_sum[n_b]:
            trac_angles_b[n_sb] = bearings[n_b]
            n_b = n_b + 1
        else:
            print("Error")
    trac_angles_b = (trac_angles_b + np.pi) % (2*np.pi)
    sig_angles_b = (bearings + np.pi) % (2*np.pi)

    # Save in a zip file to be used in the analysis
    np.savez("angles_"+section_name, trac_angles_a=trac_angles_a, trac_angles_b=trac_angles_b, sig_angles_a=sig_angles_a, sig_angles_b=sig_angles_b)


#for section in ["west_coast_main_line", "east_coast_main_line", "glasgow_edinburgh_falkirk"]:
#    for cond in ["dry", "moderate", "wet"]:
#        build_network_two_track(section_name=section, conditions=cond)

