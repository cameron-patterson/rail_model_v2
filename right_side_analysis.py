import numpy as np


def right_side_two_track(exs, eys, section_name, conditions):
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

    # Currents
    # Set up current matrix
    j_matrix = np.zeros([len(exs), n_nodes])

    # "a" first
    e_x_par_trac_a = np.outer(exs, np.cos(0.5 * np.pi - trac_angles_a))
    e_y_par_trac_a = np.outer(eys, np.cos(trac_angles_a))
    e_par_trac_a = e_x_par_trac_a + e_y_par_trac_a
    e_x_par_sig_a = np.outer(exs, np.cos(0.5 * np.pi - sig_angles_a))
    e_y_par_sig_a = np.outer(eys, np.cos(sig_angles_a))
    e_par_sig_a = e_x_par_sig_a + e_y_par_sig_a
    i_sig_a = e_par_sig_a / parameters["z_sig"]
    i_trac_a = e_par_trac_a / parameters["z_trac"]

    # "b" second
    e_x_par_trac_b = np.outer(exs, np.cos(0.5 * np.pi - trac_angles_b))
    e_y_par_trac_b = np.outer(eys, np.cos(trac_angles_b))
    e_par_trac_b = e_x_par_trac_b + e_y_par_trac_b
    e_x_par_sig_b = np.outer(exs, np.cos(0.5 * np.pi - sig_angles_b))
    e_y_par_sig_b = np.outer(eys, np.cos(sig_angles_b))
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
