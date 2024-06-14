import numpy as np


def e_field(exs, eys, section_name, conditions):
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
        # Load network admittance matrix
        y_matrix = np.load("data\\network_parameters\\" + section_name + "\\nodal_admittance_matrix_" + section_name + "_" + conditions + ".npy")

        # Calculate inverse of admittance matrix
        y_matrix_inv = np.linalg.inv(y_matrix)

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


ex = np.arange(-20, 20, 0.1)
ey = np.arange(-20, 20, 0.1)
ia, ib = e_field(ex, ey, "west_coast_main_line", "moderate")


ex = np.arange(-20, 20, 0.1)
ey = np.arange(-20, 20, 0.1)
ia, ib = e_field(ex, ey, "west_coast_main_line", "moderate")