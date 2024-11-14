import numpy as np
import pandas as pd
from numpy.f2py.cb_rules import cb_arg_rules


def dcmod14a_trains(e_par):
    # Set rail parameters
    r_rail = 0.25  # ohm/km
    r_power = 7.2  # ohm
    r_relay = 9  # ohm
    v_power = 6  # volt

    y_cb = 1e-3
    y_rw_connector = 1e-3
    y_return_wire = 0.091  # For each 60m stretch
    y_axle_trac = 0.1e-3
    y_axle_sig = 25e-3
    y_power = r_power
    y_relay = r_relay
    y_train_current = 0.28
    y_feeder = 0.28
    ye_sig = 0.015  # For each 60m stretch
    y_100k = 1e5

    # Load the sub blocks Excel file
    df = pd.read_excel(f'../data/nr_test_network/nr_test_network_block_lengths.xlsx', sheet_name='a_direction')
    spreadsheet_columns_as_arrays = df.values.T
    sub_blocks_a = spreadsheet_columns_as_arrays[2]
    df = pd.read_excel(f'../data/nr_test_network/nr_test_network_block_lengths.xlsx', sheet_name='b_direction')
    spreadsheet_columns_as_arrays = df.values.T
    sub_blocks_b = spreadsheet_columns_as_arrays[2]

    # Calculate the admittance of the rails based on the type
    # 0.25 ohm/km for jointed, 0.035 ohm/km for CWR
    y_sub_blocks_a = (sub_blocks_a * r_rail)
    y_sub_blocks_b = (sub_blocks_b * r_rail)

    # Calculate the traction rail sub block ground admittances in 'a' direction rail
    yg_traction_a = np.empty(len(y_sub_blocks_a) + 1)
    yg_traction_a[0:2] = 0
    yg_traction_a[2:8] = 0.65
    for i in range(8, 17):
        yg_traction_a[i] = (sub_blocks_a[i]/0.06)*0.05
    yg_traction_a[17] = 0.05
    yg_traction_a[18:26] = 0.65
    # Calculate the traction rail sub block ground admittances in 'b' direction rail
    yg_traction_b = np.empty(len(y_sub_blocks_b) + 1)
    yg_traction_b[0:2] = 0
    yg_traction_b[2:8] = 0.65
    for i in range(8, 22):
        yg_traction_b[i] = (sub_blocks_b[i] / 0.06) * 0.05
    yg_traction_b[22] = 0.05
    yg_traction_b[23:31] = 0.65

    # Set up the arrays for the whole network
    yg = np.zeros(78)
    yg[0:26] = yg_traction_a
    yg[29:60] = yg_traction_b
    ye_sum = np.zeros(78)
    ye_sum[0] = y_sub_blocks_a[0]
    for i in range(1, 25):
        ye_sum[i] = y_sub_blocks_a[i-1] + y_sub_blocks_a[i]
    ye_sum[25] = y_sub_blocks_a[24]
    ye_sum[29] = y_sub_blocks_b[29-29]
    for i in range(30, 59):
        ye_sum[i] = y_sub_blocks_b[i-30] + y_sub_blocks_b[i-29]
    ye_sum[59] = y_sub_blocks_b[59-30]

    # Define the nodal network
    node_indices_cb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 12, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 41, 44, 48, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59])
    node_indices_return_wire = np.array([2, 6, 12, 17, 21, 25, 31, 35, 41, 51, 55, 59])
    node_indices_axle_trac = np.array([11])
    node_indices_axle_sig = np.array([26])
    node_indices_power = np.array([11, 26, 40, 60])
    node_indices_relay = np.array([13, 27])
    node_indices_train_current = np.array([10, 28, 39])
    node_indices_feeder = np.array([17, 51])

    # Set up admittance arrays for the different components
    y_cb_array = np.zeros(78)
    y_cb_array[node_indices_cb] = y_cb
    y_rw_array = np.zeros(78)
    y_rw_array[node_indices_return_wire] = y_rw_connector
    y_axle_trac_array = np.zeros(78)
    y_axle_trac_array[node_indices_axle_trac] = y_axle_trac
    y_axle_sig_array = np.zeros(78)
    y_axle_sig_array[node_indices_axle_sig] = y_axle_sig
    y_power_array = np.zeros(78)
    y_power_array[node_indices_power] = y_power
    y_relay_array = np.zeros(78)
    y_relay_array[node_indices_relay] = y_relay
    y_train_current_array = np.zeros(78)
    y_train_current_array[node_indices_train_current] = y_train_current
    y_feeder_array = np.zeros(78)
    y_feeder_array[node_indices_feeder] = y_feeder

    # TEST
    yg = 1/yg
    yg[0:2] = 0

    # Set up y matrix of zeroes
    y_matrix = np.zeros((78, 78))

    # Sum of admittances
    y_sum = np.empty(78)
    # Traction rail nodes (a)
    for i in range(0, 26):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_rw_array[i] + y_axle_trac_array[i] + y_axle_sig_array[i] + y_power_array[i] + y_relay_array[i] + y_train_current_array[i] + y_feeder_array[i]
    # Signal rail nodes (a)
    y_sum[26] = y_power + y_axle_sig + ye_sig*10
    y_sum[27] = y_relay + ye_sig*10
    y_sum[28] = y_train_current + y_axle_trac + y_axle_sig
    # Traction rail nodes (b)
    for i in range(29, 60):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_rw_array[i] + y_axle_trac_array[i] + y_axle_sig_array[i] + y_power_array[i] + y_relay_array[i] + y_train_current_array[i] + y_feeder_array[i]
    # Signal rail nodes (b)
    y_sum[60] = y_power + y_100k*2 + ye_sig*5
    y_sum[61] = y_100k + ye_sig*5 + ye_sig*5
    y_sum[62] = y_100k + ye_sig*5 + ye_sig*5
    y_sum[63] = y_100k + ye_sig*5 + ye_sig*5
    y_sum[64] = y_100k + ye_sig*5 + ye_sig*5
    y_sum[65] = y_100k + ye_sig*5 + ye_sig*5
    y_sum[66] = ye_sig*5 + ye_sig*5
    # Other nodes
    y_sum[67] = y_return_wire*4 + y_cb + y_return_wire*4 + y_cb
    y_sum[68] = y_return_wire*4 + y_cb + y_return_wire*4
    y_sum[69] = y_return_wire*4 + y_cb + y_return_wire*4
    y_sum[70] = y_return_wire*4 + y_cb + y_return_wire*4
    y_sum[71] = y_return_wire*4 + y_cb
    y_sum[72] = y_return_wire + 1/1 + y_100k
    y_sum[73] = y_return_wire*4 + y_cb + y_return_wire*5
    y_sum[74] = y_return_wire*5 + y_cb + y_return_wire*4
    y_sum[75] = y_return_wire*4 + y_cb + y_return_wire*4
    y_sum[76] = y_return_wire*4 + y_cb + y_return_wire*5
    y_sum[77] = y_return_wire*4 + y_cb + 1
    # Additions
    y_sum[2] = y_sum[2] + y_return_wire*4
    y_sum[40] = y_sum[40] + y_100k*2
    y_sum[42] = y_sum[42] + y_100k
    y_sum[43] = y_sum[43] + y_100k
    y_sum[45] = y_sum[45] + y_100k
    y_sum[46] = y_sum[46] + y_100k
    y_sum[47] = y_sum[47] + y_100k
    y_sum[49] = y_sum[49] + y_100k

    # Add sums to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances
    for i in range(0, len(y_sub_blocks_a)):
        y_matrix[i, i+1] = -y_sub_blocks_a[i]
        y_matrix[i+1, i] = -y_sub_blocks_a[i]

    # Signal rail admittances
    for i in range(0, len(y_sub_blocks_b)):
        y_matrix[i+29, i+30] = -y_sub_blocks_b[i]
        y_matrix[i+30, i+29] = -y_sub_blocks_b[i]

    # Cross bonds
    cb_a = node_indices_cb[node_indices_cb < 28]
    cb_b = node_indices_cb[node_indices_cb > 28]
    y_matrix[cb_a, cb_b] = -y_cb
    y_matrix[cb_b, cb_a] = -y_cb

    # Return conductor vertical
    

    # Return conductor horizontal


    # Track circuit


    # Currents
    current = e_par / r_rail

    j_matrix = np.zeros(66)
    j_matrix[61] = -0.84
    j_matrix[63] = 0.84 - current
    j_matrix[14] = -1
    j_matrix[65] = 1

    j_matrix[0] = -current
    j_matrix[22] = current
    j_matrix[23] = -current
    j_matrix[45] = current
    j_matrix[64] = current

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    v_relay = v_matrix[64] - v_matrix[62]
    i_relay = v_relay / 9

    print(i_relay)

    pass


dcmod14a_trains(np.array([1]))
