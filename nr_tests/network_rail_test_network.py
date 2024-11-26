import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.gridspec import GridSpec


# DCMOD14a: double track railway, jointed rail, booster sections with return conductor.
def dcmod14a_trains(r_power, r_relay, v_power, e_par):
    # Set parameters
    r_rail = 0.25  # ohm/km
    i_train = 1  # ampere

    y_cb = 1 / 1e-3
    y_rw_connector = 1 / 1e-3
    r_return_wire = 0.091  # For each 60m stretch
    y_axle_trac = 1 / 0.1e-3
    y_axle_sig = 1 / 25e-3
    y_power = 1 / r_power
    y_relay = 1 / r_relay
    y_train_current = 1 / 0.28
    y_feeder = 1 / 0.28
    r_sig = 0.015  # For each 60m stretch
    y_100k = 1 / 1e5
    y_200k = 1 / 2e5
    y_1_ohm = 1 / 1

    # Calculate the admittance of the rails based on the type
    # 0.25 ohm/km for jointed, 0.035 ohm/km for CWR
    ye_long = ((780/1000) * r_rail)
    ye_short = ((60/1000) * r_rail)

    # Define parallel admittances
    ye_sum = np.zeros(218)
    # Traction rails
    # Top track
    ye_sum[0] = ye_long
    ye_sum[1:8] = ye_long * 2
    ye_sum[8] = ye_long + ye_short
    ye_sum[9:88] = ye_short * 2
    ye_sum[88] = ye_short + ye_long
    ye_sum[89:96] = ye_long * 2
    ye_sum[96] = ye_long
    # Bottom track
    ye_sum[99] = ye_long
    ye_sum[100:107] = ye_long * 2
    ye_sum[107] = ye_long + ye_short
    ye_sum[108:187] = ye_short * 2
    ye_sum[187] = ye_short + ye_long
    ye_sum[188:195] = ye_long * 2
    ye_sum[195] = ye_long

    # Signal rails
    for i in [196, 202]:
        ye_sum[i] = 1 / (r_sig * 5)
    for i in [97, 98, 197, 198, 199, 200, 201]:
        ye_sum[i] = 1 / (r_sig * 10)

    # Define ground admittances
    yg_long = 1 / 1.5
    yg_short = 1 / 20

    # Set up the ground admittance array
    yg = np.zeros(218)
    # Top track
    # No ground
    yg[0:2] = 0
    # Long sections 1
    yg[2:8] = yg_long
    # Short sections
    yg[8:89] = yg_short
    # Long sections 2
    yg[89:97] = yg_long
    # Bottom track
    # No ground
    yg[99:101] = 0
    # Long sections 1
    yg[101:107] = yg_long
    # Short sections
    yg[107:188] = yg_short
    # Long sections 2
    yg[188:196] = yg_long
    # Return wire end nodes
    yg[209] = y_1_ohm + y_100k
    yg[216] = y_1_ohm

    # Define the nodal network
    node_indices_cb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 36, 48, 61, 74, 88, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 107, 120, 135, 147, 160, 173, 187, 188, 189, 190, 191, 192, 193, 194, 195])
    node_indices_return_wire_vertical = np.array([2, 6, 36, 88, 92, 96, 101, 105, 135, 187, 191, 195, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215])
    node_indices_axle_trac = np.array([35, 217])
    node_indices_axle_sig = np.array([97, 217])
    node_indices_power = np.array([35, 97, 134, 196])
    node_indices_relay = np.array([45, 98])
    node_indices_train_current = np.array([24, 123, 217])
    node_indices_feeder_double = np.array([88])
    node_indices_feeder_single = np.array([187])
    node_indices_100k = np.array([139, 197, 144, 198, 149, 199, 154, 200, 159, 201, 164, 202])
    node_indices_200k = np.array([134, 196])

    # Set up admittance arrays for the different components
    y_cb_array = np.zeros(218)
    y_cb_array[node_indices_cb] = y_cb
    y_rw_array = np.zeros(218)
    y_rw_array[node_indices_return_wire_vertical] = y_rw_connector
    y_axle_trac_array = np.zeros(218)
    y_axle_trac_array[node_indices_axle_trac] = y_axle_trac
    y_axle_sig_array = np.zeros(218)
    y_axle_sig_array[node_indices_axle_sig] = y_axle_sig
    y_power_array = np.zeros(218)
    y_power_array[node_indices_power] = y_power
    y_relay_array = np.zeros(218)
    y_relay_array[node_indices_relay] = y_relay
    y_train_current_array = np.zeros(218)
    y_train_current_array[node_indices_train_current] = y_train_current
    y_feeder_single_array = np.zeros(218)
    y_feeder_single_array[node_indices_feeder_single] = y_feeder
    y_feeder_double_array = np.zeros(218)
    y_feeder_double_array[node_indices_feeder_double] = y_feeder * 2
    y_100k_array = np.zeros(218)
    y_100k_array[node_indices_100k] = y_100k
    y_200k_array = np.zeros(218)
    y_200k_array[node_indices_200k] = y_200k

    # Set up y matrix of zeroes
    y_matrix = np.zeros((218, 218))

    # Sum of admittances
    y_sum = np.empty(218)

    # Uniform nodes
    for i in range(0, 218):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_rw_array[i] + y_axle_trac_array[i] + y_axle_sig_array[i] + \
                   y_power_array[i] + y_relay_array[i] + y_train_current_array[i] + y_feeder_single_array[i] + \
                   y_feeder_double_array[i] + y_100k_array[i] + y_200k_array[i]

    # Add-ons
    # Return wire
    y_sum[203] = y_sum[203] + 1 / (r_return_wire * 4)
    y_sum[204] = y_sum[204] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 5))
    y_sum[205] = y_sum[205] + (1 / (r_return_wire * 5)) + (1 / (r_return_wire * 4))
    y_sum[206] = y_sum[206] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 4))
    y_sum[207] = y_sum[207] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 3))
    y_sum[208] = y_sum[208] + (1 / (r_return_wire * 3)) + (1 / (r_return_wire * 1))
    y_sum[209] = y_sum[209] + (1 / (r_return_wire * 1))

    y_sum[210] = y_sum[210] + 1 / (r_return_wire * 4)
    y_sum[211] = y_sum[211] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 5))
    y_sum[212] = y_sum[212] + (1 / (r_return_wire * 5)) + (1 / (r_return_wire * 4))
    y_sum[213] = y_sum[213] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 4))
    y_sum[214] = y_sum[214] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 3))
    y_sum[215] = y_sum[215] + (1 / (r_return_wire * 3)) + (1 / (r_return_wire * 1))
    y_sum[216] = y_sum[216] + (1 / (r_return_wire * 1))

    # Add sums to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances (top)
    for i in range(0, 8):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(8, 88):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(88, 96):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    # Traction rail admittances (bottom)
    for i in range(99, 107):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(107, 187):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(187, 195):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long

    # Signal Rail admittances (top)
    y_matrix[97, 98] = -1 * (1 / (r_sig * 10))
    y_matrix[98, 97] = -1 * (1 / (r_sig * 10))

    # Signal Rail admittances (b)
    for i in range(196, 201):
        y_matrix[i, i + 1] = -1 * (1 / (r_sig * 5))
        y_matrix[i + 1, i] = -1 * (1 / (r_sig * 5))

    # Cross bonds
    cb_a = node_indices_cb[node_indices_cb < 98]
    cb_b = node_indices_cb[node_indices_cb > 98]
    y_matrix[cb_a, cb_b] = -y_cb
    y_matrix[cb_b, cb_a] = -y_cb

    # Return conductor vertical
    # Top
    y_matrix[
        node_indices_return_wire_vertical[node_indices_return_wire_vertical < 97], node_indices_return_wire_vertical[
            (node_indices_return_wire_vertical > 195) & (node_indices_return_wire_vertical < 210)]] = -y_rw_connector
    # Bottom
    y_matrix[
        node_indices_return_wire_vertical[(node_indices_return_wire_vertical > 96) & (node_indices_return_wire_vertical < 196)], node_indices_return_wire_vertical[
            node_indices_return_wire_vertical > 209]] = -y_rw_connector

    # Return conductor horizontal
    for i in [203, 205, 206, 210, 212, 213]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 4)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 4)
    for i in [204, 211]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 5)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 5)
    for i in [207, 214]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 3)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 3)
    for i in [208, 215]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 1)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 1)

    # Track circuit
    y_matrix[35, 97] = -y_power
    y_matrix[97, 35] = -y_power
    y_matrix[45, 98] = -y_relay
    y_matrix[98, 45] = -y_relay
    y_matrix[35, 217] = -y_axle_trac
    y_matrix[217, 35] = -y_axle_trac
    y_matrix[97, 217] = -y_axle_sig
    y_matrix[217, 97] = -y_axle_sig
    y_matrix[134, 196] = -1 * (y_power + y_200k)
    y_matrix[196, 134] = -1 * (y_power + y_200k)

    # 100k ohm connections
    for i, j in zip([139, 144, 149, 154, 159, 164], [197, 198, 199, 200, 201, 202]):
        y_matrix[i, j] = -y_100k
        y_matrix[j, i] = -y_100k

    # Train current
    for i, j in zip([24, 217, 123], [88, 88, 187]):
        y_matrix[i, j] = -y_train_current
        y_matrix[j, i] = -y_train_current

    # Currents
    current = e_par / r_rail
    j_matrix = np.zeros(218)
    j_matrix[0] = -current
    j_matrix[24] = i_train
    j_matrix[35] = -(v_power / r_power)
    j_matrix[88] = -1 * (i_train * 2)
    j_matrix[96] = current
    j_matrix[97] = (v_power / r_power) - current
    j_matrix[98] = current
    j_matrix[217] = i_train
    j_matrix[99] = -current
    j_matrix[123] = i_train
    j_matrix[134] = -(v_power / r_power)
    j_matrix[187] = -i_train
    j_matrix[195] = current
    j_matrix[196] = (v_power / r_power) - current
    j_matrix[202] = current

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    v_relay = v_matrix[98] - v_matrix[45]
    i_relay = v_relay / r_relay

    return i_relay


def dcmod14a_no_trains(r_power, r_relay, v_power, e_par):
    # Set parameters
    r_rail = 0.25  # ohm/km

    y_cb = 1 / 1e-3
    y_rw_connector = 1 / 1e-3
    r_return_wire = 0.091  # For each 60m stretch
    y_power = 1 / r_power
    y_relay = 1 / r_relay
    r_sig = 0.015  # For each 60m stretch
    y_100k = 1 / 1e5
    y_200k = 1 / 2e5
    y_1_ohm = 1 / 1

    # Calculate the admittance of the rails based on the type
    # 0.25 ohm/km for jointed, 0.035 ohm/km for CWR
    ye_long = ((780/1000) * r_rail)
    ye_short = ((60/1000) * r_rail)

    # Define parallel admittances
    ye_sum = np.zeros(217)
    # Traction rails
    # Top track
    ye_sum[0] = ye_long
    ye_sum[1:8] = ye_long * 2
    ye_sum[8] = ye_long + ye_short
    ye_sum[9:88] = ye_short * 2
    ye_sum[88] = ye_short + ye_long
    ye_sum[89:96] = ye_long * 2
    ye_sum[96] = ye_long
    # Bottom track
    ye_sum[99] = ye_long
    ye_sum[100:107] = ye_long * 2
    ye_sum[107] = ye_long + ye_short
    ye_sum[108:187] = ye_short * 2
    ye_sum[187] = ye_short + ye_long
    ye_sum[188:195] = ye_long * 2
    ye_sum[195] = ye_long

    # Signal rails
    for i in [196, 202]:
        ye_sum[i] = 1 / (r_sig * 5)
    for i in [97, 98, 197, 198, 199, 200, 201]:
        ye_sum[i] = 1 / (r_sig * 10)

    # Define ground admittances
    yg_long = 1 / 1.5
    yg_short = 1 / 20

    # Set up the ground admittance array
    yg = np.zeros(217)
    # Top track
    # No ground
    yg[0:2] = 0
    # Long sections 1
    yg[2:8] = yg_long
    # Short sections
    yg[8:89] = yg_short
    # Long sections 2
    yg[89:97] = yg_long
    # Bottom track
    # No ground
    yg[99:101] = 0
    # Long sections 1
    yg[101:107] = yg_long
    # Short sections
    yg[107:188] = yg_short
    # Long sections 2
    yg[188:196] = yg_long
    # Return wire end nodes
    yg[209] = y_1_ohm + y_100k
    yg[216] = y_1_ohm

    # Define the nodal network
    node_indices_cb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 36, 48, 61, 74, 88, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 107, 120, 135, 147, 160, 173, 187, 188, 189, 190, 191, 192, 193, 194, 195])
    node_indices_return_wire_vertical = np.array([2, 6, 36, 88, 92, 96, 101, 105, 135, 187, 191, 195, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215])
    node_indices_power = np.array([35, 97, 134, 196])
    node_indices_relay = np.array([45, 98])
    node_indices_100k = np.array([139, 197, 144, 198, 149, 199, 154, 200, 159, 201, 164, 202])
    node_indices_200k = np.array([134, 196])

    # Set up admittance arrays for the different components
    y_cb_array = np.zeros(217)
    y_cb_array[node_indices_cb] = y_cb
    y_rw_array = np.zeros(217)
    y_rw_array[node_indices_return_wire_vertical] = y_rw_connector
    y_power_array = np.zeros(217)
    y_power_array[node_indices_power] = y_power
    y_relay_array = np.zeros(217)
    y_relay_array[node_indices_relay] = y_relay
    y_100k_array = np.zeros(217)
    y_100k_array[node_indices_100k] = y_100k
    y_200k_array = np.zeros(217)
    y_200k_array[node_indices_200k] = y_200k

    # Set up y matrix of zeroes
    y_matrix = np.zeros((217, 217))

    # Sum of admittances
    y_sum = np.empty(217)

    # Uniform nodes
    for i in range(0, 217):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_rw_array[i] + y_power_array[i] + y_relay_array[i] + y_100k_array[i] + y_200k_array[i]

    # Add-ons
    # Return wire
    y_sum[203] = y_sum[203] + 1 / (r_return_wire * 4)
    y_sum[204] = y_sum[204] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 5))
    y_sum[205] = y_sum[205] + (1 / (r_return_wire * 5)) + (1 / (r_return_wire * 4))
    y_sum[206] = y_sum[206] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 4))
    y_sum[207] = y_sum[207] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 3))
    y_sum[208] = y_sum[208] + (1 / (r_return_wire * 3)) + (1 / (r_return_wire * 1))
    y_sum[209] = y_sum[209] + (1 / (r_return_wire * 1))

    y_sum[210] = y_sum[210] + 1 / (r_return_wire * 4)
    y_sum[211] = y_sum[211] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 5))
    y_sum[212] = y_sum[212] + (1 / (r_return_wire * 5)) + (1 / (r_return_wire * 4))
    y_sum[213] = y_sum[213] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 4))
    y_sum[214] = y_sum[214] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 3))
    y_sum[215] = y_sum[215] + (1 / (r_return_wire * 3)) + (1 / (r_return_wire * 1))
    y_sum[216] = y_sum[216] + (1 / (r_return_wire * 1))

    # Add sums to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances (top)
    for i in range(0, 8):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(8, 88):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(88, 96):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    # Traction rail admittances (bottom)
    for i in range(99, 107):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(107, 187):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(187, 195):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long

    # Signal Rail admittances (top)
    y_matrix[97, 98] = -1 * (1 / (r_sig * 10))
    y_matrix[98, 97] = -1 * (1 / (r_sig * 10))

    # Signal Rail admittances (b)
    for i in range(196, 201):
        y_matrix[i, i + 1] = -1 * (1 / (r_sig * 5))
        y_matrix[i + 1, i] = -1 * (1 / (r_sig * 5))

    # Cross bonds
    cb_a = node_indices_cb[node_indices_cb < 98]
    cb_b = node_indices_cb[node_indices_cb > 98]
    y_matrix[cb_a, cb_b] = -y_cb
    y_matrix[cb_b, cb_a] = -y_cb

    # Return conductor vertical
    # Top
    y_matrix[
        node_indices_return_wire_vertical[node_indices_return_wire_vertical < 97], node_indices_return_wire_vertical[
            (node_indices_return_wire_vertical > 195) & (node_indices_return_wire_vertical < 210)]] = -y_rw_connector
    # Bottom
    y_matrix[
        node_indices_return_wire_vertical[(node_indices_return_wire_vertical > 96) & (node_indices_return_wire_vertical < 196)], node_indices_return_wire_vertical[
            node_indices_return_wire_vertical > 209]] = -y_rw_connector

    # Return conductor horizontal
    for i in [203, 205, 206, 210, 212, 213]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 4)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 4)
    for i in [204, 211]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 5)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 5)
    for i in [207, 214]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 3)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 3)
    for i in [208, 215]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 1)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 1)

    # Track circuit
    y_matrix[35, 97] = -y_power
    y_matrix[97, 35] = -y_power
    y_matrix[45, 98] = -y_relay
    y_matrix[98, 45] = -y_relay
    y_matrix[134, 196] = -1 * (y_power + y_200k)
    y_matrix[196, 134] = -1 * (y_power + y_200k)

    # 100k ohm connections
    for i, j in zip([139, 144, 149, 154, 159, 164], [197, 198, 199, 200, 201, 202]):
        y_matrix[i, j] = -y_100k
        y_matrix[j, i] = -y_100k

    # Currents
    current = e_par / r_rail
    j_matrix = np.zeros(217)
    j_matrix[0] = -current
    j_matrix[35] = -(v_power / r_power)
    j_matrix[96] = current
    j_matrix[97] = (v_power / r_power) - current
    j_matrix[98] = current
    j_matrix[99] = -current
    j_matrix[134] = -(v_power / r_power)
    j_matrix[195] = current
    j_matrix[196] = (v_power / r_power) - current
    j_matrix[202] = current

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    v_relay = v_matrix[98] - v_matrix[45]
    i_relay = v_relay / r_relay

    return i_relay


# DCMOD16a: double track railway, CWR, booster sections with return conductor.
def dcmod16a_trains(r_power, r_relay, v_power, e_par):
    # Set parameters
    r_rail = 0.035  # ohm/km
    i_train = 1  # ampere

    y_cb = 1 / 1e-3
    y_rw_connector = 1 / 1e-3
    r_return_wire = 0.091  # For each 60m stretch
    y_axle_trac = 1 / 0.1e-3
    y_axle_sig = 1 / 25e-3
    y_power = 1 / r_power
    y_relay = 1 / r_relay
    y_train_current = 1 / 0.28
    y_feeder = 1 / 0.28
    r_sig = 0.015  # For each 60m stretch
    y_100k = 1 / 1e5
    y_200k = 1 / 2e5
    y_1_ohm = 1 / 1

    # Calculate the admittance of the rails based on the type
    # 0.25 ohm/km for jointed, 0.035 ohm/km for CWR
    ye_long = ((780/1000) * r_rail)
    ye_short = ((60/1000) * r_rail)

    # Define parallel admittances
    ye_sum = np.zeros(218)
    # Traction rails
    # Top track
    ye_sum[0] = ye_long
    ye_sum[1:8] = ye_long * 2
    ye_sum[8] = ye_long + ye_short
    ye_sum[9:88] = ye_short * 2
    ye_sum[88] = ye_short + ye_long
    ye_sum[89:96] = ye_long * 2
    ye_sum[96] = ye_long
    # Bottom track
    ye_sum[99] = ye_long
    ye_sum[100:107] = ye_long * 2
    ye_sum[107] = ye_long + ye_short
    ye_sum[108:187] = ye_short * 2
    ye_sum[187] = ye_short + ye_long
    ye_sum[188:195] = ye_long * 2
    ye_sum[195] = ye_long

    # Signal rails
    for i in [196, 202]:
        ye_sum[i] = 1 / (r_sig * 5)
    for i in [97, 98, 197, 198, 199, 200, 201]:
        ye_sum[i] = 1 / (r_sig * 10)

    # Define ground admittances
    yg_long = 1 / 1.5
    yg_short = 1 / 20

    # Set up the ground admittance array
    yg = np.zeros(218)
    # Top track
    # No ground
    yg[0:2] = 0
    # Long sections 1
    yg[2:8] = yg_long
    # Short sections
    yg[8:89] = yg_short
    # Long sections 2
    yg[89:97] = yg_long
    # Bottom track
    # No ground
    yg[99:101] = 0
    # Long sections 1
    yg[101:107] = yg_long
    # Short sections
    yg[107:188] = yg_short
    # Long sections 2
    yg[188:196] = yg_long
    # Return wire end nodes
    yg[209] = y_1_ohm + y_100k
    yg[216] = y_1_ohm

    # Define the nodal network
    node_indices_cb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 36, 48, 61, 74, 88, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 107, 120, 135, 147, 160, 173, 187, 188, 189, 190, 191, 192, 193, 194, 195])
    node_indices_return_wire_vertical = np.array([2, 6, 36, 88, 92, 96, 101, 105, 135, 187, 191, 195, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215])
    node_indices_axle_trac = np.array([35, 217])
    node_indices_axle_sig = np.array([97, 217])
    node_indices_power = np.array([35, 97, 134, 196])
    node_indices_relay = np.array([45, 98])
    node_indices_train_current = np.array([24, 123, 217])
    node_indices_feeder_double = np.array([88])
    node_indices_feeder_single = np.array([187])
    node_indices_100k = np.array([139, 197, 144, 198, 149, 199, 154, 200, 159, 201, 164, 202])
    node_indices_200k = np.array([134, 196])

    # Set up admittance arrays for the different components
    y_cb_array = np.zeros(218)
    y_cb_array[node_indices_cb] = y_cb
    y_rw_array = np.zeros(218)
    y_rw_array[node_indices_return_wire_vertical] = y_rw_connector
    y_axle_trac_array = np.zeros(218)
    y_axle_trac_array[node_indices_axle_trac] = y_axle_trac
    y_axle_sig_array = np.zeros(218)
    y_axle_sig_array[node_indices_axle_sig] = y_axle_sig
    y_power_array = np.zeros(218)
    y_power_array[node_indices_power] = y_power
    y_relay_array = np.zeros(218)
    y_relay_array[node_indices_relay] = y_relay
    y_train_current_array = np.zeros(218)
    y_train_current_array[node_indices_train_current] = y_train_current
    y_feeder_single_array = np.zeros(218)
    y_feeder_single_array[node_indices_feeder_single] = y_feeder
    y_feeder_double_array = np.zeros(218)
    y_feeder_double_array[node_indices_feeder_double] = y_feeder * 2
    y_100k_array = np.zeros(218)
    y_100k_array[node_indices_100k] = y_100k
    y_200k_array = np.zeros(218)
    y_200k_array[node_indices_200k] = y_200k

    # Set up y matrix of zeroes
    y_matrix = np.zeros((218, 218))

    # Sum of admittances
    y_sum = np.empty(218)

    # Uniform nodes
    for i in range(0, 218):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_rw_array[i] + y_axle_trac_array[i] + y_axle_sig_array[i] + \
                   y_power_array[i] + y_relay_array[i] + y_train_current_array[i] + y_feeder_single_array[i] + \
                   y_feeder_double_array[i] + y_100k_array[i] + y_200k_array[i]

    # Add-ons
    # Return wire
    y_sum[203] = y_sum[203] + 1 / (r_return_wire * 4)
    y_sum[204] = y_sum[204] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 5))
    y_sum[205] = y_sum[205] + (1 / (r_return_wire * 5)) + (1 / (r_return_wire * 4))
    y_sum[206] = y_sum[206] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 4))
    y_sum[207] = y_sum[207] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 3))
    y_sum[208] = y_sum[208] + (1 / (r_return_wire * 3)) + (1 / (r_return_wire * 1))
    y_sum[209] = y_sum[209] + (1 / (r_return_wire * 1))

    y_sum[210] = y_sum[210] + 1 / (r_return_wire * 4)
    y_sum[211] = y_sum[211] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 5))
    y_sum[212] = y_sum[212] + (1 / (r_return_wire * 5)) + (1 / (r_return_wire * 4))
    y_sum[213] = y_sum[213] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 4))
    y_sum[214] = y_sum[214] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 3))
    y_sum[215] = y_sum[215] + (1 / (r_return_wire * 3)) + (1 / (r_return_wire * 1))
    y_sum[216] = y_sum[216] + (1 / (r_return_wire * 1))

    # Add sums to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances (top)
    for i in range(0, 8):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(8, 88):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(88, 96):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    # Traction rail admittances (bottom)
    for i in range(99, 107):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(107, 187):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(187, 195):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long

    # Signal Rail admittances (top)
    y_matrix[97, 98] = -1 * (1 / (r_sig * 10))
    y_matrix[98, 97] = -1 * (1 / (r_sig * 10))

    # Signal Rail admittances (b)
    for i in range(196, 201):
        y_matrix[i, i + 1] = -1 * (1 / (r_sig * 5))
        y_matrix[i + 1, i] = -1 * (1 / (r_sig * 5))

    # Cross bonds
    cb_a = node_indices_cb[node_indices_cb < 98]
    cb_b = node_indices_cb[node_indices_cb > 98]
    y_matrix[cb_a, cb_b] = -y_cb
    y_matrix[cb_b, cb_a] = -y_cb

    # Return conductor vertical
    # Top
    y_matrix[
        node_indices_return_wire_vertical[node_indices_return_wire_vertical < 97], node_indices_return_wire_vertical[
            (node_indices_return_wire_vertical > 195) & (node_indices_return_wire_vertical < 210)]] = -y_rw_connector
    # Bottom
    y_matrix[
        node_indices_return_wire_vertical[(node_indices_return_wire_vertical > 96) & (node_indices_return_wire_vertical < 196)], node_indices_return_wire_vertical[
            node_indices_return_wire_vertical > 209]] = -y_rw_connector

    # Return conductor horizontal
    for i in [203, 205, 206, 210, 212, 213]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 4)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 4)
    for i in [204, 211]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 5)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 5)
    for i in [207, 214]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 3)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 3)
    for i in [208, 215]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 1)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 1)

    # Track circuit
    y_matrix[35, 97] = -y_power
    y_matrix[97, 35] = -y_power
    y_matrix[45, 98] = -y_relay
    y_matrix[98, 45] = -y_relay
    y_matrix[35, 217] = -y_axle_trac
    y_matrix[217, 35] = -y_axle_trac
    y_matrix[97, 217] = -y_axle_sig
    y_matrix[217, 97] = -y_axle_sig
    y_matrix[134, 196] = -1 * (y_power + y_200k)
    y_matrix[196, 134] = -1 * (y_power + y_200k)

    # 100k ohm connections
    for i, j in zip([139, 144, 149, 154, 159, 164], [197, 198, 199, 200, 201, 202]):
        y_matrix[i, j] = -y_100k
        y_matrix[j, i] = -y_100k

    # Train current
    for i, j in zip([24, 217, 123], [88, 88, 187]):
        y_matrix[i, j] = -y_train_current
        y_matrix[j, i] = -y_train_current

    # Currents
    current = e_par / r_rail
    j_matrix = np.zeros(218)
    j_matrix[0] = -current
    j_matrix[24] = i_train
    j_matrix[35] = -(v_power / r_power)
    j_matrix[88] = -1 * (i_train * 2)
    j_matrix[96] = current
    j_matrix[97] = (v_power / r_power) - current
    j_matrix[98] = current
    j_matrix[217] = i_train
    j_matrix[99] = -current
    j_matrix[123] = i_train
    j_matrix[134] = -(v_power / r_power)
    j_matrix[187] = -i_train
    j_matrix[195] = current
    j_matrix[196] = (v_power / r_power) - current
    j_matrix[202] = current

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    v_relay = v_matrix[98] - v_matrix[45]
    i_relay = v_relay / r_relay

    return i_relay


def dcmod16a_no_trains(r_power, r_relay, v_power, e_par):
    # Set parameters
    r_rail = 0.035  # ohm/km

    y_cb = 1 / 1e-3
    y_rw_connector = 1 / 1e-3
    r_return_wire = 0.091  # For each 60m stretch
    y_power = 1 / r_power
    y_relay = 1 / r_relay
    r_sig = 0.015  # For each 60m stretch
    y_100k = 1 / 1e5
    y_200k = 1 / 2e5
    y_1_ohm = 1 / 1

    # Calculate the admittance of the rails based on the type
    # 0.25 ohm/km for jointed, 0.035 ohm/km for CWR
    ye_long = ((780/1000) * r_rail)
    ye_short = ((60/1000) * r_rail)

    # Define parallel admittances
    ye_sum = np.zeros(217)
    # Traction rails
    # Top track
    ye_sum[0] = ye_long
    ye_sum[1:8] = ye_long * 2
    ye_sum[8] = ye_long + ye_short
    ye_sum[9:88] = ye_short * 2
    ye_sum[88] = ye_short + ye_long
    ye_sum[89:96] = ye_long * 2
    ye_sum[96] = ye_long
    # Bottom track
    ye_sum[99] = ye_long
    ye_sum[100:107] = ye_long * 2
    ye_sum[107] = ye_long + ye_short
    ye_sum[108:187] = ye_short * 2
    ye_sum[187] = ye_short + ye_long
    ye_sum[188:195] = ye_long * 2
    ye_sum[195] = ye_long

    # Signal rails
    for i in [196, 202]:
        ye_sum[i] = 1 / (r_sig * 5)
    for i in [97, 98, 197, 198, 199, 200, 201]:
        ye_sum[i] = 1 / (r_sig * 10)

    # Define ground admittances
    yg_long = 1 / 1.5
    yg_short = 1 / 20

    # Set up the ground admittance array
    yg = np.zeros(217)
    # Top track
    # No ground
    yg[0:2] = 0
    # Long sections 1
    yg[2:8] = yg_long
    # Short sections
    yg[8:89] = yg_short
    # Long sections 2
    yg[89:97] = yg_long
    # Bottom track
    # No ground
    yg[99:101] = 0
    # Long sections 1
    yg[101:107] = yg_long
    # Short sections
    yg[107:188] = yg_short
    # Long sections 2
    yg[188:196] = yg_long
    # Return wire end nodes
    yg[209] = y_1_ohm + y_100k
    yg[216] = y_1_ohm

    # Define the nodal network
    node_indices_cb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 36, 48, 61, 74, 88, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 107, 120, 135, 147, 160, 173, 187, 188, 189, 190, 191, 192, 193, 194, 195])
    node_indices_return_wire_vertical = np.array([2, 6, 36, 88, 92, 96, 101, 105, 135, 187, 191, 195, 203, 204, 205, 206, 207, 208, 210, 211, 212, 213, 214, 215])
    node_indices_power = np.array([35, 97, 134, 196])
    node_indices_relay = np.array([45, 98])
    node_indices_100k = np.array([139, 197, 144, 198, 149, 199, 154, 200, 159, 201, 164, 202])
    node_indices_200k = np.array([134, 196])

    # Set up admittance arrays for the different components
    y_cb_array = np.zeros(217)
    y_cb_array[node_indices_cb] = y_cb
    y_rw_array = np.zeros(217)
    y_rw_array[node_indices_return_wire_vertical] = y_rw_connector
    y_power_array = np.zeros(217)
    y_power_array[node_indices_power] = y_power
    y_relay_array = np.zeros(217)
    y_relay_array[node_indices_relay] = y_relay
    y_100k_array = np.zeros(217)
    y_100k_array[node_indices_100k] = y_100k
    y_200k_array = np.zeros(217)
    y_200k_array[node_indices_200k] = y_200k

    # Set up y matrix of zeroes
    y_matrix = np.zeros((217, 217))

    # Sum of admittances
    y_sum = np.empty(217)

    # Uniform nodes
    for i in range(0, 217):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_rw_array[i] + y_power_array[i] + y_relay_array[i] + y_100k_array[i] + y_200k_array[i]

    # Add-ons
    # Return wire
    y_sum[203] = y_sum[203] + 1 / (r_return_wire * 4)
    y_sum[204] = y_sum[204] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 5))
    y_sum[205] = y_sum[205] + (1 / (r_return_wire * 5)) + (1 / (r_return_wire * 4))
    y_sum[206] = y_sum[206] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 4))
    y_sum[207] = y_sum[207] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 3))
    y_sum[208] = y_sum[208] + (1 / (r_return_wire * 3)) + (1 / (r_return_wire * 1))
    y_sum[209] = y_sum[209] + (1 / (r_return_wire * 1))

    y_sum[210] = y_sum[210] + 1 / (r_return_wire * 4)
    y_sum[211] = y_sum[211] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 5))
    y_sum[212] = y_sum[212] + (1 / (r_return_wire * 5)) + (1 / (r_return_wire * 4))
    y_sum[213] = y_sum[213] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 4))
    y_sum[214] = y_sum[214] + (1 / (r_return_wire * 4)) + (1 / (r_return_wire * 3))
    y_sum[215] = y_sum[215] + (1 / (r_return_wire * 3)) + (1 / (r_return_wire * 1))
    y_sum[216] = y_sum[216] + (1 / (r_return_wire * 1))

    # Add sums to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances (top)
    for i in range(0, 8):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(8, 88):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(88, 96):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    # Traction rail admittances (bottom)
    for i in range(99, 107):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(107, 187):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(187, 195):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long

    # Signal Rail admittances (top)
    y_matrix[97, 98] = -1 * (1 / (r_sig * 10))
    y_matrix[98, 97] = -1 * (1 / (r_sig * 10))

    # Signal Rail admittances (b)
    for i in range(196, 201):
        y_matrix[i, i + 1] = -1 * (1 / (r_sig * 5))
        y_matrix[i + 1, i] = -1 * (1 / (r_sig * 5))

    # Cross bonds
    cb_a = node_indices_cb[node_indices_cb < 98]
    cb_b = node_indices_cb[node_indices_cb > 98]
    y_matrix[cb_a, cb_b] = -y_cb
    y_matrix[cb_b, cb_a] = -y_cb

    # Return conductor vertical
    # Top
    y_matrix[
        node_indices_return_wire_vertical[node_indices_return_wire_vertical < 97], node_indices_return_wire_vertical[
            (node_indices_return_wire_vertical > 195) & (node_indices_return_wire_vertical < 210)]] = -y_rw_connector
    # Bottom
    y_matrix[
        node_indices_return_wire_vertical[(node_indices_return_wire_vertical > 96) & (node_indices_return_wire_vertical < 196)], node_indices_return_wire_vertical[
            node_indices_return_wire_vertical > 209]] = -y_rw_connector

    # Return conductor horizontal
    for i in [203, 205, 206, 210, 212, 213]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 4)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 4)
    for i in [204, 211]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 5)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 5)
    for i in [207, 214]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 3)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 3)
    for i in [208, 215]:
        y_matrix[i, i + 1] = -1 / (r_return_wire * 1)
        y_matrix[i + 1, i] = -1 / (r_return_wire * 1)

    # Track circuit
    y_matrix[35, 97] = -y_power
    y_matrix[97, 35] = -y_power
    y_matrix[45, 98] = -y_relay
    y_matrix[98, 45] = -y_relay
    y_matrix[134, 196] = -1 * (y_power + y_200k)
    y_matrix[196, 134] = -1 * (y_power + y_200k)

    # 100k ohm connections
    for i, j in zip([139, 144, 149, 154, 159, 164], [197, 198, 199, 200, 201, 202]):
        y_matrix[i, j] = -y_100k
        y_matrix[j, i] = -y_100k

    # Currents
    current = e_par / r_rail
    j_matrix = np.zeros(217)
    j_matrix[0] = -current
    j_matrix[35] = -(v_power / r_power)
    j_matrix[96] = current
    j_matrix[97] = (v_power / r_power) - current
    j_matrix[98] = current
    j_matrix[99] = -current
    j_matrix[134] = -(v_power / r_power)
    j_matrix[195] = current
    j_matrix[196] = (v_power / r_power) - current
    j_matrix[202] = current

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    v_relay = v_matrix[98] - v_matrix[45]
    i_relay = v_relay / r_relay

    return i_relay


# DCMOD18a: double track railway, jointed rail with rail return.
def dcmod18a_trains(r_power, r_relay, v_power, e_par):
    # Set parameters
    r_rail = 0.25  # ohm/km
    i_train = 1  # ampere

    y_cb = 1 / 1e-3
    y_axle_trac = 1 / 0.1e-3
    y_axle_sig = 1 / 25e-3
    y_power = 1 / r_power
    y_relay = 1 / r_relay
    y_train_current = 1 / 0.28
    y_feeder = 1 / 0.28
    r_sig = 0.015  # For each 60m stretch
    y_100k = 1 / 1e5
    y_200k = 1 / 2e5

    # Calculate the admittance of the rails based on the type
    # 0.25 ohm/km for jointed, 0.035 ohm/km for CWR
    ye_long = ((780/1000) * r_rail)
    ye_short = ((60/1000) * r_rail)

    # Define parallel admittances
    ye_sum = np.zeros(204)
    # Traction rails
    # Top track
    ye_sum[0] = ye_long
    ye_sum[1:8] = ye_long * 2
    ye_sum[8] = ye_long + ye_short
    ye_sum[9:88] = ye_short * 2
    ye_sum[88] = ye_short + ye_long
    ye_sum[89:96] = ye_long * 2
    ye_sum[96] = ye_long
    # Bottom track
    ye_sum[99] = ye_long
    ye_sum[100:107] = ye_long * 2
    ye_sum[107] = ye_long + ye_short
    ye_sum[108:187] = ye_short * 2
    ye_sum[187] = ye_short + ye_long
    ye_sum[188:195] = ye_long * 2
    ye_sum[195] = ye_long

    # Signal rails
    for i in [196, 202]:
        ye_sum[i] = 1 / (r_sig * 5)
    for i in [97, 98, 197, 198, 199, 200, 201]:
        ye_sum[i] = 1 / (r_sig * 10)

    # Define ground admittances
    yg_long = 1 / 1.5
    yg_short = 1 / 20

    # Set up the ground admittance array
    yg = np.zeros(204)
    # Top track
    # No ground
    yg[0:2] = 0
    # Long sections 1
    yg[2:8] = yg_long
    # Short sections
    yg[8:89] = yg_short
    # Long sections 2
    yg[89:97] = yg_long
    # Bottom track
    # No ground
    yg[99:101] = 0
    # Long sections 1
    yg[101:107] = yg_long
    # Short sections
    yg[107:188] = yg_short
    # Long sections 2
    yg[188:196] = yg_long

    # Define the nodal network
    node_indices_cb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 36, 48, 61, 74, 88, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 107, 120, 135, 147, 160, 173, 187, 188, 189, 190, 191, 192, 193, 194, 195])
    node_indices_axle_trac = np.array([35, 203])
    node_indices_axle_sig = np.array([97, 203])
    node_indices_power = np.array([35, 97, 134, 196])
    node_indices_relay = np.array([45, 98])
    node_indices_train_current = np.array([24, 123, 203])
    node_indices_feeder_double = np.array([88])
    node_indices_feeder_single = np.array([187])
    node_indices_100k = np.array([139, 197, 144, 198, 149, 199, 154, 200, 159, 201, 164, 202])
    node_indices_200k = np.array([134, 196])

    # Set up admittance arrays for the different components
    y_cb_array = np.zeros(204)
    y_cb_array[node_indices_cb] = y_cb
    y_axle_trac_array = np.zeros(204)
    y_axle_trac_array[node_indices_axle_trac] = y_axle_trac
    y_axle_sig_array = np.zeros(204)
    y_axle_sig_array[node_indices_axle_sig] = y_axle_sig
    y_power_array = np.zeros(204)
    y_power_array[node_indices_power] = y_power
    y_relay_array = np.zeros(204)
    y_relay_array[node_indices_relay] = y_relay
    y_train_current_array = np.zeros(204)
    y_train_current_array[node_indices_train_current] = y_train_current
    y_feeder_single_array = np.zeros(204)
    y_feeder_single_array[node_indices_feeder_single] = y_feeder
    y_feeder_double_array = np.zeros(204)
    y_feeder_double_array[node_indices_feeder_double] = y_feeder * 2
    y_100k_array = np.zeros(204)
    y_100k_array[node_indices_100k] = y_100k
    y_200k_array = np.zeros(204)
    y_200k_array[node_indices_200k] = y_200k

    # Set up y matrix of zeroes
    y_matrix = np.zeros((204, 204))

    # Sum of admittances
    y_sum = np.empty(204)

    # Uniform nodes
    for i in range(0, 204):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_axle_trac_array[i] + y_axle_sig_array[i] + \
                   y_power_array[i] + y_relay_array[i] + y_train_current_array[i] + y_feeder_single_array[i] + \
                   y_feeder_double_array[i] + y_100k_array[i] + y_200k_array[i]

    # Add sums to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances (top)
    for i in range(0, 8):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(8, 88):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(88, 96):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    # Traction rail admittances (bottom)
    for i in range(99, 107):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(107, 187):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(187, 195):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long

    # Signal Rail admittances (top)
    y_matrix[97, 98] = -1 * (1 / (r_sig * 10))
    y_matrix[98, 97] = -1 * (1 / (r_sig * 10))

    # Signal Rail admittances (b)
    for i in range(196, 201):
        y_matrix[i, i + 1] = -1 * (1 / (r_sig * 5))
        y_matrix[i + 1, i] = -1 * (1 / (r_sig * 5))

    # Cross bonds
    cb_a = node_indices_cb[node_indices_cb < 98]
    cb_b = node_indices_cb[node_indices_cb > 98]
    y_matrix[cb_a, cb_b] = -y_cb
    y_matrix[cb_b, cb_a] = -y_cb

    # Track circuit
    y_matrix[35, 97] = -y_power
    y_matrix[97, 35] = -y_power
    y_matrix[45, 98] = -y_relay
    y_matrix[98, 45] = -y_relay
    y_matrix[35, 203] = -y_axle_trac
    y_matrix[203, 35] = -y_axle_trac
    y_matrix[97, 203] = -y_axle_sig
    y_matrix[203, 97] = -y_axle_sig
    y_matrix[134, 196] = -1 * (y_power + y_200k)
    y_matrix[196, 134] = -1 * (y_power + y_200k)

    # 100k ohm connections
    for i, j in zip([139, 144, 149, 154, 159, 164], [197, 198, 199, 200, 201, 202]):
        y_matrix[i, j] = -y_100k
        y_matrix[j, i] = -y_100k

    # Train current
    for i, j in zip([24, 203, 123], [88, 88, 187]):
        y_matrix[i, j] = -y_train_current
        y_matrix[j, i] = -y_train_current

    # Currents
    current = e_par / r_rail
    j_matrix = np.zeros(204)
    j_matrix[0] = -current
    j_matrix[24] = i_train
    j_matrix[35] = -(v_power / r_power)
    j_matrix[88] = -1 * (i_train * 2)
    j_matrix[96] = current
    j_matrix[97] = (v_power / r_power) - current
    j_matrix[98] = current
    j_matrix[203] = i_train
    j_matrix[99] = -current
    j_matrix[123] = i_train
    j_matrix[134] = -(v_power / r_power)
    j_matrix[187] = -i_train
    j_matrix[195] = current
    j_matrix[196] = (v_power / r_power) - current
    j_matrix[202] = current

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    v_relay = v_matrix[98] - v_matrix[45]
    i_relay = v_relay / r_relay

    return i_relay


def dcmod18a_no_trains(r_power, r_relay, v_power, e_par):
    # Set parameters
    r_rail = 0.25  # ohm/km

    y_cb = 1 / 1e-3
    y_power = 1 / r_power
    y_relay = 1 / r_relay
    r_sig = 0.015  # For each 60m stretch
    y_100k = 1 / 1e5
    y_200k = 1 / 2e5
    y_1_ohm = 1 / 1

    # Calculate the admittance of the rails based on the type
    # 0.25 ohm/km for jointed, 0.035 ohm/km for CWR
    ye_long = ((780/1000) * r_rail)
    ye_short = ((60/1000) * r_rail)

    # Define parallel admittances
    ye_sum = np.zeros(203)
    # Traction rails
    # Top track
    ye_sum[0] = ye_long
    ye_sum[1:8] = ye_long * 2
    ye_sum[8] = ye_long + ye_short
    ye_sum[9:88] = ye_short * 2
    ye_sum[88] = ye_short + ye_long
    ye_sum[89:96] = ye_long * 2
    ye_sum[96] = ye_long
    # Bottom track
    ye_sum[99] = ye_long
    ye_sum[100:107] = ye_long * 2
    ye_sum[107] = ye_long + ye_short
    ye_sum[108:187] = ye_short * 2
    ye_sum[187] = ye_short + ye_long
    ye_sum[188:195] = ye_long * 2
    ye_sum[195] = ye_long

    # Signal rails
    for i in [196, 202]:
        ye_sum[i] = 1 / (r_sig * 5)
    for i in [97, 98, 197, 198, 199, 200, 201]:
        ye_sum[i] = 1 / (r_sig * 10)

    # Define ground admittances
    yg_long = 1 / 1.5
    yg_short = 1 / 20

    # Set up the ground admittance array
    yg = np.zeros(203)
    # Top track
    # No ground
    yg[0:2] = 0
    # Long sections 1
    yg[2:8] = yg_long
    # Short sections
    yg[8:89] = yg_short
    # Long sections 2
    yg[89:97] = yg_long
    # Bottom track
    # No ground
    yg[99:101] = 0
    # Long sections 1
    yg[101:107] = yg_long
    # Short sections
    yg[107:188] = yg_short
    # Long sections 2
    yg[188:196] = yg_long

    # Define the nodal network
    node_indices_cb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 36, 48, 61, 74, 88, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 107, 120, 135, 147, 160, 173, 187, 188, 189, 190, 191, 192, 193, 194, 195])
    node_indices_power = np.array([35, 97, 134, 196])
    node_indices_relay = np.array([45, 98])
    node_indices_100k = np.array([139, 197, 144, 198, 149, 199, 154, 200, 159, 201, 164, 202])
    node_indices_200k = np.array([134, 196])

    # Set up admittance arrays for the different components
    y_cb_array = np.zeros(203)
    y_cb_array[node_indices_cb] = y_cb
    y_power_array = np.zeros(203)
    y_power_array[node_indices_power] = y_power
    y_relay_array = np.zeros(203)
    y_relay_array[node_indices_relay] = y_relay
    y_100k_array = np.zeros(203)
    y_100k_array[node_indices_100k] = y_100k
    y_200k_array = np.zeros(203)
    y_200k_array[node_indices_200k] = y_200k

    # Set up y matrix of zeroes
    y_matrix = np.zeros((203, 203))

    # Sum of admittances
    y_sum = np.empty(203)

    # Uniform nodes
    for i in range(0, 203):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_power_array[i] + y_relay_array[i] + y_100k_array[i] + y_200k_array[i]

    # Add sums to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances (top)
    for i in range(0, 8):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(8, 88):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(88, 96):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    # Traction rail admittances (bottom)
    for i in range(99, 107):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(107, 187):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(187, 195):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long

    # Signal Rail admittances (top)
    y_matrix[97, 98] = -1 * (1 / (r_sig * 10))
    y_matrix[98, 97] = -1 * (1 / (r_sig * 10))

    # Signal Rail admittances (b)
    for i in range(196, 201):
        y_matrix[i, i + 1] = -1 * (1 / (r_sig * 5))
        y_matrix[i + 1, i] = -1 * (1 / (r_sig * 5))

    # Cross bonds
    cb_a = node_indices_cb[node_indices_cb < 98]
    cb_b = node_indices_cb[node_indices_cb > 98]
    y_matrix[cb_a, cb_b] = -y_cb
    y_matrix[cb_b, cb_a] = -y_cb

    # Track circuit
    y_matrix[35, 97] = -y_power
    y_matrix[97, 35] = -y_power
    y_matrix[45, 98] = -y_relay
    y_matrix[98, 45] = -y_relay
    y_matrix[134, 196] = -1 * (y_power + y_200k)
    y_matrix[196, 134] = -1 * (y_power + y_200k)

    # 100k ohm connections
    for i, j in zip([139, 144, 149, 154, 159, 164], [197, 198, 199, 200, 201, 202]):
        y_matrix[i, j] = -y_100k
        y_matrix[j, i] = -y_100k

    # Currents
    current = e_par / r_rail
    j_matrix = np.zeros(203)
    j_matrix[0] = -current
    j_matrix[35] = -(v_power / r_power)
    j_matrix[96] = current
    j_matrix[97] = (v_power / r_power) - current
    j_matrix[98] = current
    j_matrix[99] = -current
    j_matrix[134] = -(v_power / r_power)
    j_matrix[195] = current
    j_matrix[196] = (v_power / r_power) - current
    j_matrix[202] = current

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    v_relay = v_matrix[98] - v_matrix[45]
    i_relay = v_relay / r_relay

    return i_relay


# DCMOD20a: double track railway, CWR with rail return.
def dcmod20a_trains(r_power, r_relay, v_power, e_par):
    # Set parameters
    r_rail = 0.035  # ohm/km
    i_train = 1  # ampere

    y_cb = 1 / 1e-3
    y_axle_trac = 1 / 0.1e-3
    y_axle_sig = 1 / 25e-3
    y_power = 1 / r_power
    y_relay = 1 / r_relay
    y_train_current = 1 / 0.28
    y_feeder = 1 / 0.28
    r_sig = 0.015  # For each 60m stretch
    y_100k = 1 / 1e5
    y_200k = 1 / 2e5

    # Calculate the admittance of the rails based on the type
    # 0.25 ohm/km for jointed, 0.035 ohm/km for CWR
    ye_long = ((780/1000) * r_rail)
    ye_short = ((60/1000) * r_rail)

    # Define parallel admittances
    ye_sum = np.zeros(204)
    # Traction rails
    # Top track
    ye_sum[0] = ye_long
    ye_sum[1:8] = ye_long * 2
    ye_sum[8] = ye_long + ye_short
    ye_sum[9:88] = ye_short * 2
    ye_sum[88] = ye_short + ye_long
    ye_sum[89:96] = ye_long * 2
    ye_sum[96] = ye_long
    # Bottom track
    ye_sum[99] = ye_long
    ye_sum[100:107] = ye_long * 2
    ye_sum[107] = ye_long + ye_short
    ye_sum[108:187] = ye_short * 2
    ye_sum[187] = ye_short + ye_long
    ye_sum[188:195] = ye_long * 2
    ye_sum[195] = ye_long

    # Signal rails
    for i in [196, 202]:
        ye_sum[i] = 1 / (r_sig * 5)
    for i in [97, 98, 197, 198, 199, 200, 201]:
        ye_sum[i] = 1 / (r_sig * 10)

    # Define ground admittances
    yg_long = 1 / 1.5
    yg_short = 1 / 20

    # Set up the ground admittance array
    yg = np.zeros(204)
    # Top track
    # No ground
    yg[0:2] = 0
    # Long sections 1
    yg[2:8] = yg_long
    # Short sections
    yg[8:89] = yg_short
    # Long sections 2
    yg[89:97] = yg_long
    # Bottom track
    # No ground
    yg[99:101] = 0
    # Long sections 1
    yg[101:107] = yg_long
    # Short sections
    yg[107:188] = yg_short
    # Long sections 2
    yg[188:196] = yg_long

    # Define the nodal network
    node_indices_cb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 36, 48, 61, 74, 88, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 107, 120, 135, 147, 160, 173, 187, 188, 189, 190, 191, 192, 193, 194, 195])
    node_indices_axle_trac = np.array([35, 203])
    node_indices_axle_sig = np.array([97, 203])
    node_indices_power = np.array([35, 97, 134, 196])
    node_indices_relay = np.array([45, 98])
    node_indices_train_current = np.array([24, 123, 203])
    node_indices_feeder_double = np.array([88])
    node_indices_feeder_single = np.array([187])
    node_indices_100k = np.array([139, 197, 144, 198, 149, 199, 154, 200, 159, 201, 164, 202])
    node_indices_200k = np.array([134, 196])

    # Set up admittance arrays for the different components
    y_cb_array = np.zeros(204)
    y_cb_array[node_indices_cb] = y_cb
    y_axle_trac_array = np.zeros(204)
    y_axle_trac_array[node_indices_axle_trac] = y_axle_trac
    y_axle_sig_array = np.zeros(204)
    y_axle_sig_array[node_indices_axle_sig] = y_axle_sig
    y_power_array = np.zeros(204)
    y_power_array[node_indices_power] = y_power
    y_relay_array = np.zeros(204)
    y_relay_array[node_indices_relay] = y_relay
    y_train_current_array = np.zeros(204)
    y_train_current_array[node_indices_train_current] = y_train_current
    y_feeder_single_array = np.zeros(204)
    y_feeder_single_array[node_indices_feeder_single] = y_feeder
    y_feeder_double_array = np.zeros(204)
    y_feeder_double_array[node_indices_feeder_double] = y_feeder * 2
    y_100k_array = np.zeros(204)
    y_100k_array[node_indices_100k] = y_100k
    y_200k_array = np.zeros(204)
    y_200k_array[node_indices_200k] = y_200k

    # Set up y matrix of zeroes
    y_matrix = np.zeros((204, 204))

    # Sum of admittances
    y_sum = np.empty(204)

    # Uniform nodes
    for i in range(0, 204):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_axle_trac_array[i] + y_axle_sig_array[i] + \
                   y_power_array[i] + y_relay_array[i] + y_train_current_array[i] + y_feeder_single_array[i] + \
                   y_feeder_double_array[i] + y_100k_array[i] + y_200k_array[i]

    # Add sums to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances (top)
    for i in range(0, 8):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(8, 88):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(88, 96):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    # Traction rail admittances (bottom)
    for i in range(99, 107):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(107, 187):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(187, 195):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long

    # Signal Rail admittances (top)
    y_matrix[97, 98] = -1 * (1 / (r_sig * 10))
    y_matrix[98, 97] = -1 * (1 / (r_sig * 10))

    # Signal Rail admittances (b)
    for i in range(196, 201):
        y_matrix[i, i + 1] = -1 * (1 / (r_sig * 5))
        y_matrix[i + 1, i] = -1 * (1 / (r_sig * 5))

    # Cross bonds
    cb_a = node_indices_cb[node_indices_cb < 98]
    cb_b = node_indices_cb[node_indices_cb > 98]
    y_matrix[cb_a, cb_b] = -y_cb
    y_matrix[cb_b, cb_a] = -y_cb

    # Track circuit
    y_matrix[35, 97] = -y_power
    y_matrix[97, 35] = -y_power
    y_matrix[45, 98] = -y_relay
    y_matrix[98, 45] = -y_relay
    y_matrix[35, 203] = -y_axle_trac
    y_matrix[203, 35] = -y_axle_trac
    y_matrix[97, 203] = -y_axle_sig
    y_matrix[203, 97] = -y_axle_sig
    y_matrix[134, 196] = -1 * (y_power + y_200k)
    y_matrix[196, 134] = -1 * (y_power + y_200k)

    # 100k ohm connections
    for i, j in zip([139, 144, 149, 154, 159, 164], [197, 198, 199, 200, 201, 202]):
        y_matrix[i, j] = -y_100k
        y_matrix[j, i] = -y_100k

    # Train current
    for i, j in zip([24, 203, 123], [88, 88, 187]):
        y_matrix[i, j] = -y_train_current
        y_matrix[j, i] = -y_train_current

    # Currents
    current = e_par / r_rail
    j_matrix = np.zeros(204)
    j_matrix[0] = -current
    j_matrix[24] = i_train
    j_matrix[35] = -(v_power / r_power)
    j_matrix[88] = -1 * (i_train * 2)
    j_matrix[96] = current
    j_matrix[97] = (v_power / r_power) - current
    j_matrix[98] = current
    j_matrix[203] = i_train
    j_matrix[99] = -current
    j_matrix[123] = i_train
    j_matrix[134] = -(v_power / r_power)
    j_matrix[187] = -i_train
    j_matrix[195] = current
    j_matrix[196] = (v_power / r_power) - current
    j_matrix[202] = current

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    v_relay = v_matrix[98] - v_matrix[45]
    i_relay = v_relay / r_relay

    return i_relay


def dcmod20a_no_trains(r_power, r_relay, v_power, e_par):
    # Set parameters
    r_rail = 0.035  # ohm/km

    y_cb = 1 / 1e-3
    y_power = 1 / r_power
    y_relay = 1 / r_relay
    r_sig = 0.015  # For each 60m stretch
    y_100k = 1 / 1e5
    y_200k = 1 / 2e5
    y_1_ohm = 1 / 1

    # Calculate the admittance of the rails based on the type
    # 0.25 ohm/km for jointed, 0.035 ohm/km for CWR
    ye_long = ((780/1000) * r_rail)
    ye_short = ((60/1000) * r_rail)

    # Define parallel admittances
    ye_sum = np.zeros(203)
    # Traction rails
    # Top track
    ye_sum[0] = ye_long
    ye_sum[1:8] = ye_long * 2
    ye_sum[8] = ye_long + ye_short
    ye_sum[9:88] = ye_short * 2
    ye_sum[88] = ye_short + ye_long
    ye_sum[89:96] = ye_long * 2
    ye_sum[96] = ye_long
    # Bottom track
    ye_sum[99] = ye_long
    ye_sum[100:107] = ye_long * 2
    ye_sum[107] = ye_long + ye_short
    ye_sum[108:187] = ye_short * 2
    ye_sum[187] = ye_short + ye_long
    ye_sum[188:195] = ye_long * 2
    ye_sum[195] = ye_long

    # Signal rails
    for i in [196, 202]:
        ye_sum[i] = 1 / (r_sig * 5)
    for i in [97, 98, 197, 198, 199, 200, 201]:
        ye_sum[i] = 1 / (r_sig * 10)

    # Define ground admittances
    yg_long = 1 / 1.5
    yg_short = 1 / 20

    # Set up the ground admittance array
    yg = np.zeros(203)
    # Top track
    # No ground
    yg[0:2] = 0
    # Long sections 1
    yg[2:8] = yg_long
    # Short sections
    yg[8:89] = yg_short
    # Long sections 2
    yg[89:97] = yg_long
    # Bottom track
    # No ground
    yg[99:101] = 0
    # Long sections 1
    yg[101:107] = yg_long
    # Short sections
    yg[107:188] = yg_short
    # Long sections 2
    yg[188:196] = yg_long

    # Define the nodal network
    node_indices_cb = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 21, 36, 48, 61, 74, 88, 89, 90, 91, 92, 93, 94, 95, 96, 99, 100, 101, 102, 103, 104, 105, 106, 107, 120, 135, 147, 160, 173, 187, 188, 189, 190, 191, 192, 193, 194, 195])
    node_indices_power = np.array([35, 97, 134, 196])
    node_indices_relay = np.array([45, 98])
    node_indices_100k = np.array([139, 197, 144, 198, 149, 199, 154, 200, 159, 201, 164, 202])
    node_indices_200k = np.array([134, 196])

    # Set up admittance arrays for the different components
    y_cb_array = np.zeros(203)
    y_cb_array[node_indices_cb] = y_cb
    y_power_array = np.zeros(203)
    y_power_array[node_indices_power] = y_power
    y_relay_array = np.zeros(203)
    y_relay_array[node_indices_relay] = y_relay
    y_100k_array = np.zeros(203)
    y_100k_array[node_indices_100k] = y_100k
    y_200k_array = np.zeros(203)
    y_200k_array[node_indices_200k] = y_200k

    # Set up y matrix of zeroes
    y_matrix = np.zeros((203, 203))

    # Sum of admittances
    y_sum = np.empty(203)

    # Uniform nodes
    for i in range(0, 203):
        y_sum[i] = ye_sum[i] + yg[i] + y_cb_array[i] + y_power_array[i] + y_relay_array[i] + y_100k_array[i] + y_200k_array[i]

    # Add sums to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances (top)
    for i in range(0, 8):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(8, 88):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(88, 96):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    # Traction rail admittances (bottom)
    for i in range(99, 107):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long
    for i in range(107, 187):
        y_matrix[i, i + 1] = -ye_short
        y_matrix[i + 1, i] = -ye_short
    for i in range(187, 195):
        y_matrix[i, i + 1] = -ye_long
        y_matrix[i + 1, i] = -ye_long

    # Signal Rail admittances (top)
    y_matrix[97, 98] = -1 * (1 / (r_sig * 10))
    y_matrix[98, 97] = -1 * (1 / (r_sig * 10))

    # Signal Rail admittances (b)
    for i in range(196, 201):
        y_matrix[i, i + 1] = -1 * (1 / (r_sig * 5))
        y_matrix[i + 1, i] = -1 * (1 / (r_sig * 5))

    # Cross bonds
    cb_a = node_indices_cb[node_indices_cb < 98]
    cb_b = node_indices_cb[node_indices_cb > 98]
    y_matrix[cb_a, cb_b] = -y_cb
    y_matrix[cb_b, cb_a] = -y_cb

    # Track circuit
    y_matrix[35, 97] = -y_power
    y_matrix[97, 35] = -y_power
    y_matrix[45, 98] = -y_relay
    y_matrix[98, 45] = -y_relay
    y_matrix[134, 196] = -1 * (y_power + y_200k)
    y_matrix[196, 134] = -1 * (y_power + y_200k)

    # 100k ohm connections
    for i, j in zip([139, 144, 149, 154, 159, 164], [197, 198, 199, 200, 201, 202]):
        y_matrix[i, j] = -y_100k
        y_matrix[j, i] = -y_100k

    # Currents
    current = e_par / r_rail
    j_matrix = np.zeros(203)
    j_matrix[0] = -current
    j_matrix[35] = -(v_power / r_power)
    j_matrix[96] = current
    j_matrix[97] = (v_power / r_power) - current
    j_matrix[98] = current
    j_matrix[99] = -current
    j_matrix[134] = -(v_power / r_power)
    j_matrix[195] = current
    j_matrix[196] = (v_power / r_power) - current
    j_matrix[202] = current

    # Calculate voltage matrix
    # Calculate inverse of admittance matrix
    y_matrix_inv = np.linalg.inv(y_matrix)

    # Calculate nodal voltages
    v_matrix = np.matmul(y_matrix_inv, j_matrix.T)

    v_relay = v_matrix[98] - v_matrix[45]
    i_relay = v_relay / r_relay

    return i_relay


def plot_br939():
    es = np.arange(-10, 10.1, 0.1)
    drop_out = 0.055
    pick_up = 0.081

    i_14a_t = np.empty(len(es))
    i_14a_nt = np.empty(len(es))
    i_16a_t = np.empty(len(es))
    i_16a_nt = np.empty(len(es))
    i_18a_t = np.empty(len(es))
    i_18a_nt = np.empty(len(es))
    i_20a_t = np.empty(len(es))
    i_20a_nt = np.empty(len(es))
    for i in range(0, len(es)):
        i_14a_t[i] = dcmod14a_trains(r_power=7.2, r_relay=20, v_power=10, e_par=np.array(es[i]))
        i_14a_nt[i] = dcmod14a_no_trains(r_power=7.2, r_relay=20, v_power=10, e_par=np.array(es[i]))
        i_16a_t[i] = dcmod16a_trains(r_power=7.2, r_relay=20, v_power=10, e_par=np.array(es[i]))
        i_16a_nt[i] = dcmod16a_no_trains(r_power=7.2, r_relay=20, v_power=10, e_par=np.array(es[i]))
        i_18a_t[i] = dcmod18a_trains(r_power=7.2, r_relay=20, v_power=10, e_par=np.array(es[i]))
        i_18a_nt[i] = dcmod18a_no_trains(r_power=7.2, r_relay=20, v_power=10, e_par=np.array(es[i]))
        i_20a_t[i] = dcmod20a_trains(r_power=7.2, r_relay=20, v_power=10, e_par=np.array(es[i]))
        i_20a_nt[i] = dcmod20a_no_trains(r_power=7.2, r_relay=20, v_power=10, e_par=np.array(es[i]))

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))

    gs = GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[:, :-1])
    ax1 = fig.add_subplot(gs[:, 1:])

    ax0.plot(es, i_14a_t, label='DCMOD14a (JR, RC)')
    ax0.plot(es, i_16a_t, label='DCMOD16a (CWR, RC)')
    ax0.plot(es, i_18a_t, label='DCMOD18a (JR, RR)')
    ax0.plot(es, i_20a_t, label='DCMOD20a (CWR, RR)')
    ax0.set_title("Trains")
    ax1.plot(es, i_14a_nt, label='DCMOD14a (JR, RC)')
    ax1.plot(es, i_16a_nt, label='DCMOD16a (CWR, RC)')
    ax1.plot(es, i_18a_nt, label='DCMOD18a (JR, RR)')
    ax1.plot(es, i_20a_nt, label='DCMOD20a (CWR, RR)')
    ax1.set_title("No Trains")

    def plot_params(ax):
        # Current thresholds
        ax.axhline(drop_out, color='red')
        ax.axhline(pick_up, color='green')

        # Legend
        ax.legend()

        # Horizontal axis limit
        ax.set_xlim(-10, 10)

        # Axis labels
        ax.set_xlabel("Electric Field Strength (V/km)")
        ax.set_ylabel("Current Through Relay (A)")

        # Major and minor ticks
        major_ticks = np.arange(-10, 11, 2)
        minor_ticks = np.arange(-10, 11, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

    plot_params(ax0)
    plot_params(ax1)

    plt.suptitle("BR939 (20 ohm)")
    plt.show()


def plot_br966_f2():
    es = np.arange(-10, 10.1, 0.1)
    drop_out = 0.081
    pick_up = 0.12

    i_14a_t = np.empty(len(es))
    i_14a_nt = np.empty(len(es))
    i_16a_t = np.empty(len(es))
    i_16a_nt = np.empty(len(es))
    i_18a_t = np.empty(len(es))
    i_18a_nt = np.empty(len(es))
    i_20a_t = np.empty(len(es))
    i_20a_nt = np.empty(len(es))
    for i in range(0, len(es)):
        i_14a_t[i] = dcmod14a_trains(r_power=7.2, r_relay=9, v_power=6, e_par=np.array(es[i]))
        i_14a_nt[i] = dcmod14a_no_trains(r_power=7.2, r_relay=9, v_power=6, e_par=np.array(es[i]))
        i_16a_t[i] = dcmod16a_trains(r_power=7.2, r_relay=9, v_power=6, e_par=np.array(es[i]))
        i_16a_nt[i] = dcmod16a_no_trains(r_power=7.2, r_relay=9, v_power=6, e_par=np.array(es[i]))
        i_18a_t[i] = dcmod18a_trains(r_power=7.2, r_relay=9, v_power=6, e_par=np.array(es[i]))
        i_18a_nt[i] = dcmod18a_no_trains(r_power=7.2, r_relay=9, v_power=6, e_par=np.array(es[i]))
        i_20a_t[i] = dcmod20a_trains(r_power=7.2, r_relay=9, v_power=6, e_par=np.array(es[i]))
        i_20a_nt[i] = dcmod20a_no_trains(r_power=7.2, r_relay=9, v_power=6, e_par=np.array(es[i]))

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))

    gs = GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[:, :-1])
    ax1 = fig.add_subplot(gs[:, 1:])

    ax0.plot(es, i_14a_t, label='DCMOD14a (JR, RC)')
    ax0.plot(es, i_16a_t, label='DCMOD16a (CWR, RC)')
    ax0.plot(es, i_18a_t, label='DCMOD18a (JR, RR)')
    ax0.plot(es, i_20a_t, label='DCMOD20a (CWR, RR)')
    ax0.set_title("Trains")
    ax1.plot(es, i_14a_nt, label='DCMOD14a (JR, RC)')
    ax1.plot(es, i_16a_nt, label='DCMOD16a (CWR, RC)')
    ax1.plot(es, i_18a_nt, label='DCMOD18a (JR, RR)')
    ax1.plot(es, i_20a_nt, label='DCMOD20a (CWR, RR)')
    ax1.set_title("No Trains")

    def plot_params(ax):
        # Current thresholds
        ax.axhline(drop_out, color='red')
        ax.axhline(pick_up, color='green')

        # Legend
        ax.legend()

        # Horizontal axis limit
        ax.set_xlim(-10, 10)

        # Axis labels
        ax.set_xlabel("Electric Field Strength (V/km)")
        ax.set_ylabel("Current Through Relay (A)")

        # Major and minor ticks
        major_ticks = np.arange(-10, 11, 2)
        minor_ticks = np.arange(-10, 11, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

    plot_params(ax0)
    plot_params(ax1)

    plt.suptitle("BR966 F2 (9 ohm)")
    plt.show()


def plot_br966_f9():
    es = np.arange(-10, 10.1, 0.1)
    drop_out = 0.032
    pick_up = 0.047

    i_14a_t = np.empty(len(es))
    i_14a_nt = np.empty(len(es))
    i_16a_t = np.empty(len(es))
    i_16a_nt = np.empty(len(es))
    i_18a_t = np.empty(len(es))
    i_18a_nt = np.empty(len(es))
    i_20a_t = np.empty(len(es))
    i_20a_nt = np.empty(len(es))
    for i in range(0, len(es)):
        i_14a_t[i] = dcmod14a_trains(r_power=7.2, r_relay=60, v_power=10, e_par=np.array(es[i]))
        i_14a_nt[i] = dcmod14a_no_trains(r_power=7.2, r_relay=60, v_power=10, e_par=np.array(es[i]))
        i_16a_t[i] = dcmod16a_trains(r_power=7.2, r_relay=60, v_power=10, e_par=np.array(es[i]))
        i_16a_nt[i] = dcmod16a_no_trains(r_power=7.2, r_relay=60, v_power=10, e_par=np.array(es[i]))
        i_18a_t[i] = dcmod18a_trains(r_power=7.2, r_relay=60, v_power=10, e_par=np.array(es[i]))
        i_18a_nt[i] = dcmod18a_no_trains(r_power=7.2, r_relay=60, v_power=10, e_par=np.array(es[i]))
        i_20a_t[i] = dcmod20a_trains(r_power=7.2, r_relay=60, v_power=10, e_par=np.array(es[i]))
        i_20a_nt[i] = dcmod20a_no_trains(r_power=7.2, r_relay=60, v_power=10, e_par=np.array(es[i]))

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))

    gs = GridSpec(1, 2)
    ax0 = fig.add_subplot(gs[:, :-1])
    ax1 = fig.add_subplot(gs[:, 1:])

    ax0.plot(es, i_14a_t, label='DCMOD14a (JR, RC)')
    ax0.plot(es, i_16a_t, label='DCMOD16a (CWR, RC)')
    ax0.plot(es, i_18a_t, label='DCMOD18a (JR, RR)')
    ax0.plot(es, i_20a_t, label='DCMOD20a (CWR, RR)')
    ax0.set_title("Trains")
    ax1.plot(es, i_14a_nt, label='DCMOD14a (JR, RC)')
    ax1.plot(es, i_16a_nt, label='DCMOD16a (CWR, RC)')
    ax1.plot(es, i_18a_nt, label='DCMOD18a (JR, RR)')
    ax1.plot(es, i_20a_nt, label='DCMOD20a (CWR, RR)')
    ax1.set_title("No Trains")

    def plot_params(ax):
        # Current thresholds
        ax.axhline(drop_out, color='red')
        ax.axhline(pick_up, color='green')

        # Legend
        ax.legend()

        # Horizontal axis limit
        ax.set_xlim(-10, 10)

        # Axis labels
        ax.set_xlabel("Electric Field Strength (V/km)")
        ax.set_ylabel("Current Through Relay (A)")

        # Major and minor ticks
        major_ticks = np.arange(-10, 11, 2)
        minor_ticks = np.arange(-10, 11, 1)
        ax.set_xticks(major_ticks)
        ax.set_xticks(minor_ticks, minor=True)

        # And a corresponding grid
        ax.grid(which='both')

        # Or if you want different settings for the grids:
        ax.grid(which='minor', alpha=0.2)
        ax.grid(which='major', alpha=0.5)

    plot_params(ax0)
    plot_params(ax1)

    plt.suptitle("BR966 F9 (60 ohm)")
    plt.show()


plot_br939()
plot_br966_f2()
plot_br966_f9()
