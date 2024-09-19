import numpy as np


def dcmod14a_train(e_par):
    current = e_par / 0.25

    # Set up y matrix of zeroes
    y_matrix = np.zeros((66, 66))

    # Sum of admittances
    y_sum = np.empty(66)

    # Traction rail return conductor nodes
    for i in [2, 6, 18, 25, 29, 33, 37, 41]:
        y_sum[i] = 1e3 + 5 + 5 + 0.65 + 1e3

    # Traction rail cross bond nodes
    for i in [3, 4, 5, 7, 8, 12, 13, 15, 16, 17, 19, 20, 21, 26, 27, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43, 44]:
        y_sum[i] = 1e3 + 5 + 5 + 0.65

    # Traction rail other nodes
    for i in [0, 23]:
        y_sum[i] = 5 + 1e3
    for i in [1, 24]:
        y_sum[i] = 5 + 5 + 1e3
    for i in [22, 45]:
        y_sum[i] = 5 + 1e3 + 1e3 + 0.65

    y_sum[9] = 5 + 1e3 + 0.63 + 5.4
    y_sum[61] = 5.4 + 0.139 + 1e4 + 66.67 + 0.33
    y_sum[10] = 66.67 + 1e3 + 0.28 + 6.49 + 1e3
    y_sum[62] = 0.11 + 21.74 + 0.33 + 6.49
    y_sum[11] = 0.4 + 5 + 1e3 + 6.49
    y_sum[14] = 5 + 3.57 + 1e3 + 0.65 + 5 + 1e3

    # Return conductor
    for i in [47, 48, 49, 50, 51, 54, 55, 56, 57, 58]:
        y_sum[i] = 2.75 + 1e3 + 2.75
    for i in [46, 53]:
        y_sum[i] = 2.75 + 1e3
    y_sum[52] = 2.75 + 1e-5 + 1
    y_sum[59] = 2.75 + 1
    y_sum[60] = 1e-5 + 1

    # Track circuit nodes
    y_sum[65] = 1e4 + 40 + 3.57

    # Signalling rail nodes
    y_sum[63] = 0.139 + 40 + 6.67
    y_sum[64] = 6.67 + 0.11

    # Add to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances
    for i in range(0, 9):
        y_matrix[i, i+1] = -5
        y_matrix[i+1, i] = -5
    for i in range(11, 22):
        y_matrix[i, i+1] = -5
        y_matrix[i+1, i] = -5
    y_matrix[9, 61] = -5.4
    y_matrix[61, 9] = -5.4
    y_matrix[61, 10] = -66.67
    y_matrix[10, 61] = -66.67
    y_matrix[10, 62] = -6.49
    y_matrix[62, 10] = -6.49
    y_matrix[62, 11] = -21.74
    y_matrix[11, 62] = -21.74

    # Cross bonds
    for i in range(23, 45):
        y_matrix[i, i+1] = -5
        y_matrix[i+1, i] = -5
    for i in range(0, 23):
        y_matrix[i, i+23] = -1e3
        y_matrix[i+23, i] = -1e3

    # Return conductor vertical
    for i, j in zip([2, 6, 10, 14, 18, 22, 25, 29, 33, 37, 41, 45], [46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58]):
        y_matrix[i, j] = -1e3
        y_matrix[j, i] = -1e3
    y_matrix[52, 60] = -1e-5
    y_matrix[60, 52] = -1e-5
    y_matrix[59, 60] = -1
    y_matrix[60, 59] = -1

    # Return conductor horizontal
    for i in range(46, 53):
        y_matrix[i, i+1] = -2.75
        y_matrix[i+1, i] = -2.75
    for i in range(53, 60):
        y_matrix[i, i+1] = -2.75
        y_matrix[i+1, i] = -2.75

    # Track circuit
    y_matrix[14, 65] = -3.57
    y_matrix[65, 14] = -3.57
    y_matrix[63, 65] = -40
    y_matrix[65, 63] = -40
    y_matrix[61, 65] = -1e4
    y_matrix[65, 61] = -1e4
    y_matrix[63, 64] = -6.67
    y_matrix[64, 63] = -6.67
    y_matrix[62, 64] = -0.11
    y_matrix[64, 62] = -0.11
    y_matrix[61, 63] = -0.139
    y_matrix[63, 61] = -0.139

    # Currents
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


def dcmod14a_no_train(e_par):
    current = e_par / 0.25

    # Set up y matrix of zeroes
    y_matrix = np.zeros((65, 65))

    # Sum of admittances
    y_sum = np.empty(65)

    # Traction rail return conductor nodes
    for i in [2, 6, 18, 25, 29, 33, 37, 41]:
        y_sum[i] = 1e3 + 5 + 5 + 0.65 + 1e3

    # Traction rail cross bond nodes
    for i in [3, 4, 5, 7, 8, 12, 13, 15, 16, 17, 19, 20, 21, 26, 27, 28, 30, 31, 32, 34, 35, 36, 38, 39, 40, 42, 43, 44]:
        y_sum[i] = 1e3 + 5 + 5 + 0.65

    # Traction rail other nodes
    for i in [0, 23]:
        y_sum[i] = 5 + 1e3
    for i in [1, 24]:
        y_sum[i] = 5 + 5 + 1e3
    for i in [22, 45]:
        y_sum[i] = 5 + 1e3 + 1e3 + 0.65

    y_sum[9] = 5 + 1e3 + 0.63 + 5.4
    y_sum[61] = 5.4 + 0.139 + 66.67 + 0.33
    y_sum[10] = 66.67 + 1e3 + 0.28 + 6.49 + 1e3
    y_sum[62] = 0.11 + 21.74 + 0.33 + 6.49
    y_sum[11] = 0.4 + 5 + 1e3 + 6.49
    y_sum[14] = 5 + 1e3 + 0.65 + 5 + 1e3

    # Return conductor
    for i in [47, 48, 49, 50, 51, 54, 55, 56, 57, 58]:
        y_sum[i] = 2.75 + 1e3 + 2.75
    for i in [46, 53]:
        y_sum[i] = 2.75 + 1e3
    y_sum[52] = 2.75 + 1e-5 + 1
    y_sum[59] = 2.75 + 1
    y_sum[60] = 1e-5 + 1

    # Signalling rail nodes
    y_sum[63] = 0.139 + 6.67
    y_sum[64] = 6.67 + 0.11

    # Add to y matrix as diagonal elements
    for i in range(0, len(y_sum)):
        y_matrix[i, i] = y_sum[i]

    # Set y matrix off-diagonal elements
    # Traction rail admittances
    for i in range(0, 9):
        y_matrix[i, i + 1] = -5
        y_matrix[i + 1, i] = -5
    for i in range(11, 22):
        y_matrix[i, i + 1] = -5
        y_matrix[i + 1, i] = -5
    y_matrix[9, 61] = -5.4
    y_matrix[61, 9] = -5.4
    y_matrix[61, 10] = -66.67
    y_matrix[10, 61] = -66.67
    y_matrix[10, 62] = -6.49
    y_matrix[62, 10] = -6.49
    y_matrix[62, 11] = -21.74
    y_matrix[11, 62] = -21.74

    # Cross bonds
    for i in range(23, 45):
        y_matrix[i, i + 1] = -5
        y_matrix[i + 1, i] = -5
    for i in range(0, 23):
        y_matrix[i, i + 23] = -1e3
        y_matrix[i + 23, i] = -1e3

    # Return conductor vertical
    for i, j in zip([2, 6, 10, 14, 18, 22, 25, 29, 33, 37, 41, 45], [46, 47, 48, 49, 50, 51, 53, 54, 55, 56, 57, 58]):
        y_matrix[i, j] = -1e3
        y_matrix[j, i] = -1e3
    y_matrix[52, 60] = -1e-5
    y_matrix[60, 52] = -1e-5
    y_matrix[59, 60] = -1
    y_matrix[60, 59] = -1

    # Return conductor horizontal
    for i in range(46, 53):
        y_matrix[i, i + 1] = -2.75
        y_matrix[i + 1, i] = -2.75
    for i in range(53, 60):
        y_matrix[i, i + 1] = -2.75
        y_matrix[i + 1, i] = -2.75

    # Track circuit
    y_matrix[63, 64] = -6.67
    y_matrix[64, 63] = -6.67
    y_matrix[62, 64] = -0.11
    y_matrix[64, 62] = -0.11
    y_matrix[61, 63] = -0.139
    y_matrix[63, 61] = -0.139

    # Currents
    j_matrix = np.zeros(65)
    j_matrix[61] = -0.84
    j_matrix[63] = 0.84 - current
    j_matrix[14] = -1

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


dcmod14a_train(0)
dcmod14a_no_train(0)
#dcmod14a_train_rail_break(0)
