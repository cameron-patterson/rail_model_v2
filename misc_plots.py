import numpy as np
import matplotlib.pyplot as plt
from find_thresholds import e_field_parallel


def plot_current_difference():
    e = np.array([10])
    ia, ib = e_field_parallel(e, "west_coast_main_line", "moderate")
    ia0, ib0 = e_field_parallel(np.array([0]), "west_coast_main_line", "moderate")
    tcb = range(0, len(ia[0]))

    plt.rcParams['font.size'] = '15'
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    fig.suptitle('Difference in current between 0 and 10 V/km applied electric field parallel to each block')

    ax[0].plot(tcb, ia[0] - ia0[0], '.', color='blue')
    ax[0].axhline(0, linestyle='--', color='black')
    ax[0].set_xlabel = 'Track Circuit Block'
    ax[0].set_ylabel = 'Current Difference (A)'
    ax[0].set_xlim(-2, 528)

    ax[1].plot(tcb, ib[0] - ib0[0], '.', color='blue')
    ax[1].axhline(0, linestyle='--', color='black')
    ax[1].set_xlabel = 'Track Circuit Block'
    ax[1].set_ylabel = 'Current Difference (A)'
    ax[1].set_xlim(-2, 528)

    stops = ['Gla', 'Loc', 'Pen', 'Lan', 'Pre', 'Wig', 'War', 'Lon']
    stop_locs = [0, 118, 172, 231, 256, 274, 288, 526]
    for i in range(0, len(stops)):
        ax[0].axvline(stop_locs[i], linestyle='--', color='red')
        ax[0].text(stop_locs[i] + 0.1, max(ia[0] - ia0[0]), stops[i])
        ax[1].axvline(stop_locs[i], linestyle='--', color='red')
        ax[1].text(stop_locs[i] + 0.1, max(ib[0] - ib0[0]), stops[i])

    plt.savefig('cd.pdf')


def plot_parallel_e():
    e = np.array([-10])
    ia, ib = e_field_parallel(e, "west_coast_main_line", "moderate")
    tcb = range(0, len(ia[0]))

    plt.rcParams['font.size'] = '15'
    fig, ax = plt.subplots(2, 1, figsize=(20, 15))
    fig.suptitle('Electric field: ' + str(e) + ' V/km')

    ax[0].plot(tcb, ia[0], '.', color='blue')
    ax[0].axhline(0.055, linestyle='-', color='red')
    ax[0].axhline(0.081, linestyle='--', color='green')
    ax[0].set_xlabel = 'Track Circuit Block'
    ax[0].set_ylabel = 'Current (A)'
    ax[0].set_xlim(-2, 528)

    ax[1].plot(tcb, ib[0], '.', color='blue')
    ax[1].axhline(0.055, linestyle='-', color='red')
    ax[1].axhline(0.081, linestyle='--', color='green')
    ax[1].set_xlabel = 'Track Circuit Block'
    ax[1].set_ylabel = 'Current (A)'
    ax[1].set_xlim(-2, 528)

    stops = ['Gla', 'Loc', 'Pen', 'Lan', 'Pre', 'Wig', 'War', 'Lon']
    stop_locs = [0, 118, 172, 231, 256, 274, 288, 526]
    for i in range(0, len(stops)):
        ax[0].axvline(stop_locs[i], linestyle='--', color='red')
        ax[0].text(stop_locs[i] + 0.1, max(ia[0]), stops[i])
        ax[1].axvline(stop_locs[i], linestyle='--', color='red')
        ax[1].text(stop_locs[i] + 0.1, max(ib[0]), stops[i])

    plt.savefig(fname='e_par_' + str(e) + '.pdf')


def plot_block_lengths(section_name):
    data = np.load("data/rail_data/" + section_name + "/" + section_name + "_lengths_angles.npz")
    blocks = data["block_lengths"]
    angles = data["angles"]

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(blocks, '.')
    ax.set_title(str(section_name))
    ax.set_xlabel("Track Circuit Block")
    ax.set_ylabel("Length (km)")
    plt.show()



plot_block_lengths("west_coast_main_line")
plot_block_lengths("east_coast_main_line")
plot_block_lengths("glasgow_edinburgh_falkirk")
plot_block_lengths("bristol_parkway_london")
