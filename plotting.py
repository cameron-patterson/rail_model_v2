import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from right_side_analysis import right_side_two_track
from wrong_side_analysis import wrong_side_two_track_currents


def plot_right_side(ex, ey, section_name, conditions):
    ia0, ib0 = right_side_two_track(ex, ey, section_name, conditions)
    ia = ia0[0]
    ib = ib0[0]

    double_locs = np.load("data/rail_data/" + section_name + "/" + section_name + "_double_locs.npy")
    misoperate_a = np.array([], dtype=bool)
    j = 0
    for i in range(0, int(len(ia) - 0.5 * len(double_locs))):
        if j in double_locs:
            if ia[j] < 0.055 or ia[j + 1] < 0.055:
                misoperate_a = np.append(misoperate_a, True)
                j += 2
            else:
                misoperate_a = np.append(misoperate_a, False)
                j += 2
        else:
            if ia[j] < 0.055:
                misoperate_a = np.append(misoperate_a, True)
                j += 1
            else:
                misoperate_a = np.append(misoperate_a, False)
                j += 1
    misoperate_b = np.array([], dtype=bool)
    j = 0
    for i in range(0, int(len(ib) - 0.5 * len(double_locs))):
        if j in double_locs:
            if ib[j] < 0.055 or ib[j + 1] < 0.055:
                misoperate_b = np.append(misoperate_b, True)
                j += 2
            else:
                misoperate_b = np.append(misoperate_b, False)
                j += 2
        else:
            if ib[j] < 0.055:
                misoperate_b = np.append(misoperate_b, True)
                j += 1
            else:
                misoperate_b = np.append(misoperate_b, False)
                j += 1

    fig = plt.figure(figsize=(16, 6))
    gs = gridspec.GridSpec(2, 2, width_ratios=[1, 1], height_ratios=[4, 1])
    ax0 = plt.subplot(gs[0, 0])
    ax1 = plt.subplot(gs[0, 1])
    ax2 = plt.subplot(gs[1, 0])
    ax3 = plt.subplot(gs[1, 1])

    ax0.plot(ia, '.')
    ax1.plot(ib, '.')
    ax0.axhline(0.055, color="tomato", linestyle="-")
    ax0.axhline(0.081, color="limegreen", linestyle="--")
    ax0.set_xlabel("Track Circuit Index")
    ax0.set_ylabel("Current Through Relay (A)")
    ax1.axhline(0.055, color="tomato", linestyle="-")
    ax1.axhline(0.081, color="limegreen", linestyle="--")
    ax1.set_xlabel("Track Circuit Index")
    ax1.set_ylabel("Current Through Relay (A)")

    index_array = np.arange(len(misoperate_a))
    colors_a = np.where(misoperate_a, 'red', 'green')
    y_values_a = np.where(misoperate_a, -1, 1)
    ax2.scatter(index_array, y_values_a, c=colors_a, s=1)
    colors_b = np.where(misoperate_b, 'red', 'green')
    y_values_b = np.where(misoperate_b, -1, 1)
    ax3.scatter(index_array, y_values_b, c=colors_b, s=1)
    ax2.set_ylim(-1.5, 2)
    ax2.set_yticks([-1, 1], ['Misoperation', 'Normal'])
    ax2.set_xlabel("Signal Index")
    ax3.set_ylim(-1.5, 2)
    ax3.set_yticks([-1, 1], ['Misoperation', 'Normal'])
    ax3.set_xlabel("Signal Index")

    count_all = len(misoperate_a)
    count_red_a = np.sum(misoperate_a)
    label_text_a = f"{count_red_a}/{count_all} signal misoperations"
    ax2.text(0.5, 1.75, label_text_a, ha='left', va='top', fontsize=10, bbox=dict(facecolor='white', alpha=0, edgecolor='none'))
    count_red_b = np.sum(misoperate_b)
    label_text_b = f"{count_red_b}/{count_all} signal misoperations"
    ax3.text(0.5, 1.75, label_text_b, ha='left', va='top', fontsize=10,
                  bbox=dict(facecolor='white', alpha=0, edgecolor='none'))

    if section_name == "west_coast_main_line":
        section_label = "West Coast Main Line"
    elif section_name == "east_coast_main_line":
        section_label = "East Coast Main Line"
    elif section_name == "glasgow_edinburgh_falkirk":
        section_label = "Glasgow to Edinburgh via Falkirk High"
    else:
        section_label = "ERROR"
    plt.suptitle(f"{section_label}: Ex = {ex[0]} V/km; Ey = {ey[0]} V/km")
    plt.subplots_adjust(hspace=0.3)
    plt.savefig(fname=f"{section_name}_{ex}_{ey}_right_side.jpg")


def plot_wrong_side_block_centre(ex, ey, section_name):
    starts = np.load("data/axle_positions/" + section_name + "_front_axle_pos_block_centre.npy")
    for i in range(0, len(starts)):
        ia, ib = wrong_side_two_track_currents(section_name, i, ex, ey)
        plt.plot(ia[0], '.')
        plt.show()



#exs = np.array([-10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10])
#eys = np.array([0])
#for ex in exs:
#    for section in ["west_coast_main_line", "east_coast_main_line", "glasgow_edinburgh_falkirk"]:
#        plot_right_side(np.array([ex]), eys, section, "moderate")


exs = np.array([-10, -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10])
eys = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
for ex in exs:
    for section in ["glasgow_edinburgh_falkirk"]:
        plot_wrong_side_block_centre(exs, eys, section)
