from misoperation_analysis import rail_model_two_track_e_parallel
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.ticker import PercentFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


def save_right_side_threshold_currents(section_name):
    e_par = np.arange(-40, 40, 0.1)
    ia_all, ib_all = rail_model_two_track_e_parallel(section_name=section_name, conditions="moderate", e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))

    np.savez(f"data\\thresholds\\rs_threshold_currents_{section_name}", threshold_currents_a=ia_all, threshold_currents_b=ib_all)


def save_wrong_side_thresholds_currents(section_name):
    e_par = np.arange(-40, 40, 0.1)

    axle_pos_mid = np.load(f"data\\axle_positions\\block_centre\\{section_name}_axle_positions_two_track_back_axle_block_centre.npz", allow_pickle=True)
    axle_pos_all_a = axle_pos_mid["axle_pos_a_all"]
    axle_pos_all_b = axle_pos_mid["axle_pos_b_all"]

    threshold_e_field_currents_all_a = np.empty((len(axle_pos_all_a), len(e_par)))
    threshold_e_field_currents_all_b = np.empty((len(axle_pos_all_b), len(e_par)))
    for n_ax in range(0, len(axle_pos_all_a)):
        print(n_ax)
        axles_a = axle_pos_all_a[n_ax]
        axles_b = axle_pos_all_b[n_ax]

        ia_all, ib_all = rail_model_two_track_e_parallel(section_name=section_name, conditions="moderate", e_parallel=e_par, axle_pos_a=axles_a, axle_pos_b=axles_b)
        threshold_e_field_currents_all_a[n_ax] = ia_all[:, n_ax]
        threshold_e_field_currents_all_b[n_ax] = ib_all[:, n_ax]

    np.savez(f"data\\thresholds\\ws_threshold_currents_{section_name}", threshold_currents_a=threshold_e_field_currents_all_a, threshold_currents_b=threshold_e_field_currents_all_b)


def save_right_side_threshold_e_fields(section_name):
    e_par = np.arange(-40, 40, 0.1)

    threshold_currents = np.load(f"data\\thresholds\\rs_threshold_currents_{section_name}.npz")
    threshold_currents_a = threshold_currents["threshold_currents_a"]
    threshold_currents_b = threshold_currents["threshold_currents_b"]

    plt.plot()

    threshold_e_fields_a = np.empty(len(threshold_currents_a[0, :]))
    threshold_e_fields_b = np.empty(len(threshold_currents_b[0, :]))
    for i in range(0, len(threshold_currents_a[0, :])):
        misoperation_currents_a = threshold_currents_a[:, i][threshold_currents_a[:, i] < 0.055]
        if len(misoperation_currents_a) > 0:
            threshold_e_fields_a[i] = e_par[np.where(threshold_currents_a == np.max(misoperation_currents_a))[0]]
        else:
            threshold_e_fields_a[i] = np.nan
        misoperation_currents_b = threshold_currents_b[:, i][threshold_currents_b[:, i] < 0.055]
        if len(misoperation_currents_b) > 0:
            threshold_e_fields_b[i] = e_par[np.where(threshold_currents_b == np.max(misoperation_currents_b))[0]]
        else:
            threshold_e_fields_b[i] = np.nan

    np.savez(f"data\\thresholds\\rs_threshold_e_fields_{section_name}", threshold_e_fields_a=threshold_e_fields_a, threshold_e_fields_b=threshold_e_fields_b)


def plot_right_side_thresholds(section_name):
    e_par = np.arange(-40, 40, 0.1)
    threshold_e_fields = np.load(f"data\\thresholds\\rs_threshold_e_fields_{section_name}.npz")
    threshold_e_fields_a = threshold_e_fields["threshold_e_fields_a"]
    threshold_e_fields_b = threshold_e_fields["threshold_e_fields_b"]

    threshold_e_fields_a_pos = threshold_e_fields_a[threshold_e_fields_a > 0]
    n_misoperations = np.empty(len(threshold_e_fields_a[threshold_e_fields_a > 0]))
    for i in range(0, len(e_par[e_par > 0])):
        e = e_par[e_par > 0][i]
        print(e)
        n_misoperations[i] = len(np.where(threshold_e_fields_a_pos < e)[0])


    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, figure=fig)
    ax0 = fig.add_subplot(gs[:1, :])
    ax1 = fig.add_subplot(gs[1:, :])

    ax0.plot(n_misoperations, '.')


    plt.show()

    pass

def plot_right_side_thresholds_e_parallel_histogram(section_name):
    e_par = np.arange(-40, 40, 0.1)
    ia_all, ib_all = rail_model_two_track_e_parallel(section_name=section_name, conditions="moderate", e_parallel=e_par, axle_pos_a=np.array([]), axle_pos_b=np.array([]))

    thresholds_a = np.empty(len(ia_all[0, :]))
    thresholds_b = np.empty(len(ib_all[0, :]))
    for i in range(0, len(ia_all[0, :])):
        ia = ia_all[:, i]
        ib = ib_all[:, i]
        threshold = 0.055
        threshold_difference_a = ia - threshold
        threshold_difference_b = ib - threshold
        negative_indices_a = np.where(threshold_difference_a < 0)[0]
        negative_indices_b = np.where(threshold_difference_b < 0)[0]

        if negative_indices_a.size > 0:
            index_of_closest_to_zero_a = negative_indices_a[np.argmax(threshold_difference_a[negative_indices_a])]
            thresholds_a[i] = e_par[index_of_closest_to_zero_a]
        else:
            thresholds_a[i] = np.nan
            print(f"No negative numbers found in array {i}.")

        if negative_indices_b.size > 0:
            index_of_closest_to_zero_b = negative_indices_b[np.argmax(threshold_difference_b[negative_indices_b])]
            thresholds_b[i] = e_par[index_of_closest_to_zero_b]
        else:
            thresholds_b[i] = np.nan
            print(f"No negative numbers found in array {i}.")

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig)
    ax0 = fig.add_subplot(gs[:1, :1])
    ax1 = fig.add_subplot(gs[1:, :1])
    ax2 = fig.add_subplot(gs[:1, 1:])
    ax3 = fig.add_subplot(gs[1:, 1:])

    ax0.plot(thresholds_a, '.', ms=3)
    ax1.plot(thresholds_b, '.', ms=3)

    bins = np.arange(-30, 30.5, 0.1)
    N2, bins2, patches2 = ax2.hist(thresholds_a, bins, histtype="bar", width=0.15, zorder=2)
    N3, bins3, patches3 = ax3.hist(thresholds_b, bins, histtype="bar", width=0.15, zorder=2)
    if section_name == "west_coast_main_line":
        axins1 = inset_axes(ax2, width=2, height=0.8, loc='lower left', bbox_to_anchor=(2, 12.5, 2, 0.8), bbox_transform=ax2.transData)
        N4, bins4, patches4 = axins1.hist(thresholds_a, bins, histtype="bar", width=0.15, zorder=2)
        axins2 = inset_axes(ax3, width=2, height=0.8, loc='lower left', bbox_to_anchor=(-26, 11.5, 2, 0.8), bbox_transform=ax3.transData)
        N5, bins5, patches5 = axins2.hist(thresholds_b, bins, histtype="bar", width=0.15, zorder=2)
    if section_name == "east_coast_main_line":
        axins1 = inset_axes(ax2, width=2, height=0.8, loc='lower left', bbox_to_anchor=(2, 17.5, 2, 0.8), bbox_transform=ax2.transData)
        N4, bins4, patches4 = axins1.hist(thresholds_a, bins, histtype="bar", width=0.15, zorder=2)
        axins2 = inset_axes(ax3, width=2, height=0.8, loc='lower left', bbox_to_anchor=(-26, 20, 2, 0.8), bbox_transform=ax3.transData)
        N5, bins5, patches5 = axins2.hist(thresholds_b, bins, histtype="bar", width=0.15, zorder=2)
    if section_name == "glasgow_edinburgh_falkirk":
        axins1 = inset_axes(ax2, width=2, height=0.8, loc='lower left', bbox_to_anchor=(2, 2.5, 2, 0.8), bbox_transform=ax2.transData)
        N4, bins4, patches4 = axins1.hist(thresholds_a, bins, histtype="bar", width=0.15, zorder=2)
        axins2 = inset_axes(ax3, width=2, height=0.8, loc='lower left', bbox_to_anchor=(-26, 2.5, 2, 0.8), bbox_transform=ax3.transData)
        N5, bins5, patches5 = axins2.hist(thresholds_b, bins, histtype="bar", width=0.15, zorder=2)

    fracs2 = N2 / N2.max()
    norm2 = colors.Normalize(fracs2.min(), fracs2.max())
    for thisfrac, thispatch in zip(fracs2, patches2):
        color2 = plt.cm.viridis(norm2(thisfrac))
        thispatch.set_facecolor(color2)
    fracs3 = N3 / N3.max()
    norm3 = colors.Normalize(fracs3.min(), fracs3.max())
    for thisfrac, thispatch in zip(fracs3, patches3):
        color3 = plt.cm.viridis(norm3(thisfrac))
        thispatch.set_facecolor(color3)
    fracs4 = N4 / N4.max()
    norm4 = colors.Normalize(fracs4.min(), fracs4.max())
    for thisfrac, thispatch in zip(fracs4, patches4):
        color4 = plt.cm.viridis(norm4(thisfrac))
        thispatch.set_facecolor(color4)
    fracs5 = N5 / N5.max()
    norm5 = colors.Normalize(fracs5.min(), fracs5.max())
    for thisfrac, thispatch in zip(fracs5, patches5):
        color5 = plt.cm.viridis(norm5(thisfrac))
        thispatch.set_facecolor(color5)

    ax2.grid(zorder=1)
    ax3.grid(zorder=1)
    ax2.set_xlim(-30, 30)
    ax3.set_xlim(-30, 30)
    axins1.set_xlim(-8, -2)
    axins2.set_xlim(2, 8)
    axins1.set_xticks([-2, -3, -4, -5, -6, -7, -8])
    axins2.set_xticks([2, 3, 4, 5, 6, 7, 8])
    ax2.set_xlabel("Electric Field (V/km)")
    ax3.set_xlabel("Electric Field (V/km)")
    ax2.set_ylabel("Number of Right Side Failures")
    ax3.set_ylabel("Number of Right Side Failures")

    def add_common_elements(ax):
        ax.grid()
        ax.set_xlim(-1, len(thresholds_a))
        ax.set_ylim(np.min(e_par) + 0.1, np.max(e_par) + 0.1)
        ax.set_xlabel("Track Circuit Block")
        ax.set_ylabel("Misoperation Threshold (V/km)")

    add_common_elements(ax0)
    add_common_elements(ax1)

    if section_name == "west_coast_main_line":
        fig.suptitle("West Coast Main Line")
    elif section_name == "east_coast_main_line":
        fig.suptitle("East Coast Main Line")
    elif section_name == "glasgow_edinburgh_falkirk":
        fig.suptitle("Glasgow to Edinburgh via Falkirk High")
    else:
        print("Unrecognised route")

    #plt.savefig(f"{section_name}_thresholds.jpg")
    plt.show()



for sec in ["west_coast_main_line", "east_coast_main_line", "glasgow_edinburgh_falkirk"]:
    #save_right_side_threshold_currents(sec)
    #save_right_side_threshold_e_fields(sec)
    plot_right_side_thresholds(sec)