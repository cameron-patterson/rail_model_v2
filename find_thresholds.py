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


def save_wrong_side_thresholds_currents(section_name, position):
    e_par = np.arange(-40, 40, 0.1)

    axle_pos_mid = np.load(f"data\\axle_positions\\{position}\\{section_name}_axle_positions_two_track_back_axle_{position}.npz", allow_pickle=True)
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

    np.savez(f"data\\thresholds\\ws_threshold_currents_{section_name}_{position}", threshold_currents_a=threshold_e_field_currents_all_a, threshold_currents_b=threshold_e_field_currents_all_b)


def save_right_side_threshold_e_fields(section_name):
    e_par = np.arange(-40, 40, 0.1)

    threshold_currents = np.load(f"data\\thresholds\\rs_threshold_currents_{section_name}.npz")
    threshold_currents_a = threshold_currents["threshold_currents_a"]
    threshold_currents_b = threshold_currents["threshold_currents_b"]

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

    #np.savez(f"data\\thresholds\\rs_threshold_e_fields_{section_name}", threshold_e_fields_a=threshold_e_fields_a, threshold_e_fields_b=threshold_e_fields_b)
    pass


def save_wrong_side_threshold_e_fields(section_name, position):
    e_par = np.arange(-40, 40, 0.1)

    threshold_currents = np.load(f"data\\thresholds\\ws_threshold_currents_{section_name}_{position}.npz")
    threshold_currents_a = threshold_currents["threshold_currents_a"]
    threshold_currents_b = threshold_currents["threshold_currents_b"]

    threshold_e_fields_a = np.empty(len(threshold_currents_a[:, 0]))
    threshold_e_fields_b = np.empty(len(threshold_currents_b[:, 0]))

    for i in range(0, len(threshold_currents_a[:, 0])):

        misoperation_currents_a = threshold_currents_a[i, :][threshold_currents_a[i, :] > 0.081]
        if len(misoperation_currents_a) > 0:
            threshold_e_fields_a[i] = e_par[np.where(threshold_currents_a[i, :] == np.min(misoperation_currents_a))[0]]
        else:
            threshold_e_fields_a[i] = np.nan
        misoperation_currents_b = threshold_currents_b[i, :][threshold_currents_b[i, :] > 0.081]
        if len(misoperation_currents_b) > 0:
            threshold_e_fields_b[i] = e_par[np.where(threshold_currents_b[i, :] == np.min(misoperation_currents_b))[0]]
        else:
            threshold_e_fields_b[i] = np.nan

    np.savez(f"data\\thresholds\\ws_threshold_e_fields_{section_name}_{position}", threshold_e_fields_a=threshold_e_fields_a, threshold_e_fields_b=threshold_e_fields_b)


def plot_right_side_thresholds(section_name):
    e_par = np.arange(-40, 40, 0.1)
    threshold_e_fields = np.load(f"data\\thresholds\\rs_threshold_e_fields_{section_name}.npz")
    threshold_e_fields_a = threshold_e_fields["threshold_e_fields_a"]
    threshold_e_fields_b = threshold_e_fields["threshold_e_fields_b"]

    threshold_e_fields_a_pos = threshold_e_fields_a[threshold_e_fields_a > 0]
    n_misoperations_pos_a = np.empty(len(e_par[e_par > 0]))
    for i in range(0, len(e_par[e_par > 0])):
        e = e_par[e_par > 0][i]
        n_misoperations_pos_a[i] = len(np.where(threshold_e_fields_a_pos < e)[0])

    threshold_e_fields_a_neg = threshold_e_fields_a[threshold_e_fields_a < 0]
    n_misoperations_neg_a = np.empty(len(e_par[e_par < 0]))
    for i in range(0, len(e_par[e_par < 0])):
        e = e_par[e_par < 0][i]
        n_misoperations_neg_a[i] = len(np.where(threshold_e_fields_a_neg > e)[0])

    threshold_e_fields_b_pos = threshold_e_fields_b[threshold_e_fields_b > 0]
    n_misoperations_pos_b = np.empty(len(e_par[e_par > 0]))
    for i in range(0, len(e_par[e_par > 0])):
        e = e_par[e_par > 0][i]
        n_misoperations_pos_b[i] = len(np.where(threshold_e_fields_b_pos < e)[0])

    threshold_e_fields_b_neg = threshold_e_fields_b[threshold_e_fields_b < 0]
    n_misoperations_neg_b = np.empty(len(e_par[e_par < 0]))
    for i in range(0, len(e_par[e_par < 0])):
        e = e_par[e_par < 0][i]
        n_misoperations_neg_b[i] = len(np.where(threshold_e_fields_b_neg > e)[0])

    first_misoperation_pos_a = np.min(e_par[e_par > 0][np.isin(n_misoperations_pos_a, np.min(n_misoperations_pos_a[n_misoperations_pos_a > 0]))])
    first_misoperation_neg_a = np.max(e_par[e_par < 0][np.isin(n_misoperations_neg_a, np.min(n_misoperations_neg_a[n_misoperations_neg_a > 0]))])
    first_misoperation_pos_b = np.min(e_par[e_par > 0][np.isin(n_misoperations_pos_b, np.min(n_misoperations_pos_b[n_misoperations_pos_b > 0]))])
    first_misoperation_neg_b = np.max(e_par[e_par < 0][np.isin(n_misoperations_neg_b, np.min(n_misoperations_neg_b[n_misoperations_neg_b > 0]))])

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, figure=fig)
    ax0 = fig.add_subplot(gs[:1, :])
    ax1 = fig.add_subplot(gs[1:, :])

    ax0.plot(e_par[e_par > 0], n_misoperations_pos_a, '.', color="tomato")
    ax0.plot(e_par[e_par < 0], n_misoperations_neg_a, '.', color="steelblue")
    ax0.axvline(first_misoperation_pos_a, linestyle="--", color="tomato")
    ax0.axvline(first_misoperation_neg_a, linestyle="--", color="steelblue")
    ax0.text(first_misoperation_pos_a + 0.1, len(threshold_e_fields_a_pos)*0.75, str(np.round(first_misoperation_pos_a, 1)))
    ax0.text(first_misoperation_neg_a + 0.1, len(threshold_e_fields_a_pos)*0.75, str(np.round(first_misoperation_neg_a, 1)))

    ax1.plot(e_par[e_par > 0], n_misoperations_pos_b, '.', color="tomato")
    ax1.plot(e_par[e_par < 0], n_misoperations_neg_b, '.', color="steelblue")
    ax1.axvline(first_misoperation_pos_b, linestyle="--", color="tomato")
    ax1.axvline(first_misoperation_neg_b, linestyle="--", color="steelblue")
    ax1.text(first_misoperation_pos_b + 0.1, len(threshold_e_fields_b_neg)*0.75, str(np.round(first_misoperation_pos_b, 1)))
    ax1.text(first_misoperation_neg_b + 0.1, len(threshold_e_fields_b_neg)*0.75, str(np.round(first_misoperation_neg_b, 1)))

    def common_features(ax):
        ax.set_xlim(np.min(e_par), np.max(e_par))
        ax.grid()
        ax.set_xlabel("Electric Field Strength (V/km)")
        ax.set_ylabel("Number of Right Side Failures")
        if section_name == "glasgow_edinburgh_falkirk":
            fig.suptitle("Glasgow to Edinburgh via Falkirk High")
        elif section_name == "west_coast_main_line":
            fig.suptitle("West Coast Main Line")
        elif section_name == "east_coast_main_line":
            fig.suptitle("East Coast Main Line")
        else:
            print("Route not configured")

    common_features(ax0)
    common_features(ax1)

    plt.savefig(f"{section_name}_rs_thresholds.jpg")

    pass


def plot_wrong_side_thresholds(section_name, position):
    e_par = np.arange(-40, 40, 0.1)
    threshold_e_fields = np.load(f"data\\thresholds\\ws_threshold_e_fields_{section_name}_{position}.npz")
    threshold_e_fields_a = threshold_e_fields["threshold_e_fields_a"]
    threshold_e_fields_b = threshold_e_fields["threshold_e_fields_b"]

    threshold_e_fields_a_pos = threshold_e_fields_a[threshold_e_fields_a > 0]
    n_misoperations_pos_a = np.empty(len(e_par[e_par > 0]))
    for i in range(0, len(e_par[e_par > 0])):
        e = e_par[e_par > 0][i]
        n_misoperations_pos_a[i] = len(np.where(threshold_e_fields_a_pos < e)[0])

    threshold_e_fields_a_neg = threshold_e_fields_a[threshold_e_fields_a < 0]
    n_misoperations_neg_a = np.empty(len(e_par[e_par < 0]))
    for i in range(0, len(e_par[e_par < 0])):
        e = e_par[e_par < 0][i]
        n_misoperations_neg_a[i] = len(np.where(threshold_e_fields_a_neg > e)[0])

    threshold_e_fields_b_pos = threshold_e_fields_b[threshold_e_fields_b > 0]
    n_misoperations_pos_b = np.empty(len(e_par[e_par > 0]))
    for i in range(0, len(e_par[e_par > 0])):
        e = e_par[e_par > 0][i]
        n_misoperations_pos_b[i] = len(np.where(threshold_e_fields_b_pos < e)[0])

    threshold_e_fields_b_neg = threshold_e_fields_b[threshold_e_fields_b < 0]
    n_misoperations_neg_b = np.empty(len(e_par[e_par < 0]))
    for i in range(0, len(e_par[e_par < 0])):
        e = e_par[e_par < 0][i]
        n_misoperations_neg_b[i] = len(np.where(threshold_e_fields_b_neg > e)[0])

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1, figure=fig)
    ax0 = fig.add_subplot(gs[:1, :])
    ax1 = fig.add_subplot(gs[1:, :])

    ax0.plot(e_par[e_par > 0], n_misoperations_pos_a, '.', color="tomato")
    ax0.plot(e_par[e_par < 0], n_misoperations_neg_a, '.', color="steelblue")

    ax1.plot(e_par[e_par > 0], n_misoperations_pos_b, '.', color="tomato")
    ax1.plot(e_par[e_par < 0], n_misoperations_neg_b, '.', color="steelblue")

    if np.max(n_misoperations_pos_a) != 0:
        first_misoperation_pos_a = np.min(e_par[e_par > 0][np.isin(n_misoperations_pos_a, np.min(n_misoperations_pos_a[n_misoperations_pos_a > 0]))])
        ax0.axvline(first_misoperation_pos_a, linestyle="--", color="tomato")
        ax0.text(first_misoperation_pos_a + 0.1, len(threshold_e_fields_a_neg) * 0.75, str(np.round(first_misoperation_pos_a, 1)))
    else:
        pass
    if np.max(n_misoperations_neg_a) != 0:
        first_misoperation_neg_a = np.max(e_par[e_par < 0][np.isin(n_misoperations_neg_a, np.min(n_misoperations_neg_a[n_misoperations_neg_a > 0]))])
        ax0.axvline(first_misoperation_neg_a, linestyle="--", color="steelblue")
        ax0.text(first_misoperation_neg_a + 0.1, len(threshold_e_fields_a_neg) * 0.75, str(np.round(first_misoperation_neg_a, 1)))
    else:
        pass
    if np.max(n_misoperations_pos_b) != 0:
        first_misoperation_pos_b = np.min(e_par[e_par > 0][np.isin(n_misoperations_pos_b, np.min(n_misoperations_pos_b[n_misoperations_pos_b > 0]))])
        ax1.axvline(first_misoperation_pos_b, linestyle="--", color="tomato")
        ax1.text(first_misoperation_pos_b + 0.1, len(threshold_e_fields_b_pos) * 0.75, str(np.round(first_misoperation_pos_b, 1)))
    else:
        pass
    if np.max(n_misoperations_neg_b) != 0:
        first_misoperation_neg_b = np.max(e_par[e_par < 0][np.isin(n_misoperations_neg_b, np.min(n_misoperations_neg_b[n_misoperations_neg_b > 0]))])
        ax1.axvline(first_misoperation_neg_b, linestyle="--", color="steelblue")
        ax1.text(first_misoperation_neg_b + 0.1, len(threshold_e_fields_b_pos) * 0.75, str(np.round(first_misoperation_neg_b, 1)))
    else:
        pass

    def common_features(ax):
        ax.set_xlim(np.min(e_par), np.max(e_par))
        ax.grid()
        ax.set_xlabel("Electric Field Strength (V/km)")
        ax.set_ylabel("Number of Wrong Side Failures")
        if position == "at_end":
            if section_name == "glasgow_edinburgh_falkirk":
                fig.suptitle("Glasgow to Edinburgh via Falkirk High: Trains at Block End")
            elif section_name == "west_coast_main_line":
                fig.suptitle("West Coast Main Line: Trains at Block End")
            elif section_name == "east_coast_main_line":
                fig.suptitle("East Coast Main Line: Trains at Block End")
            else:
                print("Route not configured")
        elif position == "block_centre":
            if section_name == "glasgow_edinburgh_falkirk":
                fig.suptitle("Glasgow to Edinburgh via Falkirk High: Trains at Block Centre")
            elif section_name == "west_coast_main_line":
                fig.suptitle("West Coast Main Line: Trains at Block Centre")
            elif section_name == "east_coast_main_line":
                fig.suptitle("East Coast Main Line: Trains at Block Centre")
            else:
                print("Route not configured")
        else:
            print("Axle position not configured")

    common_features(ax0)
    common_features(ax1)

    plt.savefig(f"ws_thresholds_{section_name}_{position}.jpg")

    pass


for sec in ["glasgow_edinburgh_falkirk", "east_coast_main_line", "west_coast_main_line"]:
    #save_wrong_side_thresholds_currents(sec, "at_end")
    #save_wrong_side_threshold_e_fields(sec, "at_end")
    plot_wrong_side_thresholds(sec, "block_centre")
