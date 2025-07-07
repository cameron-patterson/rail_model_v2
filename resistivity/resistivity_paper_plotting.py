from misoperation_analysis import tc_currents_two_track_e_parallel, tc_currents_two_track_e_parallel_uni_leak, tc_currents_two_track_e_parallel_voltages, tc_currents_two_track_e_parallel_uni_leak_voltages
from geopy.distance import distance
from geopy import Point as Point
from shapely.geometry import Point as shapelyPoint
import matplotlib.pyplot as plt
import numpy as np
import geopandas as gpd
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from geographiclib.geodesic import Geodesic
import math
geod = Geodesic.WGS84  # Define the WGS84 ellipsoid


def plot_resistivity_along_line(route_name):
    resistivities = np.load(f"../data/resistivity/{route_name}_mast_resistivities.npz")
    resistivity_a20 = resistivities['a20']
    resistivity_a50 = resistivities['a50']
    resistivity_a80 = resistivities['a80']
    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[:, :])
    plt.scatter(range(0, len(resistivity_a50)), resistivity_a50, s=1, zorder=3, color='steelblue')
    for i in range(0, len(resistivity_a50)):
        plt.plot([i, i], [resistivity_a50[i], resistivity_a20[i]], color='lightsteelblue', linewidth=.4, zorder=1, alpha=.5)
        plt.plot([i, i], [resistivity_a50[i], resistivity_a80[i]], color='lightsteelblue', linewidth=.4, zorder=1, alpha=.5)
    ax0.axhline(16, color='cornflowerblue', zorder=2, linestyle='--')
    ax0.axhline(32, color='cyan', zorder=2, linestyle='--')
    ax0.axhline(64, color='greenyellow', zorder=2, linestyle='--')
    ax0.axhline(125, color='gold', zorder=2, linestyle='--')
    ax0.axhline(250, color='orange', zorder=2, linestyle='--')
    ax0.axhline(500, color='red', zorder=2, linestyle='--')
    ax0.set_xlabel("Mast Number")
    ax0.set_ylabel("Resistivity (\u03A9\u22C5m)")
    ax0.set_xlim(-2, len(resistivity_a50)+1)
    if route_name == "glasgow_edinburgh_falkirk":
        ax0.set_title("Glasgow to Edinburgh via Falkirk High")
    elif route_name == "east_coast_main_line":
        ax0.set_title("East Coast Main Line")
    elif route_name == "west_coast_main_line":
        ax0.set_title("West Coast Main Line")
    else:
        ax0.set_title("{route_name}")

    legend_elements = [Line2D([0], [0], label='16', color='cornflowerblue'),
                       Line2D([0], [0], label='32', color='cyan'),
                       Line2D([0], [0], label='64', color='greenyellow'),
                       Line2D([0], [0], label='125', color='gold'),
                       Line2D([0], [0], label='250', color='orange'),
                       Line2D([0], [0], label='500', color='red')]
    ax0.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.05), fancybox=True, shadow=True, ncol=6)

    plt.yscale("log")
    #plt.show()
    plt.savefig(f"mast_resistivities_{route_name}.pdf")


def plot_leaks(route_name):
    block_leak = np.load(f'../data/resistivity/{route_name}_block_leakage.npz')
    bl_a20 = block_leak['a20']
    bl_a50 = block_leak['a50']
    bl_a80 = block_leak['a80']

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(1, 1)
    ax0 = fig.add_subplot(gs[:, :])
    ax0.set_xlabel('Block Number')
    ax0.set_ylabel('Leakage (S km\u207B\u00B9)')
    ax0.scatter(range(0, len(bl_a50)), bl_a50, s=1, zorder=3, color='steelblue')
    for i in range(0, len(bl_a50)):
        ax0.plot([i, i], [bl_a50[i], bl_a20[i]], color='lightsteelblue', linewidth=1, zorder=1, alpha=.5)
        ax0.plot([i, i], [bl_a50[i], bl_a80[i]], color='lightsteelblue', linewidth=1, zorder=1, alpha=.5)

    if route_name == "glasgow_edinburgh_falkirk":
        ax0.set_title("Glasgow to Edinburgh via Falkirk High")
    elif route_name == "east_coast_main_line":
        ax0.set_title("East Coast Main Line")
    elif route_name == "west_coast_main_line":
        ax0.set_title("West Coast Main Line")
    else:
        ax0.set_title("{route_name}")

    plt.savefig(f'block_leakages_{route_name}.pdf')
    #plt.show()


def plot_currents_rs(route_name):
    e_par = np.array([0, 5])
    axle_pos_a = []
    axle_pos_b = []
    ia_a50, ib_a50 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=True, cross_bonds=True)
    ia_uni, ib_uni = tc_currents_two_track_e_parallel_uni_leak(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=True, cross_bonds=True)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.plot(ia_a50[0, :], '.')
    ax0.plot(ia_uni[0, :], 'x')

    ax1.plot(ia_a50[1, :], '.')
    ax1.plot(ia_uni[1, :], 'x')

    stag_dif = ia_a50[0, :] - ia_uni[0, :]
    ax2.plot(range(0, len(stag_dif)), stag_dif, '.')
    stag_dif_e = ia_a50[1, :] - ia_uni[1, :]
    ax3.plot(range(0, len(stag_dif_e)), stag_dif_e, '.')

    ax0.set_ylabel('Current Through Relay (A)')
    ax1.set_ylabel('Current Through Relay (A)')
    ax2.set_ylabel('Current Difference (A)')
    ax3.set_ylabel('Current Difference (A)')

    ax0.set_title('E = 0 V km\u207B\u00B9')
    ax1.set_title('E = 5 V km\u207B\u00B9')
    ax2.set_title('E = 0 V km\u207B\u00B9')
    ax3.set_title('E = 5 V km\u207B\u00B9')

    def ax_labels(ax):
        ax.set_xlabel('Block Number')

    ax_labels(ax0)
    ax_labels(ax1)
    ax_labels(ax2)
    ax_labels(ax3)

    ax2.axhline(0, linestyle='--', color='grey', alpha=.5)
    ax3.axhline(0, linestyle='--', color='grey', alpha=.5)

    if route_name == 'west_coast_main_line':
        ax2.set_ylim(-.006, .006)
        ax3.set_ylim(-.05, .2)

    plt.subplots_adjust(wspace=0.225, hspace=0.35)
    fig.align_ylabels()
    #plt.show()
    plt.savefig(f'currents_rs_{route_name}.pdf')


def plot_currents_rs_no_stag(route_name):
    e_par = np.array([0, 5])
    axle_pos_a = []
    axle_pos_b = []
    ia_a50, ib_a50 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=False, cross_bonds=True)
    ia_uni, ib_uni = tc_currents_two_track_e_parallel_uni_leak(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=False, cross_bonds=True)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(16, 8))
    gs = GridSpec(2, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.plot(ia_a50[0, :], '.')
    ax0.plot(ia_uni[0, :], 'x')

    ax1.plot(ia_a50[1, :], '.')
    ax1.plot(ia_uni[1, :], 'x')

    stag_dif = ia_a50[0, :] - ia_uni[0, :]
    ax2.plot(range(0, len(stag_dif)), stag_dif, '.')
    stag_dif_e = ia_a50[1, :] - ia_uni[1, :]
    ax3.plot(range(0, len(stag_dif_e)), stag_dif_e, '.')

    ax0.set_ylabel('Current Through Relay (A)')
    ax1.set_ylabel('Current Through Relay (A)')
    ax2.set_ylabel('Current Difference (A)')
    ax3.set_ylabel('Current Difference (A)')

    ax0.set_title('E = 0 V km\u207B\u00B9')
    ax1.set_title('E = 5 V km\u207B\u00B9')
    ax2.set_title('E = 0 V km\u207B\u00B9')
    ax3.set_title('E = 5 V km\u207B\u00B9')

    def ax_labels(ax):
        ax.set_xlabel('Block Number')

    ax_labels(ax0)
    ax_labels(ax1)
    ax_labels(ax2)
    ax_labels(ax3)

    ax2.axhline(0, linestyle='--', color='grey', alpha=.5)
    ax3.axhline(0, linestyle='--', color='grey', alpha=.5)

    if route_name == 'west_coast_main_line':
        ax2.set_ylim(-.006, .006)
        ax3.set_ylim(-.05, .2)

    plt.subplots_adjust(wspace=0.225, hspace=0.35)
    fig.align_ylabels()
    #plt.show()
    plt.savefig(f'currents_rs_no_stag_{route_name}.pdf')


def plot_traction_rail_voltages(route_name):
    if route_name == 'west_coast_main_line':
        plt.rcParams['font.size'] = '15'
        fig = plt.figure(figsize=(18, 12))
        gs = GridSpec(3, 4)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1:3])
        ax2 = fig.add_subplot(gs[0, 3])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1:3])
        ax5 = fig.add_subplot(gs[1, 3])

        e_par = np.array([0, 5])
        axle_pos_a = []
        axle_pos_b = []
        ia_a50, ib_a50, v = tc_currents_two_track_e_parallel_voltages(section_name=route_name, leakage='a50', e_parallel=e_par,
                                                          axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A',
                                                          electrical_staggering=True, cross_bonds=True)
        ia_uni_no_stag, ib_uni_no_stag, v_no_stag = tc_currents_two_track_e_parallel_voltages(section_name=route_name, leakage='a50', e_parallel=e_par,
                                                                   axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                                                   relay_type='BR939A', electrical_staggering=False,
                                                                   cross_bonds=True)
        ia_a50_unileak, ib_a50_unileak, v_unileak = tc_currents_two_track_e_parallel_uni_leak_voltages(section_name=route_name, leakage='a50',
                                                                      e_parallel=e_par,
                                                                      axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                                                      relay_type='BR939A',
                                                                      electrical_staggering=True, cross_bonds=True)
        ia_uni_no_stag_unileak, ib_uni_no_stag_unileak, v_no_stag_unileak = tc_currents_two_track_e_parallel_uni_leak_voltages(section_name=route_name,
                                                                                              leakage='a50',
                                                                                              e_parallel=e_par,
                                                                                              axle_pos_a=axle_pos_a,
                                                                                              axle_pos_b=axle_pos_b,
                                                                                              relay_type='BR939A',
                                                                                              electrical_staggering=False,
                                                                                              cross_bonds=True)
        ax0.plot(range(0, 250), v_unileak[0:250, 0], '.', color='red')
        ax0.plot(range(0, 250), v_no_stag_unileak[0:250, 0], 'x', color='tomato')
        ax0.plot(range(0, 250), v[0:250, 0], '.', color='blue')
        ax0.plot(range(0, 250), v_no_stag[0:250, 0], 'x', color='cornflowerblue')
        ax0.set_xlim(0, 250)
        ax1.plot(range(250, 2250), v_unileak[250:2250, 0], '.', color='red')
        ax1.plot(range(250, 2250), v_no_stag_unileak[250:2250, 0], 'x', color='tomato')
        ax1.plot(range(250, 2250), v[250:2250, 0], '.', color='blue')
        ax1.plot(range(250, 2250), v_no_stag[250:2250, 0], 'x', color='cornflowerblue')
        ax1.set_xlim(250, 2250)
        ax2.plot(range(2250, 2538), v_unileak[2250:2538, 0], '.', color='red')
        ax2.plot(range(2250, 2538), v_no_stag_unileak[2250:2538, 0], 'x', color='tomato')
        ax2.plot(range(2250, 2538), v[2250:2538, 0], '.', color='blue')
        ax2.plot(range(2250, 2538), v_no_stag[2250:2538, 0], 'x', color='cornflowerblue')
        ax2.set_xlim(2250, 2538)
        ax3.plot(range(0, 250), v_unileak[0:250, 1], '.', color='red')
        ax3.plot(range(0, 250), v_no_stag_unileak[0:250, 1], 'x', color='tomato')
        ax3.plot(range(0, 250), v[0:250, 1], '.', color='blue')
        ax3.plot(range(0, 250), v_no_stag[0:250, 1], 'x', color='cornflowerblue')
        ax3.set_xlim(0, 250)
        ax4.plot(range(250, 2250), v_unileak[250:2250, 1], '.', color='red')
        ax4.plot(range(250, 2250), v_no_stag_unileak[250:2250, 1], 'x', color='tomato')
        ax4.plot(range(250, 2250), v[250:2250, 1], '.', color='blue')
        ax4.plot(range(250, 2250), v_no_stag[250:2250, 1], 'x', color='cornflowerblue')
        ax4.set_xlim(250, 2250)
        ax5.plot(range(2250, 2538), v_unileak[2250:2538, 1], '.', color='red')
        ax5.plot(range(2250, 2538), v_no_stag_unileak[2250:2538, 1], 'x', color='tomato')
        ax5.plot(range(2250, 2538), v[2250:2538, 1], '.', color='blue')
        ax5.plot(range(2250, 2538), v_no_stag[2250:2538, 1], 'x', color='cornflowerblue')
        ax5.set_xlim(2250, 2538)

        legend_elements = [Line2D([0], [0], marker='.', linestyle='None', label='Constant leakage; Staggered', color='red'),
                           Line2D([0], [0], marker='x', linestyle='None', label='Constant leakage; Not staggered', color='tomato'),
                           Line2D([0], [0], marker='.', linestyle='None', label='Realistic leakage; Staggered', color='blue'),
                           Line2D([0], [0], marker='x', linestyle='None', label='Realistic leakage; Not staggered', color='cornflowerblue')]
        ax4.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=2)

        ax1.set_xlabel('Traction Rail Node')
        ax0.set_ylabel('Voltage (V)')
        ax4.set_xlabel('Traction Rail Node')
        ax3.set_ylabel('Voltage (V)')

        ax1.set_title('E = 0 V km\u207B\u00B9')
        ax4.set_title('E = 5 V km\u207B\u00B9')

        plt.subplots_adjust(hspace=0.42)
        #plt.show()
        plt.savefig(f'traction_rail_voltages_{route_name}.pdf')
    else:
        pass


def plot_signal_rail_voltages(route_name):
    if route_name == 'west_coast_main_line':
        plt.rcParams['font.size'] = '15'
        fig = plt.figure(figsize=(16, 10))
        gs = GridSpec(3, 4)
        ax0 = fig.add_subplot(gs[0, 0])
        ax1 = fig.add_subplot(gs[0, 1:3])
        ax2 = fig.add_subplot(gs[0, 3])
        ax3 = fig.add_subplot(gs[1, 0])
        ax4 = fig.add_subplot(gs[1, 1:3])
        ax5 = fig.add_subplot(gs[1, 3])
        e_par = np.array([0, 5])
        axle_pos_a = []
        axle_pos_b = []
        ia_a50, ib_a50, v = tc_currents_two_track_e_parallel_voltages(section_name=route_name, leakage='a50', e_parallel=e_par,
                                                          axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A',
                                                          electrical_staggering=True, cross_bonds=True)
        ia_uni_no_stag, ib_uni_no_stag, v_no_stag = tc_currents_two_track_e_parallel_voltages(section_name=route_name, leakage='a50', e_parallel=e_par,
                                                                   axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                                                   relay_type='BR939A', electrical_staggering=False,
                                                                   cross_bonds=True)
        ia_a50_unileak, ib_a50_unileak, v_unileak = tc_currents_two_track_e_parallel_uni_leak_voltages(section_name=route_name, leakage='a50',
                                                                      e_parallel=e_par,
                                                                      axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                                                      relay_type='BR939A',
                                                                      electrical_staggering=True, cross_bonds=True)
        ia_uni_no_stag_unileak, ib_uni_no_stag_unileak, v_no_stag_unileak = tc_currents_two_track_e_parallel_uni_leak_voltages(section_name=route_name,
                                                                                              leakage='a50',
                                                                                              e_parallel=e_par,
                                                                                              axle_pos_a=axle_pos_a,
                                                                                              axle_pos_b=axle_pos_b,
                                                                                              relay_type='BR939A',
                                                                                              electrical_staggering=False,
                                                                                              cross_bonds=True)
        j = 0
        for i in range(2538, 2538+50, 2):
            ax0.plot([j, j+1], [v_unileak[i, 0], v_unileak[i+1, 0]], color='red')
            ax0.plot([j, j + 1], [v_no_stag_unileak[i, 0], v_no_stag_unileak[i + 1, 0]], color='tomato', linestyle='--')
            ax0.plot([j, j + 1], [v[i, 0], v[i + 1, 0]], color='blue')
            ax0.plot([j, j + 1], [v_no_stag[i, 0], v_no_stag[i + 1, 0]], color='cornflowerblue', linestyle='--')
            ax3.plot([j, j + 1], [v_unileak[i, 1], v_unileak[i + 1, 1]], color='red')
            ax3.plot([j, j + 1], [v_no_stag_unileak[i, 1], v_no_stag_unileak[i + 1, 1]], color='tomato', linestyle='--')
            ax3.plot([j, j + 1], [v[i, 1], v[i + 1, 1]], color='blue')
            ax3.plot([j, j + 1], [v_no_stag[i, 1], v_no_stag[i + 1, 1]], color='cornflowerblue', linestyle='--')
            j += 2
        ax0.set_xlim(0, 50)
        ax3.set_xlim(0, 50)

        j = 950
        for i in range(2538+950, 2538+1050, 2):
            ax1.plot([j, j + 1], [v_unileak[i, 0], v_unileak[i + 1, 0]], color='red')
            ax1.plot([j, j + 1], [v_no_stag_unileak[i, 0], v_no_stag_unileak[i + 1, 0]], color='tomato', linestyle='--')
            ax1.plot([j, j + 1], [v[i, 0], v[i + 1, 0]], color='blue')
            ax1.plot([j, j + 1], [v_no_stag[i, 0], v_no_stag[i + 1, 0]], color='cornflowerblue', linestyle='--')
            ax4.plot([j, j + 1], [v_unileak[i, 1], v_unileak[i + 1, 1]], color='red')
            ax4.plot([j, j + 1], [v_no_stag_unileak[i, 1], v_no_stag_unileak[i + 1, 1]], color='tomato', linestyle='--')
            ax4.plot([j, j + 1], [v[i, 1], v[i + 1, 1]], color='blue')
            ax4.plot([j, j + 1], [v_no_stag[i, 1], v_no_stag[i + 1, 1]], color='cornflowerblue', linestyle='--')
            j += 2
        ax1.set_xlim(950, 1050)
        ax4.set_xlim(950, 1050)

        j = 1820
        for i in range(2538+1820, 4408+1870, 2):
            ax2.plot([j, j + 1], [v_unileak[i, 0], v_unileak[i + 1, 0]], color='red')
            ax2.plot([j, j + 1], [v_no_stag_unileak[i, 0], v_no_stag_unileak[i + 1, 0]], color='tomato', linestyle='--')
            ax2.plot([j, j + 1], [v[i, 0], v[i + 1, 0]], color='blue')
            ax2.plot([j, j + 1], [v_no_stag[i, 0], v_no_stag[i + 1, 0]], color='cornflowerblue', linestyle='--')
            ax5.plot([j, j + 1], [v_unileak[i, 1], v_unileak[i + 1, 1]], color='red')
            ax5.plot([j, j + 1], [v_no_stag_unileak[i, 1], v_no_stag_unileak[i + 1, 1]], color='tomato', linestyle='--')
            ax5.plot([j, j + 1], [v[i, 1], v[i + 1, 1]], color='blue')
            ax5.plot([j, j + 1], [v_no_stag[i, 1], v_no_stag[i + 1, 1]], color='cornflowerblue', linestyle='--')
            j += 2
        ax2.set_xlim(1820, 1870)
        ax5.set_xlim(1820, 1870)

        legend_elements = [Line2D([0], [0], linestyle='-', label='Constant leakage; Staggered', color='red'),
                           Line2D([0], [0], linestyle='--', label='Constant leakage; Not staggered', color='tomato'),
                           Line2D([0], [0], linestyle='-', label='Realistic leakage; Staggered', color='blue'),
                           Line2D([0], [0], linestyle='--', label='Realistic leakage; Not staggered', color='cornflowerblue')]
        ax4.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=2)

        ax1.set_xlabel('Signal Rail Node')
        ax0.set_ylabel('Voltage (V)')
        ax4.set_xlabel('Signal Rail Node')
        ax3.set_ylabel('Voltage (V)')

        ax1.set_title('E = 0 V km\u207B\u00B9')
        ax4.set_title('E = 5 V km\u207B\u00B9')

        plt.subplots_adjust(hspace=0.42)
        #plt.show()
        plt.savefig(f'signal_rail_voltages_{route_name}.pdf')
    else:
        pass


def plot_currents_ws(route_name):
    e_par = np.array([0, 5])

    axles = np.load(f'../data/axle_positions/block_centre/axle_positions_two_track_back_axle_at_centre_{route_name}.npz', allow_pickle=True)
    axles_a = axles['axle_pos_a_all']
    axle_pos_a = []
    axle_pos_b = []
    for i in range(0, len(axles_a)):
        for axa in axles_a[i]:
            axle_pos_a.append(axa)

    ia_a50, ib_a50 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=True, cross_bonds=True)
    ia_uni, ib_uni = tc_currents_two_track_e_parallel_uni_leak(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=True, cross_bonds=True)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(3, 2)
    ax0 = fig.add_subplot(gs[0, 0])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])

    ax0.plot(ia_a50[0, :], '.', color='cornflowerblue')
    ax0.plot(ia_uni[0, :], '.', color='tomato')
    ax1.plot(ia_a50[1, :], '.', color='cornflowerblue')
    ax1.plot(ia_uni[1, :], '.', color='tomato')

    uni_dif_0 = ia_a50[0, :] - ia_uni[0, :]
    uni_dif_5 = ia_a50[1, :] - ia_uni[1, :]
    ax2.plot(uni_dif_0, '.', color='violet')
    ax3.plot(uni_dif_5, '.', color='violet')

    ax0.set_ylabel('Current Through Relay (A)')
    ax1.set_ylabel('Current Through Relay (A)')
    ax2.set_ylabel('Current Difference (A)')
    ax3.set_ylabel('Current Difference (A)')
    ax1.set_xlabel('Block Number')
    ax3.set_xlabel('Block Number')

    ax0.set_ylim(-0.1, 0.1)
    ax2.set_ylim(-0.001, 0.001)
    ax1.set_ylim(-0.2, 0.05)
    ax3.set_ylim(-0.03, 0.005)

    def ax_labels(ax):
        ax.set_xlim(0, len(ia_a50[0, :]))

    ax_labels(ax0)
    ax_labels(ax1)
    ax_labels(ax2)
    ax_labels(ax3)

    legend_elements = [Line2D([0], [0], marker='.', linestyle='None', label='Constant leakage', color='tomato'),
                       Line2D([0], [0], marker='.', linestyle='None', label='Realistic leakage', color='cornflowerblue')]
    ax1.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.25), fancybox=True, shadow=True, ncol=2)

    plt.subplots_adjust(wspace=0.3)

    fig.align_ylabels()
    plt.show()


def plot_thresholds_rs(route_name):
    e_par = np.arange(-100, 100.1, 0.1)
    axle_pos_a = []
    axle_pos_b = []
    ia_a50, ib_a50 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a50',
                                              e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                              relay_type='BR939A',
                                              electrical_staggering=True, cross_bonds=True)
    ia_a20, ib_a20 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a20',
                                                      e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                                      relay_type='BR939A',
                                                      electrical_staggering=True, cross_bonds=True)
    ia_a80, ib_a80 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a80',
                                                      e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                                      relay_type='BR939A',
                                                      electrical_staggering=True, cross_bonds=True)
    ia_unileak, ib_unileak = tc_currents_two_track_e_parallel_uni_leak(section_name=route_name, leakage='a80',
                                                                               e_parallel=e_par,
                                                                               axle_pos_a=axle_pos_a,
                                                                               axle_pos_b=axle_pos_b,
                                                                               relay_type='BR939A',
                                                                               electrical_staggering=True,
                                                                               cross_bonds=True)

    # a first
    e_thresh_ia_a20 = []
    for i in range(0, len(ia_a20[0, :])):
        e_mis_ia_a20_block = e_par[np.where((ia_a20[:, i] < 0.055) & (ia_a20[:, i] > -0.055))[0]]
        if len(e_mis_ia_a20_block) > 0:
            e_thresh_ia_a20.append(e_mis_ia_a20_block[np.argmin(np.abs(e_mis_ia_a20_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ia_a20.append(np.nan)

    e_thresh_ia_a50 = []
    for i in range(0, len(ia_a50[0, :])):
        e_mis_ia_a50_block = e_par[np.where((ia_a50[:, i] < 0.055) & (ia_a50[:, i] > -0.055))[0]]
        if len(e_mis_ia_a50_block) > 0:
            e_thresh_ia_a50.append(e_mis_ia_a50_block[np.argmin(np.abs(e_mis_ia_a50_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ia_a50.append(np.nan)

    e_thresh_ia_a80 = []
    for i in range(0, len(ia_a80[0, :])):
        e_mis_ia_a80_block = e_par[np.where((ia_a80[:, i] < 0.055) & (ia_a80[:, i] > -0.055))[0]]
        if len(e_mis_ia_a80_block) > 0:
            e_thresh_ia_a80.append(e_mis_ia_a80_block[np.argmin(np.abs(e_mis_ia_a80_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ia_a80.append(np.nan)

    e_thresh_ia_unileak = []
    for i in range(0, len(ia_unileak[0, :])):
        e_mis_ia_unileak_block = e_par[np.where((ia_unileak[:, i] < 0.055) & (ia_unileak[:, i] > -0.055))[0]]
        if len(e_mis_ia_unileak_block) > 0:
            e_thresh_ia_unileak.append(e_mis_ia_unileak_block[np.argmin(np.abs(e_mis_ia_unileak_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ia_unileak.append(np.nan)

    # b second
    e_thresh_ib_a20 = []
    for i in range(0, len(ib_a20[0, :])):
        e_mis_ib_a20_block = e_par[np.where((ib_a20[:, i] < 0.055) & (ib_a20[:, i] > -0.055))[0]]
        if len(e_mis_ib_a20_block) > 0:
            e_thresh_ib_a20.append(e_mis_ib_a20_block[np.argmin(np.abs(e_mis_ib_a20_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ib_a20.append(np.nan)

    e_thresh_ib_a50 = []
    for i in range(0, len(ib_a50[0, :])):
        e_mis_ib_a50_block = e_par[np.where((ib_a50[:, i] < 0.055) & (ib_a50[:, i] > -0.055))[0]]
        if len(e_mis_ib_a50_block) > 0:
            e_thresh_ib_a50.append(e_mis_ib_a50_block[np.argmin(np.abs(e_mis_ib_a50_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ib_a50.append(np.nan)

    e_thresh_ib_a80 = []
    for i in range(0, len(ib_a80[0, :])):
        e_mis_ib_a80_block = e_par[np.where((ib_a80[:, i] < 0.055) & (ib_a80[:, i] > -0.055))[0]]
        if len(e_mis_ib_a80_block) > 0:
            e_thresh_ib_a80.append(e_mis_ib_a80_block[np.argmin(np.abs(e_mis_ib_a80_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ib_a80.append(np.nan)

    e_thresh_ib_unileak = []
    for i in range(0, len(ib_unileak[0, :])):
        e_mis_ib_unileak_block = e_par[np.where((ib_unileak[:, i] < 0.055) & (ib_unileak[:, i] > -0.055))[0]]
        if len(e_mis_ib_unileak_block) > 0:
            e_thresh_ib_unileak.append(e_mis_ib_unileak_block[np.argmin(np.abs(e_mis_ib_unileak_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ib_unileak.append(np.nan)

    #np.savez(f'e_thresh_rs_{route_name}.npz', e_thresh_a_a20=e_thresh_ia_a20, e_thresh_a_a50=e_thresh_ia_a50,
    #         e_thresh_a_a80=e_thresh_ia_a80, e_thresh_a_unileak=e_thresh_ia_unileak, e_thresh_b_a20=e_thresh_ib_a20,
    #         e_thresh_b_a50=e_thresh_ib_a50, e_thresh_b_a80=e_thresh_ib_a80, e_thresh_b_unileak=e_thresh_ib_unileak)

    plt.plot(e_thresh_ia_a20, '.')
    plt.plot(e_thresh_ia_a50, '.')
    plt.plot(e_thresh_ia_a80, '.')
    plt.plot(e_thresh_ia_unileak, 'x')
    plt.xticks(range(0, len(e_thresh_ia_a20)), minor=True)
    plt.grid(axis='x', which='minor', alpha=0.2)
    plt.show()


def plot_thresholds_rs_no_stag(route_name):
    e_par = np.arange(-100, 100.1, 0.1)
    axle_pos_a = []
    axle_pos_b = []
    ia_a50, ib_a50 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a50',
                                              e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                              relay_type='BR939A',
                                              electrical_staggering=False, cross_bonds=True)
    ia_a20, ib_a20 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a20',
                                                      e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                                      relay_type='BR939A',
                                                      electrical_staggering=False, cross_bonds=True)
    ia_a80, ib_a80 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a80',
                                                      e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b,
                                                      relay_type='BR939A',
                                                      electrical_staggering=False, cross_bonds=True)
    ia_unileak, ib_unileak = tc_currents_two_track_e_parallel_uni_leak(section_name=route_name, leakage='a80',
                                                                               e_parallel=e_par,
                                                                               axle_pos_a=axle_pos_a,
                                                                               axle_pos_b=axle_pos_b,
                                                                               relay_type='BR939A',
                                                                               electrical_staggering=False,
                                                                               cross_bonds=True)

    # a first
    e_thresh_ia_a20 = []
    for i in range(0, len(ia_a20[0, :])):
        e_mis_ia_a20_block = e_par[np.where(ia_a20[:, i] < 0.055)[0]]
        if len(e_mis_ia_a20_block) > 0:
            e_thresh_ia_a20.append(e_mis_ia_a20_block[np.argmin(np.abs(e_mis_ia_a20_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ia_a20.append(np.nan)

    e_thresh_ia_a50 = []
    for i in range(0, len(ia_a50[0, :])):
        e_mis_ia_a50_block = e_par[np.where(ia_a50[:, i] < 0.055)[0]]
        if len(e_mis_ia_a50_block) > 0:
            e_thresh_ia_a50.append(e_mis_ia_a50_block[np.argmin(np.abs(e_mis_ia_a50_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ia_a50.append(np.nan)

    e_thresh_ia_a80 = []
    for i in range(0, len(ia_a80[0, :])):
        e_mis_ia_a80_block = e_par[np.where(ia_a80[:, i] < 0.055)[0]]
        if len(e_mis_ia_a80_block) > 0:
            e_thresh_ia_a80.append(e_mis_ia_a80_block[np.argmin(np.abs(e_mis_ia_a80_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ia_a80.append(np.nan)

    e_thresh_ia_unileak = []
    for i in range(0, len(ia_unileak[0, :])):
        e_mis_ia_unileak_block = e_par[np.where(ia_unileak[:, i] < 0.055)[0]]
        if len(e_mis_ia_unileak_block) > 0:
            e_thresh_ia_unileak.append(e_mis_ia_unileak_block[np.argmin(np.abs(e_mis_ia_unileak_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ia_unileak.append(np.nan)

    # b second
    e_thresh_ib_a20 = []
    for i in range(0, len(ib_a20[0, :])):
        e_mis_ib_a20_block = e_par[np.where(ib_a20[:, i] < 0.055)[0]]
        if len(e_mis_ib_a20_block) > 0:
            e_thresh_ib_a20.append(e_mis_ib_a20_block[np.argmin(np.abs(e_mis_ib_a20_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ib_a20.append(np.nan)

    e_thresh_ib_a50 = []
    for i in range(0, len(ib_a50[0, :])):
        e_mis_ib_a50_block = e_par[np.where(ib_a50[:, i] < 0.055)[0]]
        if len(e_mis_ib_a50_block) > 0:
            e_thresh_ib_a50.append(e_mis_ib_a50_block[np.argmin(np.abs(e_mis_ib_a50_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ib_a50.append(np.nan)

    e_thresh_ib_a80 = []
    for i in range(0, len(ib_a80[0, :])):
        e_mis_ib_a80_block = e_par[np.where(ib_a80[:, i] < 0.055)[0]]
        if len(e_mis_ib_a80_block) > 0:
            e_thresh_ib_a80.append(e_mis_ib_a80_block[np.argmin(np.abs(e_mis_ib_a80_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ib_a80.append(np.nan)

    e_thresh_ib_unileak = []
    for i in range(0, len(ib_unileak[0, :])):
        e_mis_ib_unileak_block = e_par[np.where(ib_unileak[:, i] < 0.055)[0]]
        if len(e_mis_ib_unileak_block) > 0:
            e_thresh_ib_unileak.append(e_mis_ib_unileak_block[np.argmin(np.abs(e_mis_ib_unileak_block))])
        else:
            print(f'No value for block {i}')
            e_thresh_ib_unileak.append(np.nan)

    #np.savez(f'e_thresh_rs_{route_name}_no_stag.npz', e_thresh_a_a20=e_thresh_ia_a20, e_thresh_a_a50=e_thresh_ia_a50,
    #         e_thresh_a_a80=e_thresh_ia_a80, e_thresh_a_unileak=e_thresh_ia_unileak, e_thresh_b_a20=e_thresh_ib_a20,
    #         e_thresh_b_a50=e_thresh_ib_a50, e_thresh_b_a80=e_thresh_ib_a80, e_thresh_b_unileak=e_thresh_ib_unileak)

    #plt.plot(e_thresh_ia_a20, '.')
    #plt.plot(e_thresh_ia_a50, '.')
    #plt.plot(e_thresh_ia_a80, '.')
    #plt.plot(e_thresh_ia_unileak, 'x')
    #plt.xticks(range(0, len(e_thresh_ia_a20)), minor=True)
    #plt.grid(axis='x', which='minor', alpha=0.2)
    plt.show()

    pass


def find_thresholds_ws(route_name, leakage, direction, division, div_step):
    if direction == 'a':
        e_par = np.arange(-100, 100.1, 0.1)
        axles = np.load(f'../data/axle_positions/block_centre/axle_positions_two_track_back_axle_at_centre_{route_name}.npz', allow_pickle=True)
        axles_a = axles['axle_pos_a_all']
        axle_pos_a = []
        ia = np.zeros((len(e_par), len(axles_a)))

        for i in range(div_step, len(axles_a), division):
            for axa in axles_a[i]:
                axle_pos_a.append(axa)

        if leakage == 'unileak':
            ia_partial, ib_partial = tc_currents_two_track_e_parallel_uni_leak(section_name=route_name, leakage=leakage, e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=[], relay_type='BR939A', electrical_staggering=True, cross_bonds=True)
        else:
            ia_partial, ib_partial = tc_currents_two_track_e_parallel(section_name=route_name, leakage=leakage, e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=[], relay_type='BR939A', electrical_staggering=True, cross_bonds=True)
        ia[:, div_step::division] = ia_partial[:, div_step::division]

        e_thresh_ia_pos = []
        e_thresh_ia_neg = []
        for i in range(0, len(ia[0, :])):
            e_mis_ia_block = e_par[np.where((ia[:, i] > 0.081) | (ia[:, i] < -0.081))[0]]
            e_mis_ia_block_pos = e_mis_ia_block[np.where(e_mis_ia_block > 0)]
            if len(e_mis_ia_block_pos) > 0:
                e_thresh_ia_pos.append(np.min(e_mis_ia_block_pos))
            else:
                e_thresh_ia_pos.append(np.nan)
            e_mis_ia_block_neg = e_mis_ia_block[np.where(e_mis_ia_block < 0)]
            if len(e_mis_ia_block_neg) > 0:
                e_thresh_ia_neg.append(np.max(e_mis_ia_block_neg))
            else:
                e_thresh_ia_neg.append(np.nan)

        e_thresh_a_pos = np.round(np.array(e_thresh_ia_pos), 1)
        e_thresh_a_neg = np.round(np.array(e_thresh_ia_neg), 1)

        return e_thresh_a_pos, e_thresh_a_neg

    elif direction == 'b':
        e_par = np.arange(-100, 100.1, 0.1)
        axles = np.load(
            f'../data/axle_positions/block_centre/axle_positions_two_track_back_axle_at_centre_{route_name}.npz',
            allow_pickle=True)
        axles_b = axles['axle_pos_b_all']
        axle_pos_b = []
        ib = np.zeros((len(e_par), len(axles_b)))

        for i in range(div_step, len(axles_b), division):
            for axb in axles_b[i]:
                axle_pos_b.append(axb)

        if leakage == 'unileak':
            ib_partibl, ib_partibl = tc_currents_two_track_e_parallel_uni_leak(section_name=route_name, leakage=leakage, e_parallel=e_par, axle_pos_a=[], axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=True, cross_bonds=True)
        else:
            ib_partibl, ib_partibl = tc_currents_two_track_e_parallel(section_name=route_name, leakage=leakage, e_parallel=e_par, axle_pos_a=[], axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=True, cross_bonds=True)
        ib[:, div_step::division] = ib_partibl[:, div_step::division]

        e_thresh_ib_pos = []
        e_thresh_ib_neg = []
        for i in range(0, len(ib[0, :])):
            e_mis_ib_block = e_par[np.where((ib[:, i] > 0.081) | (ib[:, i] < -0.081))[0]]
            e_mis_ib_block_pos = e_mis_ib_block[np.where(e_mis_ib_block > 0)]
            if len(e_mis_ib_block_pos) > 0:
                e_thresh_ib_pos.append(np.min(e_mis_ib_block_pos))
            else:
                e_thresh_ib_pos.append(np.nan)
            e_mis_ib_block_neg = e_mis_ib_block[np.where(e_mis_ib_block < 0)]
            if len(e_mis_ib_block_neg) > 0:
                e_thresh_ib_neg.append(np.max(e_mis_ib_block_neg))
            else:
                e_thresh_ib_neg.append(np.nan)

        e_thresh_b_pos = np.round(np.array(e_thresh_ib_pos), 1)
        e_thresh_b_neg = np.round(np.array(e_thresh_ib_neg), 1)

        return e_thresh_b_pos, e_thresh_b_neg


def save_thresholds_ws():
    for name in ['glasgow_edinburgh_falkirk', 'west_coast_main_line', 'east_coast_main_line']:
        for leak in ['a20', 'a50', 'a80', 'unileak']:
            for dir in ['a', 'b']:
                print(f'Starting {name}; {leak}; direction {dir}')
                e_thresh_part_0_pos, e_thresh_part_0_neg = find_thresholds_ws(name, leak, dir, 2, 0)
                e_thresh_part_1_pos, e_thresh_part_1_neg = find_thresholds_ws(name, leak, dir, 2, 1)
                e_thresh_pos = np.zeros(len(e_thresh_part_0_pos))
                e_thresh_neg = np.zeros(len(e_thresh_part_0_neg))
                e_thresh_pos[0::2] = e_thresh_part_0_pos[0::2]
                e_thresh_pos[1::2] = e_thresh_part_1_pos[1::2]
                e_thresh_neg[0::2] = e_thresh_part_0_neg[0::2]
                e_thresh_neg[1::2] = e_thresh_part_1_neg[1::2]
                np.save(f'e_thresh_{dir}_{leak}_{name}_pos.npy', e_thresh_pos)
                np.save(f'e_thresh_{dir}_{leak}_{name}_neg.npy', e_thresh_neg)


def combine_thresholds_ws():
    for name in ['glasgow_edinburgh_falkirk', 'west_coast_main_line', 'east_coast_main_line']:
        e_thresh_a_a20_pos = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_a_a20_{name}_pos.npy')
        e_thresh_a_a20_neg = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_a_a20_{name}_neg.npy')
        e_thresh_a_a50_pos = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_a_a50_{name}_pos.npy')
        e_thresh_a_a50_neg = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_a_a50_{name}_neg.npy')
        e_thresh_a_a80_pos = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_a_a80_{name}_pos.npy')
        e_thresh_a_a80_neg = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_a_a80_{name}_neg.npy')
        e_thresh_a_unileak_pos = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_a_a20_{name}_pos.npy')
        e_thresh_a_unileak_neg = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_a_a20_{name}_neg.npy')
        e_thresh_b_a20_pos = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_b_a20_{name}_pos.npy')
        e_thresh_b_a20_neg = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_b_a20_{name}_neg.npy')
        e_thresh_b_a50_pos = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_b_a50_{name}_pos.npy')
        e_thresh_b_a50_neg = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_b_a50_{name}_neg.npy')
        e_thresh_b_a80_pos = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_b_a80_{name}_pos.npy')
        e_thresh_b_a80_neg = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_b_a80_{name}_neg.npy')
        e_thresh_b_unileak_pos = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_b_a20_{name}_pos.npy')
        e_thresh_b_unileak_neg = np.load(f'../data/resistivity/e_thresh/partial/e_thresh_b_a20_{name}_neg.npy')

        np.savez(f'e_thresh_ws_{name}.npz', e_thresh_a_a20_pos=e_thresh_a_a20_pos,
                 e_thresh_a_a20_neg=e_thresh_a_a20_neg, e_thresh_a_a50_pos=e_thresh_a_a50_pos,
                 e_thresh_a_a50_neg=e_thresh_a_a50_neg, e_thresh_a_a80_pos=e_thresh_a_a80_pos,
                 e_thresh_a_a80_neg=e_thresh_a_a80_neg, e_thresh_a_unileak_pos=e_thresh_a_unileak_pos,
                 e_thresh_a_unileak_neg=e_thresh_a_unileak_neg, e_thresh_b_a20_pos=e_thresh_b_a20_pos,
                 e_thresh_b_a20_neg=e_thresh_b_a20_neg, e_thresh_b_a50_pos=e_thresh_b_a50_pos,
                 e_thresh_b_a50_neg=e_thresh_b_a50_neg, e_thresh_b_a80_pos=e_thresh_b_a80_pos,
                 e_thresh_b_a80_neg=e_thresh_b_a80_neg, e_thresh_b_unileak_pos=e_thresh_b_unileak_pos,
                 e_thresh_b_unileak_neg=e_thresh_b_unileak_neg)


def plot_thresholds_ws(route_name):
    e_thresh = np.load(f'../data/resistivity/e_thresh/e_thresh_ws_{route_name}.npz')
    e_thresh_a_a20_pos = e_thresh['e_thresh_a_a20_pos']
    e_thresh_a_a50_pos = e_thresh['e_thresh_a_a50_pos']
    e_thresh_a_a80_pos = e_thresh['e_thresh_a_a80_pos']
    e_thresh_a_a20_neg = e_thresh['e_thresh_a_a20_neg']
    e_thresh_a_a50_neg = e_thresh['e_thresh_a_a50_neg']
    e_thresh_a_a80_neg = e_thresh['e_thresh_a_a80_neg']
    e_thresh_a_unileak_pos = e_thresh['e_thresh_a_unileak_pos']
    e_thresh_a_unileak_neg = e_thresh['e_thresh_a_unileak_neg']

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax0.scatter(x=range(0, len(e_thresh_a_a50_pos)), y=e_thresh_a_a50_pos, zorder=3, alpha=0.5, s=5, facecolor='tomato')
    ax0.scatter(x=range(0, len(e_thresh_a_a50_neg)), y=e_thresh_a_a50_neg, zorder=3, alpha=0.5, s=5, facecolor='cornflowerblue')
    for i in range(0, len(e_thresh_a_a50_pos)):
        if (e_thresh_a_a20_pos[i] != np.nan) | (e_thresh_a_a20_pos[i] != e_thresh_a_a50_pos[i]):
            ax0.plot([i, i], [e_thresh_a_a20_pos[i], e_thresh_a_a50_pos[i]], color='tomato', zorder=2)
        else:
            pass
        if (e_thresh_a_a80_pos[i] != np.nan) | (e_thresh_a_a80_pos[i] != e_thresh_a_a50_pos[i]):
            ax0.plot([i, i], [e_thresh_a_a50_pos[i], e_thresh_a_a80_pos[i]], color='tomato', zorder=2)
        else:
            pass
    for i in range(0, len(e_thresh_a_a50_neg)):
        if (e_thresh_a_a20_neg[i] != np.nan) | (e_thresh_a_a20_neg[i] != e_thresh_a_a50_neg[i]):
            ax0.plot([i, i], [e_thresh_a_a20_neg[i], e_thresh_a_a50_neg[i]], color='cornflowerblue', zorder=2)
        else:
            pass
        if (e_thresh_a_a80_neg[i] != np.nan) | (e_thresh_a_a80_neg[i] != e_thresh_a_a50_neg[i]):
            ax0.plot([i, i], [e_thresh_a_a50_neg[i], e_thresh_a_a80_neg[i]], color='cornflowerblue', zorder=2)
        else:
            pass
    ax0.set_xlabel('Block Number')
    ax0.set_ylabel('Leakage (S/km)')

    uni_leak_dif_pos = e_thresh_a_unileak_pos - e_thresh_a_a50_pos
    uni_leak_dif_neg = e_thresh_a_unileak_neg - e_thresh_a_a50_neg
    uni_leak_dif_pos_non_zero_loc = np.where(uni_leak_dif_pos != 0)
    uni_leak_dif_neg_non_zero_loc = np.where(uni_leak_dif_neg != 0)
    ax1.plot(uni_leak_dif_pos_non_zero_loc[0], uni_leak_dif_pos[uni_leak_dif_pos_non_zero_loc], '.', color='tomato')
    ax1.plot(uni_leak_dif_neg_non_zero_loc[0], uni_leak_dif_neg[uni_leak_dif_neg_non_zero_loc], '.', color='cornflowerblue')

    # plt.savefig(f'block_leakages_{route_name}.pdf)
    plt.show()
    plt.close()


def arc_plot_leakage_current_compare(route_name):
    block_leak = np.load(f'../data/resistivity/{route_name}_block_leakage.npz')
    bl_a50 = block_leak['a50']

    e_par = np.array([5])
    axle_pos_a = []
    axle_pos_b = []
    ia_a50, ib_a50 = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=True, cross_bonds=True)
    ia_uni, ib_uni = tc_currents_two_track_e_parallel_uni_leak(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=True, cross_bonds=True)
    ia_a50_no_stag, ib_a50_no_stag = tc_currents_two_track_e_parallel(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=False, cross_bonds=True)
    ia_uni_no_stag, ib_uni_no_stag = tc_currents_two_track_e_parallel_uni_leak(section_name=route_name, leakage='a50', e_parallel=e_par, axle_pos_a=axle_pos_a, axle_pos_b=axle_pos_b, relay_type='BR939A', electrical_staggering=False, cross_bonds=True)

    plt.rcParams['font.size'] = '15'
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 1)
    ax0 = fig.add_subplot(gs[0])
    ax1 = fig.add_subplot(gs[1])
    ax0.plot(bl_a50, '.')
    ax0.set_xlim(0, len(ia_a50))
    ax0.axhline(1.6, linestyle='--')
    ax0.set_ylabel('Leakage (S)')
    ax0.set_xlabel('Block Number')
    ax0.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
    ax1.plot((ia_a50 - ia_uni), '.', label='Polarity Staggered')
    ax1.plot((ia_a50_no_stag - ia_uni_no_stag), 'x', label='Polarity Not Staggered')
    ax1.legend()
    ax1.set_xlim(0, len(ia_a50))
    ax1.set_ylabel('Current Difference (A)')
    ax1.set_xlabel('Block Number')
    plt.subplots_adjust(hspace=0)
    fig.align_ylabels()
    plt.show()
    #plt.close()
    #plt.savefig(f'leakage_currents_compare_rs._{route_name}.pdf')


#save_thresholds_ws()
#combine_thresholds_ws()
for name in ['west_coast_main_line', 'glasgow_edinburgh_falkirk', 'east_coast_main_line']:
    #plot_resistivity_along_line(name)
    #plot_leaks(name)
    #plot_currents_rs(name)
    #plot_currents_rs_no_stag(name)
    #plot_traction_rail_voltages(name)
    #plot_signal_rail_voltages(name)
    #plot_currents_ws(name)
    plot_thresholds_rs(name)
    plot_thresholds_ws(name)
