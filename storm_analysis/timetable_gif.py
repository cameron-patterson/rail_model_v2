import numpy as np
import matplotlib.pyplot as plt
import imageio
from matplotlib.gridspec import GridSpec
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from misoperation_analysis import rail_model_two_track_e_blocks
from scipy.io import loadmat
from matplotlib import cm
import matplotlib.colors as mcolors
import pandas as pd


def generate_timetable_currents(section_name, storm):
    timetable_axles = np.load(f'../data/axle_positions/timetable/{section_name}_axle_positions_timetable.npz', allow_pickle=True)
    axle_positions_a = timetable_axles['axle_pos_a_all']
    axle_positions_b = timetable_axles['axle_pos_b_all']

    if storm == 'may2024':
        axle_positions_a_all = np.concatenate((axle_positions_a, axle_positions_a))
        axle_positions_b_all = np.concatenate((axle_positions_b, axle_positions_b))
    elif storm == 'sep2017':
        axle_positions_a_all = np.concatenate((axle_positions_a, axle_positions_a))
        axle_positions_b_all = np.concatenate((axle_positions_b, axle_positions_b))
    elif storm == 'oct2003':
        axle_positions_a_all = np.concatenate((axle_positions_a, axle_positions_a))
        axle_positions_a_all = np.concatenate((axle_positions_a_all, axle_positions_a))
        axle_positions_b_all = np.concatenate((axle_positions_b, axle_positions_b))
        axle_positions_b_all = np.concatenate((axle_positions_b_all, axle_positions_b))
    elif storm == 'mar1989':
        axle_positions_a_all = np.concatenate((axle_positions_a, axle_positions_a))
        axle_positions_b_all = np.concatenate((axle_positions_b, axle_positions_b))
    else:
        print("Storm not configured")

    # Load in storm e_field data
    storm_es = np.load(f'../data/storm_e_fields/bgs_{storm}/{section_name}_{storm}_e_blocks.npz')
    ex_blocks_all = storm_es['ex_blocks'] / 1000
    ey_blocks_all = storm_es['ey_blocks'] / 1000

    ia_all = np.empty(len(axle_positions_a_all), dtype=object)
    ib_all = np.empty(len(axle_positions_b_all), dtype=object)
    for i in range(0, len(axle_positions_a_all)):
        axle_pos_a = axle_positions_a_all[i]
        axle_pos_b = axle_positions_b_all[i]
        ex_blocks = ex_blocks_all[:, i:i+1]
        ey_blocks = ey_blocks_all[:, i:i+1]

        ia, ib = rail_model_two_track_e_blocks(section_name, "moderate", ex_blocks, ey_blocks, axle_pos_a, axle_pos_b)
        ia_all[i] = ia[0]
        ib_all[i] = ib[0]

        print(str(i/len(axle_positions_a_all) * 100) + '%')

    np.savez(f'../data/currents/{section_name}_i_all_timetable_{storm}.npz', ia_all=ia_all, ib_all=ib_all)


def save_frames(section, storm):
    bl = np.load(f'../data/rail_data/{section}/{section}_distances_bearings.npz')
    blocks = bl["distances"]
    blocks_sum = np.cumsum(blocks)
    blocks_sum = np.insert(blocks_sum, 0, 0)

    lola = loadmat(f'../data/storm_e_fields/bgs_{storm}/{storm}.mat')
    lon_grid = lola['longic']
    lat_grid = lola['latgic']

    es = loadmat(f'../data/storm_e_fields/bgs_{storm}/{storm}.mat')
    exs = es['Ex']
    eys = es['Ey']

    timetable_axles = np.load(f'../data/axle_positions/timetable/{section}_axle_positions_timetable.npz',allow_pickle=True)
    axle_positions_a = timetable_axles['axle_pos_a_all']
    axle_positions_b = timetable_axles['axle_pos_b_all']

    if storm == 'may2024':
        axle_positions_a_all = np.concatenate((axle_positions_a, axle_positions_a))
        axle_positions_b_all = np.concatenate((axle_positions_b, axle_positions_b))
    elif storm == 'sep2017':
        axle_positions_a_all = np.concatenate((axle_positions_a, axle_positions_a))
        axle_positions_b_all = np.concatenate((axle_positions_b, axle_positions_b))
    elif storm == 'oct2003':
        axle_positions_a_all = np.concatenate((axle_positions_a, axle_positions_a))
        axle_positions_a_all = np.concatenate((axle_positions_a_all, axle_positions_a))
        axle_positions_b_all = np.concatenate((axle_positions_b, axle_positions_b))
        axle_positions_b_all = np.concatenate((axle_positions_b_all, axle_positions_b))
    elif storm == 'mar1989':
        axle_positions_a_all = np.concatenate((axle_positions_a, axle_positions_a))
        axle_positions_b_all = np.concatenate((axle_positions_b, axle_positions_b))
    else:
        print("Storm not configured")

    if storm == 'may2024':
        start_time = datetime.strptime('2024-05-10 00:00', '%Y-%m-%d %H:%M')
    elif storm == 'sep2017':
        start_time = datetime.strptime('2017-09-07 00:00', '%Y-%m-%d %H:%M')
    elif storm == 'oct2003':
        start_time = datetime.strptime('2003-10-29 00:00', '%Y-%m-%d %H:%M')
    elif storm == 'mar1989':
        start_time = datetime.strptime('1989-03-13 00:00', '%Y-%m-%d %H:%M')
    else:
        print("Storm not configured")

    time_increment = timedelta(minutes=1)

    i_all = np.load(f'../data/currents/{section}_i_all_timetable_{storm}.npz', allow_pickle=True)
    ia_all = i_all['ia_all']
    ib_all = i_all['ib_all']

    lon_lats = np.load(f'../data/rail_data/{section}/{section}_sub_block_lons_lats.npz')
    lon_points = lon_lats['lons']
    lat_points = lon_lats['lats']

    coast = np.loadtxt(f'../data/storm_e_fields/coastline.txt')

    # Load the sym-h Excel file
    df = pd.read_excel(f'../data/sym_h/sym_h_{storm}.xlsx')
    spreadsheet_columns_as_arrays = df.values.T
    sym_h = spreadsheet_columns_as_arrays[6]

    for i in range(0, len(ia_all)):
        print(str(np.round((i/len(ia_all))*100, 2)) + '%')
        ia = ia_all[i]
        ib = ib_all[i]

        axle_positions_a = axle_positions_a_all[i]
        axle_positions_b = axle_positions_b_all[i]

        current_time = start_time + i * time_increment
        time_str = current_time.strftime("%Y-%m-%d %H:%M")
        num_values = len(sym_h)
        datetimes = [start_time + i * time_increment for i in range(num_values)]

        train_indices_a = []
        for value in axle_positions_a:
            valid_indices = np.where(blocks_sum < value)[0]

            if len(valid_indices) > 0:
                # Get the index of the closest value that is less than the target value
                closest_index = valid_indices[np.argmax(blocks_sum[valid_indices])]
                train_indices_a.append(closest_index)
            else:
                # If no valid index is found, append None
                train_indices_a.append(None)
        train_indices_a = np.unique(train_indices_a)
        train_indices_a = train_indices_a.astype(int)

        train_indices_b = []
        for value in axle_positions_b:
            valid_indices = np.where(blocks_sum < value)[0]

            if len(valid_indices) > 0:
                # Get the index of the closest value that is less than the target value
                closest_index = valid_indices[np.argmax(blocks_sum[valid_indices])]
                train_indices_b.append(closest_index)
            else:
                # If no valid index is found, append None
                train_indices_b.append(None)
        train_indices_b = np.unique(train_indices_b)
        train_indices_b = train_indices_b.astype(int)

        xs = np.arange(0, len(ia))
        xs_train_a = xs[train_indices_a]
        xs_train_b = xs[train_indices_b]
        xs_no_train_a = np.delete(xs, train_indices_a)
        xs_no_train_b = np.delete(xs, train_indices_b)

        ia_train = ia[train_indices_a]
        ia_no_train = np.delete(ia, train_indices_a)
        ib_train = ib[train_indices_b]
        ib_no_train = np.delete(ib, train_indices_b)

        plt.rcParams['font.size'] = '15'
        fig = plt.figure(figsize=(15, 16))

        if section == "east_coast_main_line" or section == "west_coast_main_line":
            gs = GridSpec(3, 3)
            ax0 = fig.add_subplot(gs[1:-1, :-1])
            ax1 = fig.add_subplot(gs[2:, :-1])
            ax2 = fig.add_subplot(gs[1:-1, 2:])
            ax3 = fig.add_subplot(gs[2:, 2:])
            ax0.plot(xs_no_train_a, ia_no_train, linestyle='', marker='o', markersize=2, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
            ax0.plot(xs_train_a, ia_train, linestyle='', marker='>', markersize=2, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
            ax1.plot(xs_no_train_b, ib_no_train, linestyle='', marker='o', markersize=2, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
            ax1.plot(xs_train_b, ib_train, linestyle='', marker='<', markersize=2, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
        else:
            gs = GridSpec(3, 2)
            ax0 = fig.add_subplot(gs[1:-1, :-1])
            ax1 = fig.add_subplot(gs[2:, :-1])
            ax2 = fig.add_subplot(gs[1:-1, 1:])
            ax3 = fig.add_subplot(gs[2:, 1:])
            ax0.plot(xs_no_train_a, ia_no_train, linestyle='', marker='o', markersize=4, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
            ax0.plot(xs_train_a, ia_train, linestyle='', marker='>', markersize=4, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
            ax1.plot(xs_no_train_b, ib_no_train, linestyle='', marker='o', markersize=4, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
            ax1.plot(xs_train_b, ib_train, linestyle='', marker='<', markersize=4, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)

        if storm == 'may2024':
            ex_levels = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000]
            ey_levels = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000]
        elif storm == 'sep2017':
            ex_levels = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000]
            ey_levels = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000]
        elif storm == 'oct2003':
            ex_levels = [-2000, -1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000]
            ey_levels = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000, 2000]
        elif storm == 'mar1989':
            ex_levels = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000]
            ey_levels = [-1000, -500, -400, -300, -200, -100, 0, 100, 200, 300, 400, 500, 1000, 2000]
        else:
            print("Storm not configured")

        cs1 = ax2.contourf(lon_grid, lat_grid, exs[:, :, i], zorder=1, norm=mcolors.CenteredNorm(), levels=ex_levels, cmap="RdBu_r", extend='both')
        cbar1 = fig.colorbar(cs1, ax=ax2)
        cbar1.set_ticks(ex_levels)
        cbar1.set_label("Ex (mV/km)")
        cs2 = ax3.contourf(lon_grid, lat_grid, eys[:, :, i], zorder=1, norm=mcolors.CenteredNorm(), levels=ey_levels, cmap="RdBu_r", extend='both')
        cbar2 = fig.colorbar(cs2, ax=ax3)
        cbar2.set_ticks(ey_levels)
        cbar2.set_label("Ey (mV/km)")

        ax2.plot(coast[:, 0], coast[:, 1], color='grey', linewidth=1, zorder=2)
        ax2.plot(lon_points, lat_points, '-', linewidth=4, color="black", zorder=3)
        ax2.set_xlabel("Geographic Longitude")
        ax2.set_ylabel("Geographic Latitude")
        ax3.plot(coast[:, 0], coast[:, 1], color='grey', linewidth=1, zorder=2)
        ax3.plot(lon_points, lat_points, '-', linewidth=4, color="black", zorder=3)
        ax3.set_xlabel("Geographic Longitude")
        ax3.set_ylabel("Geographic Latitude")

        ax4 = fig.add_subplot(gs[:1, :])
        ax4.plot(datetimes, sym_h, linestyle="-", color="black")
        ax4.axvline(current_time, linestyle="--", color="red")
        ax4.set_xlim([datetimes[0], datetimes[-1] + time_increment])
        ax4.set_ylabel("SYM-H (nT)")
        ax4.xaxis.set_major_locator(mdates.HourLocator(interval=12))
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))

        def add_common_elements(ax):
            ax.axhline(0.055, color="tomato", linestyle="--")
            ax.axhline(0.081, color="limegreen")
            ax.set_ylim(-0.1, 0.4)
            ax.set_xlim(-1, len(ia_all[1000]))
            ax.set_xlabel("Track Circuit Block")
            ax.set_ylabel("Current Through Relay (A)")
            ax.grid()
            ax.grid()

        add_common_elements(ax0)
        add_common_elements(ax1)

        if section == "glasgow_edinburgh_falkirk":
            name = "Glasgow to Edinburgh via Falkirk High"
            ax2.set_xlim(-5, -2.5)
            ax2.set_ylim(55, 57)
            ax3.set_xlim(-5, -2.5)
            ax3.set_ylim(55, 57)
        if section == "east_coast_main_line":
            name = "East Coast Main Line"
            ax2.set_xlim(-3.5, 0)
            ax2.set_ylim(51, 57)
            ax3.set_xlim(-3.5, 0)
            ax3.set_ylim(51, 57)
        plt.suptitle(f'{name}: {time_str}')
        plt.subplots_adjust(top=0.93)
        #plt.show()
        plt.savefig(f'../frames/{section}/{storm}/{section}_{storm}_frame_{i:03d}.png')
        plt.close()


def make_gif(section, storm):
    # Get the list of saved frame files
    frame_files = [f'../frames/{section}/{storm}/{section}_{storm}_frame_{i:03d}.png' for i in range(0, 2880)]

    # Create a GIF
    with imageio.get_writer(f'{section}_{storm}_animation.gif', mode='I', duration=0.1) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)


for storm in ['sep2017', 'may2024']:
    #generate_timetable_currents("glasgow_edinburgh_falkirk", storm)
    save_frames("glasgow_edinburgh_falkirk", storm)
    #make_gif('glasgow_edinburgh_falkirk', storm)
