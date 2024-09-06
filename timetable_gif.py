import numpy as np
import matplotlib.pyplot as plt
import imageio
from datetime import datetime, timedelta
from misoperation_analysis import rail_model_two_track_e_blocks


def generate_timetable_currents(section_name, storm):
    timetable_axles = np.load("data/axle_positions/timetable/" + section_name + "_axle_positions_timetable.npz", allow_pickle=True)
    axle_positions_a_all = timetable_axles["axle_pos_a_all"]
    axle_positions_b_all = timetable_axles["axle_pos_b_all"]

    axle_positions_a_all = np.concatenate((axle_positions_a_all, axle_positions_a_all))
    axle_positions_b_all = np.concatenate((axle_positions_b_all, axle_positions_b_all))

    # Load in storm e_field data
    storm_es = np.load('data/storm_e_fields/bgs_' + storm + '/' + section_name + '_' + storm + '_e_blocks.npz')
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

        print(i)

    np.savez(section_name + "_i_all_timetable_" + storm + ".npz", ia_all=ia_all, ib_all=ib_all)


def save_frames(section, storm):
    data = np.load("data/rail_data/" + section + "/" + section + "_distances_bearings.npz")
    blocks = data["distances"]
    blocks_sum = np.cumsum(blocks)
    blocks_sum = np.insert(blocks_sum, 0, 0)

    timetable_axles = np.load("data/axle_positions/timetable/" + section + "_axle_positions_timetable.npz", allow_pickle=True)
    axle_positions_a_all = timetable_axles["axle_pos_a_all"]
    axle_positions_b_all = timetable_axles["axle_pos_b_all"]

    axle_positions_a_all = np.concatenate((axle_positions_a_all, axle_positions_a_all))
    axle_positions_b_all = np.concatenate((axle_positions_b_all, axle_positions_b_all))

    start_time = datetime.strptime("2024-05-10 00:00", "%Y-%m-%d %H:%M")
    time_increment = timedelta(minutes=1)

    i_all = np.load("data/currents/" + section + "_i_all_timetable_" + storm + ".npz", allow_pickle=True)
    ia_all = i_all["ia_all"]
    ib_all = i_all["ib_all"]

    for i in range(0, len(ia_all)):
        print(i)
        ia = ia_all[i]
        ib = ib_all[i]

        axle_positions_a = axle_positions_a_all[i]
        axle_positions_b = axle_positions_b_all[i]

        current_time = start_time + i * time_increment
        time_str = current_time.strftime("%Y-%m-%d %H:%M")

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

        plt.rcParams['font.size'] = '15'
        fig, ax = plt.subplots(2, 1, figsize=(10, 8))

        ax[0].plot(np.delete(xs, train_indices_a), np.delete(ia, train_indices_a), linestyle='', marker='o', markersize=6, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
        ax[0].plot(train_indices_a, ia[train_indices_a], linestyle='', marker='>', markersize=6, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
        ax[0].axhline(0.055, color="tomato", linestyle="--")
        ax[0].axhline(0.081, color="limegreen")
        ax[0].set_ylim(-0.15, 0.5)
        #ax[0].set_xlim(-1, 72)
        ax[0].set_xlabel("Track Circuit Block")
        ax[0].set_ylabel("Current Through Relay (A)")
        ax[1].plot(np.delete(xs, train_indices_b), np.delete(ib, train_indices_b), linestyle='', marker='o', markersize=6, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
        ax[1].plot(train_indices_b, ib[train_indices_b], linestyle='', marker='<', markersize=6, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
        ax[1].axhline(0.055, color="tomato", linestyle="--")
        ax[1].axhline(0.081, color="limegreen")
        ax[1].set_ylim(-0.15, 0.5)
        #ax[1].set_xlim(-1, 72)
        ax[1].set_xlabel("Track Circuit Block")
        ax[1].set_ylabel("Current Through Relay (A)")
        ax[0].grid()
        ax[1].grid()
        plt.suptitle(f'{time_str}')
        plt.show()
        #plt.savefig(f'frames/{section}/{storm}/{section}_{storm}_frame_{i:03d}.png')
        plt.close()


def make_gif(section, storm):
    # Get the list of saved frame files
    frame_files = [f'frames/{section}/{storm}/{section}_{storm}_frame_{i:03d}.png' for i in range(2880)]

    # Create a GIF
    with imageio.get_writer('my_animation.gif', mode='I', duration=0.1) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)


#generate_timetable_currents("east_coast_main_line", "may2024")
#save_frames("east_coast_main_line", "may2024")
#make_gif("glasgow_edinburgh_falkirk", "may2024")
