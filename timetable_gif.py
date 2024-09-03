import numpy as np
from wrong_side_analysis import wrong_side_two_track
from right_side_analysis import right_side_two_track
import matplotlib.pyplot as plt
import imageio
from datetime import datetime, timedelta


def save_frames(section):
    data = np.load("data/rail_data/" + section + "/" + section + "_distances_bearings.npz")
    blocks = data["distances"]
    blocks_sum = np.cumsum(blocks)
    blocks_sum = np.insert(blocks_sum, 0, 0)

    timetable_axles = np.load("data/axle_positions/timetable/" + section + "_axle_positions_timetable.npz", allow_pickle=True)
    axle_positions_a_all = timetable_axles["axle_pos_a_all"]
    axle_positions_b_all = timetable_axles["axle_pos_b_all"]

    start_time = datetime.strptime("00:00", "%H:%M")
    time_increment = timedelta(minutes=1)

    for i in range(4, 1440):
        axle_positions_a = axle_positions_a_all[i]
        axle_positions_b = axle_positions_b_all[i]

        current_time = start_time + i * time_increment
        time_str = current_time.strftime("%H:%M")

        ex = np.array([0])
        ey = np.array([0])

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

        if len(axle_positions_a) == 0 & len(axle_positions_b) == 0:
            print("right")
            ia, ib = right_side_two_track(section, "moderate", ex, ey)
        else:
            print("wrong")
            ia, ib = wrong_side_two_track(section, "moderate", ex, ey, axle_positions_a, axle_positions_b)
        xs = np.arange(0, len(ia[0]))

        ia_train = ia[0, train_indices_a]



        plt.rcParams['font.size'] = '15'
        fig, ax = plt.subplots(2, 1, figsize=(20, 10))
        ax[0].plot(np.delete(xs, train_indices_a), np.delete(ia[0], train_indices_a), linestyle='', marker='o', markersize=3, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
        ax[0].plot(train_indices_a, ia[0, train_indices_a], linestyle='', marker='>', markersize=3, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
        ax[0].axhline(0.055, color="tomato", linestyle="--")
        ax[0].axhline(0.081, color="limegreen")
        #ax[0].set_ylim(-0.05, 0.4)
        #ax[0].set_xlim(-1, 72)
        ax[0].set_xlabel("Track Circuit Block")
        ax[0].set_ylabel("Current Through Relay (A)")
        ax[1].plot(np.delete(xs, train_indices_b), np.delete(ib[0], train_indices_b), linestyle='', marker='o', markersize=3, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
        ax[1].plot(train_indices_b, ib[0, train_indices_b], linestyle='', marker='<', markersize=3, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
        ax[1].axhline(0.055, color="tomato", linestyle="--")
        ax[1].axhline(0.081, color="limegreen")
        #ax[1].set_ylim(-0.05, 0.4)
        #ax[1].set_xlim(-1, 72)
        ax[1].set_xlabel("Track Circuit Block")
        ax[1].set_ylabel("Current Through Relay (A)")
        plt.suptitle(f'Time = {time_str}')
        plt.savefig(f'frames/{section}_frame_{i:03d}.png')
        plt.close()


def make_gif():
    # Get the list of saved frame files
    frame_files = [f'frames/frame_{i:03d}.png' for i in range(1054)]

    # Create a GIF
    with imageio.get_writer('my_animation.gif', mode='I', duration=0.1) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)


save_frames("east_coast_main_line")
#make_gif()
