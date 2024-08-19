import numpy as np
from wrong_side_analysis import wrong_side_two_track
import matplotlib.pyplot as plt
import imageio
from datetime import datetime, timedelta


def save_frames(section):
    data = np.load("data/rail_data/" + section + "/" + section + "_lengths_angles.npz")
    blocks = data["block_lengths"]
    blocks_sum = np.cumsum(blocks)
    blocks_sum = np.insert(blocks_sum, 0, 0)

    train_pos_day_a = np.load(section + "_train_pos_day_direction_a.npy")
    train_pos_day_b = np.load(section + "_train_pos_day_direction_b.npy")

    start_time = datetime.strptime("06:00", "%H:%M")
    time_increment = timedelta(minutes=1)

    for i in range(0, 1080):
        current_time = start_time + i * time_increment
        time_str = current_time.strftime("%H:%M")

        train_pos_min_a = train_pos_day_a[:, i]
        train_pos_min_a = train_pos_min_a[~np.isnan(train_pos_min_a)]
        train_pos_min_b = train_pos_day_b[:, i]
        train_pos_min_b = train_pos_min_b[~np.isnan(train_pos_min_b)]

        axle_positions_a = train_pos_min_a
        axle_positions_b = train_pos_min_b
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

        ia, ib = wrong_side_two_track(section, "moderate", axle_positions_a, axle_positions_b, ex, ey)
        xs = np.arange(0, len(ia[0]))

        plt.rcParams['font.size'] = '15'
        fig, ax = plt.subplots(2, 1, figsize=(12, 10))
        ax[0].plot(np.delete(xs, train_indices_a), np.delete(ia[0], train_indices_a), linestyle='', marker='o', markersize=6, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
        ax[0].plot(train_indices_a, ia[0, train_indices_a], linestyle='', marker='>', markersize=6, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
        ax[0].axhline(0.055, color="tomato", linestyle="--")
        ax[0].axhline(0.081, color="limegreen")
        ax[0].set_ylim(-0.05, 0.4)
        ax[0].set_xlim(-1, 72)
        ax[0].set_xlabel("Track Circuit Block")
        ax[0].set_ylabel("Current Through Relay (A)")
        ax[1].plot(np.delete(xs, train_indices_b), np.delete(ib[0], train_indices_b), linestyle='', marker='o', markersize=6, markerfacecolor='white', markeredgecolor='limegreen', markeredgewidth=1)
        ax[1].plot(train_indices_b, ib[0, train_indices_b], linestyle='', marker='<', markersize=6, markerfacecolor='white', markeredgecolor='red', markeredgewidth=1)
        ax[1].axhline(0.055, color="tomato", linestyle="--")
        ax[1].axhline(0.081, color="limegreen")
        ax[1].set_ylim(-0.05, 0.4)
        ax[1].set_xlim(-1, 72)
        ax[1].set_xlabel("Track Circuit Block")
        ax[1].set_ylabel("Current Through Relay (A)")
        plt.suptitle(f'Time = {time_str}')
        plt.savefig(f'frames/frame_{i:03d}.png')
        plt.close()


def make_gif():
    # Get the list of saved frame files
    frame_files = [f'frames/frame_{i:03d}.png' for i in range(1054)]

    # Create a GIF
    with imageio.get_writer('my_animation.gif', mode='I', duration=0.1) as writer:
        for filename in frame_files:
            image = imageio.imread(filename)
            writer.append_data(image)


#save_frames("glasgow_edinburgh_falkirk")
make_gif()
