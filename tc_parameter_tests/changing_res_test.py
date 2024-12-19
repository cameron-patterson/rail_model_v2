import numpy as np
import matplotlib.pyplot as plt


def ground_admittance_test(res):
    leakage_trac = 1.6
    leakage_sig = 0.1
    l_block = 1

    z0_trac = np.sqrt(res / leakage_trac)
    gamma_trac = np.sqrt(res * leakage_trac)

    z0_sig = np.sqrt(res / leakage_sig)
    gamma_sig = np.sqrt(res * leakage_sig)

    ye_trac = 1 / (z0_trac * np.sinh(gamma_trac * 1))  # Series admittance for traction return rail
    ye_sig = 1 / (z0_sig * np.sinh(gamma_sig * 1))  # Series admittance for signalling rail
    yg_trac = 2 * ((np.cosh(gamma_trac * 1) - 1) * (1 / (z0_trac * np.sinh(gamma_trac * 1))))  # Parallel admittance for traction return rail
    yg_sig = 2 * ((np.cosh(gamma_sig * 1) - 1) * (1 / (z0_sig * np.sinh(gamma_sig * 1))))  # Parallel admittance for signalling rail

    return yg_trac, yg_sig, ye_trac, ye_sig


ress = np.linspace(0.0289, 0.35, 1000)

ygs_t = np.empty(len(ress))
ygs_s = np.empty(len(ress))
yes_t = np.empty(len(ress))
yes_s = np.empty(len(ress))

for i in range(0, len(ress)):
    yg_t, yg_s, ye_t, ye_s = ground_admittance_test(ress[i])
    ygs_t[i] = yg_t
    ygs_s[i] = yg_s
    yes_t[i] = ye_t
    yes_s[i] = ye_s

fig, ax = plt.subplots(2, 2, figsize=(6, 12))

ax[0, 0].plot(ress, yes_t, color='cornflowerblue', label='Traction rail')
ax[0, 0].plot(ress, yes_s, color='tomato', label='Signal rail')
ax[0, 0].set_title("Series admittance")

ax[1, 0].plot(ress, ygs_t, color='cornflowerblue', label='Traction rail')
ax[1, 0].plot(ress, ygs_s, color='tomato', label='Signal rail')
ax[1, 0].set_title("Parallel admittance")

ax[0, 1].plot(ress, yes_s - yes_t, color='purple', label="Signal - Traction")
ax[0, 1].set_title("Difference in series admittance")

ax[1, 1].plot(ress, ygs_s - ygs_t, color='purple', label="Signal - Traction")
ax[1, 1].set_title("Difference in parallel admittance")


def ax_add(ax):
    ax.set_xlabel("Rail resistance (\u03A9/km)")
    ax.set_ylabel("Admittance (S)")
    ax.legend()


ax_add(ax[0, 0])
ax_add(ax[0, 1])
ax_add(ax[1, 0])
ax_add(ax[1, 1])

plt.show()
