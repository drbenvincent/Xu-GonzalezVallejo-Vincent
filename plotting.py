import numpy as np
import matplotlib.pyplot as plt


def plot_data(data, ax, legend=True):
    D = data["R"] == 1
    I = data["R"] == 0

    if np.sum(D) > 0:
        ax.scatter(
            x=data["DB"][D],
            y=data["RA"][D],
            c="k",
            edgecolors="k",
            label="chose delayed prospect",
        )
    if np.sum(I) > 0:
        ax.scatter(
            x=data["DB"][I],
            y=data["RA"][I],
            c="w",
            edgecolors="k",
            label="chose immediate prospect",
        )

    # deal with losses
    if data["RB"].values[0] < 0:
        ax.invert_yaxis()

    # deal with y axis limit
    ax.set_ylim(bottom=0)
