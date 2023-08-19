# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import os
import sys
import argparse
import json
from matplotlib import pyplot as plt
import numpy as np


sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import scienceplots
import matplotlib.font_manager

import os

try:
    matplotlib.font_manager._rebuild()
    print("cache rebuilt")
except:
    pass

try:
    print(
        "matplotlib.font_manager.get_cachedir(): ",
        matplotlib.font_manager.get_cachedir(),
    )
except:
    pass

try:
    print("matplotlib.get_cachedir(): ", matplotlib.get_cachedir())
except:
    pass

try:
    print("matplotlib.font_manager._fmcache: ", matplotlib.font_manager._fmcache)
except:
    pass


plt.style.use(["ieee", "vibrant"])
plt.rcParams["grid.alpha"] = 0.5
plt.rcParams["grid.color"] = "#000000"
plt.rcParams["xtick.color"] = "#000000"


plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.size"] = 14  # for general font size
plt.rcParams["axes.labelsize"] = 16  # for x and y labels
plt.rcParams["xtick.labelsize"] = 14  # for x tick labels
plt.rcParams["ytick.labelsize"] = 14  # for y tick labels
plt.rcParams["legend.fontsize"] = 14  # for legend
plt.rcParams["axes.titlesize"] = 18  # for plot title

fig, ax = plt.subplots(figsize=(8, 8))

pparam = dict(xlabel="Voltage (mV)", ylabel=r"Current ($\mu$A)")

x = np.linspace(0.75, 1.25, 201)


def model(x, p):
    return x ** (2 * p + 1) / (1 + x ** (2 * p))


for p in [5, 7, 10, 15, 20, 30, 38, 50, 100]:
    ax.plot(x, model(x, p), label=p)
ax.legend(title="Order", fontsize=7)
ax.autoscale(tight=True)
ax.set(**pparam)
fig.savefig("fig13.jpg", dpi=300)
plt.close()
print("Done")
