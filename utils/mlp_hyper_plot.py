import json
from pathlib import Path
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# get and calculate training data
base_path = Path("/mnt/emc01/zeyu/mlcw/")
matrix = np.zeros((4, 9))

index = 0
for i, h in enumerate([1, 2, 3, 4]):
    # index = index + 1
    for j, c in enumerate([16, 32, 64 ,128, 256, 512, 1024, 1536, 2048]):
        print(i, j, "mlp_h{}_c{}/pre.json".format(h, c))
        data = json.load(open(Path("/mnt/emc01/zeyu/mlcw/exp/")
                              .joinpath("mlp_h{}_c{}/pre.json".format(h, c)),
                              "r"))
        matrix[i][j] = data["best_test_acc"]

# %%

# set general theme
sns.set_theme(style="white")

# Set up the matplotlib figure
f, ax = plt.subplots(figsize=(16, 9))

# Generate a color map
cmap = sns.color_palette("Blues", as_cmap=True)

# set global font
plt.rcParams.update({'font.size': 25})

# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(matrix, cmap=cmap,
            # vmin=.3, vmax=.5,
            square=True,
            # linewidths=.5,
            cbar=False, annot=True, fmt='.4g', vmin=0.32,
            yticklabels=[1, 2, 3, 4], xticklabels=[16, 32, 64 ,128, 256, 512, 1024, 1536, 2048]
            )
# set labels
plt.xlabel("Num of Channels",fontsize=30)
plt.ylabel("Num of Hidden Layers",fontsize=30)

# set ticks
plt.xticks(fontsize=25)
plt.yticks(fontsize=25)

# plot figure
plt.tight_layout()
plt.show()
