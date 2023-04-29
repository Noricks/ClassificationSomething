import json
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# %%
###### Figure 1

# set path of the data
base_path = Path("/mnt/emc01/zeyu/mlcw/exp")
mlp_path = base_path.joinpath("mlp_h2_c2048_b")
cnn_path = base_path.joinpath("resnet2")
# %%
# choose path
path = cnn_path

# load data
data = json.load(open(path.joinpath("train_test.json"), "r"))
data_key = pd.DataFrame(data)[["train_loss", "test_loss", "train_acc", "test_acc"]]

# %%
# initialize figure
fig, ax = plt.subplots(figsize=(7, 6))

# initialize theme
sns.set_theme(style="ticks")

# add line
sns.lineplot(
    data=data_key,
)

plt.xlabel("Epoch", fontsize=18)
plt.ylabel("Value", fontsize=18)
plt.legend(fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
# %%
# plot figure
plt.tight_layout()
plt.show()
