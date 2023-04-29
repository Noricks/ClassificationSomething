import json
from pathlib import Path
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# %%
###### Figure 1
# process data
base_path = Path("/mnt/emc01/zeyu/mlcw/")
info_kept = [1, 0.95, 0.9, 0.7, 0.5, 0.3]
num_dim = [1024, 160, 77, 12, 3, 1]
data_dict = {
    "test_acc": [],
    "best_test_acc": [],
    "val_acc": [],
    "best_train_acc": [],
}

# get testing results of different dims
index = 0
for i in info_kept:
    index = index + 1
    data = json.load(open(Path("/mnt/emc01/zeyu/mlcw/exp/")
                          .joinpath("mlp_info_{}_t/pre.json".format(index)),
                          "r"))
    for k in data_dict.keys():
        data_dict[k].append(data[k])

data_dict["train_acc"] = data_dict.pop("best_train_acc")
data_dict["information_kept"] = [1, 0.95, 0.9, 0.7, 0.5, 0.3]
data_frame = pd.DataFrame(data_dict)

# %%
# initialize figure and axis
fig, ax = plt.subplots(figsize=(7, 6))

# add custom labels to x axis
ticks = [0, 1, 2, 3, 4, 5]
ax.set_xticks(ticks)
ax.set_xticklabels(["1024(100%)", "160(15.62%)", "77(7.52%)", "12(1.17%)", "3(0.29%)", "1(0.1%)"])
sns.set_theme(style="ticks")

# get color palette
palette = sns.color_palette("rocket_r")

# add line
sns.lineplot(
    data=data_frame,
    # palette=palette,
    # height=5, aspect=.75, facet_kws=dict(sharex=False),
)

# set labels, ticks and legend
plt.xlabel("Num of Dim", fontsize=18)
plt.ylabel("Value", fontsize=18)
plt.legend(fontsize=15)
plt.xticks(fontsize=12)
plt.yticks(fontsize=15)
# %%
# plot figure
plt.tight_layout()
plt.show()

# %%
###### Figure 2

# process data
base_path = Path("/mnt/emc01/zeyu/mlcw/")
info_kept = [1, 0.99806282618556, 0.9932790914717771, 0.979589567927178, 0.9216661660466343]
num_dim = [1024, 716, 512, 307, 102]
data_dict = {
    "test_acc": [],
    "best_test_acc": [],
    "val_acc": [],
    "best_train_acc": [],
}

index = 0
for i in info_kept:
    index = index + 1
    data = json.load(open(Path("/mnt/emc01/zeyu/mlcw/exp/")
                          .joinpath("mlp_info_{}_t_p/pre.json".format(index)),
                          "r"))
    for k in data_dict.keys():
        data_dict[k].append(data[k])

data_dict["train_acc"] = data_dict.pop("best_train_acc")
data_dict["information_kept"] = info_kept
data_frame = pd.DataFrame(data_dict)

# %%
# initialize figure
fig, ax = plt.subplots(figsize=(7, 6))

# add custom labels to x axis
ticks = [0, 1, 2, 3, 4]
ax.set_xticks(ticks)
ax.set_xticklabels(["1024(100%)", "716(70%)", "512(50%)", "307(30%)", "102(10%)"])
sns.set_theme(style="ticks")

# set color
palette = sns.color_palette("rocket_r")

# add line
sns.lineplot(
    data=data_frame,
    # kind="line",
    # palette=palette,
    # height=5, aspect=.75, facet_kws=dict(sharex=False),
)

# set labels, ticks and legend
plt.xlabel("Num of Dim", fontsize=18)
plt.ylabel("Value", fontsize=18)
plt.legend(fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# plot figure
plt.tight_layout()
plt.show()

# %% Figure 3

# process data
base_path = Path("/mnt/emc01/zeyu/mlcw/exp/")
info_kept = [1, 0.99806282618556, 0.9932790914717771, 0.979589567927178, 0.9216661660466343]
num_dim = [1024, 716, 512, 307, 102]
data_dict = {
    # "test_f1": [],
    # "best_test_f1": [],
    "test_acc": [],
    # "best_train_acc": [],
}

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

index = 0
for c in classes:
    data_dict[c] = []

for i in info_kept:
    index = index + 1
    data = json.load(open(Path("/mnt/emc01/zeyu/mlcw/exp/")
                          .joinpath("mlp_info_{}_t_p/pre.json".format(index)),
                          "r"))
    data_dict["test_acc"].append(data["test_acc"])
    class_index = 0
    for c in classes:
        data_dict[c].append(data["test_f1"][class_index])
        class_index = class_index + 1

data_frame = pd.DataFrame(data_dict)

# %%
# initialize figure
fig, ax = plt.subplots(figsize=(9, 6))
ticks = [0, 1, 2, 3, 4]
ax.set_xticks(ticks)
ax.set_xticklabels(["1024(100%)", "716(70%)", "512(50%)", "307(30%)", "102(10%)"])
sns.set_theme(style="ticks")

# plot line
sns.lineplot(
    data=data_frame,
    # kind="line",
    # palette=palette,
    # height=5, aspect=.75, facet_kws=dict(sharex=False),
)

# set labels
plt.xlabel("Num of Dim", fontsize=18)
plt.ylabel("Value", fontsize=18)

# move legend out of the figure (right)
num1 = 1.01
num2 = 0.30
num3 = 3
num4 = 0
plt.legend(bbox_to_anchor=(num1, num2), loc=num3, borderaxespad=num4, fontsize=15)
plt.xticks(fontsize=15)
plt.yticks(fontsize=15)

# plot figure
plt.tight_layout()
plt.show()
