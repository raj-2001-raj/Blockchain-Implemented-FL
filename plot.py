import matplotlib.pyplot as plt
import seaborn as sns

# ------------------------------
# 1) Define Final Results (Placeholders for missing data)
# ------------------------------
# Data structure: data_no_bc[dataset][config][metric]
# config can be "50r1e", "50r2e", "100r1e", "100r2e"
# metric can be: "time", "accuracy", "precision", "recall", "auc"
# All values are stored in PERCENT if they are accuracy/precision/recall/auc. Time remains in seconds.

data_no_bc = {
    "MNIST": {
        "50r1e": {"time": 1314.04, "accuracy": 98.35, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "50r2e": {"time": 2060.19, "accuracy": 98.56, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r1e": {"time": 2567.15, "accuracy": 98.60, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r2e": {"time": 5443.41, "accuracy": 98.80, "precision": 0.0, "recall": 0.0, "auc": 0.0},
    },
    "FashionMNIST": {
        "50r1e": {"time": 1342.85, "accuracy": 86.12, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "50r2e": {"time": 2014.01, "accuracy": 88.66, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r1e": {"time": 2974.96, "accuracy": 89.37, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r2e": {"time": 4842.19, "accuracy": 90.64, "precision": 0.0, "recall": 0.0, "auc": 0.0},
    },
    "CIFAR10": {
        # Provided data for "50r2e" only. We'll fill placeholders for others.
        "50r1e": {"time": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "50r2e": {
            "time": 2722.4880,
            # converting average 0.5196 → 51.96%
            "accuracy": 51.96,
            "precision": 54.28,
            "recall": 51.96,
            "auc": 73.31
        },
        "100r1e": {"time": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r2e": {"time": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0},
    },
}

data_bc = {
    "MNIST": {
        "50r1e": {"time": 1468.11, "accuracy": 98.25, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "50r2e": {"time": 2330.63, "accuracy": 98.46, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r1e": {"time": 3188.06, "accuracy": 98.63, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r2e": {"time": 5665.68, "accuracy": 98.80, "precision": 0.0, "recall": 0.0, "auc": 0.0},
    },
    "FashionMNIST": {
        "50r1e": {"time": 1493.37, "accuracy": 87.19, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "50r2e": {"time": 2493.06, "accuracy": 88.94, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r1e": {"time": 3128.00, "accuracy": 88.65, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r2e": {"time": 5069.45, "accuracy": 89.65, "precision": 0.0, "recall": 0.0, "auc": 0.0},
    },
    "CIFAR10": {
        "50r1e": {"time": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "50r2e": {
            "time": 2991.6748,
            # converting average 0.5158 → 51.58%
            "accuracy": 51.58,
            "precision": 53.26,
            "recall": 51.58,
            "auc": 73.10
        },
        "100r1e": {"time": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0},
        "100r2e": {"time": 0.0, "accuracy": 0.0, "precision": 0.0, "recall": 0.0, "auc": 0.0},
    },
}

# We have placeholders for missing data.
# Fill them with real values if you have them, or keep them at 0.0.

# ------------------------------
# 2) Compute Percentage Changes
# ------------------------------
# For each dataset & config, compute %change = 100*(bc - no_bc)/no_bc
# If no_bc is 0.0 (placeholder), we skip to avoid division by zero.
datasets = ["MNIST", "FashionMNIST", "CIFAR10"]
configs = ["50r1e", "50r2e", "100r1e", "100r2e"]
metrics = ["time", "accuracy", "precision", "recall", "auc"]

# We'll store the % changes in a structure parallel to data_no_bc
percent_change = {
    ds: {
        cf: {mt: 0.0 for mt in metrics}
        for cf in configs
    }
    for ds in datasets
}

for ds in datasets:
    for cf in configs:
        for mt in metrics:
            base_val = data_no_bc[ds][cf][mt]
            bc_val = data_bc[ds][cf][mt]
            if base_val != 0.0:
                change = 100.0 * (bc_val - base_val) / base_val
            else:
                # If we have a 0.0 placeholder, keep it 0.0 or skip
                change = 0.0
            percent_change[ds][cf][mt] = change

# ------------------------------
# 3) Create “Metric Changes” Bar Charts for each Dataset
# ------------------------------
# Each chart: x-axis has 4 configs, each with 5 bars (time, accuracy, precision, recall, auc).
# y-axis is %Change.

import numpy as np


def plot_metric_changes_bar(dataset):
    # e.g., "MNIST - Metric Changes"
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs))  # 4 configurations
    bar_width = 0.15

    # We'll offset each metric's bars
    offsets = np.linspace(-2 * bar_width, 2 * bar_width, len(metrics))

    for i, mt in enumerate(metrics):
        # Gather y-values for the 4 configs
        yvals = [percent_change[dataset][cf][mt] for cf in configs]
        ax.bar(x + offsets[i], yvals, width=bar_width, label=mt.capitalize())

    ax.set_xticks(x)
    ax.set_xticklabels(configs)
    ax.set_ylabel("% Change")
    ax.set_title(f"{dataset} - Metric Changes")
    ax.legend(loc='best')
    plt.tight_layout()
    plt.show()


# Plot for each dataset
for ds in datasets:
    plot_metric_changes_bar(ds)

# ------------------------------
# 4) Create Heatmaps
# ------------------------------
# We'll create 5 separate heatmaps: one for each metric
# The rows are the 3 datasets, the columns are the 4 configs.

import pandas as pd


def plot_metric_heatmap(metric):
    # Build a 3x4 data matrix
    mat = []
    for ds in datasets:
        row_vals = [percent_change[ds][cf][metric] for cf in configs]
        mat.append(row_vals)
    df = pd.DataFrame(mat, index=datasets, columns=configs)

    plt.figure(figsize=(6, 4))
    sns.heatmap(df, annot=True, cmap="RdYlGn", center=0, fmt=".1f")
    plt.title(f"{metric.capitalize()} Change (%)")
    plt.ylabel("Dataset")
    plt.xlabel("Config")
    plt.tight_layout()
    plt.show()


for mt in metrics:
    plot_metric_heatmap(mt)
