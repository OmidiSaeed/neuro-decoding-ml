
from typing import Optional, Sequence, Dict, Any
import numpy as np
import matplotlib.pyplot as plt
from IPython.core.pylabtools import figsize


def plot_heatmaps_by_label(
        X,
        y,
        labels: Optional[Sequence] = None,
        cmap: str = "viridis",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
        figsize_per_plot=(6, 3)):

    uniq = np.unique(y) if labels is None else np.asarray(labels)
    n_labels = len(uniq)
    fig, axes = plt.subplots(n_labels, 1, figsize=(figsize_per_plot[0], figsize_per_plot[1] * n_labels))
    if n_labels == 1:
        axes = [axes]

    plotted_idx = {}
    for ax, label in zip(axes, uniq):
        idx = np.where(y == label)[0]
        if idx.size == 0:
            ax.set_title(f"{label} (no trials)", fontsize=10)
            ax.axis('off')
            plotted_idx[label] = np.array([], dtype=int)
            continue
        X_selected = X[idx, :]
        im = ax.imshow(X_selected, aspect='auto', cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_ylabel(f"Trials (n={len(idx)})", fontsize=9)
        ax.set_title(f"Label: {label}", fontsize=10)
        ax.set_xlabel("Features")
        # colorbar on the right of each subplot
        cbar = fig.colorbar(im, ax=ax)
        cbar.ax.tick_params(labelsize=8)
    plt.tight_layout()
    plt.show()
    return plotted_idx

def plot_accuracy_bars(results: Dict[str, Any]):

    names = list(results.keys())
    means = [results[n]['mean'] for n in names]
    stds = [results[n]['std'] for n in names]

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(names))
    ax.bar(x, means, yerr=stds)
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45)
    ax.set_ylabel("Accuracy")
    ax.set_xlabel("Model accuracies (mean ± std)")
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(results: Dict[str, Any], cmap="Blues", figsize=(12, 8)):

    names = list(results.keys())
    n = len(names)
    cols = min(3, n)
    rows = int(np.ceil(n / cols))

    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    for ax, name in zip(axes, names):
        mat = results[name]['confusion_matrix']
        labels = results[name]['labels']
        im = ax.imshow(mat, aspect='auto', cmap=cmap)
        ax.set_title(name)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right")
        ax.set_yticklabels(labels)
        ax.set_ylabel("True")
        ax.set_xlabel("Pred")
    plt.tight_layout()
    plt.show()



