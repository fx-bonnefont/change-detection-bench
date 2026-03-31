import pandas as pd
import matplotlib.pyplot as plt
from src.config import EXTRACTED_FEATURES_DIR


def plot_masks_stats_compare(df):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    styles = {
        ("train", 0): ("royalblue",  "-",  "Train 2018"),
        ("train", 1): ("darkorange", "-",  "Train 2019"),
        ("valid", 0): ("royalblue",  "--", "Valid 2018"),
        ("valid", 1): ("darkorange", "--", "Valid 2019"),
    }

    for ax, ch in zip(axes, [0, 1]):
        for (split, c), (color, ls, label) in styles.items():
            if c != ch:
                continue
            row = df[(df["split"] == split) & (df["channel"] == ch)].sort_values("value")
            pct = row["count"] / row["count"].sum() * 100
            ax.plot(row["value"].values, pct.values, color=color, linestyle=ls, marker="o", label=label)
        ax.set_title(f"Canal {ch}")
        ax.set_xlabel("Classe")
        ax.set_xticks(range(10))
        ax.grid(axis="y", linestyle="--", alpha=0.6)
        ax.legend()

    axes[0].set_ylabel("% pixels")
    plt.suptitle("Distribution des classes — train vs valid", fontsize=14)
    plt.tight_layout()
    plt.show()


def main():
    path = EXTRACTED_FEATURES_DIR / "masks" / "mask_stats.csv"
    df = pd.read_csv(path)

    plot_masks_stats_compare(df)


if __name__ == "__main__":
    main()