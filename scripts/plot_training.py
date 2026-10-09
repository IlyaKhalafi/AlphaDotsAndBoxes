"""Plot recorded losses; requires the optional analysis dependencies."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


def smooth(values: list[float], window: int = 10) -> list[float]:
    return [
        sum(values[max(0, i - window + 1) : i + 1]) / min(window, i + 1) for i in range(len(values))
    ]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, default=Path("docs/assets/training.svg"))
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": "#465246",
            "text.color": "#29362d",
            "xtick.color": "#62715f",
            "ytick.color": "#62715f",
        }
    )
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), constrained_layout=True)
    figure.set_facecolor("#f6f3ed")
    for ax, metric, title, color in zip(
        axes,
        ["policy_loss", "value_loss"],
        ["Search policy loss", "Outcome prediction loss"],
        ["#316a86", "#c95435"],
        strict=True,
    ):
        ax.set_facecolor("#fcfaf6")
        for index, path in enumerate(args.input):
            with path.open() as file:
                rows = list(csv.DictReader(file))
            iterations = [int(row["iteration"]) for row in rows]
            values = [float(row[metric]) for row in rows]
            ax.plot(iterations, values, color=color, alpha=0.18, linewidth=0.8)
            ax.plot(
                iterations,
                smooth(values),
                color=color,
                linewidth=2.2,
                label="10-iteration rolling mean" if index == 0 else None,
            )
            if index > 0:
                ax.axvspan(iterations[0] - 0.5, iterations[-1] + 0.5, color="#d6e2d4", alpha=0.3)
                ax.axvline(iterations[0] - 0.5, color="#879e7e", linewidth=1, linestyle="--")
        ax.set_title(title, loc="left", pad=16, fontsize=15)
        ax.set_xlabel("Training iteration")
        ax.set_ylabel("Loss (final minibatch each iteration)")
        ax.grid(axis="y", alpha=0.15)
        ax.legend(frameon=False, fontsize=9)
    figure.suptitle("Self-play learning · shaded region: solver-guided refinement", fontsize=13)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, metadata={"Date": None})
    plt.close(figure)


if __name__ == "__main__":
    main()
