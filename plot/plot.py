#!/usr/bin/env python3
"""
For plotting the comparison figure.
"""
from typing import List

import matplotlib.pyplot as plt


def main() -> None:
    # todo: get scores.
    old_scores: List[float] = []
    new_scores_yahoo: List[float] = []
    new_scores_simple: List[float] = []

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=(5.0, 14.0))

    for ax in axes:
        ax.plot([-0.04, 1.04], [-0.04, 1.04], c="k", lw=1)

    axes[0].scatter(old_scores, new_scores_yahoo)
    axes[0].set_xlabel("TF1")
    axes[0].set_ylabel("TF2 (YAHOO pre.)")
    axes[0].set_aspect("equal")

    axes[1].scatter(old_scores, new_scores_simple)
    axes[1].set_xlabel("TF1")
    axes[1].set_ylabel("TF2 (SIMPLE pre.)")
    axes[1].set_aspect("equal")

    axes[2].scatter(new_scores_yahoo, new_scores_simple)
    axes[2].set_xlabel("TF2 (YAHOO pre.)")
    axes[2].set_ylabel("TF2 (SIMPLE pre.)")
    axes[2].set_aspect("equal")

    for ax in axes:
        ax.set(xlim=(-0.04, 1.04), ylim=(-0.04, 1.04))

    fig.set_tight_layout(True)
    plt.savefig("nsfw_probabilities_comparison.png")


if __name__ == "__main__":
    main()
