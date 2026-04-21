"""
Generates side-by-side plots showing the effect of regularization on a degree-8
polynomial fit to the polyreg dataset.
"""
import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset

if __name__ == "__main__":
    from polyreg import PolynomialRegression  # type: ignore
else:
    from .polyreg import PolynomialRegression

if __name__ == "__main__":
    allData = load_dataset("polyreg")
    X = allData[:, [0]]
    y = allData[:, [1]]

    xpoints = np.linspace(np.min(X), np.max(X), 200).reshape(-1, 1)

    configs = [
        (0,   "Before: $\\lambda = 0$ (no regularization)"),
        (100, "After: $\\lambda = 100$ (high regularization)"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Effect of Increasing Regularization (degree = 8)", fontsize=14)

    for ax, (reg_lambda, title) in zip(axes, configs):
        model = PolynomialRegression(degree=8, reg_lambda=reg_lambda)
        model.fit(X, y)
        ypoints = model.predict(xpoints)

        ax.plot(X, y, "rx", markersize=8, label="Data")
        ax.plot(xpoints, ypoints, "b-", linewidth=2, label=f"$\\lambda={reg_lambda}$")
        ax.set_title(title)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.legend()
        ax.grid(True, linestyle="dotted")

    plt.tight_layout()
    plt.savefig("reg_effect.png", dpi=150, bbox_inches="tight")
    print("Saved reg_effect.png")
    plt.show()
