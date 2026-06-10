if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np

from utils import load_dataset, problem


@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    # Separate target and features
    y_train = df_train["ViolentCrimesPerPop"].values
    X_train = df_train.drop("ViolentCrimesPerPop", axis=1).values

    y_test = df_test["ViolentCrimesPerPop"].values
    X_test = df_test.drop("ViolentCrimesPerPop", axis=1).values

    feature_names = list(df_train.drop("ViolentCrimesPerPop", axis=1).columns)

    # Compute lambda_max (smallest lambda for which solution is all zeros)
    y_mean = np.mean(y_train)
    lam_max = float(np.max(np.abs(2 * X_train.T @ (y_train - y_mean))))

    # Features to track for plot d
    features_d = ["agePct12t29", "pctWSocSec", "pctUrban", "agePct65up", "householdsize"]
    feat_idx = {f: feature_names.index(f) for f in features_d}

    # Storage for regularization path
    lambdas, nnzs, train_errors, test_errors = [], [], [], []
    coef_paths = {f: [] for f in features_d}

    weight, bias = None, None
    # Warm-start state to use for part f (lambda=30) computation
    prev_w_for_30, prev_b_for_30 = None, None

    lam = lam_max
    while lam >= 0.01:
        # Save state right before we pass through lambda=30
        if weight is not None and lam <= 60 and prev_w_for_30 is None:
            prev_w_for_30 = np.copy(weight)
            prev_b_for_30 = bias

        if weight is None:
            weight, bias = train(X_train, y_train, _lambda=lam)
        else:
            weight, bias = train(X_train, y_train, _lambda=lam,
                                 start_weight=weight, start_bias=bias)

        lambdas.append(lam)
        nnzs.append(int(np.sum(weight != 0)))

        train_pred = X_train @ weight + bias
        test_pred  = X_test  @ weight + bias
        train_errors.append(float(np.sum((train_pred - y_train) ** 2)))
        test_errors.append(float(np.sum((test_pred  - y_test)  ** 2)))

        for f in features_d:
            coef_paths[f].append(float(weight[feat_idx[f]]))

        lam /= 2

    # --- Part f: retrain at exactly lambda = 30 ---
    if prev_w_for_30 is not None:
        w30, b30 = train(X_train, y_train, _lambda=30,
                         start_weight=prev_w_for_30, start_bias=prev_b_for_30)
    else:
        w30, b30 = train(X_train, y_train, _lambda=30)

    max_idx = int(np.argmax(w30))
    min_idx = int(np.argmin(w30))
    print(f"[lambda=30] Most positive coef: {feature_names[max_idx]:30s} = {w30[max_idx]:+.4f}")
    print(f"[lambda=30] Most negative coef: {feature_names[min_idx]:30s} = {w30[min_idx]:+.4f}")

    # --- Plot c: number of non-zeros vs lambda ---
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, nnzs, "b-o", markersize=3)
    plt.xscale("log")
    plt.xlabel(r"$\lambda$ (log scale)")
    plt.ylabel(r"Number of non-zeros in $\hat{w}$")
    plt.title(r"Crime Data: Sparsity vs $\lambda$")
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("a6_plot_c_nonzeros.png", dpi=150, bbox_inches="tight")
    plt.clf()
    plt.close()

    # --- Plot d: coefficient paths for selected features ---
    plt.figure(figsize=(9, 5))
    for f in features_d:
        plt.plot(lambdas, coef_paths[f], label=f)
    plt.xscale("log")
    plt.axhline(0, color="k", linewidth=0.6, linestyle="--")
    plt.xlabel(r"$\lambda$ (log scale)")
    plt.ylabel("Coefficient value")
    plt.title("Regularization Paths for Selected Features")
    plt.legend(fontsize=9)
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("a6_plot_d_coef_paths.png", dpi=150, bbox_inches="tight")
    plt.clf()
    plt.close()

    # --- Plot e: squared errors vs lambda ---
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, train_errors, label="Training squared error")
    plt.plot(lambdas, test_errors,  label="Test squared error")
    plt.xscale("log")
    plt.xlabel(r"$\lambda$ (log scale)")
    plt.ylabel("Squared error")
    plt.title(r"Squared Error on Train and Test vs $\lambda$")
    plt.legend()
    plt.grid(True, which="both", alpha=0.3)
    plt.tight_layout()
    plt.savefig("a6_plot_e_errors.png", dpi=150, bbox_inches="tight")
    plt.clf()
    plt.close()


if __name__ == "__main__":
    main()
