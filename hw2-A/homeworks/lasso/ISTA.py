from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from utils import problem


@problem.tag("hw2-A")
def step(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float, eta: float
) -> Tuple[np.ndarray, float]:
    """Single step in ISTA algorithm.
    It should update every entry in weight, and then return an updated version of weight along with calculated bias on input weight!

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Weight returned from the step before.
        bias (float): Bias returned from the step before.
        _lambda (float): Regularization constant. Determines when weight is updated to 0, and when to other values.
        eta (float): Step-size. Determines how far the ISTA iteration moves for each step.

    Returns:
        Tuple[np.ndarray, float]: Tuple with 2 entries. First represents updated weight vector, second represents bias.

    """
    # Compute residuals using current weight and bias
    r = X @ weight + bias - y  # shape (n,)

    # Update bias: gradient of SSE w.r.t. b is 2 * sum(r)
    bias_new = bias - 2 * eta * np.sum(r)

    # Compute gradient for all weights at once: a_k = X[:, k]^T @ r
    a = X.T @ r  # shape (d,)

    # Gradient step on weights
    w_step = weight - 2 * eta * a  # shape (d,)

    # Soft thresholding with threshold = 2 * eta * lambda
    threshold = 2 * eta * _lambda
    weight_new = np.sign(w_step) * np.maximum(np.abs(w_step) - threshold, 0.0)

    return weight_new, bias_new


@problem.tag("hw2-A")
def loss(
    X: np.ndarray, y: np.ndarray, weight: np.ndarray, bias: float, _lambda: float
) -> float:
    """L-1 (Lasso) regularized SSE loss.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        weight (np.ndarray): An (d,) array. Currently predicted weights.
        bias (float): Currently predicted bias.
        _lambda (float): Regularization constant. Should be used along with L1 norm of weight.

    Returns:
        float: value of the loss function
    """
    r = X @ weight + bias - y
    return float(np.sum(r ** 2) + _lambda * np.sum(np.abs(weight)))


@problem.tag("hw2-A", start_line=5)
def train(
    X: np.ndarray,
    y: np.ndarray,
    _lambda: float = 0.01,
    eta: float = 0.00001,
    convergence_delta: float = 1e-4,
    start_weight: np.ndarray = None,
    start_bias: float = None
) -> Tuple[np.ndarray, float]:
    """Trains a model and returns predicted weight and bias.

    Args:
        X (np.ndarray): An (n x d) matrix, with n observations each with d features.
        y (np.ndarray): An (n, ) array, with n observations of targets.
        _lambda (float): Regularization constant. Should be used for both step and loss.
        eta (float): Step size.
        convergence_delta (float, optional): Defines when to stop training algorithm.
            The smaller the value the longer algorithm will train.
            Defaults to 1e-4.
        start_weight (np.ndarray, optional): Weight for hot-starting model.
            If None, defaults to array of zeros. Defaults to None.
            It can be useful when testing for multiple values of lambda.
        start_bias (float, optional): Bias for hot-starting model.
            If None, defaults to zero. Defaults to None.
            It can be useful when testing for multiple values of lambda.

    Returns:
        Tuple[np.ndarray, float]: A tuple with first item being array of shape (d,) representing predicted weights,
            and second item being a float representing the bias.

    Note:
        - You will have to keep an old copy of weights for convergence criterion function.
            Please use `np.copy(...)` function, since numpy might sometimes copy by reference,
            instead of by value leading to bugs.
        - You will also have to keep an old copy of bias for convergence criterion function.
        - You might wonder why do we also return bias here, if we don't need it for this problem.
            There are two reasons for it:
                - Model is fully specified only with bias and weight.
                    Otherwise you would not be able to make predictions.
                    Training function that does not return a fully usable model is just weird.
                - You will use bias in next problem.
    """
    if start_weight is None:
        start_weight = np.zeros(X.shape[1])
        start_bias = 0
    old_w: Optional[np.ndarray] = None
    old_b: float = None

    weight = np.copy(start_weight)
    bias = float(start_bias)

    while True:
        old_w = np.copy(weight)
        old_b = bias
        weight, bias = step(X, y, weight, bias, _lambda, eta)
        if convergence_criterion(weight, old_w, bias, old_b, convergence_delta):
            break

    return weight, bias


@problem.tag("hw2-A")
def convergence_criterion(
    weight: np.ndarray, old_w: np.ndarray, bias: float, old_b: float, convergence_delta: float
) -> bool:
    """Function determining whether weight and bias has converged or not.
    It should calculate the maximum absolute change between weight and old_w vector, and compare it to convergence delta.
    It should also calculate the maximum absolute change between the bias and old_b, and compare it to convergence delta.

    Args:
        weight (np.ndarray): Weight from current iteration of gradient descent.
        old_w (np.ndarray): Weight from previous iteration of gradient descent.
        bias (float): Bias from current iteration of gradient descent.
        old_b (float): Bias from previous iteration of gradient descent.
        convergence_delta (float): Aggressiveness of the check.

    Returns:
        bool: False, if weight and bias has not converged yet. True otherwise.
    """
    max_w_change = np.max(np.abs(weight - old_w))
    b_change = np.abs(bias - old_b)
    return bool(max_w_change < convergence_delta and b_change < convergence_delta)


@problem.tag("hw2-A")
def main():
    """
    Use all of the functions above to make plots.
    """
    np.random.seed(546)
    n, d, k = 500, 1000, 100
    sigma = 1.0

    # True weights: w_j = j/k for j in {1,...,k}, 0 otherwise (1-indexed)
    w_true = np.zeros(d)
    for j in range(1, k + 1):
        w_true[j - 1] = j / k

    # Generate and standardize data
    X = np.random.randn(n, d)
    X = (X - X.mean(axis=0)) / X.std(axis=0)

    # Generate targets (b=0 in the model)
    eps = sigma * np.random.randn(n)
    y = X @ w_true + eps

    # Compute lambda_max: smallest lambda for which solution is all zeros
    y_mean = np.mean(y)
    lam_max = float(np.max(np.abs(2 * X.T @ (y - y_mean))))

    # --- Regularization path ---
    lambdas = []
    nnzs = []
    fdrs = []
    tprs = []

    true_nonzero = w_true != 0  # first k features

    lam = lam_max
    weight = None
    bias = None

    while True:
        lambdas.append(lam)

        if weight is None:
            weight, bias = train(X, y, _lambda=lam)
        else:
            weight, bias = train(X, y, _lambda=lam, start_weight=weight, start_bias=bias)

        nonzero_mask = weight != 0
        nnz = int(np.sum(nonzero_mask))
        nnzs.append(nnz)

        # FDR: number of incorrect nonzeros / total nonzeros
        incorrect_nz = nonzero_mask & ~true_nonzero
        fdr = float(np.sum(incorrect_nz) / max(nnz, 1))

        # TPR: number of correct nonzeros / k
        correct_nz = nonzero_mask & true_nonzero
        tpr = float(np.sum(correct_nz) / k)

        fdrs.append(fdr)
        tprs.append(tpr)

        # Stop when nearly all features are chosen
        if nnz >= d - 10:
            break

        lam /= 2

        if lam < 1e-4:  # safety stop
            break

    # --- Plot 1: number of non-zeros vs lambda (log x-axis) ---
    plt.figure(figsize=(8, 5))
    plt.plot(lambdas, nnzs, 'b-o', markersize=4)
    plt.xscale('log')
    plt.xlabel(r'$\lambda$ (log scale)')
    plt.ylabel(r'Number of non-zeros in $\hat{w}$')
    plt.title('Lasso Regularization Path: Sparsity vs $\\lambda$')
    plt.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot1_nonzeros_vs_lambda.png', dpi=150, bbox_inches='tight')
    plt.show()

    # --- Plot 2: FDR vs TPR ---
    plt.figure(figsize=(6, 6))
    plt.plot(fdrs, tprs, 'r-o', markersize=4)
    plt.xlabel('FDR (False Discovery Rate)')
    plt.ylabel('TPR (True Positive Rate)')
    plt.title('FDR vs TPR for Lasso Regularization Path')
    plt.xlim(-0.02, 1.02)
    plt.ylim(-0.02, 1.02)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('plot2_fdr_vs_tpr.png', dpi=150, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    main()
