import os
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

LossFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]

_HW3_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


@problem.tag("hw3-A")
def J_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the joint negative log-likelihood loss.

    This is the sum of per-example negative log-likelihoods for the
    multinomial logistic regression model.

    Args:
        logits: FloatTensor of shape (n, k). Raw class scores.
        y: LongTensor of shape (n,). Class labels.

    Returns:
        A scalar tensor containing the loss.
    """
    return F.cross_entropy(logits, y, reduction="sum")


@problem.tag("hw3-A")
def L_loss(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Compute the standard average cross-entropy loss.

    This is the mean per-example negative log-likelihood. You may use
    torch.nn.functional.cross_entropy or write the softmax/log expression
    directly.

    Args:
        logits: FloatTensor of shape (n, k). Raw class scores.
        y: LongTensor of shape (n,). Class labels.

    Returns:
        A scalar tensor containing the loss.
    """
    return F.cross_entropy(logits, y, reduction="mean")


@problem.tag("hw3-A")
def accuracy(logits: torch.Tensor, y: torch.Tensor) -> float:
    """
    Compute classification accuracy.

    Args:
        logits: FloatTensor of shape (n, k). Raw class scores.
        y: LongTensor of shape (n,). Class labels.

    Returns:
        Fraction of examples classified correctly.
    """
    return (logits.argmax(dim=1) == y).float().mean().item()


@problem.tag("hw3-A")
def train(
    x_train: torch.Tensor,
    y_train: torch.Tensor,
    loss_fn: LossFunction,
    *,
    learning_rate: float = 0.1,
    epochs: int = 50,
    batch_size: int = 256,
) -> Tuple[torch.Tensor, List[float]]:
    """
    Train a multinomial logistic regression model on MNIST.

    Args:
        x_train: FloatTensor of shape (n, d).
        y_train: LongTensor of shape (n,).
        loss_fn: Either J_loss or L_loss.
        learning_rate: Step size for gradient descent.
        epochs: Number of training epochs.
        batch_size: Mini-batch size.

    Returns:
        A tuple (W, losses), where W has shape (k, d) and losses stores one
        average training loss per epoch.
    """
    d = x_train.shape[1]
    k = int(y_train.max().item()) + 1

    # W stored as (d, k): logits = x @ W  →  shape (n, k)
    W = torch.zeros(d, k, requires_grad=True)
    optimizer = torch.optim.SGD([W], lr=learning_rate)

    dataset = TensorDataset(x_train, y_train)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    epoch_losses: List[float] = []
    for _ in range(epochs):
        running_loss = 0.0
        for x_batch, y_batch in loader:
            optimizer.zero_grad()
            logits = x_batch @ W
            loss = loss_fn(logits, y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        epoch_losses.append(running_loss / len(loader))

    # Return W as (k, d) per docstring convention
    return W.detach().T.clone(), epoch_losses


@problem.tag("hw3-A", start_line=5)
def main() -> Dict[str, Dict[str, float]]:
    """
    Train multinomial logistic regression models with J_loss and L_loss.

    For each loss, this function should:
        1. Load MNIST.
        2. Train a multinomial logistic regression model.
        3. Plot training loss vs. epoch.
        4. Report training and test accuracy.

    Returns:
        A dictionary mapping loss names to accuracy summaries.
    """
    (x_train, y_train), (x_test, y_test) = load_dataset("mnist")
    x_train = torch.from_numpy(x_train).float()
    y_train = torch.from_numpy(y_train).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    results: Dict[str, Dict[str, float]] = {}

    for loss_name, loss_fn in [("J_loss", J_loss), ("L_loss", L_loss)]:
        print(f"Training with {loss_name}...")
        W, epoch_losses = train(x_train, y_train, loss_fn)

        # W is (k, d), so logits = x @ W.T
        with torch.no_grad():
            train_logits = x_train @ W.T
            test_logits = x_test @ W.T
            train_acc = accuracy(train_logits, y_train)
            test_acc = accuracy(test_logits, y_test)

        results[loss_name] = {"train": train_acc, "test": test_acc}
        print(f"  train accuracy: {train_acc:.4f}  test accuracy: {test_acc:.4f}")

        # Plot training loss vs epoch
        plt.figure(figsize=(8, 5))
        plt.plot(range(1, len(epoch_losses) + 1), epoch_losses)
        plt.xlabel("Epoch")
        plt.ylabel(loss_name)
        plt.title(f"Training Loss ({loss_name}) vs. Epoch — MNIST Logistic Regression")
        plt.tight_layout()
        tag = "j" if loss_name == "J_loss" else "l"
        out_path = os.path.join(_HW3_DIR, f"a6_{tag}_loss.png")
        plt.savefig(out_path, dpi=150)
        plt.close()
        print(f"  Saved {out_path}")

    return results


if __name__ == "__main__":
    main()
