# When taking sqrt for initialization you might want to use math package,
# since torch.sqrt requires a tensor, and math.sqrt is ok with integer
import math
from typing import List

import matplotlib.pyplot as plt
import torch
from torch.distributions import Uniform
from torch.nn import Module
from torch.nn.functional import cross_entropy, relu
from torch.nn.parameter import Parameter
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem


class F1(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h: int, d: int, k: int):
        """Create a F1 model as described in pdf.

        Args:
            h (int): Hidden dimension.
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        # W0: (d, h), b0: (h,)  — fan_in = d
        alpha0 = 1.0 / math.sqrt(d)
        self.W0 = Parameter(Uniform(-alpha0, alpha0).sample((d, h)))
        self.b0 = Parameter(Uniform(-alpha0, alpha0).sample((h,)))
        # W1: (h, k), b1: (k,)  — fan_in = h
        alpha1 = 1.0 / math.sqrt(h)
        self.W1 = Parameter(Uniform(-alpha1, alpha1).sample((h, k)))
        self.b1 = Parameter(Uniform(-alpha1, alpha1).sample((k,)))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F1 model.

        It should perform operation:
        W_1(sigma(W_0*x + b_0)) + b_1

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        return relu(x @ self.W0 + self.b0) @ self.W1 + self.b1


class F2(Module):
    @problem.tag("hw3-A", start_line=1)
    def __init__(self, h0: int, h1: int, d: int, k: int):
        """Create a F2 model as described in pdf.

        Args:
            h0 (int): First hidden dimension (between first and second layer).
            h1 (int): Second hidden dimension (between second and third layer).
            d (int): Input dimension/number of features.
            k (int): Output dimension/number of classes.
        """
        super().__init__()
        alpha0 = 1.0 / math.sqrt(d)
        self.W0 = Parameter(Uniform(-alpha0, alpha0).sample((d, h0)))
        self.b0 = Parameter(Uniform(-alpha0, alpha0).sample((h0,)))

        alpha1 = 1.0 / math.sqrt(h0)
        self.W1 = Parameter(Uniform(-alpha1, alpha1).sample((h0, h1)))
        self.b1 = Parameter(Uniform(-alpha1, alpha1).sample((h1,)))

        alpha2 = 1.0 / math.sqrt(h1)
        self.W2 = Parameter(Uniform(-alpha2, alpha2).sample((h1, k)))
        self.b2 = Parameter(Uniform(-alpha2, alpha2).sample((k,)))

    @problem.tag("hw3-A")
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Pass input through F2 model.

        It should perform operation:
        W_2(sigma(W_1(sigma(W_0*x + b_0)) + b_1) + b_2)

        Note that in this coding assignment, we use the same convention as previous
        assignments where a linear module is of the form xW + b. This differs from the
        general forward pass operation defined above, which assumes the form Wx + b.
        When implementing the forward pass, make sure that the correct matrices and
        transpositions are used.

        Args:
            x (torch.Tensor): FloatTensor of shape (n, d). Input data.

        Returns:
            torch.Tensor: FloatTensor of shape (n, k). Prediction.
        """
        h = relu(x @ self.W0 + self.b0)
        h = relu(h @ self.W1 + self.b1)
        return h @ self.W2 + self.b2


@problem.tag("hw3-A")
def train(model: Module, optimizer: Adam, train_loader: DataLoader) -> List[float]:
    """
    Train a model until it reaches 99% accuracy on train set, and return list of training crossentropy losses for each epochs.

    Args:
        model (Module): Model to train. Either F1, or F2 in this problem.
        optimizer (Adam): Optimizer that will adjust parameters of the model.
        train_loader (DataLoader): DataLoader with training data.
            You can iterate over it like a list, and it will produce tuples (x, y),
            where x is FloatTensor of shape (n, d) and y is LongTensor of shape (n,).
            Note that y contains the classes as integers.

    Returns:
        List[float]: List containing average loss for each epoch.
    """
    model.train()
    epoch_losses: List[float] = []
    while True:
        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            logits = model(x_batch)
            loss = cross_entropy(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(y_batch)
            preds = logits.argmax(dim=1)
            total_correct += (preds == y_batch).sum().item()
            total_samples += len(y_batch)
        avg_loss = total_loss / total_samples
        accuracy = total_correct / total_samples
        epoch_losses.append(avg_loss)
        if accuracy >= 0.99:
            break
    return epoch_losses


@problem.tag("hw3-A", start_line=5)
def main():
    """
    Main function of this problem.
    For both F1 and F2 models it should:
        1. Train a model
        2. Plot per epoch losses
        3. Report accuracy and loss on test set
        4. Report total number of parameters for each network

    Note that we provided you with code that loads MNIST and changes x's and y's to correct type of tensors.
    We strongly advise that you use torch functionality such as datasets, but as mentioned in the pdf you cannot use anything from torch.nn other than what is imported here.
    """
    (x, y), (x_test, y_test) = load_dataset("mnist")
    x = torch.from_numpy(x).float()
    y = torch.from_numpy(y).long()
    x_test = torch.from_numpy(x_test).float()
    y_test = torch.from_numpy(y_test).long()

    d, k = 784, 10
    batch_size = 256

    train_loader = DataLoader(TensorDataset(x, y), batch_size=batch_size, shuffle=True)

    # ── F1: wide shallow network (h = 64) ──────────────────────────────
    print("Training F1 (h=64)...")
    f1 = F1(h=64, d=d, k=k)
    opt1 = Adam(f1.parameters(), lr=1e-3)
    losses_f1 = train(f1, opt1, train_loader)

    # test evaluation
    f1.eval()
    with torch.no_grad():
        logits_f1 = f1(x_test)
        test_loss_f1 = cross_entropy(logits_f1, y_test).item()
        test_acc_f1 = (logits_f1.argmax(1) == y_test).float().mean().item()

    params_f1 = sum(p.numel() for p in f1.parameters())
    print(f"F1 — test accuracy: {test_acc_f1:.4f}, test loss: {test_loss_f1:.4f}, params: {params_f1}")

    # ── F2: narrow deeper network (h0=h1=32) ───────────────────────────
    print("Training F2 (h0=h1=32)...")
    f2 = F2(h0=32, h1=32, d=d, k=k)
    opt2 = Adam(f2.parameters(), lr=1e-3)
    losses_f2 = train(f2, opt2, train_loader)

    f2.eval()
    with torch.no_grad():
        logits_f2 = f2(x_test)
        test_loss_f2 = cross_entropy(logits_f2, y_test).item()
        test_acc_f2 = (logits_f2.argmax(1) == y_test).float().mean().item()

    params_f2 = sum(p.numel() for p in f2.parameters())
    print(f"F2 — test accuracy: {test_acc_f2:.4f}, test loss: {test_loss_f2:.4f}, params: {params_f2}")

    # ── Plot training loss curves ───────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(losses_f1) + 1), losses_f1, label=f"F1 (h=64, {params_f1} params)")
    plt.plot(range(1, len(losses_f2) + 1), losses_f2, label=f"F2 (h0=h1=32, {params_f2} params)")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training Loss vs. Epoch — MNIST Neural Networks")
    plt.legend()
    plt.tight_layout()
    plt.savefig("a5_training_loss.png", dpi=150)
    plt.close()
    print("Saved a5_training_loss.png")


if __name__ == "__main__":
    main()
