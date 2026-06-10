if __name__ == "__main__":
    from layers import LinearLayer, ReLULayer, SigmoidLayer
    from losses import MSELossLayer
    from optimizers import SGDOptimizer
    from train import plot_model_guesses, train
else:
    from .layers import LinearLayer, ReLULayer, SigmoidLayer
    from .optimizers import SGDOptimizer
    from .losses import MSELossLayer
    from .train import plot_model_guesses, train


import os
from typing import Any, Dict

import numpy as np

_HW3_DIR = os.path.join(os.path.dirname(__file__), "..", "..")
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from utils import load_dataset, problem

RNG = torch.Generator()
RNG.manual_seed(446)


@problem.tag("hw3-A")
def accuracy_score(model: nn.Module, dataloader: DataLoader) -> float:
    """Calculates accuracy of model on dataloader. Returns it as a fraction.

    Args:
        model (nn.Module): Model to evaluate.
        dataloader (DataLoader): Dataloader for MSE.
            Each example is a tuple consiting of (observation, target).
            Observation is a 2-d vector of floats.
            Target is also a 2-d vector of floats, but specifically with one being 1.0, while other is 0.0.
            Index of 1.0 in target corresponds to the true class.

    Returns:
        float: Vanilla python float resprenting accuracy of the model on given dataset/dataloader.
            In range [0, 1].

    Note:
        - For a single-element tensor you can use .item() to cast it to a float.
        - This is similar to CrossEntropy accuracy_score function,
            but there will be differences due to slightly different targets in dataloaders.
    """
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for x_batch, y_batch in dataloader:
            y_pred = model(x_batch)
            preds = torch.argmax(y_pred, dim=1)
            true_labels = torch.argmax(y_batch, dim=1)
            correct += (preds == true_labels).sum().item()
            total += len(y_batch)
    return correct / total


@problem.tag("hw3-A")
def mse_parameter_search(
    dataset_train: TensorDataset, dataset_val: TensorDataset
) -> Dict[str, Any]:
    """
    Main subroutine of the MSE problem.
    It's goal is to perform a search over hyperparameters, and return a dictionary containing training history of models, as well as models themselves.

    Models to check (please try them in this order):
        - Linear Regression Model
        - Network with one hidden layer of size 2 and sigmoid activation function after the hidden layer
        - Network with one hidden layer of size 2 and ReLU activation function after the hidden layer
        - Network with two hidden layers (each with size 2)
            and Sigmoid, ReLU activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU, Sigmoid activation function after corresponding hidden layers
        - Network with two hidden layers (each with size 2)
            and ReLU activation function after each hidden layer

    Notes:
        - When choosing the number of epochs, consider effect of other hyperparameters on it.
            For example as learning rate gets smaller you will need more epochs to converge.

    Args:
        dataset_train (TensorDataset): Training dataset.
        dataset_val (TensorDataset): Validation dataset.

    Returns:
        Dict[str, Any]: Dictionary/Map containing history of training of all models.
            You are free to employ any structure of this dictionary, but we suggest the following:
            {
                name_of_model: {
                    "train": Per epoch losses of model on train set,
                    "val": Per epoch losses of model on validation set,
                    "model": Actual PyTorch model (type: nn.Module),
                }
            }
    """
    EPOCHS = 200
    LR = 0.1
    BATCH = 32

    train_loader = DataLoader(dataset_train, batch_size=BATCH, shuffle=True, generator=RNG)
    val_loader = DataLoader(dataset_val, batch_size=BATCH, shuffle=False)

    criterion = MSELossLayer()

    # MSE models do NOT end with Softmax; output is (n, 2) matching one-hot targets
    architectures = [
        ("Linear",
         nn.Sequential(LinearLayer(2, 2, RNG))),
        ("1-hidden-Sigmoid",
         nn.Sequential(LinearLayer(2, 2, RNG), SigmoidLayer(), LinearLayer(2, 2, RNG))),
        ("1-hidden-ReLU",
         nn.Sequential(LinearLayer(2, 2, RNG), ReLULayer(), LinearLayer(2, 2, RNG))),
        ("2-hidden-Sigmoid-ReLU",
         nn.Sequential(LinearLayer(2, 2, RNG), SigmoidLayer(), LinearLayer(2, 2, RNG), ReLULayer(), LinearLayer(2, 2, RNG))),
        ("2-hidden-ReLU-Sigmoid",
         nn.Sequential(LinearLayer(2, 2, RNG), ReLULayer(), LinearLayer(2, 2, RNG), SigmoidLayer(), LinearLayer(2, 2, RNG))),
        ("2-hidden-ReLU-ReLU",
         nn.Sequential(LinearLayer(2, 2, RNG), ReLULayer(), LinearLayer(2, 2, RNG), ReLULayer(), LinearLayer(2, 2, RNG))),
    ]

    configs: Dict[str, Any] = {}
    for name, model in architectures:
        optimizer = SGDOptimizer(model.parameters(), lr=LR)
        history = train(train_loader, model, criterion, optimizer, val_loader=val_loader, epochs=EPOCHS)
        configs[name] = {"train": history["train"], "val": history["val"], "model": model}

    return configs


@problem.tag("hw3-A", start_line=11)
def main():
    """
    Main function of the MSE problem.
    It should:
        1. Call mse_parameter_search routine and get dictionary for each model architecture/configuration.
        2. Plot Train and Validation losses for each model all on single plot (it should be 12 lines total).
            x-axis should be epochs, y-axis should me MSE loss, REMEMBER to add legend
        3. Choose and report the best model configuration based on validation losses.
            In particular you should choose a model that achieved the lowest validation loss at ANY point during the training.
        4. Plot best model guesses on test set (using plot_model_guesses function from train file)
        5. Report accuracy of the model on test set.

    Starter code loads dataset, converts it into PyTorch Datasets, and those into DataLoaders.
    You should use these dataloaders, for the best experience with PyTorch.
    """
    (x, y), (x_val, y_val), (x_test, y_test) = load_dataset("xor")

    dataset_train = TensorDataset(torch.from_numpy(x).float(), torch.from_numpy(to_one_hot(y)))
    dataset_val = TensorDataset(
        torch.from_numpy(x_val).float(), torch.from_numpy(to_one_hot(y_val))
    )
    dataset_test = TensorDataset(
        torch.from_numpy(x_test).float(), torch.from_numpy(to_one_hot(y_test))
    )

    mse_configs = mse_parameter_search(dataset_train, dataset_val)

    # Plot 12 lines (train + val for 6 models)
    plt.figure(figsize=(12, 7))
    colors = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown"]
    for (name, data), color in zip(mse_configs.items(), colors):
        epochs = range(1, len(data["train"]) + 1)
        plt.plot(epochs, data["train"], color=color, label=f"{name} train")
        plt.plot(epochs, data["val"], color=color, linestyle="--", label=f"{name} val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("MSE Loss — Training and Validation (all 6 architectures)")
    plt.legend(fontsize=7, ncol=2)
    plt.tight_layout()
    plt.savefig(os.path.join(_HW3_DIR, "a4_mse_loss.png"), dpi=150)
    plt.close()
    print("Saved a4_mse_loss.png")

    # Best model by lowest validation loss at any epoch
    best_name = min(mse_configs.keys(), key=lambda n: min(mse_configs[n]["val"]))
    best_model = mse_configs[best_name]["model"]
    best_val_loss = min(mse_configs[best_name]["val"])
    print(f"Best MSE model: {best_name}, best val loss: {best_val_loss:.6f}")

    test_loader = DataLoader(dataset_test, batch_size=len(dataset_test))
    test_acc = accuracy_score(best_model, test_loader)
    print(f"MSE best model test accuracy: {test_acc:.4f}")

    # Scatter plot of predictions on test set
    fig = plt.figure(figsize=(7, 6))
    plot_model_guesses(test_loader, best_model, title=f"MSE Best: {best_name}\nTest Accuracy = {test_acc:.4f}")
    fig.savefig(os.path.join(_HW3_DIR, "a4_mse_predictions.png"), dpi=150)
    plt.close("all")
    print("Saved a4_mse_predictions.png")


def to_one_hot(a: np.ndarray) -> np.ndarray:
    """Helper function. Converts data from categorical to one-hot encoded.

    Args:
        a (np.ndarray): Input array of integers with shape (n,).

    Returns:
        np.ndarray: Array with shape (n, c), where c is maximal element of a.
            Each element of a, has a corresponding one-hot encoded vector of length c.
    """
    r = np.zeros((len(a), 2))
    r[np.arange(len(a)), a] = 1
    return r


if __name__ == "__main__":
    main()
