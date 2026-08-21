from torch.utils.data import DataLoader
from .dataset import H5Dataset
from .gain import transform
import torch.nn as nn
import numpy as np
import torch


def get_device():
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def count_trainable_params(model: nn.Module):
    """Return the number of trainable parameters in the model."""
    total = 0

    for param in model.parameters():
        if param.requires_grad:
            total += param.numel()

    return total

def compute_grad_norms(model : nn.Module):
    """"""
    #TODO: Add docstring to 'compute_grad_norms'.
    total_norm_squared = 0.0
    max_abs_grad = 0.0

    total = 0
    total_none = 0

    for param in model.parameters():
        total += param.numel()
        if param.grad is None:
            total_none += param.numel()
            continue

        grad = param.grad
        grad_l2_norm = grad.norm(2).item()
        total_norm_squared += grad_l2_norm ** 2

        grad_max_abs = grad.abs().max().item()
        max_abs_grad = max(max_abs_grad, grad_max_abs)

    total_grad_norm = np.sqrt(total_norm_squared)
    if total_none > 0:
        print(f"Params (total, grad = None): ({total}, {total_none}).")

    return total_grad_norm, max_abs_grad

def make_loaders(dataset : H5Dataset,
                 batch_size : int = 128,
                 val_fraction : float = 0.1,
                 seed : int = 0,
                 verbose: int = 0):
    """Return DataLoader of main and validation sets.

    Args:
        dataset: a H5Dataset from which to read the data.
        batch_size: a integer defining the size of the batches loaded
        out of the DataLoader.
        val_fraction: a float defining the share of the dataset
        allocated to model validation.
        seed: a integer for deterministic randomness.

    Returns:
        A DataLoader for the main dataset and the validation set.
    """
    generator = torch.Generator().manual_seed(seed)

    # create the training and validation sets
    n_total = len(dataset)
    n_valid = int(val_fraction * n_total)
    n_main = n_total - n_valid

    main, validation = torch.utils.data.random_split(
        dataset = dataset,
        lengths = [n_main, n_valid],
        generator = generator)

    main_loader = DataLoader(main,
                             batch_size = batch_size,
                             shuffle = True)

    val_loader = DataLoader(validation,
                            batch_size = batch_size,
                            shuffle = False)

    if verbose != 0:
        message = f"""
        Main dataset size: {n_main}.
        Validation set size: {n_valid}.
        """; print(message)

    return main_loader, val_loader

def test_run(model : nn.Module,
             device : torch.device):
    """Run one forward and backward pass to catch problems.

    The method runs one loop to catch shape mistakes or missing gradients.
    Args:
        model: a pytorch model.
        device: a pytorch device.

    Raises:
        A runtime error if the gradient of a parameter is missing.
    """
    model.train()

    # generate data with the expected shape for the GainNN class
    instances = torch.randn(size = (1, 10 * 4),
                            device = device).float()
    coalitions = torch.randint(low = 0,
                               high = 2,
                               size = (1, 3),
                               device = device).float()
    char_value = torch.randn(size = (1, 1),
                             device = device).float()

    output = model(instances, coalitions)
    assert output.shape == (1, 1), (
        f"Expected output shape (1, 1), got {tuple(output.shape)}."
    )

    criterion = nn.MSELoss()
    loss = criterion(output, char_value)
    loss.backward()

    # capture parameters with missing gradients
    missing = []

    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is None:
            missing.append(name)

    if missing:
        raise RuntimeError(f"Missing gradients for:\n {missing}.")

    model.zero_grad(set_to_none = True)

@torch.no_grad()
def evaluate(model : nn.Module,
             loader : DataLoader,
             criterion : nn.Module,
             device : torch.device):
    """Return the performance of the model on the validation/test set.

    Args:
        model: a nn.Module neural network.
        loader: a DataLoader.
        criterion: a nn.Module loss function.
        device: a device recognized by PyTorch.

    Returns:
        The average loss on the validation/test set.
    """
    model.eval()

    total_loss = 0.0
    total_n = 0

    for instances, coalitions, values in loader:
        instances, coalitions, values = transform(instances,
                                                coalitions,
                                                        values,
                                                        device)



        output = model(instances, coalitions)
        loss = criterion(output, values)

        total_loss += loss.item() * instances.size(0)
        total_n += instances.size(0)

    return total_loss / total_n