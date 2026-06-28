from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from datetime import datetime
from .dataset import H5Dataset

import os
import json
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim


class GainNN(nn.Module):
    """Approximate the value of collaboration of gain of all coalitions
    in a VRP instance with 3 players."""

    def __init__(self):
        super().__init__()

        self.instance = nn.Sequential(
            nn.Linear(48, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.coalition = nn.Sequential(
            nn.Linear(3, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU()
        )

        self.head = nn.Sequential(
            nn.Linear(256 + 256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )

    def forward(self,
                instance : torch.Tensor,
                coalition : torch.Tensor):
        """"""
        instance = self.instance(instance)
        coalition = self.coalition(coalition)
        x = torch.cat(tensors = [instance, coalition], dim = 1)
        return self.head(x)

    @staticmethod
    def transform(instances: np.ndarray | torch.Tensor,
                  coalitions: np.ndarray | torch.Tensor,
                  values: np.ndarray | torch.Tensor,
                  device: torch.device):
        """Transform the dataset into the relevant format for the neural network.

        Args:
            instances: A numpy array of the instance matrices.
            coalitions: A numpy array of the coalition vectors.
            values: a numpy array containing the value of a coalition.
            device: A device to run the neural network on.

        Returns:
            Two torch tensor of the instances and coalitions matrices.
        """
        if instances.shape[-1] == 5:
            # remove the id column if it is present
            instances = instances[:, :, 1:]

        n_batches = instances.shape[0]
        n_nodes = instances.shape[1]
        n_param = instances.shape[2]
        n_coalitions = coalitions.shape[1]
        n_depots = coalitions.shape[2]

        if isinstance(instances, np.ndarray):
            instances = torch.from_numpy(instances).to(device)
        else:
            instances = instances.to(device)
        # (n_batches, n_nodes, n_cols) => (n_batches, 1, n_nodes, n_cols)
        instances = instances.unsqueeze(1)
        # (n_batches, 1, n_nodes, n_cols) => (n_batches, 8, n_nodes, n_cols)
        instances = instances.repeat(1, 8, 1, 1)
        # (n_batches, 8, n_nodes, n_cols) => (8 * n_batches, n_nodes * n_cols)
        rows = n_batches * n_coalitions
        cols = n_nodes * n_param
        instances = instances.view(rows, cols)
        """
        [[x y d v], [x y d v], ...] becomes [ x y d v x y d v ...]
        """

        if isinstance(coalitions, np.ndarray):
            coalitions = torch.from_numpy(coalitions).to(device)
        else:
            coalitions = coalitions.to(device)

        # (n_batches, n_coalitions, n_depots) => (n_batches * n_coalitions, n_depots)
        rows = n_batches * n_coalitions
        cols = n_depots
        coalitions = coalitions.view(rows, cols)

        if isinstance(values, np.ndarray):
            values = torch.from_numpy(values).to(device)
        else:
            values = values.to(device)

        # (n_batches, n_coalitions, n_depots) => (n_batches * n_coalitions, n_depots)
        rows = n_batches * n_coalitions
        cols = 1
        values = values.view(rows, cols)

        return instances, coalitions, values

def get_device():
    """Return the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

def count_trainable_params(model : nn.Module):
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

    for param in model.parameters():
        if param.grad is None:
            continue

        grad = param.grad
        grad_l2_norm = grad.norm(2).item()
        total_norm_squared += grad_l2_norm ** 2

        grad_max_abs = grad.abs().max().item()
        max_abs_grad = max(max_abs_grad, grad_max_abs)

    total_grad_norm = np.sqrt(total_norm_squared)

    return total_grad_norm, max_abs_grad

def make_loaders(dataset : H5Dataset,
                 batch_size : int = 128,
                 val_fraction : float = 0.1,
                 seed : int = 0):
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

    message = f"""
    Main dataset size: {n_main}.\n
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
    instances = torch.randn(size = (1, 12 * 4),
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
        instances, coalitions, values = model.transform(instances,
                                                      coalitions,
                                                      values,
                                                      device)

        output = model(instances, coalitions)
        loss = criterion(output, values)

        total_loss += loss.item() * instances.size(0)
        total_n += instances.size(0)

    return total_loss / total_n

def train_one_model(model : nn.Module,
                    train : DataLoader,
                    valid : DataLoader,
                    criterion : nn.Module,
                    optimizer,
                    device : torch.device,
                    n_epochs : int = 5,
                    log_every : int = 2,
                    label : str = "model"):
    """"""
    # TODO: add docstring to 'train_one_model' method.
    history = {
        "name": label,
        "train_loss": [],
        "eval_loss": [],
        "grad_norm": [],
        "grad_absmax": []
    }

    # FIXME: taken from Leo AI
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_dir = f"data/runs/{timestamp}"
    os.makedirs(log_dir, exist_ok=True)
    # until here
    writer = SummaryWriter(log_dir = log_dir)

    for epoch in range(n_epochs):
        model.train()

        total_loss = 0.0
        total_n = 0
        total_grad_norm = 0.0
        total_grad_absmax = 0.0
        n_batches = 0
        filename = "data/models/" + label + "-" + timestamp + ".pth"

        for batch_id, (instances, coalitions, values) in enumerate(train, start = 1):
            instances = instances.to(device)
            coalitions = coalitions.to(device)

            instances, coalitions, values = model.transform(instances, coalitions, values, device)

            output = model(instances, coalitions)
            loss = criterion(output, values)

            optimizer.zero_grad(set_to_none = True)
            loss.backward()

            grad_norm, grad_absmax = compute_grad_norms(model)
            optimizer.step()

            total_loss += loss.item() * values.size(0)
            total_n += values.size(0)
            total_grad_norm += grad_norm
            total_grad_absmax += grad_absmax
            n_batches += 1

            if batch_id %  log_every == 0:
                running_loss = total_loss / total_n
                message = f"""
                Epoch {epoch:02d}: 
                Batch {batch_id:04d}:
                loss = {running_loss:.4f}; grad_norm = {grad_norm:.2e}. 
                """; print(message)

        train_loss = total_loss / total_n
        avg_grad_norm = total_grad_norm / n_batches
        avg_grad_absmax = total_grad_absmax / n_batches
        eval_loss = evaluate(model, valid, criterion, device)

        writer.add_scalar(tag = "train/loss",
                          scalar_value = train_loss,
                          global_step = epoch)

        writer.add_scalar(tag = "train/grad_norm",
                          scalar_value = avg_grad_norm,
                          global_step = epoch)

        writer.add_scalar(tag = "train/grad_absmax",
                          scalar_value = avg_grad_absmax,
                          global_step = epoch)

        writer.add_scalar(tag = "eval/loss",
                          scalar_value = eval_loss,
                          global_step = epoch)

        if epoch %  log_every == 0:
            torch.save(model.state_dict(), filename)
            writer.flush()

        history["train_loss"].append(train_loss)
        history["eval_loss"].append(eval_loss)
        history["grad_norm"].append(avg_grad_norm)
        history["grad_absmax"].append(avg_grad_absmax)
        history["name"] = label + "-" + timestamp
        # TODO: add final output string and measure time.
    writer.close()
    return history

def train_model(model : nn.Module,
                train : DataLoader,
                valid : DataLoader,
                device : torch.device,
                n_epochs : int = 5,
                learning_rate : float = 0.1,
                weight_decay : float = 0.0,
                log_every : int = 100,
                label : str = "GainNN"):
    """"""
    # TODO: add docstring to 'train_model' method.

    model = model.to(device); print(model)
    print(f"Trainable parameters: {count_trainable_params(model)}")

    test_run(model, device)
    print("Test run: ok")

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),
                           lr = learning_rate,
                           weight_decay = weight_decay)

    history = train_one_model(model = model,
                              train = train,
                              valid = valid,
                              criterion = criterion,
                              optimizer = optimizer,
                              device = device,
                              n_epochs = n_epochs,
                              log_every = log_every,
                              label = label)

    return model, history

if __name__ == "__main__":

    seeds = [0, 1, 2]
    model = GainNN()

    for run, seed in enumerate(seeds, start = 1):

        torch.manual_seed(seed)

        n_epochs = 100
        batch_size = 64
        learning_rate = 1e-4
        val_fraction = 0.1
        weight_decay = 0.01

        device = get_device(); print("device:", str(device))
        dataset = H5Dataset(path = "data/instances.h5",
                            features = ['instance', 'coalitions'],
                            target = 'char_function')

        train, valid = make_loaders(dataset,
                                    batch_size = batch_size,
                                    val_fraction = val_fraction,
                                    seed = seed)

        model, history = train_model(model = model,
                                     train = train,
                                     valid = valid,
                                     device = device,
                                     n_epochs = n_epochs,
                                     learning_rate = learning_rate,
                                     weight_decay = weight_decay)

        history["n_epochs"] = n_epochs
        history["device"] = str(device)
        history["learning_rate"] = learning_rate
        history["weight_decay"] = weight_decay
        history["data_size"] = len(dataset)
        history["run"] = run

        """
        for key, values in history.items():
            print(key, values)
        """

        filename = "data/logs/" + history["name"] + ".json"
        with open(filename, 'w') as file:
            json.dump(history, file, indent = 4)