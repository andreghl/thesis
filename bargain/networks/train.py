from .log import Logger
from tqdm import tqdm
from .utils import *


import os
import torch
import torch.nn as nn
import torch.optim as optim


def train_one_model(model : nn.Module,
                    train : DataLoader,
                    valid : DataLoader,
                    criterion : nn.Module,
                    optimizer,
                    device : torch.device,
                    tensorboard : Logger,
                    seed: int = 0,
                    n_epochs : int = 5,
                    log_every : int = 1,
                    label : str = "model",
                    lr: float | int = -1,
                    wd: float | int = -1,
                    verbose: int = 0):
    """"""
    # TODO: add docstring to 'train_one_model' method.

    torch.manual_seed(seed)
    score = 0

    for epoch in tqdm(range(n_epochs), disable = verbose != 0):
        model.train()

        total_loss = 0.0
        total_n = 0
        total_grad_norm = 0.0
        total_grad_absmax = 0.0

        n_batches = 0
        os.makedirs("data/models", exist_ok = True)
        filename = "data/models/" + label + ".pth"
        batch_size = -1

        for batch_id, (instances, coalitions, values) in enumerate(train, start = 1):

            batch_size = instances.shape[0]

            instances, coalitions, values = transform(instances, coalitions, values, device)

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

                if verbose == 1:
                    message = f"""
                    Epoch {epoch:02d}: 
                    Batch {batch_id:04d}:
                    loss = {running_loss:.4f}; grad_norm = {grad_norm:.2e}. 
                    """; print(message)

        train_loss = total_loss / total_n
        avg_grad_norm = total_grad_norm / n_batches
        avg_grad_absmax = total_grad_absmax / n_batches
        eval_loss = evaluate(model, valid, criterion, device)
        score += eval_loss

        metrics = {
            "train/loss": train_loss,
            "train/grad_norm": avg_grad_norm,
            "train/grad_absmax": avg_grad_absmax,
            "eval/loss": eval_loss,
            "param/learning_rate": lr,
            "param/weight_decay": wd,
            "param/batch_size": batch_size,
            "param/seed": seed
        }

        tensorboard.log(metrics, epoch)

        if epoch % log_every == 0:
            torch.save(model.state_dict(), filename)

    tensorboard.close()
    return model, score

def train_model(model : nn.Module,
                train : DataLoader,
                valid : DataLoader,
                device : torch.device,
                tensorboard: Logger,
                seed: int = 0,
                criterion: nn.Module = nn.MSELoss(),
                n_epochs : int = 5,
                learning_rate : float = 0.1,
                weight_decay : float = 0.0,
                log_every : int | None = 10,
                label : str = "model",
                verbose: int = 0):
    """"""
    # TODO: add docstring to 'train_model' method.

    model = model.to(device)
    test_run(model, device)
    if verbose != 0:
        print(model)
        print(f"Trainable parameters: {count_trainable_params(model)}")
        print("Test run: ok")

    log_every = n_epochs + 1 if log_every is None else log_every

    optimizer = optim.Adam(model.parameters(),
                           lr = learning_rate,
                           weight_decay = weight_decay)

    model, score = train_one_model(model = model,
                            train = train,
                            valid = valid,
                            criterion = criterion,
                            optimizer = optimizer,
                            device = device,
                            tensorboard = tensorboard,
                            n_epochs = n_epochs,
                            log_every = log_every,
                            label = label,
                            seed = seed,
                            lr = learning_rate,
                            wd = weight_decay,
                            verbose = verbose)

    return model, score

def train(model: nn.Module,
          n_epochs: int,
          data_path: str,
          features: list[str],
          target: str,
          batch_size: int = 64,
          learning_rate: float = 1e-3,
          weight_decay: float = 0.0,
          val_fraction: float = 0.2,
          seed: int = 0,
          label: str = "model",
          log_dir: str = "data/logs/networks",
          verbose: int = 0,
          log_every: int | None = 10):

    device = get_device(); print("device:", str(device))
    dataset = H5Dataset(path = data_path,
                        features = features,
                        target = target)

    tensorboard = Logger(log_dir,
                         label)

    estimation, evaluation = make_loaders(dataset,
                                          batch_size = batch_size,
                                          val_fraction = val_fraction,
                                          seed = seed,
                                          verbose = verbose)

    model, score = train_model(model = model,
                        train = estimation,
                        valid = evaluation,
                        device = device,
                        tensorboard = tensorboard,
                        n_epochs = n_epochs,
                        learning_rate = learning_rate,
                        weight_decay = weight_decay,
                        label = label,
                        seed = seed,
                        verbose = verbose,
                        log_every = log_every)

    return model, score