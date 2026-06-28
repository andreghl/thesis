"""
Session 8 - depth, regularization, pooling, and inductive bias.

Students should edit:
    - the selected model class,
    - the optimizer weight_decay,
    - whether dropout is used,
    - which experiment block is run.

The training loop is provided.
"""
import math
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms



def get_device():
    """Return the best available device: CUDA, then MPS, then CPU."""

    if torch.cuda.is_available():
        return torch.device("cuda")

    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")

# ------------------------------------------------------------
# Data
# ------------------------------------------------------------
def make_mnist_loaders(
    batch_size,
    val_fraction=0.1,
    n_train_subset=None,
    n_test_subset=None,
    seed=123,
):
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train_dataset = datasets.MNIST(
        root="./data",
        train=True,
        download=True,
        transform=transform,
    )

    full_test_dataset = datasets.MNIST(
        root="./data",
        train=False,
        download=True,
        transform=transform,
    )

    generator = torch.Generator().manual_seed(seed)

    if n_train_subset is not None:
        train_indices = torch.randperm(
            len(full_train_dataset),
            generator=generator,
        )[:n_train_subset]

        full_train_dataset = torch.utils.data.Subset(
            full_train_dataset,
            train_indices,
        )

    if n_test_subset is not None:
        test_indices = torch.randperm(
            len(full_test_dataset),
            generator=generator,
        )[:n_test_subset]

        full_test_dataset = torch.utils.data.Subset(
            full_test_dataset,
            test_indices,
        )

    n_total = len(full_train_dataset)
    n_val = int(val_fraction * n_total)
    n_train = n_total - n_val

    train_dataset, val_dataset = torch.utils.data.random_split(
        full_train_dataset,
        [n_train, n_val],
        generator=generator,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = torch.utils.data.DataLoader(
        full_test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    print("Training examples:", len(train_dataset))
    print("Validation examples:", len(val_dataset))
    print("Test examples:", len(full_test_dataset))

    return train_loader, val_loader, test_loader


# ------------------------------------------------------------
# Fully connected models
# ------------------------------------------------------------

class ShallowMLP(nn.Module):
    def __init__(self, hidden_units=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 10),
        )

    def forward(self, x):
        return self.net(x)


class DeepMLP(nn.Module):
    def __init__(self, hidden_units=256):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Linear(hidden_units, 10),
        )

    def forward(self, x):
        return self.net(x)


class DropoutMLP(nn.Module):
    def __init__(self, hidden_units=256, dropout_p=0.5):
        super().__init__()

        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_units, hidden_units),
            nn.ReLU(),
            nn.Dropout(p=dropout_p),
            nn.Linear(hidden_units, 10),
        )

    def forward(self, x):
        return self.net(x)


# ------------------------------------------------------------
# Convolutional models
# ------------------------------------------------------------

class CNNNoPooling(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 28 * 28, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class CNNWithPooling(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(16 * 7 * 7, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# ------------------------------------------------------------
# Diagnostics and evaluation
# ------------------------------------------------------------

def count_trainable_params(model):
    total = 0

    for p in model.parameters():
        if p.requires_grad:
            total += p.numel()

    return total


def test_run(model, device):
    """
    One forward and backward pass to catch:
    - shape mistakes,
    - missing gradients.
    """
    model.train()

    x = torch.randn(8, 1, 28, 28, device=device)
    y = torch.randint(0, 10, (8,), device=device)

    loss_fn = nn.CrossEntropyLoss()

    logits = model(x)

    assert logits.shape == (8, 10), (
        f"Expected output shape (8, 10), got {tuple(logits.shape)}"
    )

    loss = loss_fn(logits, y)
    loss.backward()

    missing = []

    for name, p in model.named_parameters():
        if p.requires_grad and p.grad is None:
            missing.append(name)

    if missing:
        raise RuntimeError(f"Missing gradients for: {missing}")

    model.zero_grad(set_to_none=True)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()

    total_loss = 0.0
    total_correct = 0
    total_n = 0

    for x, y in loader:
        x = x.to(device)
        y = y.to(device)

        logits = model(x)
        loss = criterion(logits, y)

        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(dim=1) == y).sum().item()
        total_n += x.size(0)

    avg_loss = total_loss / total_n
    avg_acc = total_correct / total_n

    return avg_loss, avg_acc


def compute_grad_norms(model):
    total_norm_squared = 0.0
    max_abs_grad = 0.0

    for p in model.parameters():
        if p.grad is None:
            continue

        grad = p.grad

        grad_l2_norm = grad.norm(2).item()
        total_norm_squared += grad_l2_norm ** 2

        grad_max_abs = grad.abs().max().item()
        max_abs_grad = max(max_abs_grad, grad_max_abs)

    total_grad_norm = math.sqrt(total_norm_squared)

    return total_grad_norm, max_abs_grad


@torch.no_grad()
def inspect_feature_shapes(model, device):
    """
    Print intermediate feature-map shapes.

    This is intended for CNNNoPooling and CNNWithPooling.
    """
    if not hasattr(model, "features"):
        print("This model does not have a model.features module.")
        return

    model.eval()

    x = torch.randn(1, 1, 28, 28, device=device)

    print("\nFeature-map shapes")
    print("Input:", tuple(x.shape))

    for layer in model.features:
        x = layer(x)
        print(layer.__class__.__name__, "->", tuple(x.shape))


# ------------------------------------------------------------
# Training
# ------------------------------------------------------------

def train_one_model(
    model,
    train_loader,
    eval_loader,
    criterion,
    optimizer,
    device,
    epochs=5,
    log_every=100,
):
    history = {
        "train_loss": [],
        "train_acc": [],
        "eval_loss": [],
        "eval_acc": [],
        "grad_norm": [],
        "grad_maxabs": [],
    }

    start_time = time.time()

    for epoch in range(1, epochs + 1):
        model.train()

        total_loss = 0.0
        total_correct = 0
        total_n = 0

        total_grad_norm = 0.0
        total_grad_maxabs = 0.0
        n_batches = 0

        for batch_idx, (x, y) in enumerate(train_loader, start=1):
            x = x.to(device)
            y = y.to(device)

            logits = model(x)
            loss = criterion(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()

            grad_norm, grad_maxabs = compute_grad_norms(model)

            optimizer.step()

            total_loss += loss.item() * x.size(0)
            total_correct += (logits.argmax(dim=1) == y).sum().item()
            total_n += x.size(0)

            total_grad_norm += grad_norm
            total_grad_maxabs += grad_maxabs
            n_batches += 1

            if batch_idx % log_every == 0:
                running_loss = total_loss / total_n
                running_acc = total_correct / total_n

                print(
                    f"Epoch {epoch:02d} "
                    f"Batch {batch_idx:04d} | "
                    f"loss={running_loss:.4f} | "
                    f"acc={running_acc * 100:.2f}% | "
                    f"grad_norm={grad_norm:.2e}"
                )

        train_loss = total_loss / total_n
        train_acc = total_correct / total_n

        avg_grad_norm = total_grad_norm / n_batches
        avg_grad_maxabs = total_grad_maxabs / n_batches

        eval_loss, eval_acc = evaluate(
            model,
            eval_loader,
            criterion,
            device,
        )

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["eval_loss"].append(eval_loss)
        history["eval_acc"].append(eval_acc)
        history["grad_norm"].append(avg_grad_norm)
        history["grad_maxabs"].append(avg_grad_maxabs)

        print(
            f"Epoch {epoch:02d}/{epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"train_acc={train_acc * 100:.2f}% | "
            f"eval_loss={eval_loss:.4f} | "
            f"eval_acc={eval_acc * 100:.2f}% | "
            f"avg_grad_norm={avg_grad_norm:.2e}"
        )

    elapsed = time.time() - start_time
    print(f"Training finished in {elapsed:.1f} seconds.")

    return history

# ------------------------------------------------------------
# Running experiments
# ------------------------------------------------------------
def test_run_experiment(model, device, label="model"):
    print("\n" + "=" * 72)
    print("Test run:", label)
    print("=" * 72)

    model = model.to(device)

    print(model)
    print("Trainable parameters:", f"{count_trainable_params(model):,}")

    test_run(model, device)
    print("Test run: OK")

    return model

def run_experiment(
    model,
    train_loader,
    eval_loader,
    device,
    epochs=5,
    learning_rate=0.1,
    weight_decay=0.0,
    label="model",
):
    print("\n" + "=" * 72)
    print("Experiment:", label)
    print("=" * 72)

    model = model.to(device)

    print(model)
    print("Trainable parameters:", f"{count_trainable_params(model):,}")

    test_run(model, device)
    print("Test run: OK")

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    history = train_one_model(
        model=model,
        train_loader=train_loader,
        eval_loader=eval_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=device,
        epochs=epochs,
    )

    return model, history


def main():
    # ------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------

    torch.manual_seed(123)

    epochs = 5
    batch_size = 128
    learning_rate = 0.1
    val_fraction = 0.1

    # Use smaller subsets for faster classroom experiments.
    # Set these to None to use the full MNIST datasets.
    n_train_subset = 5_000
    n_test_subset = 1_000

    # Prefer: CUDA > MPS (Apple Silicon) > CPU
    device = get_device()
    print("Device:", str(device))

    # ------------------------------------------------------------
    # Prepare data
    # ------------------------------------------------------------

    train_loader, val_loader, test_loader = make_mnist_loaders(
        batch_size=batch_size,
        val_fraction=val_fraction,
        n_train_subset=n_train_subset,
        n_test_subset=n_test_subset,
        seed=123,
    )

    # ------------------------------------------------------------
    # Default test run
    # ------------------------------------------------------------

    """
    This default block checks that the script works without training.

    For the exercises, comment out this block and uncomment the relevant
    experiment block above.
    """

    test_run_experiment(
        model=ShallowMLP(),
        device=device,
        label="Default: ShallowMLP",
    )

    # ------------------------------------------------------------
    # Default run
    # ------------------------------------------------------------

    """
    This default run checks that the script works.

    For the exercises, comment out this block and uncomment the relevant
    experiment block above.
    """

    model, history = run_experiment(
        model=ShallowMLP(),
        train_loader=train_loader,
        eval_loader=val_loader,
        device=device,
        epochs=epochs,
        learning_rate=learning_rate,
        weight_decay=0.0,
        label="Default run: ShallowMLP",
    )

    print("\nFinal history dictionary:")
    for key, values in history.items():
        print(key, values)


    # ------------------------------------------------------------
    # Experiment 1: ShallowMLP vs DeepMLP
    # ------------------------------------------------------------

    """
    Run both models and record:
        - train/test loss and accuracy,
        - number of trainable parameters,
        - average gradient norm.
    """

    # ------------------------------------------------------------
    # Experiment 2: no regularization, L2 regularization, dropout
    # ------------------------------------------------------------

    """
    Run the three settings and record:
        - train/validation loss and accuracy,
        - average gradient norm.
    """


    # ------------------------------------------------------------
    # Experiment 3: CNNNoPooling vs CNNWithPooling
    # ------------------------------------------------------------

    """
    Run both CNNs and record:
        - train/test loss and accuracy,
        - feature-map shapes,
        - number of trainable parameters,
        - average gradient norm.
    """


    # ------------------------------------------------------------
    # Experiment 4: regularized MLP vs minimally regularized CNN
    # ------------------------------------------------------------

    """
    Run both models and record:
        - train/test loss and accuracy,
        - number of trainable parameters,
        - average gradient norm.
    """

if __name__ == "__main__":
    main()