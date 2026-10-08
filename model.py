import copy
import random
import os
import json
import shutil

from datetime import datetime
from pathlib import Path

from sklearn import metrics

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader, Subset, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")

# --------------------------------------------------
# MEL-SPECTROGRAM PROCESSING
# --------------------------------------------------
mel_params = json.load(open("data/processed/mel_params.json", "r"))

def load_specs(df):
    specs = []
    labels = []
    for _, row in df.iterrows():
        spec = np.load(row['Mel_Path'])
        # Normalize
        spec = (spec - mel_params["MEAN_DB"]) / (mel_params["STD_DB"] + 1e-9)
        # Pad/truncate to fixed length
        if spec.shape[1] < mel_params["MAX_MLEN"]:
            pad_width = mel_params["MAX_MLEN"] - spec.shape[1]
            spec = np.pad(spec, pad_width=((0, 0), (0, pad_width)), mode="constant")
        else:
            spec = spec[:, :mel_params["MAX_MLEN"]]

        # Expand dimensions for channel
        spec = np.expand_dims(spec, 0)
        specs.append(spec)
        labels.append(row['Label'])

    df['features'] = specs
    df['labels'] = labels
    return df

def df_to_tf(df, batch=128, shuffle=True):
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)

    mapped_series = df['labels'].map(mel_params["LABEL_MAP"])

    if mapped_series.isna().any():
        missing = df['labels'][mapped_series.isna()].unique()
        raise ValueError(f"Found unmapped labels after shuffling: {missing}")

    X = torch.stack([torch.tensor(x, dtype=torch.float32) for x in df['features']])
    Y = torch.tensor(mapped_series.to_numpy(), dtype=torch.long)

    ds = TensorDataset(X, Y)

    dl = DataLoader(
        ds,
        batch_size=batch,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    return dl

# --------------------------------------------------
# MODEL
# --------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
    ):
        super().__init__()

        # Step 1: Create nn.Conv2d with the supplied dimensions.
        #         Set bias=False.
        # Step 2: Create nn.BatchNorm2d for out_channels features.
        # Step 3: Create nn.ReLU with inplace=True.

        # YOUR CODE HERE
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # Step 1: Apply the convolution.
        # Step 2: Apply Batch Normalization.
        # Step 3: Apply ReLU.
        # Step 4: Return the resulting tensor.

        # YOUR CODE HERE
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class SkipConnectionBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # Step 1: Create the first main-path operation using ConvBNReLU.
        #         It maps in_channels -> out_channels and uses the supplied stride.
        # Step 2: Create the second 3x3 convolution.
        #         It keeps out_channels channels, uses stride=1, padding=1, and bias=False.
        # Step 3: Create BatchNorm2d after the second convolution.
        # Step 4: If stride == 1 and in_channels == out_channels, use nn.Identity()
        #         for the shortcut.
        # Step 5: Otherwise, use a 1x1 Conv2d followed by BatchNorm2d for the shortcut.
        # Step 6: Create the final ReLU.

        # YOUR CODE HERE
        #self.first = ConvBNReLU(in_channels, in_channels)
        #self.second = ConvBNReLU(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        #self.bn = BatchNorm2d(out_channels)
        self.conv1 = ConvBNReLU(in_channels, out_channels, stride=stride)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride == 1 and (in_channels == out_channels):
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        self.Y = nn.ReLU()


    def forward(self, x):
        # Step 1: Compute the shortcut output from x.
        # Step 2: Pass x through the first main-path module.
        # Step 3: Apply the second convolution and its BatchNorm.
        # Step 4: Add the main path and shortcut tensors.
        # Step 5: Apply the final ReLU.
        # Step 6: Return the block output.

        # YOUR CODE HERE
        short = self.shortcut(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.Y(x + short)
        return x


class CustomCNN(nn.Module):
    def __init__(self, num_classes=28): ##!!! changed from 10
        super().__init__()

        # Step 1: Create the stem: ConvBNReLU(3, 32, kernel_size=3, stride=1, padding=1).
        # Step 2: Create Stage 1 with two 32 -> 32 skip-connection blocks.
        # Step 3: Create Stage 2. The first block is 32 -> 64 with stride=2;
        #         the second is 64 -> 64 with stride=1.
        # Step 4: Create Stage 3. The first block is 64 -> 128 with stride=2;
        #         the second is 128 -> 128 with stride=1.
        # Step 5: Create adaptive average pooling to output spatial size 1x1.
        # Step 6: Create the final Linear layer from 128 features to num_classes logits.

        # YOUR CODE HERE
        self.conv = ConvBNReLU(1,32, kernel_size=3, stride=1, padding=1)
        
        ##!!! could mess around with kernel size, stride, padding to see if it improves accuracy
        self.block1 = nn.Sequential(
            SkipConnectionBlock(32,32,1),
            SkipConnectionBlock(32,32,1),
        )

        self.block2 = nn.Sequential(
            SkipConnectionBlock(32,64,2),
            SkipConnectionBlock(64,64,1),
        )

        self.block3 = nn.Sequential(
            SkipConnectionBlock(64,128,2),
            SkipConnectionBlock(128,128,1),
        )

        self.avg = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # Step 1: Pass x through the stem.
        # Step 2: Pass the result through Stage 1, Stage 2, and Stage 3.
        # Step 3: Apply global average pooling.
        # Step 4: Flatten all dimensions except the batch dimension.
        # Step 5: Apply the final Linear layer.
        # Step 6: Return the logits.

        # YOUR CODE HERE
        x = self.conv(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = self.avg(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x


def initialize_weights(module):
    # Step 1: If module is nn.Conv2d, apply He normal initialization.
    # Step 2: If module is nn.BatchNorm2d, set weight to 1 and bias to 0.
    # Step 3: If module is nn.Linear, initialize weight ~ N(0, 0.01^2) and bias to 0.
    # Step 4: This function does not need to return anything.

    # YOUR CODE HERE
    if isinstance(module, nn.Conv2d):
        nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Linear):
        nn.init.normal_(module.weight, mean=0.0, std=0.01)
        if module.bias is not None:
            nn.init.zeros_(module.bias)


def train_one_epoch(model, loader, criterion, optimizer, device):
    # Step 1: Put the model in training mode with model.train().
    # Step 2: Initialize accumulators for total loss, correct predictions, and sample count.
    # Step 3: Loop over minibatches from loader.
    # Step 4: Move images and labels to device.
    # Step 5: Clear old gradients with optimizer.zero_grad().
    # Step 6: Run the forward pass and compute the loss.
    # Step 7: Call loss.backward().
    # Step 8: Call optimizer.step().
    # Step 9: Accumulate loss weighted by minibatch size and count correct predictions.
    # Step 10: Return average loss and accuracy over the entire loader.

    # YOUR CODE HERE
    model.train()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    # Step 1: Put the model in evaluation mode with model.eval().
    # Step 2: Initialize accumulators for total loss, correct predictions, and sample count.
    # Step 3: Loop over loader and move each minibatch to device.
    # Step 4: Compute logits and loss. Do not call backward().
    # Step 5: Accumulate loss and correct predictions.
    # Step 6: Return average loss and accuracy.

    # YOUR CODE HERE
    model.eval()
    total_loss = 0.0
    total_correct = 0
    total_examples = 0

    all_preds = []
    all_labels = []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        preds = logits.argmax(dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    cm = confusion_matrix(all_labels, all_preds, labels=list(range(28)))

    avg_loss = total_loss / total_examples
    avg_acc = total_correct / total_examples

    return avg_loss, avg_acc, cm


##!!! Could mess around with parameters
def fit_model(
    model,
    optimizer, #This allows for different optimizer to be used
    train_loader,
    val_loader,
    lr=0.1,
    weight_decay=5e-4,
    epochs=20,
    device=DEVICE,
    verbose=True,
):

    # YOUR CODE HERE
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=max(1, epochs // 2), gamma=0.3
    )

    anneal = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = {
        "train_loss": [], "val_loss": [],
        "train_acc": [], "val_acc": [], "lr": []
    }
    best_val_acc = -float("inf")
    best_state = copy.deepcopy(model.state_dict())

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, _ = evaluate(model, val_loader, criterion, device)

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_loss"].append(float(train_loss))
        history["val_loss"].append(float(val_loss))
        history["train_acc"].append(float(train_acc))
        history["val_acc"].append(float(val_acc))
        history["lr"].append(float(current_lr))

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        if verbose:
            print(
                f"{datetime.now().strftime("%H:%M:%S")} \t "
                f"Epoch {epoch:02d}/{epochs} | "
                f"train acc {100*train_acc:.2f}% | "
                f"val acc {100*val_acc:.2f}% | "
                f"lr {current_lr:.5f}"
            )

        torch.save(
            model.state_dict(),
            f"models/epoch_{epoch:02d}.pt"
        )
        scheduler.step()

    model.load_state_dict(best_state)
    torch.save(
        model.state_dict(),
        "final_model.pt"
    )
    return history

def optimizer_search(
        model,
        train_loader,
        val_loader,
        lr=0.1,
        weight_decay=5e-4,
        device=DEVICE,
        #verbose=True,
):
    # Create smaller loaders for hyperparameter tuning.
    TUNE_TRAIN_SIZE = min(10000, len(train_loader.dataset))
    TUNE_VAL_SIZE = min(2000, len(val_loader.dataset))

    _tune_train_dataset = Subset(train_loader.dataset, range(TUNE_TRAIN_SIZE))
    _tune_val_dataset = Subset(val_loader.dataset, range(TUNE_VAL_SIZE))

    tune_train_loader = DataLoader(
        _tune_train_dataset,
        batch_size=128,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )
    tune_val_loader = DataLoader(
        _tune_val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        persistent_workers=False,
    )

    candidate_optimizers = [
        {"name": "SGD"},
        {"name": "Adam"},
        {"name": "RMSProp"},
        {"name": "AdamW"}
    ]

    search_results = []

    for config in candidate_optimizers:
        print("Evaluating optimizer:", config["name"])
        model = CustomCNN(num_classes=num_classes)
        model.apply(initialize_weights)

        name = config["name"]
        if name == "SGD":
            optimizer = torch.optim.SGD(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif name == "Adam":
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif name == "RMSProp":
            optimizer = torch.optim.RMSprop(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )
        elif name == "AdamW":
            optimizer = torch.optim.AdamW(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay
            )

        history = fit_model(
            model,
            optimizer,
            tune_train_loader,
            tune_val_loader,
            lr=lr,
            weight_decay=weight_decay,
            epochs=2
        )

        result = {
            "optimizer_name": config["name"],
            "lr": lr,
            "weight_decay": weight_decay,
            "val_loss": history["val_loss"][-1],
            "val_acc": history["val_acc"][-1],
        }
        search_results.append(result)

    # Step 5: Select the result with the highest validation accuracy.
    # YOUR CODE HERE
    best_config = max(search_results, key=lambda r: r["val_acc"])
    print(best_config)

    return best_config

def final_training(
        optimizer_name,
        train_loader,
        val_loader,
        lr=0.1,
        weight_decay=5e-4
):
    FINAL_EPOCHS = 10

    final_model = CustomCNN(num_classes=num_classes)
    final_model.apply(initialize_weights)
    final_model.to(DEVICE)

    if optimizer_name == "SGD":
        optimizer = torch.optim.SGD(
            final_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "Adam":
        optimizer = torch.optim.Adam(
            final_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "RMSProp":
        optimizer = torch.optim.RMSprop(
            final_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
    elif optimizer_name == "AdamW":
        optimizer = torch.optim.AdamW(
            final_model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )

    final_history = fit_model(
        final_model,
        optimizer,
        train_loader,
        val_loader,
        lr=lr,
        weight_decay=weight_decay,
        epochs=FINAL_EPOCHS,
        device=DEVICE,
        verbose=True,
    )

    return final_model, final_history

def plot_history(history):
    epochs = np.arange(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_loss"], marker="o", label="Train")
    plt.plot(epochs, history["val_loss"], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Cross-Entropy Loss")
    plt.title("Training and Validation Loss")
    plt.legend()
    plt.grid(alpha=0.2)
    # plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["train_acc"], marker="o", label="Train")
    plt.plot(epochs, history["val_acc"], marker="o", label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Training and Validation Accuracy")
    plt.legend()
    plt.grid(alpha=0.2)
    # plt.show()

    plt.figure(figsize=(8, 4))
    plt.plot(epochs, history["lr"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Learning Rate")
    plt.title("Learning-Rate Schedule")
    plt.grid(alpha=0.2)
    # plt.show()

# --------------------------------------------------
# TRAINING
# --------------------------------------------------
if __name__ == "__main__":
    print(f"{datetime.now().strftime("%H:%M:%S")} \t Executing on {DEVICE}")

    # Load the CSV files
    print(f"{datetime.now().strftime("%H:%M:%S")} \t Loading the CSV files")
    test_df = pd.read_csv("data/processed/test_manifest.csv")
    train_df = pd.read_csv("data/processed/train_manifest.csv")
    val_df = pd.read_csv("data/processed/val_manifest.csv")

    # Create DataFrame
    print(f"{datetime.now().strftime("%H:%M:%S")} \t Creating the Data Frames")
    test_df = load_specs(test_df)
    train_df = load_specs(train_df)
    val_df = load_specs(val_df)

    # Data Loaders
    print(f"{datetime.now().strftime("%H:%M:%S")} \t Creating the Data Loaders")
    train_loader = df_to_tf(train_df, shuffle=True)
    test_loader = df_to_tf(test_df, shuffle=False)
    val_loader = df_to_tf(val_df, shuffle=False)

    # Initialize Model
    print(f"{datetime.now().strftime("%H:%M:%S")} \t Initializing the Model")
    num_classes = len(mel_params["LABEL_MAP"])
    model = CustomCNN(num_classes=num_classes)
    model.apply(initialize_weights)

    # Select an Optimizer
    print(f"{datetime.now().strftime("%H:%M:%S")} \t Comparing the Optimizers")
    # UNCOMMENT FOR FULL RUN
    # optimizer_results = optimizer_search(model, train_loader, val_loader)

    # UNCOMMENT FOR HARDCODED RUN
    optimizer_results = {
        "optimizer_name": "AdamW",
        "lr": 0.1,
        "weight_decay": 5e-4,
        "val_loss": 1.2582788810729981,
        "val_acc": 0.6415,
    }

    # Final Model
    print(f"{datetime.now().strftime("%H:%M:%S")} \t Training the Final Model")
    # UNCOMMENT FOR FULL RUN
    # final_model, final_history = final_training(
    #     optimizer_results["optimizer_name"],
    #     train_loader,
    #     val_loader,
    # )
    #
    # plot_history(final_history)

    # UNCOMMENT FOR HARDCODED RUN
    final_model = CustomCNN(num_classes=num_classes)
    final_model.apply(initialize_weights)
    final_model.to(DEVICE)

    state_dict = torch.load('final_model.pt', weights_only=True)

    final_model.load_state_dict(state_dict)

    # Evaluation
    print(f"{datetime.now().strftime("%H:%M:%S")} \t Evaluating the Final Model")

    criterion = nn.CrossEntropyLoss()
    test_loss, test_acc, cm = evaluate(
        final_model,
        test_loader,
        criterion,
        DEVICE
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {100 * test_acc:.2f}%")

    fig, ax = plt.subplots(figsize=(10, 8))

    plt.rc('font', size=9)
    plt.rc('axes', titlesize=12)

    cm_display = metrics.ConfusionMatrixDisplay(
        confusion_matrix=cm, display_labels=mel_params["LABEL_MAP"])

    cm_display.plot(ax=ax, cmap=plt.cm.Blues)

    ax.set_xticklabels(mel_params["LABEL_MAP"], rotation=60, ha="right")

    # plt.savefig("confusion_matrix.png", dpi=300)
    plt.tight_layout()
    plt.show()
