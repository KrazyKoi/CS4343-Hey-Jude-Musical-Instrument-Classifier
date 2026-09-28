import copy
import random

import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset
from torchvision import datasets, transforms

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
        self.conv = ConvBNReLU(3,32, kernel_size=3, stride=1, padding=1)
        
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

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_examples += batch_size

    return total_loss / total_examples, total_correct / total_examples


##!!! Could mess around with parameters
def fit_model(
    model,
    train_loader,
    val_loader,
    lr=0.1,
    weight_decay=5e-4,
    epochs=20,
    device=DEVICE,
    verbose=True,
):
    # Step 1: Move model to device.
    # Step 2: Create nn.CrossEntropyLoss().
    # Step 3: Create torch.optim.SGD with momentum=0.9, nesterov=True,
    #         the supplied learning rate, and the supplied weight decay.
    # Step 4: Create CosineAnnealingLR with T_max=epochs.
    # Step 5: Create a history dictionary for train/validation loss and accuracy and lr.
    # Step 6: For each epoch:
    #         a. call train_one_epoch,
    #         b. call evaluate on the validation loader,
    #         c. save a copy of the best model state according to validation accuracy,
    #         d. record all metrics and the current learning rate,
    #         e. call scheduler.step().
    # Step 7: Restore the best validation model weights before returning.
    # Step 8: Return history.

    # YOUR CODE HERE
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
       model.parameters(), momentum=0.9, nesterov=True, lr=lr, weight_decay=weight_decay
    )
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
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

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
                f"Epoch {epoch:02d}/{epochs} | "
                f"train acc {100*train_acc:.2f}% | "
                f"val acc {100*val_acc:.2f}% | "
                f"lr {current_lr:.5f}"
            )

        scheduler.step()

    model.load_state_dict(best_state)
    return history


