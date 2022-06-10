# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %%
from math import inf
from pathlib import Path
import csv

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn import preprocessing

from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
import torch
import torch.nn as nn

# %% [markdown]
# ## Define Constants and Hyperparams

# %% [markdown]
# ## Define a pytorch Dataset object to contain the training and testing data
#
# Pytorch handles data shuffling and batch loading, as long as the user provides a "Dataset" class. This class is just a
# wrapper for your data that casts the data into pytorch tensor format and returns slices of the data. In this case, our
# data is in numpy format, which conveniently pytorch has a method for converting to their native format.

# %%
class AngleDataset(Dataset):
    def __init__(self, csv_path: Path):
        """Create a PyTorch dataset from CSV data.

        Args:
            csv_path (Path): kinematics data file
        """
        df = pd.read_csv(csv_path)
        length = len(df)

        # The network takes a sinusoidal signal as an input
        sim_time = 10
        freq = 1
        time = torch.linspace(0, sim_time, length)
        sin_signal = torch.sin(2 * torch.pi * freq * time)

        # data order = sin (1), angles (4*3*2), torso (2*2), touch_sens (4)
        x_data = []
        y_data = []

        for sin_val, i in zip(sin_signal, range(length)):
            x_data.append(torch.hstack([sin_val, torch.tensor(df.iloc[i])]))
            # No need to predict/output touch sensor values
            y_data.append(df.iloc[i + 1][:-4] if i < length - 1 else df.iloc[0][:-4])

        self.x_data: torch.Tensor = torch.vstack(x_data).type(torch.float32)
        self.y_data: torch.Tensor = torch.tensor(y_data).type(torch.float32)
        self.length: int = len(self.x_data)

    def __getitem__(self, index) -> tuple[torch.Tensor, torch.Tensor]:
        return self.x_data[index], self.y_data[index]

    def __len__(self) -> int:
        return self.length


# %% [markdown]
# ## Define training methods for the model
#
# These methods use an initialized model and training data to iteratively perform the forward and backward pass of
# optimization. Aside from some data reformatting that depends on the input, output, and loss function, these methods will
# always be the same for any shallow neural network.

# %%
def train_batch(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    optimizer: optim.Optimizer,
    criteria: nn.Module,
) -> float:
    """Train the given model on a single batch of data.

    Args:
        model (nn.Module): model to train
        x (torch.Tensor): input data
        y (torch.Tensor): labeled output data
        optimizer (optim.Optimizer): SGD-based optimzier
        criteria (nn.Module): loss function

    Returns:
        float: mean loss for the batch
    """

    model.train()

    # Compute output
    y_predict = model(x)

    # Compute loss
    loss = criteria(y_predict, y)

    # Zero out current gradient values
    optimizer.zero_grad()

    # Compute gradients
    loss.backward()

    # Update parameters
    optimizer.step()

    return loss.data.item()


def train_loop(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    critera: nn.Module,
    num_epochs: int,
) -> list[float]:
    """Train the model.

    Args:
        model (nn.Module): model to train
        loader (DataLoader): training data
        optimizer (nn.Module): SGD-based optimizer
        critera (nn.Module): loss function
        num_epochs (int): number of epochs to train

    Returns:
        list[float]: mean loss for each batch in each epoch
    """

    losses = []

    for _ in range(num_epochs):
        for x, y in loader:
            loss = train_batch(model, x, y, optimizer, critera)
            losses.append(loss)

    return losses


# %% [markdown]
# ## Define testing methods for the model
#
# These methods are like training, but we don't need to update the parameters of the model anymore because when we call
# the test() method, the model has already been trained. Instead, this method just calculates the predicted y values and
# returns them, AKA the forward pass.

# %%
def batch_inference(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    return model(x)


def inference(model: nn.Module, dataset: AngleDataset) -> torch.Tensor:
    """Compute model outputs for the given data.

    Args:
        model (nn.Module): trained model
        dataset (AngleDataset): data to test

    Returns:
        torch.Tensor: computed output values
    """
    loader = DataLoader(dataset, batch_size=len(dataset))
    predictions = [batch_inference(model, x) for x, _ in loader]
    return torch.concat(predictions)


# %% [markdown]
# ## Define plotting method for loss
# This is a plotting method for looking at the behavior of the loss over training iterations.

# %%
def plot_losses(losses: list[float], title: str):
    plt.plot(losses)
    plt.title(title)
    plt.xlabel("Batch")
    plt.ylabel("Loss")


# %% [markdown]
# ## Define Model Architecture
# - 33 inputs = 3 joint angles per leg, 4 legs, 2 DOF per joint. 4 touch sensors. 1 sine timestamp.
# - 28 outputs = *same as above, except just the joint angles*
#

# %%
class GaitModel(nn.Module):
    def __init__(self, layer_sizes: list[int], batch_norm: bool, dropout: float):
        """A PyTorch model trained to output new joint angles.

        Args:
            layer_sizes (list[int]): number of neurons per layer
            batch_norm (bool): flag for using batch normalization
            dropout (float): flag/value for using dropout
        """
        super(GaitModel, self).__init__()

        hidden_layers = []

        # Loop over layer_sizes and create linear->relu[->batchnorm][->dropout] layers
        for nl, nlminus1 in zip(layer_sizes[1:-1], layer_sizes):
            # Required layers
            layers = [nn.Linear(nlminus1, nl), nn.ReLU()]

            # Optional batch normalization layer
            if batch_norm:
                layers.append(nn.BatchNorm1d(nl))

            # Optional dropout layer
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

            hidden_layers.append(nn.Sequential(*layers))

        output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        # Group all layers into the sequential container
        all_layers = hidden_layers + [output_layer]
        self.layers = nn.Sequential(*all_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


# %% [markdown]
# ## Define Run function

# %%
def train(
    dataset: AngleDataset,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    layer_sizes: list[int],
    batch_norm: bool,
    dropout: float,
) -> tuple[nn.Module, list[float]]:
    """Train a model on the given dataset.

    Args:
        dataset (AngleDataset): kinematics dataset
        layer_sizes (list[int]): neurons per layer
        batch_norm (bool): batch normalization flag
        dropout (float): dropout value (0 for no dropout)
        num_epochs (int): number of training epochs
        batch_size (int): training batch size
        learning_rate (float): training learning rate

    Returns:
        tuple[nn.Module, list[float]]: model and training losses
    """

    # learning_rate = 1e-3

    data_loader = DataLoader(dataset, batch_size, shuffle=True)
    model = GaitModel(layer_sizes, batch_norm, dropout)

    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()

    losses = train_loop(
        model=model,
        loader=data_loader,
        optimizer=optimizer,
        critera=criterion,
        num_epochs=num_epochs,
    )

    return model, losses


# %%

ANIMATIONS_DIR = Path("Animations/")
DATA_DIR = Path("KinematicsData/")
SANITY_DIR = Path("SanityChecks/")
MODEL_OUTPUT_DIR = Path("ModelOutputs/")
MODEL_DIR = Path("Models/")

segment_names = ["FL", "FR", "BL", "BR", "SP"]
joints_per_segment = (3, 3, 3, 3, 2)
dof_per_join = 2

CSV_HEADER = []
for segment_name, num_joints in zip(segment_names, joints_per_segment):
    for joint_index in range(num_joints):
        for dof_index in range(dof_per_join):
            CSV_HEADER.append(f"{segment_name} A{joint_index + 1} DF {dof_index + 1}")

datasets = {
    f.stem.split("_")[0]: AngleDataset(f) for f in DATA_DIR.glob("*_kinematic.csv")
}

# %%

# Model hyperparameters
layer_sizes = [33, 31, 30, 28]

# Training hyperparameters
num_epochs = 100
batch_size = 32
learning_rate = 0.01

final_losses = []

for name in datasets:

    dataset = datasets[name]

    print(f"Processing the '{name}' dataset.")

    model, losses = train(
        dataset,
        num_epochs,
        batch_size,
        learning_rate,
        layer_sizes,
        batch_norm=False,
        dropout=0,
    )

    predictions = inference(model, dataset)

    # Save the outputs to a csv for quicker comparisons later
    with open(MODEL_OUTPUT_DIR / f"{name}_output.csv", "w") as csvfile:

        writer = csv.writer(csvfile)

        writer.writerow(CSV_HEADER)

        for row in predictions:
            writer.writerow(row.tolist())

    # torch.save(model_to_save, model_path / f"{name}_model{spacer}{extra_id}.pt")

    # final_loss = sum(losses[-100:]) / 100
    # final_losses.append(final_loss)

    # print(f"Final loss for {name}: {final_loss}")
    # if plot:
    # plot_losses(losses, name)

# return sum(final_losses) / len(final_losses)


# %% [markdown]
# ## Sanity Check Data By Plotting
# Also saving to files under Sanity_Checks

# %%
kinematics = sorted([f for f in DATA_DIR.glob("*_kinematic.csv") if f.is_file()])
outputs = sorted([f for f in MODEL_OUTPUT_DIR.glob("*_output.csv") if f.is_file()])

# %%

for actual, pred in zip(kinematics, outputs):

    dfa = pd.read_csv(actual)
    dfp = pd.read_csv(pred)

    columns = dfp.columns
    num_cols = len(columns)

    fig, axes = plt.subplots(
        num_cols // 2,
        2,
        figsize=(16, 64),
        sharex=True,
        sharey=True,
        constrained_layout=True,
    )

    gait_name = actual.stem.split("_")[0]

    fig.suptitle(gait_name, fontsize=24)

    for col, ax in zip(columns, axes.flatten()):
        ax.plot(dfa[col], label="Actual")
        ax.plot(dfp[col], label="Prediction")
        ax.set_title(col)
        ax.legend()

    # fig.savefig(SANITY_DIR / f"{gait_name}_comparison.png", facecolor="white")

    break

    # plot_actual[0][0].get_figure().savefig(
    #     SANITY_DIR / f"{actual.stem}_actual.png", facecolor="white"
    # )
    # plot_predicted[0][0].get_figure().savefig(
    #     SANITY_DIR / f"{pred.stem}_predicted.png", facecolor="white"
    # )


# %%
def plot_comparisons(data_files, model_outputs, save_path, extra_id: str = None):
    for actual, pred in zip(data_files, model_outputs):

        dfa = pd.read_csv(actual)
        dfp = pd.read_csv(pred)

        name = actual.stem.split("_")[0]

        fig, axs = pyplot.subplots(
            nrows=6, ncols=10, sharey=True, sharex=True, figsize=(16, 12)
        )
        fig.suptitle(f"{name}: actual (left) vs predicted (right)", fontsize=16)
        # fig.set_size_inches(9, 9, forward=True)
        for ax_arr in axs:
            for ax in ax_arr:
                # ax.set_aspect('equal')
                # ax.set_adjustable('box')
                ax.set_box_aspect(1)
        pyplot.subplots_adjust(
            left=0.025, right=1.0, bottom=0.1, top=0.9, wspace=0.5, hspace=0.5
        )
        pyplot.ylim(-3, 3)
        fig.tight_layout()

        dfa = dfa.iloc[:, :-4]

        dfa_p1 = dfa.iloc[:, :6]
        dfa_p2 = dfa.iloc[:, 6:12]
        dfa_p3 = dfa.iloc[:, 12:18]
        dfa_p4 = dfa.iloc[:, 18:24]
        dfa_p5 = dfa.iloc[:, 24:]

        dfp_p1 = dfp.iloc[:, :6]
        dfp_p2 = dfp.iloc[:, 6:12]
        dfp_p3 = dfp.iloc[:, 12:18]
        dfp_p4 = dfp.iloc[:, 18:24]
        dfp_p5 = dfp.iloc[:, 24:]

        dfa_p1.plot(ax=axs[:, 0], subplots=True)
        dfp_p1.plot(ax=axs[:, 1], subplots=True)
        dfa_p2.plot(ax=axs[:, 2], subplots=True)
        dfp_p2.plot(ax=axs[:, 3], subplots=True)
        dfa_p3.plot(ax=axs[:, 4], subplots=True)
        dfp_p3.plot(ax=axs[:, 5], subplots=True)
        dfa_p4.plot(ax=axs[:, 6], subplots=True)
        dfp_p4.plot(ax=axs[:, 7], subplots=True)
        dfa_p5.plot(ax=axs[:-2, 8], subplots=True)
        dfp_p5.plot(ax=axs[:-2, 9], subplots=True)

        spacer = "_" if extra_id else ""

        fig.savefig(
            save_path / f"{name}_comparison{spacer}{extra_id}", facecolor="white"
        )


# plot_comparisons(data_files, model_outputs, SANITY_PATH)


# %%
configs = [
    ("bothoff", False, 0),
    ("onlydrop", False, 0.5),
    ("bothon", True, 0.5),
    ("onlybatch", True, 0),
]

tmp_path = Path("TMP/")

names_and_ds = []

for filename in DATA_DIR.glob("*_kinematic.csv"):
    gait_name = filename.stem.split("_")[0]
    train_ds, test_ds = create_datasets(filename)
    names_and_ds.append((gait_name, train_ds, test_ds))

min_loss = inf
best_config = None

for config, batch, drop in configs:

    # trains all gait types w this config
    avg_loss = train_csv_plot(
        names_and_ds,
        tmp_path,
        tmp_path,
        plot=False,
        dropout=drop,
        batch_norm=batch,
        extra_id=config,
    )

    if min_loss > avg_loss:
        min_loss = avg_loss
        best_config = config

    # now we've got a set of models trained w these specific configs and a csv for each. need to compare vs regular graphs
    config_outputs = sorted(
        [x for x in tmp_path.glob(f"*_{config}.csv") if x.is_file()]
    )

    plot_comparisons(
        kinematics, config_outputs, tmp_path, extra_id=config
    )  # plot to tmp folder

# %%
print(best_config)

# %%
# !jupytext --set-formats ipynb, py:percent gait_model.ipynb
