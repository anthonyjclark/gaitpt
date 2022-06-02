# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown] toc=true
# <h1>Table of Contents<span class="tocSkip"></span></h1>
# <div class="toc"><ul class="toc-item"><li><span><a href="#Define-a-pytorch-Dataset-object-to-contain-the-training-and-testing-data" data-toc-modified-id="Define-a-pytorch-Dataset-object-to-contain-the-training-and-testing-data-1"><span class="toc-item-num">1&nbsp;&nbsp;</span>Define a pytorch Dataset object to contain the training and testing data</a></span></li><li><span><a href="#Define-training-methods-for-the-model" data-toc-modified-id="Define-training-methods-for-the-model-2"><span class="toc-item-num">2&nbsp;&nbsp;</span>Define training methods for the model</a></span></li><li><span><a href="#Define-testing-methods-for-the-model" data-toc-modified-id="Define-testing-methods-for-the-model-3"><span class="toc-item-num">3&nbsp;&nbsp;</span>Define testing methods for the model</a></span></li><li><span><a href="#Define-plotting-method-for-loss" data-toc-modified-id="Define-plotting-method-for-loss-4"><span class="toc-item-num">4&nbsp;&nbsp;</span>Define plotting method for loss</a></span></li><li><span><a href="#Define-Model-Architecture" data-toc-modified-id="Define-Model-Architecture-5"><span class="toc-item-num">5&nbsp;&nbsp;</span>Define Model Architecture</a></span></li><li><span><a href="#Define-Run-function" data-toc-modified-id="Define-Run-function-6"><span class="toc-item-num">6&nbsp;&nbsp;</span>Define Run function</a></span></li><li><span><a href="#Create-Datasets-for-Each-Gait-File" data-toc-modified-id="Create-Datasets-for-Each-Gait-File-7"><span class="toc-item-num">7&nbsp;&nbsp;</span>Create Datasets for Each Gait File</a></span></li><li><span><a href="#Run-and-plot-results" data-toc-modified-id="Run-and-plot-results-8"><span class="toc-item-num">8&nbsp;&nbsp;</span>Run and plot results</a></span></li></ul></div>

# %%
import sys
import numpy as np
from matplotlib import pyplot
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

# from icecream import ic
import pandas as pd
from math import sqrt
import glob

from pathlib import Path

import random

# TODO: more gaits: gallop

# ic("USING pytorch VERSION: ", torch.__version__)

# %% [markdown]
# ## Define a pytorch Dataset object to contain the training and testing data
# Pytorch handles data shuffling and batch loading, as long as the user provides a "Dataset" class. This class is just a wrapper for your data that casts the data into pytorch tensor format and returns slices of the data. In this case, our data is in numpy format, which conveniently pytorch has a method for converting to their native format.
#
# The init function takes the path to the csv and creates a dataset out of it. I actually have three different options here. The dataset could be composed such that x is the 'timestamp' of the movement,the previous set of angles, or a tuple of both.

# %%
class AngleDataset(Dataset):
    def __init__(self, x, y):
        x_dtype = torch.FloatTensor
        y_dtype = torch.FloatTensor  # for MSE or L1 Loss

        self.length = x.shape[0]

        self.x_data = torch.from_numpy(x).type(x_dtype)
        self.y_data = torch.from_numpy(y).type(y_dtype)

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.length


def create_datasets(csv_path: str, train_perc: float = 0.8):
    df = pd.read_csv(csv_path)
    length = len(df)
    time = 10
    timestep = 0.005

    # x_data = np.array([])
    # y_data = np.array([])
    x_data = []
    y_data = []

    sin_test_timepoints = (
        np.random.rand(length, 1) * time
    )  # Repeat data generation for test set
    sin_test_timepoints = sin_test_timepoints.ravel()
    sin_iter = iter(sin_test_timepoints)

    # data order = sin, angles, torso, touch_sens

    # if x = curr angles and y = next angles
    for i in range(len(df)):

        x = []
        y = []

        if i < length - 1:
            x = np.append(x, df.iloc[i])
            y = np.append(y, df.iloc[i + 1][:-4])  # only include angles
        else:
            # since it loops anyway
            x = np.append(x, df.iloc[i])
            y = np.append(y, df.iloc[0][:-4])

        x = np.append([next(sin_iter)], x)

        x_data.append(x)
        y_data.append(y)

    x_data = np.array(x_data, dtype=np.float64)
    y_data = np.array(y_data, dtype=np.float64)

    last_train_idx = int(len(x_data) * train_perc)

    train_x = x_data[:last_train_idx]
    train_y = y_data[:last_train_idx]
    test_x = x_data[last_train_idx:]
    test_y = y_data[last_train_idx:]

    return AngleDataset(x=train_x, y=train_y), AngleDataset(x=test_x, y=test_y)


# %% [markdown]
# ## Define training methods for the model
# These methods use an initialized model and training data to iteratively perform the forward and backward pass of optimization. Aside from some data reformatting that depends on the input, output, and loss function, these methods will always be the same for any shallow neural network.

# %%
def train_batch(model, x, y, optimizer, loss_fn):
    # Run forward calculation
    y_predict = model.forward(x)

    # Compute loss.
    loss = loss_fn(y_predict, y)

    # Before the backward pass, use the optimizer object to zero all of the
    # gradients for the variables it will update (which are the learnable weights
    # of the model)
    optimizer.zero_grad()

    # Backward pass: compute gradient of the loss with respect to model
    # parameters
    loss.backward()

    # Calling the step function on an Optimizer makes an update to its
    # parameters
    optimizer.step()

    return loss.data.item()


def train(model, loader, optimizer, loss_fn, epochs=5):
    losses = list()

    batch_index = 0
    for e in range(epochs):
        for x, y in loader:
            loss = train_batch(
                model=model, x=x, y=y, optimizer=optimizer, loss_fn=loss_fn
            )
            losses.append(loss)

            batch_index += 1

        if e % 50 == 0:
            pass
        #   ic("Epoch: ", e+1)
        #   ic("Batches: ", batch_index)

    return losses


# %% [markdown]
# ## Define testing methods for the model
# These methods are like training, but we don't need to update the parameters of the model anymore because when we call the test() method, the model has already been trained. Instead, this method just calculates the predicted y values and returns them, AKA the forward pass.
#

# %%
def test_batch(model, x, y):
    # run forward calculation
    y_predict = model.forward(x)

    return y, y_predict


def test(model, loader):
    y_vectors = list()
    y_predict_vectors = list()

    batch_index = 0
    for x, y in loader:
        y, y_predict = test_batch(model=model, x=x, y=y)

        y_vectors.append(y.data.numpy())
        y_predict_vectors.append(y_predict.data.numpy())

        batch_index += 1

    y_predict_vector = np.concatenate(y_predict_vectors)

    return y_predict_vector


# %% [markdown]
# ## Define plotting method for loss
# This is a plotting method for looking at the behavior of the loss over training iterations.

# %%
def plot_loss(losses, title: str, show=True):
    fig = pyplot.gcf()
    fig.set_size_inches(8, 6)
    ax = pyplot.axes()
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss")
    x_loss = list(range(len(losses)))
    pyplot.title(title)

    pyplot.plot(x_loss, losses)

    if show:
        pyplot.show()

    pyplot.close()


# %% [markdown]
# ## Define Model Architecture
# - 33 inputs = 3 joint angles per leg, 4 legs, 2 DOF per joint. 4 touch sensors. 1 sine timestamp.
# - 28 outputs = *same as above, except just the joint angles*
#

# %%
class GaitModel(nn.Module):
    def __init__(self, layer_sizes, joint_mids):
        super(GaitModel, self).__init__()
        
        self.joint_mids = joint_mids

        hidden_layers = [
            nn.Sequential(nn.Linear(nlminus1, nl), nn.ReLU(), nn.BatchNorm1d(nl))
            for nl, nlminus1 in zip(layer_sizes[1:-1], layer_sizes)
        ]

        # The output layer does not include an activation function.
        # See: https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html
        output_layer = nn.Linear(layer_sizes[-2], layer_sizes[-1])

        tanh = torch.nn.Tanh()

        # Group all layers into the sequential container
        all_layers = hidden_layers + [output_layer] + [tanh]
        self.layers = nn.Sequential(*all_layers)

    def forward(self, X):
        X -= joint_mids
        return self.layers(X)


# %% [markdown]
# ## Define Run function

# %%
def run(train_dataset, test_dataset, epochs=4, layer_sizes=[33, 31, 30, 28]):
    # Batch size is the number of training examples used to calculate each iteration's gradient
    batch_size_train = 33

    data_loader_train = DataLoader(
        dataset=train_dataset, batch_size=batch_size_train, shuffle=True
    )
    data_loader_test = DataLoader(
        dataset=test_dataset, batch_size=len(test_dataset), shuffle=False
    )

    # Define the hyperparameters
    learning_rate = 1e-3

    pytorch_model = GaitModel(layer_sizes)

    # Initialize the optimizer with above parameters
    optimizer = optim.Adam(pytorch_model.parameters(), lr=learning_rate)

    # Define the loss function
    loss_fn = nn.MSELoss()  # mean squared error

    # Train and get the resulting loss per iteration
    loss = train(
        model=pytorch_model,
        loader=data_loader_train,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=epochs,
    )

    # Test and get the resulting predicted y values
    y_predict = test(model=pytorch_model, loader=data_loader_test)

    return loss, y_predict, pytorch_model


# %% [markdown]
# ## Create Datasets for Each Gait File

# %%
angles_path = Path("Data")
names_and_ds = []

for filename in angles_path.glob("*.csv"):

    gait_name = filename.stem.split("_")[0]
    train_ds, test_ds = create_datasets(filename)
    names_and_ds.append((gait_name, train_ds, test_ds))

# %% [markdown]
# ## Run and plot results

# %%
for name, train_ds, test_ds in names_and_ds:

    train_ds.x_data.shape

    losses, y_predict, model_to_save = run(
        train_dataset=train_ds, test_dataset=test_ds, epochs=400
    )
    torch.save(model_to_save, Path("Models") / f"{name}_model.pt")

    print(f"Final loss for {name}: {sum(losses[-100:])/100}")
    plot_loss(losses, name)

# %%
# !head ./Data/trot_angles.csv

# %%
df_gait = pd.read_csv("./Data/trot_angles.csv", header=None)
df_gait

# %%
df_gait.describe()

# %%
df_gait.plot(subplots=True, layout=(6, 6), figsize=(16, 16))

# %%
