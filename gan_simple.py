import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import math


def sample_generator():
    return torch.rand(1, dtype=torch.float)


def sample_data():
    return torch.randn(1, dtype=torch.float)


def descriminator(input_size=1, hidden_size=10, output_size=1):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.Sigmoid(),
    )

    return model


def generator(input_size=1, hidden_size=10, output_size=1):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
        nn.Tanh(),
    )

    return model


class PlotDistributions:
    def __init__(self, num_plots, num_samples=100, device="cpu"):
        if num_plots > 5:
            self.ncols = 5
            self.nrows = math.ceil(num_plots / ncols)
        else:
            self.ncols = num_plots
            self.nrows = 1
        self.num_samples = num_samples
        self.device = device
        self.reset()

    def reset(self):
        self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols)
        self.ax_idx = 0

    def add(self, g_model):
        with torch.no_grad():
            X, G_Z = [], []
            for i in range(num_samples):
                x = sample_data().to(self.device)
                z = sample_generator().to(self.device)
                g_z = g_model(z)
                print("Z:", z)
                print("G(z):", g_z)

                X.append(x.cpu().item())
                G_Z.append(g_z.cpu().item())

        row = self.ax_idx // self.nrows
        col = self.ax_idx % self.ncols
        self.ax[row, col].scatter(X, num_samples * [0], marker="o", color="b")
        self.ax[row, col].scatter(G_Z, num_samples * [0], marker="x", color="r")
        self.ax_idx += 1

    def show(self):
        plt.show()


if __name__ == "__main__":
    device = "cpu"
    num_epochs = 50
    d_lr = 0.1
    g_lr = 0.1
    d_steps = 1
    g_steps = 1
    num_samples = 25
    plot_step = 5

    d_model = descriminator().to(device)
    g_model = generator().to(device)

    d_optim = torch.optim.SGD(d_model.parameters(), lr=d_lr, momentum=0.9)
    g_optim = torch.optim.SGD(g_model.parameters(), lr=g_lr, momentum=0.9)

    criterion = nn.BCELoss()

    num_plots = num_epochs // plot_step + 1
    plot_dist = PlotDistributions(num_plots, num_samples=num_samples, device=device)

    for epoch in range(num_epochs):
        # Descriminator training loop
        for d_step in range(d_steps):
            # --------------------------------------------------------------------------
            # Train the descriminator on correct data
            # --------------------------------------------------------------------------
            # 1. Get data sampled from the real distribution
            x = sample_data().to(device)

            # 2. Forward propagation
            d_model.zero_grad()
            output = d_model(x)
            D_x = output.mean().item()

            # 3. Compute loss and gradient
            y = torch.ones_like(x)
            loss_real = criterion(output, y)
            loss_real.backward()

            # --------------------------------------------------------------------------
            # Train the descriminator on data from the generator
            # --------------------------------------------------------------------------
            # 1. Get data from the generator using noise
            z = sample_generator().to(device)
            # Note that there is no need to detach from g_model because the gradient
            # will be zeroed later
            g_z = g_model(z)

            # 2. Forward propagation. Can't do zero grad here, else the gradient from
            # the real data is lost
            output = d_model(g_z)
            D_G_z = output.mean().item()

            # 3. Compute loss and gradient
            y = torch.zeros_like(x)
            loss_fake = criterion(output, y)
            loss_fake.backward()
            d_optim.step()

            d_loss = (loss_real + loss_fake).item()

        # Generator training loop
        for g_step in range(g_steps):
            # --------------------------------------------------------------------------
            # Train the generator on fake data
            # --------------------------------------------------------------------------
            # 1. Get data sampled from the uniform distribution
            z = sample_generator().to(device)

            # 2. Forward propagation
            g_model.zero_grad()
            g_z = g_model(z)
            print(g_z)
            output = d_model(g_z)

            # 3. Compute loss and gradient
            y = torch.ones_like(g_z)
            g_loss = criterion(output, y)
            g_loss.backward()
            g_optim.step()

        if epoch % 5 == 0 or epoch == num_epochs - 1:
            plot_dist.add(g_model)

        print(
            "Epoch {}/{}\tLoss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}".format(
                epoch, num_epochs - 1, d_loss, g_loss, D_x, D_G_z
            )
        )
        print()

    plot_dist.show()
