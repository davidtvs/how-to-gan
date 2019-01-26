import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import math


def sample_generator(size):
    return torch.rand(size, dtype=torch.float)


def sample_data(size, mean=0, std=1):
    mean = torch.ones(size, dtype=torch.float) * torch.tensor(mean, dtype=torch.float)
    std = torch.ones(size, dtype=torch.float) * torch.tensor(std, dtype=torch.float)
    return torch.normal(mean, std)


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
    )

    return model


class PlotDistributions:
    def __init__(
        self, num_plots, mean, std, num_samples=100, num_bins=20, device="cpu"
    ):
        if num_plots > 5:
            self.ncols = 5
            self.nrows = math.ceil(num_plots / self.ncols)
        else:
            self.ncols = num_plots
            self.nrows = 1
        self.mean = mean
        self.std = std
        self.num_samples = num_samples
        self.nbins = num_bins
        self.device = device
        self.reset()

    def reset(self):
        self.fig, self.ax = plt.subplots(nrows=self.nrows, ncols=self.ncols)
        self.ax_idx = 0

    def add(self, g_model, title=None):
        # Get samples from the real data and generate the same ammount of samples using
        # the passed generator model
        with torch.no_grad():
            X, G_Z = [], []
            sample_size = (self.num_samples, 1)
            x = sample_data(sample_size, mean=self.mean, std=self.std).to(self.device)
            z = sample_generator(sample_size).to(self.device)
            g_z = g_model(z)

            X = x.cpu().numpy()
            G_Z = g_z.cpu().numpy()

        if self.nrows == 1:
            # Single row of subplots -> use current index to get the axis directly
            ax = self.ax[self.ax_idx]
        else:
            # Get row and column of subplot from the current axis index
            col = self.ax_idx % self.ncols
            row = self.ax_idx // self.ncols
            ax = self.ax[row, col]

        _, bins, _ = ax.hist(X, bins=self.nbins, color="b", alpha=0.5)
        ax.hist(G_Z, bins=bins, color="r", alpha=0.5)

        if isinstance(title, str):
            ax.set_title(title)

        self.ax_idx += 1

    def show(self):
        x_patch = mpatches.Patch(color="b", alpha=0.5, label="Real distribution")
        gz_patch = mpatches.Patch(color="r", alpha=0.5, label="Generated distribution")
        self.fig.legend(handles=[x_patch, gz_patch], loc="lower center", ncol=2)
        plt.show()


if __name__ == "__main__":
    # General settings
    device = "cuda"
    num_epochs = 2000
    num_samples = 1000
    plot_step = 200

    # Real data (normal distribution) settings
    mean = 10
    std = 2

    # Learning rate (d - descriminator; g - generator)
    d_lr = 0.001
    g_lr = 0.001

    # Input size
    d_input_samples = 1
    g_input_samples = 1

    # Number of neurons in each hidden layer
    d_hidden_size = 25
    g_hidden_size = 10

    # Number of iterations for each model in a single epoch
    d_steps = 1
    g_steps = 1

    # Batch size for each model
    d_batch_size = 256
    g_batch_size = 256

    # Compute the input dimension from the settings above
    d_sample_size = (d_batch_size, d_input_samples)
    g_sample_size = (g_batch_size, g_input_samples)

    # Initialize descriminator and generator models
    d_model = descriminator(input_size=d_input_samples, hidden_size=d_hidden_size)
    d_model.to(device)
    g_model = generator(input_size=g_input_samples, hidden_size=g_hidden_size)
    g_model.to(device)

    # Optimizers for each model
    d_optim = torch.optim.SGD(d_model.parameters(), lr=d_lr, momentum=0.9)
    g_optim = torch.optim.SGD(g_model.parameters(), lr=g_lr, momentum=0.9)

    # Binary cross entropy loss for both models
    criterion = nn.BCELoss()

    # Initialize the class responsible for plotting histograms as training progresses
    num_plots = num_epochs // plot_step
    plot_dist = PlotDistributions(
        num_plots, mean, std, num_samples=num_samples, device=device
    )

    for epoch in range(num_epochs):
        # Descriminator training loop
        for d_step in range(d_steps):
            # --------------------------------------------------------------------------
            # Train the descriminator on correct data
            # --------------------------------------------------------------------------
            # 1. Get data sampled from the real distribution
            x = sample_data(d_sample_size, mean=mean, std=std).to(device)

            # 2. Forward propagation
            d_model.zero_grad()
            output = d_model(x)
            D_x = output.mean().item()

            # 3. Compute loss and gradient
            y = torch.ones_like(output)
            loss_real = criterion(output, y)
            loss_real.backward()

            # --------------------------------------------------------------------------
            # Train the descriminator on data from the generator
            # --------------------------------------------------------------------------
            # 1. Get data from the generator using noise
            z = sample_generator(d_sample_size).to(device)
            # Note that there is no need to detach from g_model because the gradient
            # will be zeroed later
            g_z = g_model(z)

            # 2. Forward propagation. Can't do zero grad here, else the gradient from
            # the real data is lost
            output = d_model(g_z)
            D_G_z = output.mean().item()

            # 3. Compute loss and gradient
            y = torch.zeros_like(output)
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
            z = sample_generator(g_sample_size).to(device)

            # 2. Forward propagation
            g_model.zero_grad()
            g_z = g_model(z)
            output = d_model(g_z)

            # 3. Compute loss and gradient
            y = torch.ones_like(output)
            g_loss = criterion(output, y)
            g_loss.backward()
            g_optim.step()

        if (epoch + 1) % plot_step == 0 or epoch + 1 == num_epochs:
            plot_dist.add(g_model, title="Epoch " + str(epoch))

        print(
            "Epoch {}/{}\tLoss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}".format(
                epoch, num_epochs - 1, d_loss, g_loss, D_x, D_G_z
            )
        )
        print()

    plot_dist.show()
