import torch
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm
from scipy.stats.kde import gaussian_kde
from PIL import Image
import common


def descriminator(input_size, hidden_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, hidden_size),
        nn.LeakyReLU(0.2),
        nn.Linear(hidden_size, output_size),
        nn.Sigmoid(),
    )

    return model


def generator(input_size, hidden_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size),
    )

    return model


def plot_hist(xs, normal_pdf, g_z, decision_boudnary, title=None):
    """Plots two histograms: one for the real distribution (blue) and the other for the
    learned distribution (red)
    """
    # Clear the current plot
    ax = plt.gca()
    ax.clear()

    # Draw the histograms
    ax.plot(xs, normal_pdf, "--")
    ax.plot(xs, gaussian_kde(g_z)(xs))
    ax.plot(xs, decision_boudnary, linestyle=":")
    ax.set_ylim([0, 1])
    if title is not None:
        plt.title(title)
    plt.legend(["Real (normal) distribution", "Learned distribution (G(z))", "Descriminator decision boundary"], loc="upper right")
    plt.pause(0.0001)


if __name__ == "__main__":
    # General settings
    device = "cuda"
    num_iter = 5000

    # Real data (normal distribution) settings
    mean = 4
    std = 1.25

    # Learning rate (d - descriminator; g - generator)
    d_lr = 0.001
    g_lr = 0.001

    # Number of neurons in each hidden layer
    d_hidden_size = 64
    g_hidden_size = 32

    # Number of iterations for each model in a single epoch
    d_steps = 1
    g_steps = 1

    # Batch size for each model
    batch_size = 1024

    # GIF settings
    gif_path = "gif/gan_normal.gif"
    frame_ms = 100
    num_frames = 100
    num_repeat_last = 10
    figsize = (8, 8)

    # Compute the input dimension from the settings above
    sample_size = (batch_size, 1)

    # Initialize descriminator and generator models
    d_model = descriminator(1, d_hidden_size, 1)
    d_model.to(device)
    g_model = generator(1, g_hidden_size, 1)
    g_model.to(device)

    # Optimizers for each model
    d_optim = torch.optim.Adam(d_model.parameters(), lr=d_lr, betas=(0.5, 0.999))
    d_scheduler = StepLR(d_optim, step_size=2000, gamma=0.5)
    g_optim = torch.optim.Adam(g_model.parameters(), lr=g_lr, betas=(0.5, 0.999))
    g_scheduler = StepLR(g_optim, step_size=2000, gamma=0.5)

    # Binary cross entropy loss for both models
    criterion = nn.BCELoss()

    # Get the normal distribution probability density function in the relevant interval
    xs_test = np.linspace(mean - 3 * std, mean + 3 * std, batch_size)
    x_test = norm.pdf(xs_test, mean, std)

    # xs_test needs to be converted to a tensor get the descriminator's decision
    # boundary. Also need noise samples to test the generator
    xs_test = torch.tensor(xs_test, dtype=torch.float, device=device).unsqueeze(-1)
    z_test = common.sample_uniform(sample_size).to(device)

    # Step size in epochs to get a frame and figure initialization
    frame_step = num_iter // num_frames
    fig, ax = plt.subplots(figsize=figsize)
    pil_images = []

    for iteration in range(num_iter):
        # Descriminator training loop - generally, d_steps = 1
        for d_step in range(d_steps):
            d_scheduler.step()
            g_scheduler.step()

            # Get data sampled from the real distribution and the noise distribution
            x = common.sample_normal(sample_size, mean=mean, std=std).to(device)
            z = common.sample_uniform(sample_size).to(device)

            # Train the descriminator
            out = common.train_descriminator(x, z, d_model, g_model, criterion, d_optim)
            ((d_real, loss_real), (d_g_noise, loss_noise)) = out
            d_loss = loss_real + loss_noise

        # Generator training loop
        for g_step in range(g_steps):
            # Get data sampled from the noise distribution and train
            z = common.sample_uniform(sample_size).to(device)
            g_loss = common.train_generator(z, d_model, g_model, criterion, g_optim)

        print(
            "Epoch {}/{}\tLoss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}".format(
                iteration, num_iter - 1, d_loss, g_loss, d_real, d_g_noise
            )
        )

        if iteration % frame_step == 0 or iteration == num_iter - 1:
            # Display the real distribution and the the distribution learned by the
            # generator
            with torch.no_grad():
                boundary = d_model(xs_test).detach()
                g_z_test = g_model(z_test).detach()

            xs_np = xs_test.cpu().flatten().numpy()
            boundary = boundary.cpu().flatten().numpy()
            g_z_test = g_z_test.cpu().flatten().numpy()
            plot_hist(xs_np, x_test, g_z_test, boundary, title="Iteration " + str(iteration + 1))
            fig.canvas.draw()

            # Convert the plot to a PIL image and store it to create a GIF later
            img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
            pil_images.append(img)

    common.save_gif(gif_path, pil_images, duration=frame_ms, num_repeat_last=num_repeat_last)
