import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import common


def descriminator(input_size, hidden_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
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


def plot_hist(real_samples, gen_samples, title=None, nbins=20):
    """Plots two histograms: one for the real distribution (blue) and the other for the
    learned distribution (red)
    """
    # Clear the current plot
    ax = plt.gca()
    ax.clear()

    # Draw the histograms
    n, bins, _ = ax.hist(real_samples, bins=nbins, color="b", alpha=0.5)
    ax.hist(gen_samples, bins=bins, color="r", alpha=0.5)

    # Set the limits of the y-axis to a slightly larger value than the total number of
    # samples given by the sum
    ax.set_ylim([0, n.sum() * 1.1])
    if title is not None:
        plt.title(title)
    plt.legend(["Real distribution", "Generated distribution"])
    plt.pause(0.0001)


if __name__ == "__main__":
    # General settings
    device = "cuda"
    num_epochs = 5000
    num_samples_disp = 10000
    num_bins = 50

    # Real data (normal distribution) settings
    mean = 10
    std = 2

    # Learning rate (d - descriminator; g - generator)
    d_lr = 0.001
    g_lr = 0.001
    momentum = 0.9

    # Number of neurons in each hidden layer
    d_hidden_size = 25
    g_hidden_size = 10

    # Number of iterations for each model in a single epoch
    d_steps = 1
    g_steps = 1

    # Batch size for each model
    d_batch_size = 256
    g_batch_size = 256

    # GIF settings
    gif_path = "gif/gan_normal.gif"
    frame_ms = 100
    num_frames = 100
    num_repeat_last = 10
    figsize = (8, 8)

    # Compute the input dimension from the settings above
    d_sample_size = (d_batch_size, 1)
    g_sample_size = (g_batch_size, 1)

    # Initialize descriminator and generator models
    d_model = descriminator(1, d_hidden_size, 1)
    d_model.to(device)
    g_model = generator(1, g_hidden_size, 1)
    g_model.to(device)

    # Optimizers for each model
    d_optim = torch.optim.SGD(d_model.parameters(), lr=d_lr, momentum=momentum)
    g_optim = torch.optim.SGD(g_model.parameters(), lr=g_lr, momentum=momentum)

    # Binary cross entropy loss for both models
    criterion = nn.BCELoss()

    # Real and noise samples to use for visualization
    disp_size = (num_samples_disp, 1)
    x_disp = common.sample_normal(disp_size, mean=mean, std=std).numpy()
    z_disp = common.sample_uniform(disp_size).to(device)

    # Step size in epochs to get a frame and figure initialization
    frame_step = num_epochs // num_frames
    fig, ax = plt.subplots(figsize=figsize)
    pil_images = []

    for epoch in range(num_epochs):
        # Descriminator training loop - generally, d_steps = 1
        for d_step in range(d_steps):
            # Get data sampled from the real distribution and the noise distribution
            x = common.sample_normal(d_sample_size, mean=mean, std=std).to(device)
            z = common.sample_uniform(g_sample_size).to(device)

            # Train the descriminator
            out = common.train_descriminator(x, z, d_model, g_model, criterion, d_optim)
            ((d_real, loss_real), (d_g_noise, loss_noise)) = out
            d_loss = loss_real + loss_noise

        # Generator training loop
        for g_step in range(g_steps):
            # Get data sampled from the noise distribution and train
            z = common.sample_uniform(g_sample_size).to(device)
            g_loss = common.train_generator(z, d_model, g_model, criterion, g_optim)

        print(
            "Epoch {}/{}\tLoss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}".format(
                epoch, num_epochs - 1, d_loss, g_loss, d_real, d_g_noise
            )
        )

        if (epoch + 1) % frame_step == 0 or epoch + 1 == num_epochs:
            # Display the real distribution and the the distribution learned by the
            # generator
            with torch.no_grad():
                g_z_disp = g_model(z_disp).detach()
            g_z_disp = g_z_disp.cpu().numpy()
            plot_hist(x_disp, g_z_disp, title="Epoch " + str(epoch + 1), nbins=num_bins)
            fig.canvas.draw()

            # Convert the plot to a PIL image and store it to create a GIF later
            img = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            pil_images.append(img)

    common.save_gif(
        gif_path, pil_images, duration=frame_ms, num_repeat_last=num_repeat_last
    )
