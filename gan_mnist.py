import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import common


def descriminator(input_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, 240),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(240, 240),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(240, output_size),
        nn.Sigmoid(),
    )

    return model


def generator(input_size, output_size):
    model = nn.Sequential(
        nn.Linear(input_size, 1200),
        nn.ReLU(),
        nn.Linear(1200, 1200),
        nn.ReLU(),
        nn.Linear(1200, output_size),
        nn.Sigmoid(),
    )

    return model


if __name__ == "__main__":
    # General settings
    device = "cuda"
    num_epochs = 200

    # Learning rate (d - descriminator; g - generator)
    d_lr = 0.0002
    g_lr = 0.0002

    # Input size
    g_input_dim = 100

    # Batch size for each model
    d_batch_size = 256
    g_batch_size = 256

    # GIF settings
    gif_path = "gif/gan_mnist.gif"
    frame_ms = 100
    num_frames = 100
    num_repeat_last = 10
    num_images = 16
    num_img_row = 4
    figsize = (8, 8)

    # Compute the input dimension from the settings above
    g_sample_size = (g_batch_size, g_input_dim)

    tf = transforms.Compose([transforms.ToTensor(), lambda x: x.view(-1)])
    train_set = MNIST("../data", train=True, download=True, transform=tf)
    train_loader = DataLoader(
        train_set, batch_size=d_batch_size, shuffle=True, num_workers=4
    )

    # Get a batch of training data
    inputs, classes = next(iter(train_loader))
    d_input_dim = inputs[0].size()[0]
    d_sample_size = (d_batch_size, d_input_dim)

    # Initialize descriminator and generator models
    d_model = descriminator(d_input_dim, 1)
    d_model.to(device)
    g_model = generator(g_input_dim, d_input_dim)
    g_model.to(device)

    # Optimizers for each model
    d_optim = torch.optim.Adam(d_model.parameters(), lr=d_lr)
    g_optim = torch.optim.Adam(g_model.parameters(), lr=g_lr)

    # Binary cross entropy loss for both models
    criterion = nn.BCELoss()

    # Noise sample to use for visualization
    test_noise = common.sample_normal(g_sample_size).to(device)

    # Step size in epochs to get a frame and figure initialization
    frame_step = num_epochs // num_frames
    pil_grids = []
    fig, ax = plt.subplots(figsize=figsize)

    for epoch in range(num_epochs):
        for batch_idx, (inputs, _) in enumerate(train_loader):
            # Get data sampled from the real distribution and the noise distribution
            x = inputs.to(device)
            z = common.sample_normal(g_sample_size).to(device)

            # Train the descriminator
            out = common.train_descriminator(x, z, d_model, g_model, criterion, d_optim)
            ((d_real, loss_real), (d_g_noise, loss_noise)) = out
            d_loss = loss_real + loss_noise

            # Train the generator on fake data:
            # Get data sampled from the noise distribution and train
            z = common.sample_normal(g_sample_size).to(device)
            g_loss = common.train_generator(z, d_model, g_model, criterion, g_optim)

            # Print results every few batches
            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(
                    "Batch {}/{}\tLoss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}".format(
                        batch_idx,
                        len(train_loader) - 1,
                        d_loss,
                        g_loss,
                        d_real,
                        d_g_noise,
                    ),
                    end="\r",
                )

        # Clear the line (assuming 80 chars is enough) and then print the epoch results
        print(80 * " ", end="\r")
        print(
            "Epoch {}/{}\tLoss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}".format(
                epoch, num_epochs - 1, d_loss, g_loss, d_real, d_g_noise
            )
        )

        if (epoch + 1) % frame_step == 0 or epoch + 1 == num_epochs:
            # Display images from the generator
            with torch.no_grad():
                g_z = g_model(test_noise).detach().cpu()

            g_z = g_z.view(g_batch_size, 1, 28, 28)
            g_z_grid = torchvision.utils.make_grid(g_z[:num_images], nrow=num_img_row)
            g_z_grid_np = g_z_grid.numpy().transpose((1, 2, 0))
            common.imshow(g_z_grid_np, title="Epoch " + str(epoch + 1))
            fig.canvas.draw()

            # Convert the plot to a PIL image and store it to create a GIF later
            img = Image.frombytes(
                "RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb()
            )
            pil_grids.append(img)

    common.save_gif(
        gif_path, pil_grids, duration=frame_ms, num_repeat_last=num_repeat_last
    )
