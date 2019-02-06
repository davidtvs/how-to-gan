import torch
import torch.nn as nn
from torchvision.datasets import CIFAR10
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from PIL import Image
import common


def descriminator(nbase_channels=128):
    model = nn.Sequential(
        # 32x32x3
        nn.Conv2d(3, nbase_channels, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(nbase_channels),
        # 16x16x128
        nn.Conv2d(nbase_channels, 2 * nbase_channels, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(2 * nbase_channels),
        # 8x8x256
        nn.Conv2d(2 * nbase_channels, 4 * nbase_channels, 3, stride=2, padding=1),
        nn.LeakyReLU(0.2),
        nn.BatchNorm2d(4 * nbase_channels),
        # 4x4x512
        nn.Conv2d(4 * nbase_channels, 1, 4, padding=0),
        # 1x1x1
        nn.Sigmoid(),
    )

    return model


def generator(ninput_channels, nbase_channels=128):
    model = nn.Sequential(
        # 1x1x100
        nn.ConvTranspose2d(ninput_channels, 4 * nbase_channels, 4),
        nn.ReLU(),
        nn.BatchNorm2d(4 * nbase_channels),
        # 4x4x512
        nn.ConvTranspose2d(4 * nbase_channels, 2 * nbase_channels, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(2 * nbase_channels),
        # 8x8x256
        nn.ConvTranspose2d(2 * nbase_channels, nbase_channels, 3, stride=2, padding=1, output_padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(nbase_channels),
        # 16x16x128
        nn.ConvTranspose2d(nbase_channels, 3, 3, stride=2, padding=1, output_padding=1),
        nn.Tanh(),
    )

    return model


def init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        nn.init.normal_(m.weight, std=0.02)


if __name__ == "__main__":
    # General settings
    device = "cuda"
    num_epochs = 10

    # Learning rate (d - descriminator; g - generator)
    d_lr = 0.0002
    g_lr = 0.0002
    beta1 = 0.5
    beta2 = 0.999

    # Input size
    g_input_dim = 100

    # Batch size for each model
    d_batch_size = 128
    g_batch_size = 128

    # GIF settings
    gif_path = "gif/dcgan_cifar10.gif"
    frame_ms = 100
    num_frames = 100
    num_repeat_last = 10
    num_images = 16
    num_img_row = 4
    figsize = (4.8, 4.8)

    # Dataset normalization
    mean, std = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

    # Compute the input dimension from the settings above
    g_sample_size = (g_batch_size, g_input_dim, 1, 1)

    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    train_set = CIFAR10("../data", train=True, download=True, transform=tf)
    train_loader = DataLoader(train_set, batch_size=d_batch_size, shuffle=True, num_workers=4)

    # Get a batch of training data
    inputs, classes = next(iter(train_loader))
    d_input_dim = inputs[0].size()[0]
    d_sample_size = (d_batch_size, d_input_dim)

    # Initialize descriminator and generator models
    d_model = descriminator().apply(init_weights)
    d_model.to(device)
    g_model = generator(g_input_dim).apply(init_weights)
    g_model.to(device)

    # Optimizers for each model
    d_optim = torch.optim.Adam(d_model.parameters(), lr=d_lr, betas=(beta1, beta2))
    g_optim = torch.optim.Adam(g_model.parameters(), lr=g_lr, betas=(beta1, beta2))

    # Binary cross entropy loss for both models
    criterion = nn.BCELoss()

    # Noise sample to use for visualization
    test_noise = common.sample_normal(g_sample_size).to(device)

    # Step size in epochs to get a frame and figure initialization
    frame_step = (num_epochs * len(train_loader)) // num_frames
    pil_grids = []
    fig, ax = plt.subplots(figsize=figsize)

    iteration = 0
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

            if iteration % frame_step == 0 or (epoch == num_epochs - 1 and batch_idx == len(train_loader) - 1):
                # Display images from the generator
                with torch.no_grad():
                    g_z = g_model(test_noise).detach().cpu()

                g_z = g_z.view(g_batch_size, 3, 32, 32)
                g_z_grid = torchvision.utils.make_grid(g_z[:num_images], nrow=num_img_row)
                g_z_grid_np = g_z_grid.numpy().transpose((1, 2, 0))
                common.imshow(g_z_grid_np, title="Iteration " + str(iteration + 1), mean=mean, std=std)
                fig.canvas.draw()

                # Convert the plot to a PIL image and store it to create a GIF later
                img = Image.frombytes("RGB", fig.canvas.get_width_height(), fig.canvas.tostring_rgb())
                pil_grids.append(img)

            iteration += 1

        # Clear the line (assuming 80 chars is enough) and then print the epoch results
        print(80 * " ", end="\r")
        print(
            "Epoch {}/{}\tLoss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}".format(
                epoch, num_epochs - 1, d_loss, g_loss, d_real, d_g_noise
            )
        )

    common.save_gif(gif_path, pil_grids, duration=frame_ms, num_repeat_last=num_repeat_last)
