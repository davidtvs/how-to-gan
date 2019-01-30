import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torchvision.datasets import MNIST
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader


def noise_sample(size):
    return torch.randn(size, dtype=torch.float)


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


def imshow(tensor, title=None):
    images = tensor.numpy().transpose((1, 2, 0))
    plt.imshow(images)
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)


if __name__ == "__main__":
    # General settings
    device = "cuda"
    num_epochs = 200

    # Learning rate (d - descriminator; g - generator)
    d_lr = 0.00015
    g_lr = 0.00015

    # Input size
    g_input_dim = 100

    # Batch size for each model
    d_batch_size = 256
    g_batch_size = 256

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
    test_noise = noise_sample(g_sample_size).to(device)

    for epoch in range(num_epochs):
        for batch_idx, (inputs, _) in enumerate(train_loader):
            # Train the descriminator on correct data:
            # 1. Get data sampled from the real distribution
            x = inputs.to(device)

            # 2. Forward propagation
            d_model.zero_grad()
            output = d_model(x)
            D_x = output.mean().item()

            # 3. Compute loss and gradient
            y = torch.ones_like(output)
            loss_real = criterion(output, y)
            loss_real.backward()

            # Train the descriminator on data from the generator:
            # 1. Get data from the generator using noise
            z = noise_sample(g_sample_size).to(device)
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

            # Train the generator on fake data:
            # 1. Get data sampled from the uniform distribution
            z = noise_sample(g_sample_size).to(device)

            # 2. Forward propagation
            g_z = g_model(z)
            g_model.zero_grad()
            output = d_model(g_z)

            # 3. Compute loss and gradient
            y = torch.ones_like(output)
            g_loss = criterion(output, y)
            g_loss.backward()
            g_optim.step()

            if batch_idx % 10 == 0 or batch_idx == len(train_loader) - 1:
                print(
                    "Batch {}/{}\tLoss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}".format(
                        batch_idx, len(train_loader) - 1, d_loss, g_loss, D_x, D_G_z
                    ),
                    end="\r",
                )

        print(
            "Epoch {}/{}\tLoss_D: {:.4f} Loss_G: {:.4f} D(x): {:.4f} D(G(z)): {:.4f}".format(
                epoch, num_epochs - 1, d_loss, g_loss, D_x, D_G_z
            )
        )
        g_z = g_model(test_noise).detach().cpu()
        g_z = g_z.view(g_batch_size, 1, 28, 28)
        g_z_grid = torchvision.utils.make_grid(g_z[:16], nrow=4)
        imshow(g_z_grid)
