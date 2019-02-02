import torch
import matplotlib.pyplot as plt


def imshow(image, title=None):
    plt.imshow(image)
    if title is not None:
        plt.title(title)
    plt.pause(0.0001)


def save_gif(path, images, duration=30, num_repeat_last=1):
    images.extend([images[-1]] * num_repeat_last)
    images[0].save(
        path, save_all=True, append_images=images[1:], duration=duration, loop=0
    )


def sample_normal(size, mean=0, std=1):
    mean = torch.ones(size, dtype=torch.float) * torch.tensor(mean, dtype=torch.float)
    std = torch.ones(size, dtype=torch.float) * torch.tensor(std, dtype=torch.float)

    return torch.normal(mean, std)


def sample_uniform(size):
    return torch.rand(size, dtype=torch.float)


def train_descriminator(real, noise, d_model, g_model, criterion, optimizer):
    # Train the descriminator on correct data:
    # Forward propagation
    d_model.zero_grad()
    output = d_model(real)
    d_real = output.mean().item()

    # Compute loss and gradient
    y = torch.ones_like(output)
    loss_real = criterion(output, y)
    loss_real.backward()

    # Train the descriminator on data from the generator:
    # Note that there is no need to detach from g_model because the gradient will be
    # zeroed later
    g_noise = g_model(noise)

    # Forward propagation (can't do zero grad here, else the gradient from the real data
    # is lost)
    output = d_model(g_noise)
    d_g_noise = output.mean().item()

    # Compute loss and gradient
    y = torch.zeros_like(output)
    loss_noise = criterion(output, y)
    loss_noise.backward()
    optimizer.step()

    return (d_real, loss_real.item()), (d_g_noise, loss_noise.item())


def train_generator(noise, d_model, g_model, criterion, optimizer):
    # Forward propagation
    g_model.zero_grad()
    g_z = g_model(noise)
    output = d_model(g_z)

    # Compute loss and gradient
    y = torch.ones_like(output)
    g_loss = criterion(output, y)
    g_loss.backward()
    optimizer.step()

    return g_loss.item()
