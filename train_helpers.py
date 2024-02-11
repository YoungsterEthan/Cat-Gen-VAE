import torch
import torchvision.datasets as datasets
from torchvision.datasets import ImageFolder
from tqdm import tqdm
from torch import nn
from model2 import VAE
from torchvision import transforms
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision.io import read_image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import matplotlib.pyplot as plt
from model import VariationalAutoEncoder


def visualize_reconstruction(model, data_loader, device, n_images=5):
    model.eval()
    images, _ = next(iter(data_loader))
    images = images[:n_images].to(device)
    with torch.no_grad():
        reconstructed, _, _ = model(images)
    
    images = images.cpu().numpy()
    reconstructed = reconstructed.cpu().numpy()
    
    fig, axs = plt.subplots(2, n_images, figsize=(15, 4))
    for i in range(n_images):
        axs[0, i].imshow(np.transpose(images[i], (1, 2, 0)))
        axs[0, i].axis('off')
        axs[1, i].imshow(np.transpose(reconstructed[i], (1, 2, 0)))
        axs[1, i].axis('off')
    
    axs[0, 0].set_ylabel('Original')
    axs[1, 0].set_ylabel('Reconstructed')
    plt.show()

def load_data(root="dataset/", batch_size=128):
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root="C:/Users/Ethan/Desktop/VAE/train/", transform=transformations)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True)

def train(dataloader, model, num_epochs, device, lr, path):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCELoss(reduction="sum")

    for epoch in range(num_epochs):
        loop = tqdm(enumerate(dataloader))
        for i, (x,_) in loop:
            x = x.to(device)
            x_reconstructed, mu, sigma = model(x)

            #compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = -0.5 * torch.sum(1+ torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            #backprop
            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())
        if epoch % 10 == 0:
            visualize_reconstruction(model, dataloader, 'cuda', 5)

    torch.save(model.state_dict(), f'{path}/cat_generator3.pth')

def generate_images(model, num_samples=20, z_dims=30):
    with torch.no_grad():
        z = torch.randn(num_samples, z_dims)
        generate_images = model.decode(z)

        return generate_images

def show_images(images, num_images=5):
    fig, axes = plt.subplots(nrows=1, ncols=num_images, figsize=(10,2))
    for i, ax in enumerate(axes):
        img = np.moveaxis(images[i], 0, -1)
        ax.imshow(img)
        ax.axis('off')
    plt.show()