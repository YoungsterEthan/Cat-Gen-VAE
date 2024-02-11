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
    train_loader = DataLoader(dataset=root, batch_size=batch_size, shuffle=True)

    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root="C:/Users/Ethan/Desktop/VAE/train/", transform=transformations)
    return DataLoader(dataset=dataset, batch_size=32, shuffle=True)

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


    torch.save(model.state_dict(), f'{path}')

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

def reconstruct_img(model, image_path):
    image = Image.open(image_path)
    transform = transforms.Compose([
        transforms.Resize((256, 256)),  # Resize the image to 256x256
        transforms.ToTensor(),  
    ])

    image = transform(image).unsqueeze(0) 


    with torch.no_grad():  
        mu, sigma = model.encode(image)
        print("Mu", mu)
        print("Sigma", sigma)
        reconstructed_img, _, _ = model(image)

    # Convert the tensor to a displayable format
    reconstructed_img = reconstructed_img.squeeze(0)  
    reconstructed_img = reconstructed_img.detach().cpu().numpy()  
    reconstructed_img = np.moveaxis(reconstructed_img, 0, -1)  

    # Display the original and regenerated images
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.imshow(image.squeeze(0).permute(1, 2, 0))
    plt.title('Original Image')
    plt.subplot(1, 2, 2)
    plt.imshow(reconstructed_img, cmap='gray')
    plt.title('Regenerated Image')
    plt.show()
