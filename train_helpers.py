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
import os
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
    transformations = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor()
    ])
    dataset = ImageFolder(root="C:/Users/Ethan/Desktop/VAE/train/", transform=transformations)
    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader

def train_and_visualize_fixed_images(model, train_loader, num_epochs, device, fixed_images, path):
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.BCELoss(reduction="sum")
    
    plt.ion()  # Turn on interactive mode
    fig, axes = plt.subplots(2, len(fixed_images), figsize=(15, 5))  # Adjust subplot size accordingly
    
    fixed_images = fixed_images.to(device)

    plots_dir = 'training_plots'
    os.makedirs(plots_dir, exist_ok=True)
    
    for epoch in range(num_epochs):
        for batch_idx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            model.train()
            optimizer.zero_grad()
            
            # Forward pass
            x_reconstructed, mu, log_var = model(x)
            
            # Compute loss
            reconstruction_loss = loss_fn(x_reconstructed, x)
            kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
            loss = reconstruction_loss + kl_div
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:  # Log loss
                print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')
                
        # Visualization update at the end of each epoch
        with torch.no_grad():
            model.eval()
            fixed_reconstructed, _, _ = model(fixed_images)
            
            for i in range(len(fixed_images)):
                original = fixed_images[i].cpu()
                reconstructed = fixed_reconstructed[i].cpu()
                
                axes[0, i].imshow(original.permute(1, 2, 0))
                axes[1, i].imshow(reconstructed.permute(1, 2, 0))
                axes[0, i].axis('off')
                axes[1, i].axis('off')
                
                if epoch == 0:  # Set titles only in the first epoch
                    axes[0, i].set_title(f"Original {i+1}")
                    axes[1, i].set_title(f"Reconstructed {i+1}")
                    
            # Save the current figure
            fig.savefig(os.path.join(plots_dir, f'training_epoch_{epoch+1}.png'))
            plt.pause(0.1)  
    
    plt.ioff() 
    plt.show()

    # Save the model after training
    torch.save(model.state_dict(), path)

def generate_images(model, num_samples=20, z_dims=30):
    with torch.no_grad():
        z = torch.randn(num_samples, z_dims)
        print("z")
        print(z)
        print("z-dim", z.shape)
        generate_image = model.decode(z)
        print("img")
        print(generate_image)
        print(generate_image.shape)

        return generate_image


def show_images(images, num_images=5):
    # Determine the number of rows needed (each row will have up to 5 images)
    num_rows = num_images // 5 + (1 if num_images % 5 else 0)

    fig, axes = plt.subplots(nrows=num_rows, ncols=min(num_images, 5), figsize=(10, 2*num_rows), squeeze=False)

    axes_flat = axes.flatten()

    for i in range(num_images):
        row, col = divmod(i, 5)
        img = np.moveaxis(images[i], 0, -1)
        ax = axes_flat[i]
        ax.imshow(img)
        ax.axis('off')
    for j in range(num_images, len(axes_flat)):
        axes_flat[j].axis('off')
    
    plt.tight_layout()
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
        z = model.reparameterize(mu, sigma)
        print("z")
        print(z)
        print("z-dim")
        print(z.shape)
        reconstructed_img, _, _ = model(image)
        print("reconstructed")
        print(reconstructed_img)
        print(reconstructed_img.shape)

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
