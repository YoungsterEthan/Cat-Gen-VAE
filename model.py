import torch
import torch.nn.functional as functional
from torch import nn 
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, feature_dim=1024, z_dim=256):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder
        self.encConv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1) # Output: [16, 128, 128]
        self.encConv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # Output: [32, 64, 64]
        self.encConv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # Output: [64, 32, 32]
        self.encConv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Output: [128, 16, 16]
        self.flatten = nn.Flatten() # Flattens the output for the linear layer
        self.fc1 = nn.Linear(128 * 16 * 16, feature_dim) # Fully connected layer to get latent vector
        self.fc_mu = nn.Linear(feature_dim, z_dim) # Latent vector mean
        self.fc_logvar = nn.Linear(feature_dim, z_dim) # Latent vector log-variance

        # Decoder
        self.decFC1 = nn.Linear(z_dim, 128 * 16 * 16)
        self.decConv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decConv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decConv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decConv4 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.encConv1(x))
        x = F.relu(self.encConv2(x))
        x = F.relu(self.encConv3(x))
        x = F.relu(self.encConv4(x))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z):
        z = F.relu(self.decFC1(z))
        z = z.view(-1, 128, 16, 16)
        z = F.relu(self.decConv1(z))
        z = F.relu(self.decConv2(z))
        z = F.relu(self.decConv3(z))
        z = torch.sigmoid(self.decConv4(z))
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


if __name__ == "__main__":
    # Create an instance of the VAE
    vae = VariationalAutoEncoder()

    # Example input tensor
    example_input = torch.rand(1, 3, 256, 256) # Batch size of 1

    # Forward pass
    reconstructed_img, mu, logvar = vae(example_input)

