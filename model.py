import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoEncoder(nn.Module):
    def __init__(self, feature_dim=1024, z_dim=256):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder
        self.encConv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16) # BatchNorm for first conv layer
        self.encConv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32) # BatchNorm for second conv layer
        self.encConv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64) # BatchNorm for third conv layer
        self.encConv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128) # BatchNorm for fourth conv layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, feature_dim)
        self.fc1_bn = nn.BatchNorm1d(feature_dim) # BatchNorm for the fc layer
        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

        # Decoder
        self.decFC1 = nn.Linear(z_dim, 128 * 16 * 16)
        self.decFC1_bn = nn.BatchNorm1d(128 * 16 * 16) # BatchNorm for decoder fc layer
        self.decConv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decConv1_bn = nn.BatchNorm2d(64) # BatchNorm for first deconv layer
        self.decConv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decConv2_bn = nn.BatchNorm2d(32) # BatchNorm for second deconv layer
        self.decConv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decConv3_bn = nn.BatchNorm2d(16) # BatchNorm for third deconv layer
        self.decConv4 = nn.ConvTranspose2d(16, 3, kernel_size=4, stride=2, padding=1)

    def encode(self, x):
        x = F.relu(self.bn1(self.encConv1(x)))
        x = F.relu(self.bn2(self.encConv2(x)))
        x = F.relu(self.bn3(self.encConv3(x)))
        x = F.relu(self.bn4(self.encConv4(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1_bn(self.fc1(x)))
        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        z = F.relu(self.decFC1_bn(self.decFC1(z)))
        z = z.view(-1, 128, 16, 16)
        z = F.relu(self.decConv1_bn(self.decConv1(z)))
        z = F.relu(self.decConv2_bn(self.decConv2(z)))
        z = F.relu(self.decConv3_bn(self.decConv3(z)))
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

