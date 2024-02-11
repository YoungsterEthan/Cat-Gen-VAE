import torch.nn as nn
import torch.nn.functional as F
import torch

class VAE(nn.Module):
    def __init__(self, h_dim, z_dim):
        super(VAE, self).__init__()
        #encoder

        self.conv = nn.Conv2d(3, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)

        self.fc_size = 32 * 128 * 128
        self.img_2hid = nn.Linear(self.fc_size, h_dim)
        self.hid_2mu = nn.Linear(h_dim, z_dim)
        self.hid_2sigma = nn.Linear(h_dim, z_dim)

        #decoder
        self.z_2hid = nn.Linear(z_dim, h_dim)
        self.hid_2img = nn.Linear(h_dim, self.fc_size)
        self.convTranspose1 = nn.ConvTranspose2d(32, 32, kernel_size=2, stride=2)
        self.convTranspose2 = nn.ConvTranspose2d(32, 3, kernel_size=3, stride=1, padding=1)


    def encoder(self, X):
        x = F.relu(self.conv(X))
        x = self.pool(x)
        x = x.view(-1, self.fc_size)
        h = F.relu(self.img_2hid(x))
        mu, sigma = self.hid_2mu(h), self.hid_2sigma(h)

        return mu, sigma
    
    def reparameterize(self, mu, sigma):
        std = torch.exp(0.5 * sigma)
        eps = torch.randn_like(std)
        return mu + std * eps
    
    def decoder(self, z):
        x = F.relu(self.z_2hid(z))
        x = F.relu(self.hid_2img(x))
        x = x.view(-1, 32, 128, 128)
        x = F.relu(self.convTranspose1(x))
        return F.sigmoid(self.convTranspose2(x))
    
    def forward(self,x):
        mu, sigma = self.encoder(x)
        z = self.reparameterize(mu, sigma)
        return self.decoder(z), mu, sigma