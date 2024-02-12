import torch
import torch.nn as nn
import torch.nn.functional as F

class VariationalAutoEncoder_BN(nn.Module):
    def __init__(self, feature_dim=1024, z_dim=256):
        super(VariationalAutoEncoder_BN, self).__init__()
        
        # Encoder
        self.encConv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.encConv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.encConv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.encConv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, feature_dim)
        self.fc1_bn = nn.BatchNorm1d(feature_dim)
        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

        # Decoder
        self.decFC1 = nn.Linear(z_dim, 128 * 16 * 16)
        self.decFC1_bn = nn.BatchNorm1d(128 * 16 * 16)
        self.decConv1 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.decConv1_bn = nn.BatchNorm2d(64)
        self.decConv2 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.decConv2_bn = nn.BatchNorm2d(32)
        self.decConv3 = nn.ConvTranspose2d(32, 16, kernel_size=4, stride=2, padding=1)
        self.decConv3_bn = nn.BatchNorm2d(16)
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
    

class VariationalAutoEncoder(nn.Module):
    def __init__(self, feature_dim=1024, z_dim=256):
        super(VariationalAutoEncoder, self).__init__()
        
        # Encoder
        self.encConv1 = nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1) # Output: [16, 128, 128]
        self.encConv2 = nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1) # Output: [32, 64, 64]
        self.encConv3 = nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1) # Output: [64, 32, 32]
        self.encConv4 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1) # Output: [128, 16, 16]
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 16 * 16, feature_dim)
        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim) 

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

class VariationalAutoEncoder_BN_Pool(nn.Module):
    def __init__(self, feature_dim=1024, z_dim=256):
        super(VariationalAutoEncoder_BN_Pool, self).__init__()

        # Encoder
        self.encConv1 = self.create_conv_layer(3, 16)
        self.encConv2 = self.create_conv_layer(16, 32)
        self.encConv3 = self.create_conv_layer(32, 64)
        self.encConv4 = self.create_conv_layer(64, 128, apply_pooling=False)  
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128 * 2 * 2, feature_dim)  
        self.fc1_bn = nn.BatchNorm1d(feature_dim)
        self.fc_mu = nn.Linear(feature_dim, z_dim)
        self.fc_logvar = nn.Linear(feature_dim, z_dim)

        # Decoder
        self.decFC1 = nn.Linear(z_dim, 128 * 4 * 4)
        self.decFC1_bn = nn.BatchNorm1d(128 * 4 * 4)
        self.decConv1 = self.create_deconv_layer(128, 64)
        self.decConv2 = self.create_deconv_layer(64, 32)
        self.decConv3 = self.create_deconv_layer(32, 16)
        self.decConv4 = nn.ConvTranspose2d(16, 3, kernel_size=3, stride=1, padding=1)  

    def create_conv_layer(self, in_channels, out_channels, apply_pooling=True):
        """Creates a convolutional layer followed by BatchNorm and optionally MaxPooling."""
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        ]
        if apply_pooling:
            layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def create_deconv_layer(self, in_channels, out_channels):
        """Creates a deconvolutional (ConvTranspose) layer followed by BatchNorm and Upsampling."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def encode(self, x):
        x = self.encConv1(x)
        x = self.encConv2(x)
        x = self.encConv3(x)
        x = self.encConv4(x)
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
        z = z.view(-1, 128, 4, 4)  
        z = self.decConv1(z)
        z = self.decConv2(z)
        z = self.decConv3(z)
        z = torch.sigmoid(self.decConv4(z))  
        return z

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar
