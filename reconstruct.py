import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
from model2 import VAE
import numpy as np
from model import VariationalAutoEncoder

# Assuming VAE is your model class

model = VariationalAutoEncoder(200, 100)
model.load_state_dict(torch.load('cat_generator3.pth'))


model.eval()  

# Load and prepare an image
image_path = 'train\cat\cat.2.jpg'
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
