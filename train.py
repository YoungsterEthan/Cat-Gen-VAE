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
from model import VariationalAutoEncoder, VariationalAutoEncoder_BN, VariationalAutoEncoder_BN_Pool
from train_helpers import load_data, show_images, generate_images, reconstruct_img, train_and_visualize_fixed_images

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 3 * 356 * 256
H_DIM = 1024
N_DIM = 200
Z_DIM = 256
NUM_EPOCHS = 500
BATCH_SIZE = 128
LR = 0.001

model = VariationalAutoEncoder_BN_Pool(H_DIM, Z_DIM).to(DEVICE)

TRAIN = False
generate = True
train_data, val_data, test_data = load_data()
if TRAIN:
    fixed_images = next(iter(train_data))[0][:5]  
    train_and_visualize_fixed_images(model, train_data, NUM_EPOCHS, DEVICE, fixed_images, 'cat_generator6.pth')
else:
    model.load_state_dict(torch.load('cat_generator6.pth'))
    
model.cpu()
model.eval()


if generate:
    generated_images = generate_images(model, z_dims=Z_DIM, num_samples=1)
    show_images(generated_images.numpy(), num_images=1)
else:
    image_path = 'train\cat\cat.29.jpg'
    reconstruct_img(model, image_path)


