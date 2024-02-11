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
from train_helpers import load_data, train, show_images, generate_images, reconstruct_img

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

INPUT_DIM = 3 * 356 * 256
H_DIM = 200
N_DIM = 200
Z_DIM = 100
NUM_EPOCHS = 150
BATCH_SIZE = 128
LR = 0.001

model = VariationalAutoEncoder(H_DIM, Z_DIM).to(DEVICE)

TRAIN = False
generate = True

if TRAIN:
    dataloader = load_data()
    train(dataloader, model, NUM_EPOCHS, DEVICE, LR, path='cat_generator4.pth')
else:
    model.load_state_dict(torch.load('cat_generator4.pth'))
    
model.cpu()
model.eval()


if generate:
    generated_images = generate_images(model, z_dims=Z_DIM)
    show_images(generated_images.cpu().numpy())
else:
    image_path = 'train\cat\cat.4.jpg'
    reconstruct_img(model, image_path)


