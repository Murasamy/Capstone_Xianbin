import torch
import torch.optim as optim
import torch.nn as nn
import model
import torchvision.transforms as transforms
import torchvision
import matplotlib
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import pickle as pk
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from engine import train, validate
from dataclasses import dataclass, field
from utils import save_reconstructed_images, save_true_images, save_input_images, save_cat_images, image_to_vid, save_loss_plot
from making_chart_peiyang import generate_pk
import os
import warnings
import re # for regular expression
warnings.filterwarnings("ignore")

matplotlib.style.use('ggplot')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## initialize the model
## a model requires input image shape [w, h], and output image shape [w, h]
model = model.ConvVAE([64, 60], [64, 15]).to(device) 
## set the learning parameters
lr = 0.001
epochs = 10
batch_size = 1
optimizer = optim.Adam(model.parameters(), lr=lr)
# criterion = nn.BCELoss(reduction='sum')
criterion = nn.MSELoss(reduction='sum')
# a list to save all the reconstructed images in PyTorch grid format
grid_images = [] #generating the gif of the outputs

## loading data 
# symbols = ["601857", "600028"]

years = range(2008, 2022)

imgs = []
labels = []

dirPath = "./data/"
# read SSE50.txt
f = open("SSE50.txt", "r")
symbols = f.read().splitlines()[:1]
f.close()
# symbols = symbols[1:]
# symbols
path_dir = "../data/"
for symbol in symbols:
    # path = f"{path_dir}{symbol}.SH.CSV"
    # or path = f"{path_dir}{symbol}.SZ.CSV"
    # regular expression or ether "SH" or "SZ"
    # re = re.compile(r"SH|SZ")
    path = f"{path_dir}{symbol}.SH.CSV"
    # path = re.compile(r"SH|SZ").sub(symbol, f"{path_dir}SH.CSV")
    imgsTemp, labelsTemp = generate_pk(path)
    print(len(imgsTemp), len(labelsTemp))
    imgs.extend(imgsTemp)
    labels.extend(labelsTemp)
    
class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
convert_tensor = transforms.ToTensor()
# print(convert_tensor(imgs[0]).size())
imgs = [convert_tensor(img) for img in imgs]
labels = [convert_tensor(img) for img in labels]
# data = list(zip(imgs, labels))
idx_train = round(0.8*(len(labels)))
train_imgs = imgs[:idx_train]
train_labels = labels[:idx_train]
train_set = CustomDataset(train_imgs, train_labels)

test_imgs = imgs[idx_train:]
test_labels = labels[idx_train:]
test_set = CustomDataset(test_imgs, test_labels)
# testset = data[idx_train:]

trainloader = DataLoader(
    train_set, batch_size=batch_size, shuffle=False
)

testloader = DataLoader(
    test_set, batch_size=batch_size, shuffle=False
)

train_loss = []
valid_loss = []
result_path = "./outputs/" ##folder for saving the output images
if not os.path.exists(result_path):
    os.makedirs(result_path)
for epoch in range(epochs):
    print(f"Epoch {epoch+1} of {epochs}")
    train_epoch_loss = train(
        model, trainloader, train_set, device, optimizer, criterion
    )
    valid_epoch_loss, recon_images, labels, input_images = validate(
        model, testloader, test_set, device, criterion
    )
    train_loss.append(train_epoch_loss)
    valid_loss.append(valid_epoch_loss)
    # save the reconstructed images from the validation loop
    # print("images for cat:", recon_images.shape, input_images.shape)
    if (epoch+1)%100 == 0:
        # save_reconstructed_images(recon_images, epoch+1, result_path)
        # save_true_images(labels, epoch+1, result_path)
        # save_input_images(input_images, epoch+1, result_path)
        save_cat_images(recon_images, labels, epoch+1, result_path)
    
    # convert the reconstructed images to PyTorch image grid format
    image_grid = make_grid(recon_images.detach().cpu())
    grid_images.append(image_grid)
    print(f"Train Loss: {train_epoch_loss:.4f}")
    print(f"Val Loss: {valid_epoch_loss:.4f}")

# save the reconstructions as a .gif file
# result_path = "./outputs/"
image_to_vid(grid_images, result_path)
# save the loss plots to disk
save_loss_plot(train_loss, valid_loss, result_path)
# save the true label to disk
# save_true_images(labels, epoch+1, result_path)

print('TRAINING COMPLETE')