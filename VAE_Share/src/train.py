import torch
import torch.optim as optim
import torch.nn as nn
import model
import torchvision.transforms as transforms
import torchvision
import matplotlib
# import matplotlib.pyplot as plt
import pickle as pk
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import make_grid
from engine import train, validate
from utils import save_reconstructed_images, save_true_images, save_input_images, save_cat_images, image_to_vid, save_loss_plot


matplotlib.style.use('ggplot')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## initialize the model
## a model requires input image shape [w, h], and output image shape [w, h]
model = model.ConvVAE([64, 60], [64, 75]).to(device) 
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
symbols = ["601857", "600028"]

years = range(2008, 2022)
imgs = []
labels = []

for s in symbols:
    for y in years:
        f = open(f"./kchart/{s}_{y}_imgs.pk", "rb")
        # img, label = split_image(pk.load(f), 20)
        # images = pk.load(f)
        # split_imgs = [split_image(img, 20) for img in images]
        # img = [split_image[0] for split_image in split_imgs]
        # label = [split_image[1] for split_image in split_imgs]
        # imgs.extend(img)
        # labels.extend(label)
        imgs.extend((pk.load(f)))
        f = open(f"./kchart/{s}_{y}_labels.pk", "rb")
        labels.extend(pk.load(f))
        
# imgs = [img/255.0 for img in imgs]
# f = open("../data/labels.pk", "rb")
# labels = pk.load(f)
# plt.figure()
# plt.imshow(imgs[0])


## dataloader 
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

## Training 
train_loss = []
valid_loss = []
result_path = "./outputs/" ##folder for saving the output images
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