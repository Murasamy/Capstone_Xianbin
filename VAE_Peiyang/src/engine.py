from tqdm import tqdm
import torch 
from torchvision.utils import save_image


def final_loss(bce_loss, mu, logvar):
    """
    This function will add the reconstruction loss (BCELoss) and the 
    KL-Divergence.
    KL-Divergence = 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    :param bce_loss: recontruction loss
    :param mu: the mean from the latent vector
    :param logvar: log variance from the latent vector
    """
    BCE = bce_loss 
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


def train(model, dataloader, dataset, device, optimizer, criterion):
    print("training")
    model.train()
    running_loss = 0.0
    counter = 0
    for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
        counter += 1
        
        ##for different output size
        img = data[0]
        # print("img", img.size())
        # save_image(img, f"./outputs/data_loader_img{i}.jpg") #check if the img and label are correct.
        label = data[1]
        # save_image(label, f"./outputs/data_loader_label{i}.jpg")#check if the img and label are correct.
        # print("label", label.size())
        # data = data.to(device)
        img = img.to(device)
        label = label.to(device)
        optimizer.zero_grad()
        reconstruction, mu, logvar = model(img)
        print("reconstruction", reconstruction.size())
        print("label", label.size())
        bce_loss = criterion(reconstruction, label)

        loss = final_loss(bce_loss, mu, logvar)
        loss.backward()
        running_loss += loss.item()
        optimizer.step()
    train_loss = running_loss / counter 
    return train_loss


def validate(model, dataloader, dataset, device, criterion):
    model.eval()
    running_loss = 0.0
    counter = 0
    recon_images = None
    true_images = None
    input_images = None
    with torch.no_grad():
        for i, data in tqdm(enumerate(dataloader), total=int(len(dataset)/dataloader.batch_size)):
            counter += 1
            # data= data[0]
            # data = data.to(device)
            # # print(data.shape)
            # reconstruction, mu, logvar = model(data)
            # bce_loss = criterion(reconstruction, data)
            ##for different output size
            imgs = data[0]
            labels = data[1]
            imgs = imgs.to(device)
            labels = labels.to(device)
            reconstruction, mu, logvar = model(imgs)
            bce_loss = criterion(reconstruction, labels)
            loss = final_loss(bce_loss, mu, logvar)
            running_loss += loss.item()
            # true_images = 0
            # save the last batch input and output of every epoch
            # print("recon_image", i)
            if i == int(len(dataset)/dataloader.batch_size) - 1:
                recon_images = reconstruction
                true_images = labels
                input_images = imgs
    val_loss = running_loss / counter
    return val_loss, recon_images, true_images, input_images