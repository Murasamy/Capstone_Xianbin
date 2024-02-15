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
import argparse
import random
random.seed(42)
warnings.filterwarnings("ignore")
matplotlib.style.use('ggplot')

# matplotlib.style.use('ggplot')
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    

class AutEncoDeco():
    def __init__(self, 
                device, 
                model, 
                lr, 
                epochs, 
                batch_size, 
                criterion, 
                train_data_dir, 
                output_dir, 
                read_data_format, 
                txt_path, 
                save_steps, 
                debug_mode=False, 
                shuffle_train_data=True, 
                eval_data_proportion=0.2,
                save_eval_img=False,
                save_train_img=False,
                save_input_img=False,
                save_cat_img=False,):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.transforms = transforms
        self.model = model.ConvVAE([64, 60], [64, 15]).to(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.MSELoss(reduction='sum')
        self.train_data_dir = train_data_dir
        self.output_dir = output_dir
        self.read_data_format = read_data_format
        self.txt_path = txt_path
        self.save_steps = save_steps
        # self.grid_images = []
        self.imgs = []
        self.labels = []
        self.train_loader = None
        self.test_loader = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loss = []
        self.test_loss = []
        self.debug_mode = debug_mode
        self.shuffle_train_data = shuffle_train_data
        self.eval_data_proportion = eval_data_proportion
        self.save_eval_img = save_eval_img
        self.save_train_img = save_train_img
        self.save_input_img = save_input_img
        self.save_cat_img = save_cat_img

    def wrapper(
            self,

    ):
        if self.read_data_format == "txt":
            # check if self.txt_path exists and is a txt file, throw error if not
            print(self.txt_path)
            if not os.path.exists(self.txt_path):
                raise Exception("txt_path_dir does not exist")
            if not os.path.isfile(self.txt_path):
                raise Exception("txt_path_dir is not a file")
            if not self.txt_path.endswith(".txt"):
                raise Exception("txt_path_dir is not a txt file")
            self.load_data_from_txt()
        else:
            raise Exception("Peiyang doesn't know what you wanna do T_T")
        self.train_model()

    def load_data_from_txt(self):
        f = open(self.txt_path, "r")
        symbols = f.read().splitlines()
        if self.debug_mode:
            symbols = symbols[:1]
        f.close()
        for symbol in symbols:
            path_sh = f"{self.train_data_dir}/{symbol}.SH.CSV"
            path_sz = f"{self.train_data_dir}/{symbol}.SZ.CSV"
            if os.path.exists(path_sh):
                path = path_sh
            elif os.path.exists(path_sz):
                path = path_sz
            else:
                print(f"{symbol} does not exist in {self.train_data_dir}")
                continue
                # display warning
                # warnings.warn(f"{symbol} does not exist in {self.train_data_dir}")
            print(path)
            imgsTemp, labelsTemp = generate_pk(path)
            self.imgs.extend(imgsTemp)
            self.labels.extend(labelsTemp)
        #     imgsTemp, labelsTemp = generate_pk(path)
        #     print(len(imgsTemp), len(labelsTemp))
        #     imgs.extend(imgsTemp)
        #     labels.extend(labelsTemp)
        convert_tensor = self.transforms.ToTensor()
        self.imgs = [convert_tensor(img) for img in self.imgs]
        self.labels = [convert_tensor(img) for img in self.labels]
        print(len(self.imgs), len(self.labels))
        if self.shuffle_train_data:
            random.shuffle(self.imgs)
            random.shuffle(self.labels)
        print(len(self.imgs), len(self.labels))
        idx_train = round((1-self.eval_data_proportion)*(len(self.labels)))
        train_imgs = self.imgs[:idx_train]
        train_labels = self.labels[:idx_train]
        self.train_set = CustomDataset(train_imgs, train_labels)
        test_imgs = self.imgs[idx_train:]
        test_labels = self.labels[idx_train:]
        self.test_set = CustomDataset(test_imgs, test_labels)
        # testset = data[idx_train:]
        self.trainloader = DataLoader(
            self.train_set, batch_size=self.batch_size, shuffle=False
        )
        self.testloader = DataLoader(
            self.test_set, batch_size=self.batch_size, shuffle=False
        )
        # throw error if trainloader and testloader are empty
        if len(self.trainloader) == 0:
            raise Exception("trainloader is empty")
        if len(self.testloader) == 0:
            raise Exception("testloader is empty")
    
    def train_model(self):
        train_loss = []
        valid_loss = []
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # create a dir for this run: time + lr + epochs
        import time
        time = time.strftime("%Y-%m-%d-%H:%M:%S")
        result_path_run = f"{self.output_dir}/{time}_lr={self.lr}_epoch={self.epochs}/"
        if not os.path.exists(result_path_run):
            os.makedirs(result_path_run)
        
        # dir to save training state:
        result_path_run_state = f"{result_path_run}state/"
        if not os.path.exists(result_path_run_state):
            os.makedirs(result_path_run_state)
        
        # dir to save training images:
        if self.save_train_img:
            result_path_run_train_img = f"{result_path_run}train_img/"
            if not os.path.exists(result_path_run_train_img):
                os.makedirs(result_path_run_train_img)

        # dir to save eval images:
        if self.save_eval_img:
            result_path_run_eval_img = f"{result_path_run}eval_img/"
            if not os.path.exists(result_path_run_eval_img):
                os.makedirs(result_path_run_eval_img)

        # dir to save input images:
        if self.save_input_img:
            result_path_run_input_img = f"{result_path_run}input_img/"
            if not os.path.exists(result_path_run_input_img):
                os.makedirs(result_path_run_input_img)

        # dir to save cat images:
        if self.save_cat_img:
            result_path_run_cat_img = f"{result_path_run}cat_img/"
            if not os.path.exists(result_path_run_cat_img):
                os.makedirs(result_path_run_cat_img)

        for epoch in range(self.epochs):
            print(f"Epoch {epoch+1} of {self.epochs}")
            train_epoch_loss = train(
                self.model, self.trainloader, self.train_set, device, self.optimizer, self.criterion
            )
            valid_epoch_loss, recon_images, labels, input_images = validate(
                self.model, self.testloader, self.test_set, self.device, self.criterion
            )
            train_loss.append(train_epoch_loss)
            valid_loss.append(valid_epoch_loss)

            print(f"Train Loss: {train_epoch_loss:.4f}")
            print(f"Val Loss: {valid_epoch_loss:.4f}")

            if (epoch + 1) % self.save_steps == 0:
                # save
                checkpoint_path = f"{result_path_run_state}epoch={epoch+1}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'loss': train_loss,
                }, checkpoint_path)

                if self.save_train_img:
                    save_reconstructed_images(recon_images, epoch+1, result_path_run_train_img)
                if self.save_eval_img:
                    save_true_images(labels, epoch+1, result_path_run_eval_img)
                if self.save_input_img:
                    save_input_images(input_images, epoch+1, result_path_run_input_img)
                if self.save_cat_img:
                    save_cat_images(recon_images, labels, epoch+1, result_path_run_cat_img)


        save_loss_plot(train_loss, valid_loss, result_path_run)
        print('TRAINING COMPLETE')
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Autoencoder')
    parser.add_argument('--lr', type=float, default=0.0001, help='learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    # parser.add_argument('--optimizer', type=str, default='Adam', help='optimizer')
    # parser.add_argument('--criterion', type=str, default='MSELoss', help='criterion')
    parser.add_argument('--train_data_dir', type=str, default='data', help='train data path')
    parser.add_argument('--output_dir', type=str, default='checkpoints', help='output directory')
    parser.add_argument('--read_data_format', type=str, default='txt', help='read data format')
    parser.add_argument('--txt_path', type=str, default='txtfiles/SSE50.txt', help='txt path')
    parser.add_argument('--save_steps', type=bool, default=True, help='save_steps')
    parser.add_argument('--debug_mode', type=str, default='False', help='debug mode, if True, only use 1 stock')
    parser.add_argument('--shuffle_train_data', type=bool, default='True', help='whether to shuffle the train data')
    parser.add_argument('--eval_data_proportion', type=float, default='0.2', help='proportion of the eval data')
    parser.add_argument('--save_eval_img', type=bool, default='False', help='whether to save the eval images')
    parser.add_argument('--save_train_img', type=bool, default='False', help='whether to save the train images')
    parser.add_argument('--save_input_img', type=bool, default='False', help='whether to save the input images')
    parser.add_argument('--save_cat_img', type=bool, default='False', help='whether to save the cat images')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device == 'cuda':
        print(torch.cuda.get_device_name(0))
    else:
        print("warning: GPU is not activated, the training process will be slow.")
    transforms = transforms
    criterion = nn.MSELoss(reduction='sum')
    
    PeiyangAutoEncoDeco = AutEncoDeco(
        device, 
        model, 
        args.lr, 
        args.epochs, 
        args.batch_size, 
        criterion, 
        args.train_data_dir, 
        args.output_dir, 
        args.read_data_format, 
        args.txt_path, 
        args.save_steps, 
        args.debug_mode, 
        args.shuffle_train_data,
        args.eval_data_proportion,
        args.save_eval_img,
        args.save_train_img,
        args.save_input_img,
        args.save_cat_img,
        )
    PeiyangAutoEncoDeco.wrapper()
    
    