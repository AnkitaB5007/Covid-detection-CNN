# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 19:58:12 2021

@author: Ankita
"""

import os
import time
import copy
from time import sleep
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
from sklearn.metrics import f1_score, precision_score, recall_score, classification_report, confusion_matrix
from tqdm import tqdm

# Local files
import Data_aug as aug
import Logger as logg
import Metrics as met
import Model_utilities as mtils


# --------------------------------------------- # TRAINING BLOCK #------------------------------------------------------------- #
def train(model, train_loader, train_size, val_loader, val_size, device, criterion, optimizer, scheduler, num_epochs, workspace):

    epochs = []
    train_accs, train_losses, train_f1_list = [], [], []
    val_accs, val_losses, val_f1_list = [], [], []

    print("[INFO] Training started")

    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        epochs.append(epoch)
        print('\nEpoch {}/{}'.format(epoch + 1, num_epochs))
        sleep(0.2)

        # --- Switch to train mode --- #
        model.train()
        running_loss = 0.0
        running_corrects = 0
        t_train = tqdm(train_loader)
        l = 0
        tot_labels = []
        tot_preds = []

        for inputs, labels in t_train:
            l = l + len(inputs)
            t_train.set_description("Train {}/{}".format(l, train_size), True)
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            with torch.set_grad_enabled(True):
                outputs = model(inputs)
                # _ contains max value for each row of outputs
                # preds contains index of the predicted class
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                loss.backward()
                optimizer.step()

            for x in labels:
                tot_labels.append(x.item())
            for y in preds:
                tot_preds.append(y.item())

            f1_train = f1_score(tot_labels, tot_preds, average='weighted')

            # Loss values and correct prediction count
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            t_train.set_postfix({'Training loss': running_loss / l, 'Training accuracy': (running_corrects.double() / l).item(), 'F1-score': f1_train}, True)

        scheduler.step()
        train_loss = running_loss / train_size  
        train_acc = running_corrects.double() / train_size 
        train_losses.append(train_loss)
        train_accs.append(train_acc.item())
        train_f1_list.append(f1_train)

        # torch.cuda.empty_cache()

        # --- evaluation block --- #
        # --- Same as training block; just change to val_loader ---- #
        model.eval()
        running_loss = 0.0
        running_corrects = 0
        t_val = tqdm(val_loader)
        l = 0
        tot_labels = []
        tot_preds = []

        for inputs, labels in t_val:
            l = l + len(inputs)
            t_val.set_description("Validation {}/{}".format(l, val_size), True)
            inputs = inputs.to(device)
            labels = labels.to(device)

            with torch.set_grad_enabled(False):
                optimizer.zero_grad()
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

            for x in labels:
                tot_labels.append(x.item())
            for y in preds:
                tot_preds.append(y.item())

            f1_val = f1_score(tot_labels, tot_preds, average='micro')

            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels)
            t_val.set_postfix({'Val loss': running_loss / l, 'Val accu': (running_corrects.double() / l).item(), 'F1-score': f1_val}, True)

        val_loss = running_loss / val_size  
        val_acc = running_corrects.double() / val_size 
        val_losses.append(val_loss)
        val_accs.append(val_acc.item())
        val_f1_list.append(f1_val)

        if val_acc > best_acc:
            best_acc = val_acc
            best_model_wts = copy.deepcopy(model.state_dict())
        
        # save last run model parameters in checkpoints/last folder
        if epoch == num_epochs - 1:
            timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[0]
            if os.path.isdir(os.path.join(workspace, 'checkpoints/last')) != True:
                os.mkdir(os.path.join(workspace, 'checkpoints/last'))
            model_path = os.path.join(workspace, 'checkpoints/last', timestamp + '_epoch-' + str(epoch+1) + '.pth' )
            torch.save(model.state_dict(), model_path)

    # load best model weights
    model.load_state_dict(best_model_wts)
    time_elapsed = time.time() - since
    print('\n[INFO] Training complete in {:.0f}m {:.0f}s\n'.format(time_elapsed // 60, time_elapsed % 60))
    print('Training\n- max accuracy = {}\n- min loss = {}\n- max F1-score = {}'.format(max(train_accs), min(train_losses), max(train_f1_list)))
    print('Validation\n- max accuracy = {}\n- min loss = {}\n- max F1-score = {}\n'.format(max(val_accs), min(val_losses), max(val_f1_list)))

    return model, epochs, train_accs, train_losses, val_accs, val_losses, train_f1_list, val_f1_list



# ---------------- workspace path ----------------- #
# --------------------------------------------------- #

torch.manual_seed(0)
# Location of checkpoints, scripts and dataset
#workspace = os.path.abspath("../")    
workspace = '/content/drive/MyDrive/HLCV/Covid'
dataset = 'Datasets'
data_dir = data_dir = os.path.join(workspace, dataset)
torch.hub.set_dir(workspace)
torch.hub.get_dir()
# ---------------- HYPER PARAMETERS ----------------- #
# --------------------------------------------------- #
# YOUR MODEL NAME HERE WHICH YOU WANT TO TRAIN
model_name = 'ResNeXt50'
#model_list = ['ResNeXt50', 'Inception_v3', ]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

freeze = True
dropout = True
prob = 0.3
lr = 0.000001
momentum = 0.9
step_size = 20
gamma = 0.1
criterion_name = 'Cross Entropy'
optimizer_name = 'Adam'
num_classes = 3
num_epochs = 30
batch_size = 30

# -------------------------SINGLE MODEL TRAINING AND VALIDATION BLOCK-------------------- #
model, optimizer, criterion, scheduler, transform = mtils.create_model(workspace, dataset, num_classes, model_name, freeze, dropout, prob, lr, momentum, step_size, gamma, criterion_name, optimizer_name, device)
model = model.to(device)

train_dir = os.path.join(data_dir, 'Train')
val_dir = os.path.join(data_dir, 'Val')

# Credits for augmentation code - https://github.com/shervinmin/DeepCovid/blob/master/ResNet18_train.py;
#https://github.com/Edo2610/Covid-19_X-ray_Two-proposed-Databases 
# Augmentation block ; use only for first time
# augmentation to increase the number of data
#source_dirs = ['Covid-19', 'Normal', 'Pneumonia']
#aug.data_augmentation(workspace, train_dir, source_dirs)

train_set = datasets.ImageFolder(train_dir, transform)
val_set = datasets.ImageFolder(val_dir, transform)
train_size = len(train_set)
val_size = len(val_set)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)

model, epochs, train_accs, train_losses, val_accs, val_losses, train_f1_list, val_f1_list = train(model, train_loader, train_size, val_loader, val_size, device, criterion, optimizer, scheduler, num_epochs, workspace)

# Save the best model parameters
timestamp = str(datetime.datetime.now()).split('.')[0].split(' ')[0]
if os.path.isdir(os.path.join(workspace, 'checkpoints/best')) != True:
        os.mkdir(os.path.join(workspace, 'checkpoints/best'))
model_path = os.path.join(workspace, 'checkpoints/best', timestamp + '.pth')
torch.save(model.state_dict(), model_path)

met.plot_loss_acc(timestamp, workspace, model_name, optimizer_name, epochs, train_losses, val_losses, train_accs, val_accs)
logg.create_train_log(workspace, train_accs, train_losses, train_f1_list, val_accs, val_losses, val_f1_list, model_name, optimizer_name, criterion_name, lr, momentum, step_size, gamma, num_epochs)
