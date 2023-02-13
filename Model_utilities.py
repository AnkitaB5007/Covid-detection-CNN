# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 21:01:14 2021

@author: Ankita
"""


import os
import datetime
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torchvision import transforms, datasets, models
import torch.nn.functional as F
from time import sleep


class TwoInputsNet(nn.Module):

  def __init__(self, model1, model2, num_classes):

    super(TwoInputsNet, self).__init__()
    self.model1 = model1
    self.model2 = model2
    self.fc2 = nn.Linear(2048, num_classes)

  def forward(self, input1, input2):

    c = self.model1(input1)
    f = self.model2(input2)

    combined = torch.cat((c,f), dim=1)

    out = self.fc2(F.relu(combined))

    return out

    
def load_pretrained_model(model_name):
    if model_name == 'ResNeXt50':
        model = models.resnext50_32x4d(pretrained=True)
    elif model_name == 'Inception_v3':
        model = models.inception_v3(pretrained=True, aux_logits=False)
    elif model_name == 'DenseNet161':
        model = models.densenet161(pretrained=True)
    return model

def edit_model(model_name, model, dropout, prob, freeze, num_classes):
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    if freeze:
        print ("\n[INFO] Freezing feature layers...")
        for param in model.parameters():
            param.requires_grade=False
        sleep(0.5)
        print("-"*50)
    # https://pytorch.org/tutorials/beginner/finetuning_torchvision_models_tutorial.html
    # Make sure num of classes = number of output features for all the models
    if model_name == 'DenseNet161':
        num_ftrs = model.classifier.in_features
        if dropout:
            model.classifier = nn.Sequential(
                nn.Dropout(prob),
                nn.Linear(num_ftrs, num_classes))
        else:
            model.classifier = nn.linea(num_ftrs, num_classes)
    # No need to add Dropout as inception already has dropout layer
    elif model_name == 'Inception_v3':
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    else:
        num_ftrs = model.fc.in_features
        if dropout:
            model.fc = nn.Sequential(
                nn.Dropout(prob),
                nn.Linear(num_ftrs, num_classes))
        else:
            model.fc = nn.Linear(num_ftrs, num_classes)
    return model

def create_transform(model_name):
    if model_name == 'Inception_v3':
        transform = transforms.Compose([
            transforms.Resize(299),
            transforms.CenterCrop(299),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    return transform


def create_model(workspace, dataset, num_classes, model_name, freeze, dropout, prob, lr, momentum, step_size, gamma, criterion_name, optimizer_name, device):
    if model_name == 'ResNeXt50' or model_name =='Inception_v3' or model_name == 'DenseNet161':
        # Load the pre-trained models
        model = load_pretrained_model(model_name)
        # Fine tune the models as per your num of classes
        model = edit_model(model_name, model, dropout, prob, freeze, num_classes)
        # Transform image as per your model name
        transform = create_transform(model_name)

    # Optimizer and Loss block and scheduler
    if optimizer_name == 'Adam':
        optimizer_conv = optim.Adam(model.parameters(), lr=lr)
    if criterion_name == 'Cross Entropy':
        criterion = nn.CrossEntropyLoss()
    
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=step_size, gamma=gamma)

    return model, optimizer_conv, criterion, exp_lr_scheduler, transform
