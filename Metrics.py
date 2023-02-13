# -*- coding: utf-8 -*-
"""
Created on Fri Jul 16 20:47:48 2021

@author: Ankita
"""
import os
import datetime
import numpy as np
import itertools
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt


def plot_loss_acc(timestamp, workspace, model_name, optimizer_name, epochs, train_losses, val_losses, train_accs, val_accs, img_size=[12,5]):
    fig, (axs1, axs2) = plt.subplots(1, 2, figsize=(img_size[0], img_size[1]))
    fig.suptitle(model_name + ' with ' + optimizer_name + ' optimizer')
    
    # Loss graph for training and validation part
    axs1.plot(epochs, train_losses, label='Training')
    axs1.plot(epochs, val_losses, label='Validation')
    axs1.set(xlabel='Epochs', ylabel='Loss')
    axs1.legend()
    
    # Accuracy graph for training and validation part
    axs2.plot(epochs, train_accs, label='Training')
    axs2.plot(epochs, val_accs, label='Validation')
    axs2.set(xlabel='Epochs', ylabel='Accuracy')
    axs2.legend()

    # save the graph in graph folder, if not present create a 'graph' folder
    if os.path.isdir(os.path.join(workspace, 'graph')) != True:
      os.mkdir(os.path.join(workspace, 'graph'))
    # save the figure to file
    fig.savefig(os.path.join(workspace, 'graph', timestamp + '.png'))
    # plt.show()

# https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
def plot_confusion_matrix(cm, classes, timestamp, workspace, model_name, cmap=plt.cm.Blues, save=True):
    plt.figure(figsize=(10,10))
    if os.path.isdir(os.path.join(workspace, 'graph')) != True and save == True:
      os.mkdir(os.path.join(workspace, 'graph'))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)

    # https://www.kaggle.com/grfiv4/plot-a-confusion-matrix
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes, rotation=90)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j]) + '\n(' + format(round(cm[i,j]*100/207, 1)) + '%)', fontsize=20, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.title(model_name)
    # Save the confusion matrix path
    if save:
        cm_path = timestamp + '_confmatrix.png'
        plt.savefig(os.path.join(workspace, 'graph', cm_path))
    # plt.show()
