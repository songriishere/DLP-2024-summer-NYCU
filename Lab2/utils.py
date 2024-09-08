# script for drawing figures, and more if needed
import torch
import torch.nn as nn
from model.SCCNet import *
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from Dataloader import *
import matplotlib.pyplot as plt

def drawplot (loss_all , loss_all_test ,train_accuracy_plot ,test_accuracy_plot , epoch_list):
    plt.figure()
    plt.title("Learning Curve", fontsize = 20)
    plt.plot(epoch_list, loss_all , color = 'orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.figure()
    plt.title("Test loss", fontsize = 20)
    plt.plot(epoch_list, loss_all_test , color = 'orange')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.show()

    plt.figure()
    plt.title("Train Accuracy",fontsize = 20)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(epoch_list ,train_accuracy_plot , color = 'orange')
    plt.show()

    plt.figure()
    plt.title("Test Accuracy",fontsize = 20)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot(epoch_list ,test_accuracy_plot , color = 'orange')
    plt.show()