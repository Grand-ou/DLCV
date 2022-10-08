import argparse
import numpy as np
import tqdm
import os
import time
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, SubsetRandomSampler
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, confusion_matrix

from models.resnet import ResNet50
from datasets.pa1 import ImageDataset, Normalize, Transpose, ToTensor

def argument_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='C:/Users/ouchu/bucket data/post-estimate/assests/combine', help='data_directory')
    parser.add_argument('--num_classes', type = int, default=50, help='number of class')
    parser.add_argument('--batch_size', type=int, default=128, help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='initial momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    parser.add_argument('--cv', type=int, default=5, help='number of folds for cross validation')
    parser.add_argument('--preprocess', type=bool, default=False, help='True if preprocess needed False otherwise')
    parser.add_argument('--result_dir', type=str, default='ckpt', help='result directory')
    parser.add_argument('--pretrain_dir', type=str, default='', help='pretrain weight directory')
    parser.add_argument('--mode', type=str, default='cv', help='training with cv, one_fold or all')
    return parser.parse_args()

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
    

def valid_epoch(model, device, dataloader, criterion):
    y_true = np.array([])
    y_pred = np.array([])
    train_loss, train_correct = 0.0, 0
    model.eval()
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            predictions = torch.max(outputs, 1)[1]
            y_true = np.append(y_true, labels.cpu().numpy())
            y_pred = np.append(y_pred, predictions.cpu().numpy())
            train_loss += loss.item()
            train_correct += (predictions == labels).sum().item()
    return y_true, y_pred, train_loss, train_correct



def valid_model(device, dataset, args):
    model = ResNet50(num_classes=args.num_classes, pretrain_dir=args.pretrain_dir)
    model.to(device)
    criterion = nn.CrossEntropyLoss() 
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    true, pred, test_loss, test_correct = valid_epoch(model, device, data_loader, criterion)
    test_acc = test_correct / len(data_loader.sampler)
    print("Testing Acc {:.2f}".format(test_acc))


if __name__ == '__main__':
    args = argument_parser()
    device = get_device()
    transform = transforms.Compose([    
        ToTensor(),
        Normalize(),
        Transpose()
    ])
    dataset = ImageDataset(root_dir = '/content/drive/MyDrive/DLCV/HW1/hw1_data/hw1_data/p1_data/val_50',transform=transform)
    valid_model(device, dataset, args)
