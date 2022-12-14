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
    layers = torch.tensor([]).to(device)
    all_labels = torch.tensor([])
    with torch.no_grad():
        for data in tqdm.tqdm(dataloader):
            inputs, labels = data
            all_labels = torch.cat((all_labels, labels), 0)
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, logits = model(inputs)
            layers = torch.cat((layers, logits), 0)
            loss = criterion(outputs, labels)
            predictions = torch.max(outputs, 1)[1]
            y_true = np.append(y_true, labels.cpu().numpy())
            y_pred = np.append(y_pred, predictions.cpu().numpy())
            train_loss += loss.item()
            train_correct += (predictions == labels).sum().item()
    return y_true, y_pred, train_loss, train_correct, layers, all_labels



def valid_model(device, dataset, args):

    model = ResNet50(num_classes=args.num_classes, mode='val')
    model.load_state_dict(torch.load(args.pretrain_dir))
    model.to(device)
    criterion = nn.CrossEntropyLoss() 
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    true, pred, test_loss, test_correct, layers, labels = valid_epoch(model, device, data_loader, criterion)
    test_acc = test_correct / len(data_loader.sampler)
    print("Testing Acc {:.2f}".format(test_acc)) 
    return layers, labels


if __name__ == '__main__':
    args = argument_parser()
    device = get_device()
    transform = transforms.Compose([    
        ToTensor(),
        Normalize(),
        Transpose()
    ])
    dataset = ImageDataset(root_dir = '/content/drive/MyDrive/DLCV/HW1/hw1_data/p1_data/val_50',transform=transform)
    layers, labels = valid_model(device, dataset, args)
    import pandas as pd
    from sklearn.preprocessing import StandardScaler
    layers = layers.cpu()

    # Standardizing the features
    x = StandardScaler().fit_transform(layers)
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)

    principalDf = pd.DataFrame(data = principalComponents
                , columns = ['principal component 1', 'principal component 2'])
 
    finalDf = pd.concat([principalDf, pd.DataFrame(labels.numpy(), columns=['target'])['target']], axis = 1)
    # print(finalDf)
    import matplotlib.pyplot as plt

    for i in range(50):
        plt.scatter("principal component 1", "principal component 2", data=finalDf[finalDf.target==i], alpha = 0.2)
        # print(finalDf[finalDf.target==i].shape)
    
    plt.savefig('pca_visual.png')
    from sklearn import manifold
    X_tsne = manifold.TSNE(n_components=2, init='random', random_state=5, verbose=1).fit_transform(x)

    #Data Visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  #Normalize
    plt.figure(figsize=(8, 8))
    finalDf['principal component 1'] = X_norm[:, 0]
    finalDf['principal component 2'] = X_norm[:, 1]
    for i in range(50):
        plt.scatter("principal component 1", "principal component 2", data=finalDf[finalDf.target==i], alpha = 0.2)
        # print(finalDf[finalDf.target==i].shape)

    plt.savefig('tsne_visual.png')