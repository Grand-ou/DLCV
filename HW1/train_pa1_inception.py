import argparse
import numpy as np
import tqdm
import os
import time
import matplotlib.pyplot as plt
import timm 
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
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
    parser.add_argument('--batch_size', type=int, default=32, help='size for each minibatch')
    parser.add_argument('--num_epochs', type=int, default=10, help='maximum number of epochs')
    parser.add_argument('--learning_rate', type=float, default=0.05, help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.5, help='initial momentum')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='weight_decay rate')
    parser.add_argument('--cv', type=int, default=5, help='number of folds for cross validation')
    parser.add_argument('--preprocess', type=bool, default=False, help='True if preprocess needed False otherwise')
    parser.add_argument('--result_dir', type=str, default='checkpoints', help='result directory')
    parser.add_argument('--pretrain_dir', type=str, default='ckpt', help='pretrain weight directory')
    parser.add_argument('--mode', type=str, default='cv', help='training with cv, one_fold or all')
    return parser.parse_args()

def get_device():
    return 'cuda' if torch.cuda.is_available() else 'cpu'
    
def train_epoch(model, device, dataloader, criterion, optimizer):
    train_loss,train_correct = 0.0, 0
    model.train()
    for data in tqdm.tqdm(dataloader):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        model.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        predictions = torch.max(outputs, 1)[1]
        train_loss += loss.item()
        train_correct += (predictions == labels).sum().item()
    return train_loss, train_correct

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

def train_with_cv(device, dataset, args):
    criterion = nn.CrossEntropyLoss() 
    if args.cv == 1:
        model = timm.create_model('inception_v4', pretrained=True, num_classes=args.num_classes)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
        save_model(args.num_epochs, device, model, optimizer, dataset, args)
    else:
        y_true = np.array([])
        y_pred = np.array([])

        splits = KFold(n_splits=args.cv, shuffle=True)

        average_history = {'train_loss': np.zeros(args.num_epochs), 'train_acc':np.zeros(args.num_epochs), 'test_loss': np.zeros(args.num_epochs), 'test_acc':np.zeros(args.num_epochs)}

        for fold, (train_idx, test_idx) in enumerate(splits.split(np.arange(len(dataset)))):
            print('Fold {}'.format(fold + 1))


            model = timm.create_model('inception_v4', pretrained=True, num_classes=args.num_classes)

            model.to(device)
            optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
            
            train_sampler = SubsetRandomSampler(train_idx)
            test_sampler = SubsetRandomSampler(test_idx)
            train_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=train_sampler)
            test_loader = DataLoader(dataset, batch_size=args.batch_size, sampler=test_sampler)

            history = {'train_loss': [], 'train_acc':[], 'test_loss': [], 'test_acc':[]}
            for epoch in range(args.num_epochs):
                train_loss, train_correct = train_epoch(model, device, train_loader, criterion, optimizer)
                true, pred, test_loss, test_correct = valid_epoch(model, device, test_loader, criterion)

                train_loss = train_loss / len(train_loader.sampler)
                train_acc = train_correct / len(train_loader.sampler)
                test_loss = test_loss / len(test_loader.sampler)
                test_acc = test_correct / len(test_loader.sampler)

                print("  Epoch:{}/{}".format(epoch + 1, args.num_epochs))                                               
                print("    Training Loss:{:.4f},  Training Acc {:.2f}".format(train_loss, train_acc))
                print("    Testing Loss:{:.4f},  Testing Acc {:.2f}".format(test_loss, test_acc))
                
                history['train_loss'].append(train_loss)
                history['train_acc'].append(train_acc)
                history['test_loss'].append(test_loss)
                history['test_acc'].append(test_acc)

            y_true = np.append(y_true, true)
            y_pred = np.append(y_pred, pred)

            average_history['train_loss'] += history['train_loss']
            average_history['train_acc'] += history['train_acc']
            average_history['test_loss'] += history['test_loss']
            average_history['test_acc'] += history['test_acc']

        average_history['train_loss'] /= args.cv
        average_history['train_acc'] /= args.cv
        average_history['test_loss'] /= args.cv
        average_history['test_acc'] /= args.cv

        print(classification_report(y_true, y_pred))
        print(confusion_matrix(y_true, y_pred))
        draw_curve(args, average_history)

        best_epoch_num = np.argmax(average_history['test_acc'])+1
        model = timm.create_model('inception_v4', pretrained=True, num_classes=args.num_classes)
        model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay, momentum=args.momentum)
        save_model(best_epoch_num, device, model, optimizer, dataset, args)
    
def draw_curve(args, history):
    fig = plt.figure()
    epochs = np.arange(1, args.num_epochs+1)
    ax0 = fig.add_subplot(121, title="accuracy")
    ax1 = fig.add_subplot(122, title="loss")
    ax0.plot(epochs, history['train_acc'], 'b-', label='train')
    ax0.plot(epochs, history['test_acc'], 'r-', label='test')
    ax1.plot(epochs, history['train_loss'], 'b-', label='train')
    ax1.plot(epochs, history['test_loss'], 'r-', label='test')
    plt.legend()
    fig.savefig(os.path.join(args.result_dir, 'C3D_loss_graph'))

def save_model(epoch_num, device, model, optimizer, dataset, args):
    print('Best epoch: {}'.format(epoch_num))
    print('Saving model...')
    criterion = nn.CrossEntropyLoss() 
    data_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    for epoch in range(epoch_num):
        train_loss, train_correct = train_epoch(model, device, data_loader, criterion, optimizer)
    true, pred, test_loss, test_correct = valid_epoch(model, device, data_loader, criterion)
    test_acc = test_correct / len(data_loader.sampler)
    print("Testing Acc {:.2f}".format(test_acc))
    result = time.localtime(time.time())
    torch.save(model.state_dict(), os.path.join(args.result_dir, 'C3D_'+str(result.tm_mon)+'_'+str(result.tm_mday)+'_best_epoch'+str(epoch_num)+'.pt'))
    print('Model saved.')

if __name__ == '__main__':
    args = argument_parser()
    device = get_device()
    transform = transforms.Compose([    
        ToTensor(),
        Normalize(),
        
        Transpose()
    ])
    config = resolve_data_config({}, model=timm.create_model('inception_v4', pretrained=True, num_classes=args.num_classes))
    transform = create_transform(**config)
    dataset = ImageDataset(root_dir = '/content/drive/MyDrive/DLCV/HW1/hw1_data/p1_data/train_50',transform=transform)
    train_with_cv(device, dataset, args)