"""
The trainer class.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division, print_function

from util.validation import *
from util.logger import *

try:
    from tqdm import tqdm
    from tqdm import trange
except ImportError:
    print("tqdm and trange not found, disabling progress bars")

    def tqdm(iter):
        return iter

    def trange(iter):
        return iter

TQDM_COLS = 80

def cross_entropy2d(input, target):
    # input: (n, c, h, w), target: (n, h, w)
    n, c, h, w = input.size()

    # input: (n*h*w, c)
    input = input.transpose(1, 2).transpose(2, 3).contiguous()
    input = input[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    input = input.view(-1, c)

    # target: (n*h*w,)
    mask = target >= 0.0
    target = target[mask]

    func_loss = torch.nn.CrossEntropyLoss()
    loss = func_loss(input, target)

    return loss


class Trainer(object):

    def __init__(self, model, optimizer, logger, num_epochs, train_loader,
                 test_loader=None,
                 epoch=0,
                 log_batch_stride=30,
                 check_point_epoch_stride=60,
                 scheduler=None):
        """
        :param model: A network model to train.
        :param optimizer: A optimizer.
        :param logger: The logger for writing results to Tensorboard.
        :param num_epochs: iteration count.
        :param train_loader: pytorch's DataLoader
        :param test_loader: pytorch's DataLoader
        :param epoch: the start epoch number.
        :param log_batch_stride: it determines the step to write log in the batch loop.
        :param check_point_epoch_stride: it determines the step to save a model in the epoch loop.
        :param scheduler: optimizer scheduler for adjusting learning rate.
        """
        self.cuda = torch.cuda.is_available()
        self.model = model
        self.optim = optimizer
        self.logger = logger
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.num_epoches = num_epochs
        self.check_point_step = check_point_epoch_stride
        self.log_batch_stride = log_batch_stride
        self.scheduler = scheduler

        self.epoch = epoch

    def train(self):
        if not next(self.model.parameters()).is_cuda and self.cuda:
            raise ValueError("A model should be set via .cuda() before constructing optimizer.")

        for epoch in trange(self.epoch, self.num_epoches,
                          position=0,
                          desc='Train', ncols=TQDM_COLS):
            self.epoch = epoch

            # train
            self._train_epoch()

            # step forward to reduce the learning rate in the optimizer.
            if self.scheduler:
                self.scheduler.step()

            # model checkpoints
            if epoch%self.check_point_step == 0:
                self.logger.save_model_and_optimizer(self.model,
                                                     self.optim,
                                                     'epoch_{}'.format(epoch))



    def evaluate(self):
        num_batches = len(self.test_loader)
        self.model.eval()

        with torch.no_grad():
            for n_batch, (sample_batched) in tqdm(enumerate(self.test_loader),
                                total=num_batches,
                                leave=False,
                                desc="Valid epoch={}".format(self.epoch),
                                ncols=TQDM_COLS):
                self._eval_batch(sample_batched, n_batch, num_batches)

    def _train_epoch(self):

        num_batches = len(self.train_loader)

        if self.test_loader:
            dataloader_iterator = iter(self.test_loader)

        train_loss,train_acc = 0.0, 0.0
        cnt = 0
        for n_batch, (sample_batched) in tqdm(enumerate(self.train_loader),
                                              total=num_batches,
                                              leave=False,
                                              desc="Train epoch={}".format(self.epoch),
                                              ncols=TQDM_COLS):
            cnt+=1
            self.model.train()
            data = sample_batched['image']
            target = sample_batched['annotation']

            if self.cuda:
                data, target = data.cuda(), target.cuda()

            self.optim.zero_grad()

            torch.cuda.empty_cache()

            score = self.model(data)
            loss = cross_entropy2d(score, target)

            loss_data = loss.data.item()
            if np.isnan(loss_data):
                raise ValueError('loss is nan while training')

            loss.backward()
            self.optim.step()

            if n_batch%self.log_batch_stride != 0:
                continue

            self.logger.store_checkpoint_var('img_width', data.shape[3])
            self.logger.store_checkpoint_var('img_height', data.shape[2])

            self.model.img_width = data.shape[3]
            self.model.img_height = data.shape[2]

            #write logs to Tensorboard.
            lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
            lbl_true = target.data.cpu().numpy()
            acc, acc_cls, mean_iou, fwavacc = \
                label_accuracy_score(lbl_true, lbl_pred, n_class=score.shape[1])

            self.logger.log_train(loss, 'loss', self.epoch, n_batch, num_batches)
            self.logger.log_train(acc, 'acc', self.epoch, n_batch, num_batches)
            self.logger.log_train(acc_cls, 'acc_cls', self.epoch, n_batch, num_batches)
            self.logger.log_train(mean_iou, 'mean_iou', self.epoch, n_batch, num_batches)
            self.logger.log_train(fwavacc, 'fwavacc', self.epoch, n_batch, num_batches)
            train_loss += loss.item()
            train_acc += acc
            #write result images when starting epoch.
            if n_batch == 0:
                log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
                log_img = self.logger.concatenate_images([log_img, data.cpu().numpy()[:, :, :, :]])
                self.logger.log_images_train(log_img, self.epoch, n_batch, num_batches,
                                             nrows=data.shape[0])
        print("    Training Loss:{:.4f},  Training Acc {:.2f}".format(train_loss/cnt, train_acc/cnt))
            #if the trainer has the test loader, it evaluates the model using the test data.
        if self.test_loader:
            self.model.eval()
            with torch.no_grad():
                try:
                    sample_batched = next(dataloader_iterator)
                except StopIteration:
                    dataloader_iterator = iter(self.test_loader)
                    sample_batched = next(dataloader_iterator)

                self._eval_batch(sample_batched, n_batch, num_batches)
        


    def _eval_batch(self, sample_batched, n_batch, num_batches):
        data = sample_batched['image']
        target = sample_batched['annotation']
        
        if self.cuda:
            data, target = data.cuda(), target.cuda()
        torch.cuda.empty_cache()

        score = self.model(data)

        loss = cross_entropy2d(score, target)
        loss_data = loss.data.item()
        if np.isnan(loss_data):
            raise ValueError('loss is nan while training')

        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()
        acc, acc_cls, mean_iou, fwavacc = \
            label_accuracy_score(lbl_true, lbl_pred, n_class=score.shape[1])

        self.logger.log_test(loss, 'loss', self.epoch, n_batch, num_batches)
        self.logger.log_test(acc, 'acc', self.epoch, n_batch, num_batches)
        self.logger.log_test(acc_cls, 'acc_cls', self.epoch, n_batch, num_batches)
        self.logger.log_test(mean_iou, 'mean_iou', self.epoch, n_batch, num_batches)
        self.logger.log_test(fwavacc, 'fwavacc', self.epoch, n_batch, num_batches)
        print("    Testing Loss:{:.4f},  Testing Acc {:.2f}".format(loss, acc))
        if n_batch == 0:
            log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
            log_img = self.logger.concatenate_images([log_img, data.cpu().numpy()[:, :, :, :]])
            self.logger.log_images_test(log_img, self.epoch, n_batch, num_batches,
                                        nrows=data.shape[0])

    def _write_img(self, score, target, input_img, n_batch, num_batches):
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()

        log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
        log_img = self.logger.concatenate_images([log_img, input_img.cpu().numpy()[:, :, :, :]])
        self.logger.log_images(log_img, self.epoch, n_batch, num_batches, nrows=log_img.shape[0])

from torchvision import transforms

from datasets.pa2 import SegmentationDataset
from datasets.transform import Rescale, ToTensor


from models.FCN32 import pspnet_mobilenet_v2
from util.logger import Logger

train_images = r'/content/drive/MyDrive/DLCV/HW1/hw1_data/p2_data/train'
test_images = r'/content/drive/MyDrive/DLCV/HW1/hw1_data/p2_data/validation'


if __name__ == '__main__':
    model_name = "pspnet_mobilenet_v2"
    device = 'cuda'
    batch_size = 32
    n_classes = 7
    num_epochs = 10
    image_axis_minimum_size = 200
    pretrained = True
    fixed_feature = False

    logger = Logger(model_name=model_name, data_name='example')

    ### Loader
    compose = transforms.Compose([
        Rescale(image_axis_minimum_size),
        ToTensor()
         ])

    train_datasets = SegmentationDataset(train_images, n_classes, compose)
    train_loader = torch.utils.data.DataLoader(train_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    test_datasets = SegmentationDataset(test_images, n_classes, compose)
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)

    ### Model
    model = fcn32_vgg16(n_classes, batch_size,
                                                   pretrained=pretrained,
                                                   fixed_feature=fixed_feature)
    model.to(device)

    ###Load model
    ###please check the foloder: (.segmentation/test/runs/models)
    #logger.load_model(model, 'epoch_15')

    ### Optimizers
    if pretrained and fixed_feature: #fine tunning
        params_to_update = model.parameters()
        print("Params to learn:")
        params_to_update = []
        for name, param in model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
        optimizer = torch.optim.Adadelta(params_to_update)
    else:
        optimizer = torch.optim.Adadelta(model.parameters())

    ### Train
    #scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = Trainer(model, optimizer, logger, num_epochs, train_loader, test_loader)
    trainer.train()
    trainer.evaluate()

