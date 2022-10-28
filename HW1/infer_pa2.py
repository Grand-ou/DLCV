"""
The trainer class.

Library:	Tensowflow 2.2.0, pyTorch 1.5.1
Author:		Ian Yoo
Email:		thyoostar@gmail.com
"""
from __future__ import absolute_import, division, print_function
import torch
from util.validation import *
from util.viz_mask import *
import numpy as np
import imageio
import cv2
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

def convert_seg_gray_to_color(input, n_classes=7, output_path=None):
	"""
	Convert the segmented image on gray to color.
	:param input: it is available to get two type(ndarray, string), string type is a file path.
	:param n_classes: number of the classes.
	:param output_path: output path. if it is None, this function return result array(ndarray)
	:param colors: refer to 'class_colors' format. Default: random assigned color.
	:return: if out_path is None, return result array(ndarray)
	"""
	colors = {
        0:  [0, 255, 255],
        1:  [255, 255, 0],
        2:  [255, 0, 255],
        3:  [0, 255, 0],
        4:  [0, 0, 255],
        5:  [255, 255, 255],
        6: [0, 0, 0],
    }
	assert len(input.shape) == 2, "Input should be h,w "
	seg = input

	height = seg.shape[0]
	width = seg.shape[1]

	seg_img = np.zeros((height, width, 3))

	for c in range(n_classes):
		seg_arr = seg[:, :] == c
		seg_img[:, :, 0] += ((seg_arr) * colors[c][0]).astype('uint8')
		seg_img[:, :, 1] += ((seg_arr) * colors[c][1]).astype('uint8')
		seg_img[:, :, 2] += ((seg_arr) * colors[c][2]).astype('uint8')

	if output_path:
		imageio.imsave(output_path, np.uint8(seg_img))
	else:
		return seg_img

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


    def evaluate(self):
        num_batches = len(self.test_loader)

        self.model.eval()

        with torch.no_grad():
            predict = torch.tensor([])
            files = []
            for n_batch, (sample_batched) in enumerate(self.test_loader):
                pred, file = self._eval_batch(sample_batched, n_batch, num_batches)
                predict = torch.cat((predict, torch.from_numpy(pred)), 0)
                files = files + list(file)

            for i in range(predict.shape[0]):
                convert_seg_gray_to_color(predict[i,:,:].numpy(), output_path='output/pred_dir/'+files[i]+'.png')
        

    def _eval_batch(self, sample_batched, n_batch, num_batches):
        data = sample_batched['image']
        filenames = sample_batched['annotation']
        print(data.shape)

        
        if self.cuda:
            data= data.cuda()
        torch.cuda.empty_cache()

        score = self.model(data)['out']

        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]

        return lbl_pred, filenames


    def _write_img(self, score, target, input_img, n_batch, num_batches):
        lbl_pred = score.data.max(1)[1].cpu().numpy()[:, :, :]
        lbl_true = target.data.cpu().numpy()

        log_img = self.logger.concatenate_images([lbl_pred, lbl_true], input_axis='byx')
        log_img = self.logger.concatenate_images([log_img, input_img.cpu().numpy()[:, :, :, :]])
        self.logger.log_images(log_img, self.epoch, n_batch, num_batches, nrows=log_img.shape[0])

from torchvision import transforms, models

from datasets.pa2 import SegmentationDataset
from datasets.transform import Rescale, ToTensor


from models.pspnet import pspnet_mobilenet_v2
from util.checkpoint import CheckpointHandler
from util.logger import Logger
train_images = r'/content/drive/MyDrive/DLCV/HW1/hw1_data/p2_data/train'
test_images = r'/content/drive/MyDrive/DLCV/HW1/input/test_dir'


if __name__ == '__main__':
    model_name = "DeepLabV3_ResNet101"
    device = 'cuda'
    batch_size = 3
    n_classes = 7
    num_epochs = 100
    image_axis_minimum_size = 512
    pretrained = True
    fixed_feature = False

    logger = Logger(model_name=model_name, data_name='example')

    ### Loader
    compose = transforms.Compose([
        Rescale(image_axis_minimum_size),
        ToTensor()
         ])
    test_datasets = SegmentationDataset(test_images, n_classes, compose, mode='val')
    test_loader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=True, drop_last=True)
    train_loader = ''
    ### Model
    weights = models.segmentation.DeepLabV3_ResNet101_Weights
  
    model = models.segmentation.deeplabv3_resnet101(weights = weights).to(device)
    model.classifier = models.segmentation.deeplabv3.DeepLabHead(2048, n_classes).to(device)


    ###Load model
    ###please check the foloder: (.segmentation/test/runs/models)
    #logger.load_model(model, 'epoch_15')
 
    
    logger.load_model(model, 'epoch_32')
    # model.load_state_dict()
    
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

    trainer.evaluate()

