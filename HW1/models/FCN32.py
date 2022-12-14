import torch
from squeeze_extractor import *

class FCN32(torch.nn.Module):

    def __init__(self, n_classes, pretrained_model: SqueezeExtractor):
        super(FCN32, self).__init__()
        self.pretrained_model = pretrained_model
        self.features = self.pretrained_model.features
        self.upsampling32 = nn.ConvTranspose2d(n_classes, n_classes, kernel_size=64,
                                               stride=32, bias=False)

        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                channels = m.out_channels

        self.classifier = nn.Sequential(nn.Conv2d(channels, n_classes, kernel_size=1), nn.Sigmoid())
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        o = x
        for feature in self.features:
            o = feature(o)
        o = self.classifier(o)
        o = self.upsampling32(o)
        cx = int((o.shape[3] - x.shape[3]) / 2)
        cy = int((o.shape[2] - x.shape[2]) / 2)
        o = o[:, :, cy:cy + x.shape[2], cx:cx + x.shape[3]]

        return o

from VGG import *



def fcn32_vgg16(n_classes, batch_size, pretrained=False, fixed_feature=True):
    batch_norm = False if batch_size == 1 else True
    vgg = vgg_16(batch_norm, pretrained, fixed_feature)
    return FCN32(n_classes, vgg)
