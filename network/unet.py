import torch.nn as nn
from runx.logx import logx

from config import cfg

from .unet_parts import *


class UNet(nn.Module):
    # adapted from source
    # source: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_model.py
    # last accessed: 10.11.2022

    def __init__(self, num_classes, criterion=None):
        super(UNet, self).__init__()

        self.criterion = criterion
        self.n_classes = num_classes
        self.bilinear = cfg.MODEL.BILINEAR

        self.inc = DoubleConv(3, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        factor = 2 if self.bilinear else 1
        self.down4 = Down(512, 1024 // factor)
        self.up1 = Up(1024, 512 // factor, self.bilinear)
        self.up2 = Up(512, 256 // factor, self.bilinear)
        self.up3 = Up(256, 128 // factor, self.bilinear)
        self.up4 = Up(128, 64, self.bilinear)
        self.outc = OutConv(64, self.n_classes)

    def forward(self, inputs):
        x = inputs["images"]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x_up1 = self.up1(x5, x4)
        x_up2 = self.up2(x_up1, x3)
        x_up3 = self.up3(x_up2, x2)
        x_up4 = self.up4(x_up3, x1)
        logits = self.outc(x_up4)

        output_dict = {}

        if self.training:
            assert "gts" in inputs
            gts = inputs["gts"]
            loss = self.criterion(logits, gts)
            return loss
        else:
            output_dict["pred"] = logits
            return output_dict


def unet(num_classes, criterion):
    model = UNet(num_classes=num_classes, criterion=criterion)
    # model = init_weight(model)
    return model
