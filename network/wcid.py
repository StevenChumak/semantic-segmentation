import math

import torch
import torch.nn as nn
from runx.logx import logx
from torch.nn import init
from torch.nn.modules.conv import ConvTranspose2d

from config import cfg


class WCID_block(nn.Module):
    def __init__(self, num_classes):
        super(WCID_block, self).__init__()
        in_channels = 3
        self.multiplier = 1

        l1 = [in_channels, 8, 16]
        l2 = [l1[2], 16, 32, 32]
        l3 = [l2[3], 64, 64]
        l4 = [l3[2], 64, 64]
        l5 = [l4[2], 64, 64]
        l6 = [l5[2], 64, 64]
        l7 = [l6[2], 32, 32, 16]
        l8 = [l7[3], 16, num_classes]

        # Convolution
        self.wcidLayer1 = Down1(l1, self.multiplier)
        self.wcidLayer2 = Down2(l2, self.multiplier)
        self.wcidLayer3 = Down3(l3, self.multiplier)
        self.wcidLayer4 = Down4(l4, self.multiplier)

        # Transposed Convolution
        self.wcidLayer5 = Up1(l5, self.multiplier)
        self.wcidLayer6 = Up2(l6, self.multiplier)
        self.wcidLayer7 = Up3(l7, self.multiplier)
        self.wcidLayer8 = Up4(l8, self.multiplier)

        self.high_level_ch = self.wcidLayer8.out_channels

    def forward(self, x):

        x1 = self.wcidLayer1(x)
        x2 = self.wcidLayer2(x1)
        x3 = self.wcidLayer3(x2)
        x4 = self.wcidLayer4(x3)

        x5 = self.wcidLayer5(x4)
        x6 = self.wcidLayer6(x5)
        x7 = self.wcidLayer7(x6)
        x8 = self.wcidLayer8(x7)

        return x7, x8


class WCID_SE_block(nn.Module):
    def __init__(self, num_classes, se_reduction):
        super(WCID_SE_block, self).__init__()
        in_channels = 3
        self.se_reduction = se_reduction
        self.multiplier = 1.2

        l1 = [in_channels, 8, 16]
        l2 = [l1[-1], 16, 32, 32]
        l3 = [l2[-1], 64, 64]
        l4 = [l3[-1], 64, 64]

        l5 = [l4[-1], 64, 64]
        l6 = [l5[-1], 64, 64]
        l7 = [l6[-1], 32, 32, 16]
        l8 = [l7[-1], 16, num_classes]

        # l1 = [in_channels, 16, 32]
        # l2 = [l1[-1], 32, 64]
        # l3 = [l2[-1], 128, 128]
        # l4 = [l3[-1], 128, 128]

        # l5 = [l4[-1], 128, 128]
        # l6 = [l5[-1], 128, 128]
        # l7 = [l6[-1], 64, 32]
        # l8 = [l7[-1], 16, num_classes]

        # Convolution
        self.wcidLayer1 = Down1(l1, self.multiplier)
        self.wcidLayer2 = Down2(l2, self.multiplier)
        self.wcidLayer3 = Down3(l3, self.multiplier)
        self.wcidLayer4 = Down4(l4, self.multiplier)

        # Transposed Convolution
        self.wcidLayer5 = Up1(l5, self.multiplier)
        self.wcidLayer6 = Up2(l6, self.multiplier)
        self.wcidLayer7 = Up3(l7, self.multiplier)
        self.wcidLayer8 = Up4(l8, self.multiplier)

        self.high_level_ch = self.wcidLayer7.out_channels

        se_layer = {}

        # SE wcidLayers
        for i in range(1, 9):
            layerName = f"se_{i}"
            out_channels = self.__getattr__("wcidLayer" + str(i)).out_channels
            layer = []
            layer.append(
                ChannelSELayer(out_channels, reduction_ratio=self.se_reduction)
            )

            se_layer[layerName] = nn.Sequential(*layer)

        self.se_layer = nn.ModuleDict(se_layer)

    def forward(self, x):
        x1 = self.wcidLayer1(x)
        x1_se = self.se_layer["se_1"](x1)

        x2 = self.wcidLayer2(x1_se)
        x2_se = self.se_layer["se_2"](x2)

        x3 = self.wcidLayer3(x2_se)
        x3_se = self.se_layer["se_3"](x3)

        x4 = self.wcidLayer4(x3_se)
        x4_se = self.se_layer["se_4"](x4)

        x5 = self.wcidLayer5(x4_se)
        x5_se = self.se_layer["se_5"](x5)

        x6 = self.wcidLayer6(x5_se)
        x6_se = self.se_layer["se_6"](x6)

        x7 = self.wcidLayer7(x6_se)
        x7_se = self.se_layer["se_7"](x7)

        x8 = self.wcidLayer8(x7_se)
        x8_se = self.se_layer["se_8"](x8)

        return x7_se, x8_se


class WCID(nn.Module):
    # 'where can I drive' network
    # https://arxiv.org/abs/2004.07639

    def __init__(self, num_classes, criterion=None):
        super(WCID, self).__init__()
        self.criterion = criterion

        self.wcid = WCID_block(num_classes=num_classes)

    def forward(self, inputs):
        x = inputs["images"]

        _, prediction = self.wcid(x)
        output_dict = {}

        if self.training:
            assert "gts" in inputs
            gts = inputs["gts"]
            loss = self.criterion(prediction, gts)
            return loss
        else:
            output_dict["pred"] = prediction
            return output_dict


class WCID_SE(nn.Module):
    def __init__(self, num_classes, criterion=None):
        super(WCID_SE, self).__init__()

        self.criterion = criterion
        se_reduction = 8

        self.wcid = WCID_SE_block(num_classes=num_classes, se_reduction=se_reduction)

    def forward(self, inputs):
        x = inputs["images"]

        _, prediction = self.wcid(x)
        output_dict = {}

        if self.training:
            assert "gts" in inputs
            gts = inputs["gts"]
            loss = self.criterion(prediction, gts)
            return loss
        else:
            output_dict["pred"] = prediction
            return output_dict


class Conv2d_ReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier, dw=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        if in_channels != 3:
            in_channels = math.floor(in_channels ** multiplier)
        out_channels = math.floor(out_channels ** multiplier)

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding=0,
            stride=1,
            groups=self.in_channels if dw else 1,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class TransConv_ReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, multiplier, dw=False):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        in_channels = math.floor(in_channels ** multiplier)
        out_channels = math.floor(out_channels ** multiplier)

        self.conv1 = ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=1,
            groups=self.in_channels if dw else 1,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class Down1(nn.Module):
    def __init__(self, l, multiplier):
        super().__init__()
        self.batchNorm = nn.BatchNorm2d(l[0])

        self.conv1 = Conv2d_ReLu(l[0], l[1], kernel_size=3, multiplier=multiplier)
        self.conv2 = Conv2d_ReLu(l[1], l[2], kernel_size=3, multiplier=multiplier)

        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2))

        self.out_channels = self.conv2.out_channels

    def forward(self, x):
        x = self.batchNorm(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.maxPool(x)

        return x


class Down2(nn.Module):
    def __init__(self, l, multiplier):
        super().__init__()
        self.conv1 = Conv2d_ReLu(l[0], l[1], kernel_size=5, multiplier=multiplier)
        self.drop1 = nn.Dropout2d(0.2)
        self.conv2 = Conv2d_ReLu(l[1], l[2], kernel_size=3, multiplier=multiplier)
        self.drop2 = nn.Dropout2d(0.2)
        self.conv3 = Conv2d_ReLu(l[2], l[3], kernel_size=5, multiplier=multiplier)

        self.maxpol = nn.MaxPool2d(kernel_size=(2, 2))

        self.out_channels = self.conv3.out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.drop2(x)
        x = self.conv3(x)

        x = self.maxpol(x)

        return x


class Down3(nn.Module):
    def __init__(self, l, multiplier):
        super().__init__()

        self.conv1 = Conv2d_ReLu(l[0], l[1], kernel_size=3, multiplier=multiplier)
        self.drop1 = nn.Dropout2d(0.2)

        self.conv2 = Conv2d_ReLu(l[1], l[2], kernel_size=5, multiplier=multiplier)
        self.drop2 = nn.Dropout2d(0.2)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.out_channels = self.conv2.out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.drop2(x)

        x = self.maxpool(x)

        return x


class Down4(nn.Module):
    def __init__(self, l, multiplier, dw=False):
        super().__init__()
        self.conv1 = Conv2d_ReLu(
            l[0], l[1], kernel_size=3, multiplier=multiplier, dw=dw
        )
        self.drop1 = nn.Dropout2d(0.2)
        self.conv2 = Conv2d_ReLu(
            l[1], l[2], kernel_size=5, multiplier=multiplier, dw=dw
        )
        self.drop2 = nn.Dropout2d(0.2)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.out_channels = self.conv2.out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.drop2(x)

        x = self.maxpool(x)

        return x


class upsample(nn.Module):
    def __init__(self):
        super().__init__()

        if cfg.MODEL.BILINEAR:
            self.up = nn.Upsample(
                scale_factor=2, mode="bilinear", align_corners=cfg.MODEL.ALIGN_CORNERS
            )
        else:
            self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        return self.up(x)


class Up1(nn.Module):
    def __init__(self, l, multiplier, dw=False):
        super().__init__()
        self.up = upsample()
        self.conv1 = TransConv_ReLu(
            l[0], l[1], kernel_size=5, multiplier=multiplier, dw=dw
        )
        self.drop1 = nn.Dropout2d(0.2)
        self.conv2 = TransConv_ReLu(
            l[1], l[2], kernel_size=3, multiplier=multiplier, dw=dw
        )
        self.drop2 = nn.Dropout2d(0.2)

        self.out_channels = self.conv2.out_channels

    def forward(self, x):
        x = self.up(x)
        x = self.conv1(x)
        x = self.drop1(x)
        x = self.conv2(x)
        x = self.drop2(x)

        return x


class Up2(nn.Module):
    def __init__(self, l, multiplier):
        super().__init__()
        self.up = upsample()
        self.conv1 = TransConv_ReLu(l[0], l[1], kernel_size=5, multiplier=multiplier)
        self.drop1 = nn.Dropout2d(0.2)

        self.conv2 = TransConv_ReLu(l[1], l[2], kernel_size=3, multiplier=multiplier)
        self.drop2 = nn.Dropout2d(0.2)

        self.out_channels = self.conv2.out_channels

    def forward(self, x):
        x = self.up(x)

        x = self.conv1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.drop2(x)

        return x


class Up3(nn.Module):
    def __init__(self, l, multiplier):
        super().__init__()
        self.up = upsample()

        self.conv1 = TransConv_ReLu(l[0], l[1], kernel_size=5, multiplier=multiplier)
        self.drop1 = nn.Dropout2d(0.2)
        self.conv2 = TransConv_ReLu(l[1], l[2], kernel_size=3, multiplier=multiplier)
        self.drop2 = nn.Dropout2d(0.2)
        self.conv3 = TransConv_ReLu(l[2], l[3], kernel_size=5, multiplier=multiplier)
        self.drop3 = nn.Dropout2d(0.2)

        self.out_channels = self.conv3.out_channels

    def forward(self, x):
        x = self.up(x)

        x = self.conv1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.drop2(x)

        x = self.conv3(x)
        x = self.drop3(x)

        return x


class Up4(nn.Module):
    def __init__(self, l, multiplier):
        super().__init__()
        self.up = upsample()

        self.conv1 = TransConv_ReLu(l[0], l[1], kernel_size=3, multiplier=multiplier)
        self.conv2 = ConvTranspose2d(l[1], l[2], kernel_size=3, stride=(1, 1))

        self.out_channels = self.conv2.out_channels

    def forward(self, x):
        x = self.up(x)

        x = self.conv1(x)
        x = self.conv2(x)

        return x


class ChannelSELayer(nn.Module):
    # original source: https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SEAttention.py
    # some changes were made by Steven Chumak
    # paper: https://arxiv.org/abs/1709.01507

    def __init__(self, channel, reduction_ratio=8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # if channel < reduction_ration the result of the whole number
        # division will be zero.
        # to prevent this we need some form of clipping

        if channel < reduction_ratio:
            reducedChannel = 1
        else:
            reducedChannel = channel // reduction_ratio

        self.fc = nn.Sequential(
            nn.Linear(channel, reducedChannel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reducedChannel, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                init.constant_(m.weight, 1)
                init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


# def init_weight(model):
#     model_dict = model.state_dict()

#     # Load state_dict
#     torch_ = torch.load(pretrained_path, map_location={"cuda:0": "cpu"})
#     try:
#         pretrained_dict = torch_["state_dict"]
#     except:
#         pretrained_dict = torch_

#     # 1. filter out unnecessary keys
#     pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
#     # 2. overwrite entries in the existing state dict
#     model_dict.update(pretrained_dict)
#     # 3. load the new state dict
#     model.load_state_dict(model_dict)

#     logx.msg("=> loading pretrained model {}".format(pretrained_path))

#     return model


def wcid(num_classes, criterion):
    model = WCID(num_classes=num_classes, criterion=criterion)
    # model = init_weight(model)
    return model


def wcid_se(num_classes, criterion):
    model = WCID_SE(num_classes=num_classes, criterion=criterion)
    # model = init_weight(model)
    return model
