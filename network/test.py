

import torch.nn as nn
from torch.nn.modules.conv import ConvTranspose2d

from network.utils import old_make_attn_head
from network.mscale2 import MscaleBase
from config import cfg


class TestNet_block(nn.Module):
    def __init__(
        self,
        n_classes,
        se_reduction=8,
    ):
        super(TestNet_block, self).__init__()

        self.in_channels = 3 # expected 3 channel RGB image as input
        self.se_reduction = se_reduction # TODO: add to cfg?

        # changed values to be closer to previous 1,2 exp of previous values but which yield a whole number when divided by 8 (se_reduction ration)

        l1 = [self.in_channels, 16, 32]
        l2 = [l1[-1], 32, 64]
        l3 = [l2[-1], 128, 128]
        l4 = [l3[-1], 128, 128]

        l5 = [l4[-1], 128, 128]
        l6 = [l5[-1], 128, 128]
        l7 = [l6[-1], 64, 32]
        # l8 = [l7[-1], 32, 16]
        l8 = [l7[-1], 16, n_classes]

        # Convolution
        self.down1 = Down(l1, [3,3], self.se_reduction, drop=False)
        self.down2 = Down(l2, [5,3],self.se_reduction, drop=False)
        self.down3 = Down(l3, [5,3],self.se_reduction, drop=False)
        self.down4 = Down(l4, [5,3],self.se_reduction, drop=False)

        # Transposed Convolution
        self.up1 = Up(l5, [3,5],self.se_reduction, drop=False)
        self.up2 = Up(l6, [3,5],self.se_reduction, drop=False)
        self.up3 = Up(l7, [3,5],self.se_reduction, drop=False)
        self.up4 = Up(l8, [3,3],self.se_reduction, drop=False)

        # self.output = nn.Conv2d(l8[-1], n_classes, 1)

        self.second_out_ch = self.down1.out_channels

        self.init_weights()
        
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.down1(x)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)

        x5 = self.up1(x4)
        x6 = self.up2(x5)
        x7 = self.up3(x6)
        x8 = self.up4(x7)
        # x8 = self.output(x8)

        return x1, x8
        # TODO: why did I return x7 with WCID-Net? Does x1 work?

class TestNet(nn.Module):
    def __init__(
        self,
        n_classes,
        criterion=None,
    ):
        super(TestNet, self).__init__()

        self.criterion = criterion
        self.se_reduction = 8 # TODO: add to cfg?

        self.testNet = TestNet_block(n_classes, self.se_reduction)

        self.second_out_ch = self.testNet.second_out_ch
    

    def forward(self, inputs):
        x = inputs['images']

        _, prediction = self.testNet(x)
        output_dict = {}
        
        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(prediction, gts)
            return loss
        else:
            output_dict['pred'] = prediction
            return output_dict


class TestNet_mscale(MscaleBase):
    def __init__(self, n_classes, criterion=None):
        super(TestNet_mscale, self).__init__()
        
        self.criterion = criterion
        self.se_reduction = 8 # TODO: add to cfg?

        self.testNet = TestNet_block(n_classes, self.se_reduction)

        self.second_out_ch = self.testNet.second_out_ch

        self.scale_attn = old_make_attn_head(
            in_ch=self.second_out_ch*2, bot_ch=self.second_out_ch//2, out_ch=1)

    def _fwd(self, x):
            second_out, final = self.testNet(x)

            return final, second_out


class Conv2d_ReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class TransConv_ReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        self.conv1 = ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
        )
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class double_Conv(nn.Module):
    def __init__(self, l, k, se_reduction, drop=True):
        super().__init__()
        self.layer = nn.Sequential()

        # self.layer.add_module("conv1", Conv2d_ReLu(l[0], l[1], kernel_size=3, stride=1, padding=1))
        # # if drop:
        #     # self.layer.add_module("drop1", nn.Dropout(0.2))
        # self.layer.add_module("bn1", nn.BatchNorm2d(l[1]))

        # self.layer.add_module("conv2", Conv2d_ReLu(l[1], l[2], kernel_size=3, stride=1, padding=1))
        # self.layer.add_module("se", ChannelSELayer(l[2], reduction_ratio=se_reduction))
        # if drop:
        #     self.layer.add_module("drop2", nn.Dropout(0.2))
        # self.layer.add_module("bn2", nn.BatchNorm2d(l[2]))
           

        self.layer.add_module("conv1", Conv2d_ReLu(l[0], l[1], kernel_size=k[0], stride=1, padding=(k[0]//2)))
        self.layer.add_module("bn1", nn.BatchNorm2d(l[1]))

        self.layer.add_module("conv2", Conv2d_ReLu(l[1], l[2], kernel_size=k[1], stride=1, padding=k[1]//2))
        self.layer.add_module("se", ChannelSELayer(l[2], reduction_ratio=se_reduction))
        self.layer.add_module("drop2", nn.Dropout(0.2))
        self.layer.add_module("bn2", nn.BatchNorm2d(l[2]))

    def forward(self, x):
        return self.layer(x)


class Down(nn.Module):
    def __init__(self, l, k, se_reduction, drop=True):
        super().__init__()
        # l[2] = l[2] * 2
        self.dConv = double_Conv(l, k, se_reduction, drop=drop)

        # self.se = ChannelSELayer(l[2], reduction_ratio=se_reduction)
        # self.conv3 = Conv2d_ReLu(l[2], l[2]//2, kernel_size=1, stride=1, padding=0)
        # self.drop3 = nn.Dropout(0.2)

        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2))

        self.out_channels = l[2]

    def forward(self, x):

        x = self.dConv(x)

        # x = self.se(x)
        # x = self.conv3(x)
        # x = self.drop3(x)

        x = self.maxPool(x)

        return x


class upsample(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        return self.up(x)


class Up(nn.Module):
    def __init__(self, l, k, se_reduction, drop=True):
        super().__init__()

        self.up = upsample()
        self.dConv  = double_Conv(l, k, se_reduction, drop=drop)

        self.out_channels = l[2]

    def forward(self, x):
        x = self.up(x)
        x = self.dConv(x)

        return x


def _make_divisible(v, divisor, min_value=None):
    # source: https://github.com/d-li14/efficientnetv2.pytorch/blob/main/effnetv2.py
    """
    This function is taken from the original tf repo.
    It ensures that all layers have a channel number that is divisible by 8
    It can be seen here:
    https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/mobilenet.py
    :param v:
    :param divisor:
    :param min_value:
    :return:
    """
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # Make sure that round down does not go down by more than 10%.
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


class ChannelSELayer(nn.Module):
    # original source: https://github.com/xmu-xiaoma666/External-Attention-pytorch/blob/master/model/attention/SEAttention.py
    # some changes were made by me
    # paper: https://arxiv.org/abs/1709.01507

    def __init__(self, channel, reduction_ratio=8):
        super().__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)

        reducedChannel = _make_divisible(channel, reduction_ratio)

        self.excitate = nn.Sequential(
            nn.Linear(channel, reducedChannel, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reducedChannel, channel, bias=False),
            nn.Sigmoid(),
        )

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        batch_size, channel, _, _ = x.size()

        y = self.squeeze(x).view(batch_size, channel)
        y = self.excitate(y).view(batch_size, channel, 1, 1)

        return x * y.expand_as(x)


def testnet(num_classes, criterion):
    model = TestNet(n_classes=num_classes, criterion=criterion)

    return model

def testnet_mscale(num_classes, criterion):
    model = TestNet_mscale(n_classes=num_classes, criterion=criterion)

    return model