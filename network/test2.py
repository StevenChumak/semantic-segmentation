
import torch
import torch.nn as nn
from torch.nn.modules.conv import ConvTranspose2d
import torch.nn.functional as F


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

        l5 = [l4[-1]+l4[-1], 128, 128]
        l6 = [l5[-1]+l3[-1], 128, 128]
        l7 = [l6[-1]+l2[-1], 64, 32]
        l8 = [l7[-1]+l1[-1], 16, n_classes]

        # Convolution
        self.down1 = Down(l1, [3,3], self.se_reduction, dw=False, drop=False)
        self.down2 = Down(l2, [5,3], self.se_reduction, dw=False)
        self.down3 = Down(l3, [5,3], self.se_reduction, dw=True)
        self.down4 = Down(l4, [5,3], self.se_reduction, dw=True)

        # Transposed Convolution
        self.up1 = Up(l5, [3,5], self.se_reduction, dw=False)
        self.up2 = Up(l6, [3,5], self.se_reduction, dw=False)
        self.up3 = Up(l7, [3,5], self.se_reduction, dw=False)
        self.up4 = Up(l8, [3,3], self.se_reduction, dw=False, drop=False)

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
        x1, x11 = self.down1(x)
        x2, x22 = self.down2(x1)
        x3, x33 = self.down3(x2)
        x4, x44 = self.down4(x3)

        x5 = self.up1(x4, x44)
        x6 = self.up2(x5, x33)
        x7 = self.up3(x6, x22)
        x8 = self.up4(x7, x11)

        return x7, x8
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


class double_Conv(nn.Module):
    def __init__(self, l, k, se_reduction, dw=False, drop=False):
        super().__init__()
        self.layer = nn.Sequential()        

        self.layer.add_module("conv1", nn.Conv2d(l[0], l[1], kernel_size=k[0], stride=1, padding=k[0]//2, groups=l[0] if dw else 1))
        self.layer.add_module("bn1", nn.BatchNorm2d(l[1]))
        if  dw:
            self.layer.add_module("conv1x1", nn.Conv2d(l[1], l[1], kernel_size=1, stride=1, padding=0))
            self.layer.add_module("bn1x1", nn.BatchNorm2d(l[1]))

        self.layer.add_module("relu1", nn.ReLU())

        self.layer.add_module("conv2", nn.Conv2d(l[1], l[2], kernel_size=k[1], stride=1, padding=k[1]//2, groups=l[1] if dw else 1))
        self.layer.add_module("bn2", nn.BatchNorm2d(l[2]))
        if  dw:
            self.layer.add_module("conv2x1", nn.Conv2d(l[2], l[2], kernel_size=1, stride=1, padding=0))
            self.layer.add_module("bn2x1", nn.BatchNorm2d(l[2]))

        self.layer.add_module("relu2", nn.ReLU())

        self.layer.add_module("se", ChannelSELayer(l[2], reduction_ratio=se_reduction))
        if drop:
            self.layer.add_module("drop2", nn.Dropout(0.2))

    def forward(self, x):
        return self.layer(x)


class Down(nn.Module):
    def __init__(self, l, k, se_reduction, dw=False, drop=True):
        super().__init__()
        self.dConv = double_Conv(l, k, se_reduction, dw=dw)
        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2))

        self.out_channels = l[2]

    def forward(self, x):

        x = self.dConv(x)
        x_small = self.maxPool(x)

        return x_small, x


class upsample(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        return self.up(x)


class Up(nn.Module):
    def __init__(self, l, k, se_reduction, dw=False, drop=True):
        super().__init__()

        self.up = upsample()

        self.dConv  = double_Conv(l, k, se_reduction, dw=dw)

        self.out_channels = l[2]

    def forward(self, x1, x2):
        x1 = self.up(x1)

        #from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        

        x = torch.cat([x2, x1], dim=1)
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
    # some changes were made by Steven Chumak
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