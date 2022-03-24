import torch
import torch.nn as nn
import torch.nn.functional as F
from runx.logx import logx
import functools

class Attention_block(nn.Module):
    # source: https://github.com/LeeJunHyun/Image_Segmentation/blob/master/network.py
    # paper: https://arxiv.org/pdf/1804.03999.pdf
        
    def __init__(self,F_g,F_l,F_int):
        super(Attention_block,self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
            )
        
        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1,stride=1,padding=0,bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )
        
        self.relu = nn.ReLU()
        
    def forward(self,g,x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.relu(g1+x1)
        psi = self.psi(psi)

        return x*psi
    
    
class TestNet_block(nn.Module):
    def __init__(
        self,
        se_type,
        se_reduction=16,
    ):
        super(TestNet_block, self).__init__()

        self.in_channels = 3  # expected 3 channel RGB image as input
        self.se_type = se_type
        self.se_reduction = se_reduction  # TODO: add to cfg?

        # changed values to be closer to previous 1,2 exp of previous values but which yield a whole number when divided by 8 (se_reduction ration)
        d1 = [self.in_channels, 8, 16]
        d2 = [d1[-1], 16, 32]
        d3 = [d2[-1], 64, 64]
        d4 = [d3[-1], 64, 64]
        d5 = [d4[-1], 128, 128]
        
        neck = [d5[-1], 256, d5[-1]]
        
        u1 = [d5[-1], d5[-2], d5[-2]]
        u2 = [u1[-1], d4[-2], d4[-2]]
        u3 = [u2[-1], d3[-2], d3[-2]]
        u4 = [u3[-1], d2[-2], d2[-2]]
        u5 = [u4[-1], d1[-2], d1[-2]]

        # Convolution
        self.down1 = Down(d1, [3, 3], self.se_type, self.se_reduction, dw=False)
        self.down2 = Down(d2, [5, 3], self.se_type, self.se_reduction, dw=False)
        self.down3 = Down(d3, [5, 3], self.se_type, self.se_reduction, reduce=False, dw=True)
        self.down4 = Down(d4, [5, 3], self.se_type, self.se_reduction, reduce=False, dw=True)
        self.down5 = Down(d5, [5, 3], self.se_type, self.se_reduction, reduce=False, dw=True)
        
        self.bneck = BNeck(neck, [3, 3], self.se_type, self.se_reduction, reduce=False, dw=True)

        self.up1 = Up(u1, [3, 5], self.se_type, self.se_reduction, reduce=False, dw=False, UNet=self.down5.out_channels)
        self.up2 = Up(u2, [3, 5], self.se_type, self.se_reduction, reduce=False, dw=False, UNet=self.down4.out_channels)
        self.up3 = Up(u3, [3, 5], self.se_type, self.se_reduction, reduce=False, dw=False, UNet=self.down3.out_channels)
        self.up4 = Up(u4, [3, 5], self.se_type, self.se_reduction, reduce=False, dw=False, UNet=self.down2.out_channels)
        self.up5 = Up(u5, [3, 3], self.se_type, self.se_reduction, reduce=False, dw=False, UNet=self.down1.out_channels)


        self.second_out_channels = self.up4.out_channels
        self.out_channels = self.up5.out_channels

        self._init_weights()
        
    def forward(self, x):
        d1, d1_b = self.down1(x)
        d2, d2_b = self.down2(d1)
        d3, d3_b = self.down3(d2)
        d4, d4_b = self.down4(d3)
        d5, d5_b = self.down5(d4)
        
        d5 = self.bneck(d5)

        u1 = self.up1(d5, d5_b)
        u2 = self.up2(u1, d4_b)
        u3 = self.up3(u2, d3_b)
        u4 = self.up4(u3, d2_b)
        u5 = self.up5(u4, d1_b)


        return u4, u5
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias.data, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    

class TestNet(nn.Module):
    def __init__(
        self,
        n_classes,
        se_type="channel",
        se_reduction=16,
        criterion=None,
        bin=False,
    ):
        super(TestNet, self).__init__()
        self.bin = bin
        self.criterion = criterion
        self.se_type = se_type
        self.se_reduction = se_reduction

        ####################################################################################
        self.testNet = TestNet_block(self.se_type, self.se_reduction)
        
        self.second_out_channels = self.testNet.second_out_channels
        self.high_level_ch = self.testNet.out_channels

        # self.out = nn.Conv2d(self.high_level_ch, n_classes, 1, stride=1, padding=0)

    def forward(self, x_in):

        _, out = self.testNet(x_in)       
        # prediction = self.out(out)          
        
        return None, None, out
 
    
class DWConv(nn.Module):
    def __init__(self, fm_in, fm_out, kernel):
        super().__init__()
        self.layer = nn.Sequential()  
        self.layer.add_module(
            "convDW",
            nn.Conv2d(
                fm_in,
                fm_in,
                kernel_size=kernel,
                stride=1,
                padding=kernel// 2,
                groups=fm_in,
            ),
        )
        self.layer.add_module(
            "conv1x1", nn.Conv2d(fm_in, fm_out, kernel_size=1, stride=1, padding=0)
        )
        self.layer.add_module("bnDW", nn.BatchNorm2d(fm_out))
        self.layer.add_module("reluDW", nn.ReLU())
        
    def forward(self, x):
        return self.layer(x)
        

class ConvBlock(nn.Module):
    def __init__(self, fm_in, fm_out, multiplier, kernel, reduce=False, dw=False):
        super().__init__()
        self.layer = nn.Sequential()  
        
        if reduce:
            mid = fm_in//multiplier
            
            self.layer.add_module(
                "convReduce", nn.Conv2d(fm_in, mid, kernel_size=1, stride=1, padding=0)
            )
        else:
            mid=fm_in

        if dw:
            self.layer.add_module("convDW", DWConv(fm_in=mid, fm_out=fm_out, kernel=kernel))
        
        else:
            self.layer.add_module(
                "conv",
                nn.Conv2d(
                    mid,
                    fm_out,
                    kernel_size=kernel,
                    stride=1,
                    padding=kernel// 2,
                    groups=1,
                ),
            )
            self.layer.add_module("bn", nn.BatchNorm2d(fm_out))
            self.layer.add_module("relu", nn.ReLU())
            

    def forward(self, x):
        return self.layer(x)


class NConv(nn.Module):
    def __init__(self, fm_list, kernel_list, se_type, se_reduction, reduce=False, dw=False):
        super().__init__()
        self.layer = nn.Sequential()
        
        self.residual = fm_list[0] == fm_list[-1]

        for i in range(0, len(kernel_list)):
            self.layer.add_module(f"conv_block{i}", ConvBlock(fm_list[i], fm_list[i+1], multiplier=2, kernel=kernel_list[i], reduce=reduce, dw=dw,))
            
        if se_type == 'channel':
            self.layer.add_module("se", ChannelSELayer(fm_list[-1], reduction_ratio=se_reduction))
        elif se_type == 'cbam':
            self.layer.add_module("se", CBAM(fm_list[-1], reduction_ratio=se_reduction))
        else:
            logx.msg(f"{se_type} is not supported")

    def forward(self, x):
        x_out = self.layer(x)
        
        if self.residual:
            x_out += x
        
        return x_out


class Down(nn.Module):
    def __init__(self, fm_list, kernel_list, se_type, se_reduction, reduce=False, dw=False):
        super().__init__()
        self.nConv = NConv(fm_list, kernel_list, se_type, se_reduction, reduce=reduce, dw=dw)
        self.maxPool = nn.MaxPool2d(kernel_size=2)
        
        self.out_channels = fm_list[-1]

    def forward(self, x):

        x = self.nConv(x)
        x_small = self.maxPool(x)

        return x_small, x


class BNeck(nn.Module):
    def __init__(self, fm_list, kernel_list, se_type, se_reduction, reduce=False, dw=False):
        super().__init__()
        
        self.nConv = NConv(fm_list, kernel_list, se_type, se_reduction, reduce=reduce, dw=dw)
        
        self.out_channels = fm_list[-1]

    def forward(self, x):

        x = self.nConv(x)

        return x


class Upsample(nn.Module):
    def __init__(self):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

    def forward(self, x):
        return self.up(x)


class Up(nn.Module):
    def __init__(self, fm_list, kernel_list, se_type, se_reduction, reduce=False, dw=False, UNet=0,):
        super().__init__()

        self.up = Upsample()
        self.UNet=UNet
        
        if self.UNet>0:
            self.attention= Attention_block(fm_list[0], self.UNet, (fm_list[0] + self.UNet)//4)
            fm_list[0] = fm_list[0] + self.UNet
        
        self.nConv = NConv(fm_list, kernel_list, se_type, se_reduction, reduce=reduce, dw=dw)

        self.out_channels = fm_list[2]

    def forward(self, small, big):

        x1 = self.up(small)
        if self.UNet>0:
            # from: https://github.com/milesial/Pytorch-UNet/blob/master/unet/unet_parts.py
            diffY = big.size()[2] - x1.size()[2]
            diffX = big.size()[3] - x1.size()[3]

            x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
            # if you have padding issues, see
            # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
            # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd

            attention = self.attention(g=x1, x=big)
            x1 = torch.cat([attention, x1], dim=1)
        x = self.nConv(x1)

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

        self._init_weights()

    def forward(self, x):
        batch_size, channel, _, _ = x.size()

        y = self.squeeze(x).view(batch_size, channel)
        y = self.excitate(y).view(batch_size, channel, 1, 1)

        return x * y.expand_as(x)

    def _init_weights(self):
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
                    

def testnet(num_classes, criterion):
    model = TestNet(n_classes=num_classes, criterion=criterion)

    return model
