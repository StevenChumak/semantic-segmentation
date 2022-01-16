
import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.modules.conv import ConvTranspose2d
from runx.logx import logx

from network.utils import old_make_attn_head
from network.mscale2 import MscaleBase
from config import cfg


align_corners = cfg.MODEL.ALIGN_CORNERS

class WCID_block(nn.Module):
    def __init__(self, num_classes):
        super(WCID_block, self).__init__()
        in_channels=3     

        # Convolution
        self.wcidLayer1 = Down1(in_channels)
        self.wcidLayer2 = Down2()
        self.wcidLayer3 = Down3()
        self.wcidLayer4 = Down4()

        # Transposed Convolution 
        self.wcidLayer5 = Up1()
        self.wcidLayer6 = Up2()
        self.wcidLayer7 = Up3()
        self.wcidLayer8 = Up4(num_classes)

        self.high_level_ch = self.wcidLayer7.out_channels
        
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
        in_channels=3     
        self.se_reduction = se_reduction
        # Convolution
        self.wcidLayer1 = Down1(in_channels)
        self.wcidLayer2 = Down2()
        self.wcidLayer3 = Down3()
        self.wcidLayer4 = Down4()

        # Transposed Convolution 
        self.wcidLayer5 = Up1()
        self.wcidLayer6 = Up2()
        self.wcidLayer7 = Up3()
        self.wcidLayer8 = Up4(num_classes)
        
        se_layer = {}

        # SE wcidLayers
        for i in range(1, 9):
            layerName = f"se_{i}"
            out_channels = self.__getattr__("wcidLayer" + str(i)).out_channels
            se_layer[layerName] = ChannelSELayer(
                out_channels, reduction_ratio=self.se_reduction
            )

        self.se_layer = nn.ModuleDict(se_layer)

        self.high_level_ch = self.wcidLayer7.out_channels
        
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
        x = inputs['images']

        _, prediction = self.wcid(x)
        output_dict = {}
        
        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(prediction, gts)
            return loss
        else:
            output_dict['pred'] = prediction
            return output_dict


class WCID_SE(nn.Module):
    def __init__(self, num_classes, criterion=None):
        super(WCID_SE, self).__init__()

        self.criterion = criterion
        se_reduction = 8

        self.wcid = WCID_SE_block(num_classes=num_classes, se_reduction=se_reduction)

    def forward(self, inputs):
        x = inputs['images']

        _, prediction = self.wcid(x)
        output_dict = {}
        
        if self.training:
            assert 'gts' in inputs
            gts = inputs['gts']
            loss = self.criterion(prediction, gts)
            return loss
        else:
            output_dict['pred'] = prediction
            return output_dict


class WCID_mscale(MscaleBase):
    def __init__(self, num_classes, criterion=None):
        super(WCID_mscale, self).__init__()

        self.criterion = criterion

        self.wcid = WCID_block(num_classes=num_classes)
        self.high_level_ch = self.wcid.high_level_ch

        self.scale_attn = old_make_attn_head(
            in_ch=self.high_level_ch*2, bot_ch=self.high_level_ch//2 ,out_ch=1)
        

    def _fwd(self, x):
        # x_size = x.size()[2:]

        pre_seg_head, final = self.wcid(x)

        # attn = self.scale_attn(pre_seg_head)

        # final = Upsample(final, x_size)
        # attn = Upsample(attn, x_size)

        return final, pre_seg_head


class WCID_SE_mscale(MscaleBase):
    def __init__(self, num_classes, criterion=None):
        super(WCID_SE_mscale, self).__init__()

        self.criterion = criterion
        se_reduction = 8

        self.wcid = WCID_SE_block(num_classes=num_classes, se_reduction=se_reduction)
                    
        self.high_level_ch = self.wcid.high_level_ch


        self.scale_attn = old_make_attn_head(
            in_ch=self.high_level_ch*2, bot_ch=self.high_level_ch//2, out_ch=1)
        

    def _fwd(self, x):
        pre_seg_head, final = self.wcid(x)

        return final, pre_seg_head


class Conv_ReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = nn.Conv2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            padding="valid",
            stride=(1, 1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class TransConv_ReLu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size

        self.conv1 = ConvTranspose2d(
            self.in_channels,
            self.out_channels,
            kernel_size=self.kernel_size,
            stride=(1, 1),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        return x


class Down1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.batchNorm = nn.BatchNorm2d(in_channels)

        self.conv1 = Conv_ReLu(in_channels, 8, kernel_size=(3, 3))
        self.conv2 = Conv_ReLu(8, 16, kernel_size=(3, 3))

        self.maxPool = nn.MaxPool2d(kernel_size=(2, 2))

        self.out_channels = self.conv2.out_channels

    def forward(self, x):
        x = self.batchNorm(x)

        x = self.conv1(x)
        x = self.conv2(x)

        x = self.maxPool(x)

        return x


class Down2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_ReLu(16, 16, kernel_size=(5, 5))
        self.drop1 = nn.Dropout2d(0.2)
        self.conv2 = Conv_ReLu(16, 32, kernel_size=(3, 3))
        self.drop2 = nn.Dropout2d(0.2)
        self.conv3 = Conv_ReLu(32, 32, kernel_size=(5, 5))

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
    def __init__(self):
        super().__init__()

        self.conv1 = Conv_ReLu(32, 64, kernel_size=(3, 3))
        self.drop1 = nn.Dropout2d(0.2)

        self.conv2 = Conv_ReLu(64, 64, kernel_size=(5, 5))
        self.drop2 = nn.Dropout2d(0.2)

        self.maxpool = nn.MaxPool2d(kernel_size=(2, 2))

        self.out_channels = self.conv1.out_channels

    def forward(self, x):
        x = self.conv1(x)
        x = self.drop1(x)

        x = self.conv2(x)
        x = self.drop2(x)

        x = self.maxpool(x)

        return x


class Down4(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = Conv_ReLu(64, 64, kernel_size=(3, 3))
        self.drop1 = nn.Dropout2d(0.2)
        self.conv2 = Conv_ReLu(64, 64, kernel_size=(5, 5))
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


class MyUpsample(nn.Module):
    def __init__(self, scale_factor) -> None:
        super().__init__()
        self.up = nn.Upsample(
            scale_factor=scale_factor, 
            mode='nearest', 
            # align_corners=align_corners
        )

    def forward(self, x):
        x = self.up(x)

        return x


class Up1(nn.Module):
    def __init__(self):
        super().__init__()
        self.up = MyUpsample(scale_factor=2)

        self.conv1 = TransConv_ReLu(64, 64, kernel_size=(5, 5))
        self.drop1 = nn.Dropout2d(0.2)
        self.conv2 = TransConv_ReLu(64, 64, kernel_size=(3, 3))
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
    def __init__(self):
        super().__init__()
        self.up = MyUpsample(scale_factor=2)

        self.conv1 = TransConv_ReLu(64, 64, kernel_size=(5, 5))
        self.drop1 = nn.Dropout2d(0.2)

        self.conv2 = TransConv_ReLu(64, 64, kernel_size=(3, 3))
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
    def __init__(self):
        super().__init__()
        self.up = MyUpsample(scale_factor=2)

        self.conv1 = TransConv_ReLu(64, 32, kernel_size=(5, 5))
        self.drop1 = nn.Dropout2d(0.2)
        self.conv2 = TransConv_ReLu(32, 32, kernel_size=(3, 3))
        self.drop2 = nn.Dropout2d(0.2)
        self.conv3 = TransConv_ReLu(32, 16, kernel_size=(5, 5))
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
    def __init__(self, n_classes):
        super().__init__()
        self.up = MyUpsample(scale_factor=2)

        self.conv1 = TransConv_ReLu(16, 16, kernel_size=(3, 3))
        self.conv2 = ConvTranspose2d(16, n_classes, kernel_size=(3, 3), stride=(1, 1))

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


def init_weight(model):
    model_dict = model.state_dict()

    pretrained_path = "/home/s0559816/Desktop/wcid-pytorch/logs/best_runs/channel_augmented-puma_2022.01.04_05.55/720x304/channel_8/bs_2/nearest/1/best_ego.pth"
    # Load state_dict
    torch_ = torch.load(pretrained_path, map_location={'cuda:0': 'cpu'})
    try:
        pretrained_dict = torch_["state_dict"]
    except:
        pretrained_dict = torch_

    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict) 
    # 3. load the new state dict
    model.load_state_dict(model_dict)

    logx.msg('=> loading pretrained model {}'.format(pretrained_path))

    return model


def wcid(num_classes, criterion):
    model = WCID(num_classes=num_classes, criterion=criterion)
    # model = init_weight(model)
    return model

def wcid_se(num_classes, criterion):
    model = WCID_SE(num_classes=num_classes, criterion=criterion)  
    # model = init_weight(model)
    return model

def wcid_mscale(num_classes, criterion):
    model = WCID_mscale(num_classes=num_classes, criterion=criterion)
    # model = init_weight(model)
    return model

def wcid_se_mscale(num_classes, criterion):
    model = WCID_SE_mscale(num_classes=num_classes, criterion=criterion)
    # model = init_weight(model)
    return model