import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import constant_init, kaiming_init


def last_zero_init(m):
    if isinstance(m, nn.Sequential):
        constant_init(m[-1], val=0)
    else:
        constant_init(m, val=0)


class ChannelPool(nn.Module):
    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True,
                 bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class SAWeightModule(nn.Module):
    def __init__(self):
        super(SAWeightModule, self).__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = BasicConv(2, 1, kernel_size, stride=1, padding=(kernel_size - 1) // 2, relu=False)

    def forward(self, x):
        x_compress = self.compress(x)
        x_out = self.spatial(x_compress)
        weight = F.sigmoid(x_out)  # broadcasting
        return weight


class CAWeightModule(nn.Module):

    def __init__(self, channels, reduction=4):
        super(CAWeightModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.avg_pool(x)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        weight = self.sigmoid(out)
        return weight


class NLCWeightModule(nn.Module):

    def __init__(self, in_channels, ratio=1 / 4):
        super(NLCWeightModule, self).__init__()
        channels = int(in_channels * ratio)
        self.conv_mask = nn.Conv2d(in_channels, 1, kernel_size=(1, 1))
        self.softmax = nn.Softmax(dim=2)
        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=(1, 1)),
            nn.LayerNorm([channels, 1, 1]),
            nn.ReLU(inplace=True),  # yapf: disable
            nn.Conv2d(channels, in_channels, kernel_size=(1, 1)))
        self.reset_parameters()

    def reset_parameters(self):
        kaiming_init(self.conv_mask, mode='fan_in')
        self.conv_mask.inited = True
        last_zero_init(self.channel_add_conv)

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(-1)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)
        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.spatial_pool(x)
        # out0 = x
        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        # out0 = out0 + channel_add_term
        return channel_add_term


class ENLCModule(nn.Module):

    def __init__(self, in_channels, square_kernel_size=3, band_kernel_size=5, branch_ratio=1 / 4):
        super(ENLCModule, self).__init__()
        gc = int(in_channels * branch_ratio)
        self.conv_hw = nn.Conv2d(gc, gc, square_kernel_size, padding=square_kernel_size // 2)
        self.dwconv_w = nn.Conv2d(gc, gc, kernel_size=(1, band_kernel_size),
                                  padding=(0, band_kernel_size // 2), groups=gc)
        self.dwconv_h = nn.Conv2d(gc, gc, kernel_size=(band_kernel_size, 1),
                                  padding=(band_kernel_size // 2, 0), groups=gc)
        self.split_indexes = (gc, gc, gc, in_channels - 3 * gc)
        self.nlc = NLCWeightModule(gc)
        self.nlc1 = NLCWeightModule(in_channels - 3 * gc)
        self.split_channel = gc
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        batch_size = x.shape[0]
        x_hw, x_w, x_h, x_id = torch.split(x, self.split_indexes, dim=1)
        x1 = self.conv_hw(x_hw)
        x2 = self.dwconv_w(x_w)
        x3 = self.dwconv_h(x_h)
        x4 = x_id
        feats = torch.cat((x1, x2, x3, x4), dim=1)
        feats = feats.view(batch_size, 4, self.split_channel, feats.shape[2], feats.shape[3])
        x1_nlc = self.nlc(x1)
        x2_nlc = self.nlc(x2)
        x3_nlc = self.nlc(x3)
        x4_nlc = self.nlc1(x4)
        x_nlc = torch.cat((x1_nlc, x2_nlc, x3_nlc, x4_nlc), dim=1)
        attention_vectors = x_nlc.view(batch_size, 4, self.split_channel, 1, 1)
        # attention_vectors = self.softmax(attention_vectors)
        feats_weight = feats + attention_vectors
        for i in range(4):
            x_nlc_weight_fp = feats_weight[:, i, :, :]
            if i == 0:
                out = x_nlc_weight_fp
            else:
                out = torch.cat((x_nlc_weight_fp, out), 1)

        return out


class ENLCBlockA(nn.Module):
    def __init__(self, in_channels=64):
        super(ENLCBlockA, self).__init__()
        self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels, in_channels, 3, 1, 0),
            nn.InstanceNorm2d(in_channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            ENLCModule(in_channels),
            nn.InstanceNorm2d(in_channels)
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        out = self.prelu(out)
        return out


class ENLCBlockB(nn.Module):
    def __init__(self, in_channels=128, ratio=1 / 2):
        super(ENLCBlockB, self).__init__()
        channels = int(in_channels * ratio)
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, channels, 1, 1, 0),
            nn.InstanceNorm2d(channels),
            nn.PReLU()
        )
        self.conv2 = nn.Sequential(
            ENLCModule(channels),
            nn.InstanceNorm2d(channels),
            nn.PReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(channels, in_channels, 1, 1, 0),
            nn.InstanceNorm2d(in_channels)
        )
        self.prelu = nn.PReLU()

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out += identity
        out = self.prelu(out)
        return out


class JNet(nn.Module):
    def __init__(self, num=64, block=ENLCBlockA, layers=3):
        super().__init__()
        self.in_planes = num
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.PReLU()
        )
        self.layer = self._make_layers(block, layers)
        self.final = nn.Sequential(
            nn.Conv2d(num, 3, 1, 1, 0),
            nn.Sigmoid()
        )

    def _make_layers(self, block, num_blocks):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer(x)
        out = self.final(x)
        return out


class TNet(torch.nn.Module):
    def __init__(self, num=128, block=ENLCBlockB, layers=3):
        super().__init__()
        self.in_planes = num
        self.conv1 = torch.nn.Sequential(
            torch.nn.ReflectionPad2d(1),
            torch.nn.Conv2d(3, num, 3, 1, 0),
            torch.nn.InstanceNorm2d(num),
            torch.nn.PReLU()
        )
        self.layer = self._make_layers(block, layers)
        self.final = torch.nn.Sequential(
            torch.nn.Conv2d(num, 3, 1, 1, 0),
            torch.nn.Sigmoid()
        )

    def _make_layers(self, block, num_blocks):
        layers = []
        for i in range(1, num_blocks):
            layers.append(block(self.in_planes))
        return nn.Sequential(*layers)

    def forward(self, data):
        data = self.conv1(data)
        data = self.layer(data)
        data1 = self.final(data)
        return data1
