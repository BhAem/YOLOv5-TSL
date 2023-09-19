import torch
import torch.nn as nn
import torch.nn.functional as F
# from torchsummary import summary


def make_model(args, parent=False):
    return Net()


class ConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class UpsampleConvLayer(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride):
      super(UpsampleConvLayer, self).__init__()
      self.conv2d = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride=stride)

    def forward(self, x):
        out = self.conv2d(x)
        return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.relu = nn.PReLU()

    def forward(self, x):
        residual = x
        out = self.relu(self.conv1(x))
        out = self.conv2(out) * 0.1
        out = torch.add(out, residual)
        return out


class Net(nn.Module):
    def __init__(self, res_blocks=4):
        super(Net, self).__init__()

        self.conv_input = ConvLayer(3, 16, kernel_size=11, stride=1)
        self.dense0 = nn.Sequential(
            # ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv2x = ConvLayer(16, 32, kernel_size=3, stride=2)
        # self.fusion1 = Encoder_MDCBlock1(32, 2, mode='iter2')
        self.dense1 = nn.Sequential(
            # ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.conv4x = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.dense2 = nn.Sequential(
            # ResidualBlock(64),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.conv8x = ConvLayer(64, 128, kernel_size=3, stride=2)

        self.dehaze = nn.Sequential()
        for i in range(0, res_blocks):
            self.dehaze.add_module('res%d' % i, ResidualBlock(128))

        self.convd8x = UpsampleConvLayer(128, 64, kernel_size=3, stride=2)
        self.dense_4 = nn.Sequential(
            # ResidualBlock(128),
            ResidualBlock(64),
            ResidualBlock(64)
        )

        self.convd4x = UpsampleConvLayer(64, 32, kernel_size=3, stride=2)
        self.dense_2 = nn.Sequential(
            # ResidualBlock(32),
            ResidualBlock(32),
            ResidualBlock(32)
        )

        self.convd2x = UpsampleConvLayer(32, 16, kernel_size=3, stride=2)
        self.dense_1 = nn.Sequential(
            # ResidualBlock(16),
            ResidualBlock(16),
            ResidualBlock(16)
        )

        self.conv_output = ConvLayer(16, 3, kernel_size=3, stride=1)


    def forward(self, x):
        res1x = self.conv_input(x)
        x = self.dense0(res1x) + res1x

        res2x = self.conv2x(x)
        res2x = self.dense1(res2x) + res2x

        res4x = self.conv4x(res2x)
        res4x = self.dense2(res4x) + res4x

        res8x = self.conv8x(res4x)
        # res16x = self.fusion4(res16x, feature_mem)
        #res16x = self.dense4(res16x)

        res_dehaze = res8x
        in_ft = res8x*2
        res8x = self.dehaze(in_ft) + in_ft - res_dehaze
        # feature_mem_up = [res16x]

        res8x = self.convd8x(res8x)
        res8x = F.upsample(res8x, res4x.size()[2:], mode='bilinear')
        res4x = torch.add(res8x, res4x)
        res4x = self.dense_4(res4x) + res4x - res8x


        res4x = self.convd4x(res4x)
        res4x = F.upsample(res4x, res2x.size()[2:], mode='bilinear')
        res2x = torch.add(res4x, res2x)
        res2x = self.dense_2(res2x) + res2x - res4x
        # res2x = self.fusion_2(res2x, feature_mem_up)
        # feature_mem_up.append(res2x)

        res2x = self.convd2x(res2x)
        res2x = F.upsample(res2x, x.size()[2:], mode='bilinear')
        x = torch.add(res2x, x)
        x = self.dense_1(x) + x - res2x
        # x = self.fusion_1(x, feature_mem_up)

        x = self.conv_output(x)

        return x


# if __name__=='__main__':
#     # model = ResnetEncoder(18, False, "")
#     # image = torch.randn(4, 3, 192, 640)
#     # output = model(image)
#     # print(output)
#
#     net = Net()
#     summary(net, (3, 640, 640), device='cpu')