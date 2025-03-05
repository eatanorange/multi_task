import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50




class sMLPBlock(nn.Module):
    def __init__(self, W, H, channels):
        super().__init__()
        assert W == H
        self.channels = channels
        self.activation = nn.GELU()
        self.BN = nn.BatchNorm2d(channels)
        self.proj_h = nn.Conv2d(H, H, (1, 1))
        self.proh_w = nn.Conv2d(W, W, (1, 1))
        self.fuse = nn.Conv2d(channels*3, channels, (1,1), (1,1), bias=False)

    def forward(self, x):
        x1 = self.activation(self.BN(x))
        x_h = self.proj_h(x1.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
        x_w = self.proh_w(x1.permute(0, 2, 1, 3)).permute(0, 2, 1, 3)
        x1 = self.fuse(torch.cat([x1, x_h, x_w], dim=1))
        x=x+x1
        return x





class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        # if you have padding issues, see
        # https://github.com/HaiyongJiang/U-Net-Pytorch-Unstructured-Buggy/commit/0e854509c2cea854e247a9c615f175f76fbb2e3a
        # https://github.com/xiaopeng-liao/Pytorch-UNet/commit/8ebac70e633bac59fc22bb5195e513d5832fb3bd
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)


class ResNet50UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=False):
        super(ResNet50UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        # Load pre-trained ResNet50
        self.resnet = resnet50(pretrained=True)

        # Remove the fully connected layer
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

        # Encoder layers
        self.encoder1 = self.resnet[:3]  # First two layers of ResNet50
        self.encoder2 = self.resnet[3:5]  # Third and fourth layers of ResNet50
        self.encoder3 = self.resnet[5]  # Fifth layer of ResNet50
        self.encoder4 = self.resnet[6]  # Sixth layer of ResNet50
        factor = 2 if bilinear else 1
        self.classifier_rsna= nn.Sequential(nn.AdaptiveAvgPool2d((7,7)),
                                        nn.Flatten(),
                                        nn.Linear(50176, n_classes))
        self.classifier_covid= nn.Sequential(nn.AdaptiveAvgPool2d((7,7)),
                                        nn.Flatten(),
                                        nn.Linear(50176, n_classes))
        self.sMLPBlock1 = sMLPBlock(112, 112, 64)
        self.sMLPBlock2 = sMLPBlock(56, 56, 256)
        self.sMLPBlock3 = sMLPBlock(28, 28, 512)
        self.sMLPBlock4 = sMLPBlock(14, 14, 1024)
        
        # Decoder layers
        self.up1 = (Up(1024, 512 // factor, bilinear))
        self.up2 = (Up(512, 256 // factor, bilinear))
        self.c = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.up3 = (Up(256, 128 // factor, bilinear))
        self.up4 =  nn.Sequential(
            nn.Conv2d(kernel_size=(3, 3), in_channels=128, out_channels=64, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.outc = (OutConv(64, n_classes))

    def forward(self, x):
        # Encoder
        x1 = self.encoder1(x)
        x1 = self.sMLPBlock1(x1)
        x2 = self.encoder2(x1)
        x2 = self.sMLPBlock2(x2)
        x3 = self.encoder3(x2)
        x3 = self.sMLPBlock3(x3)
        x4 = self.encoder4(x3)
        x4 = self.sMLPBlock4(x4)
        classify_rsna = self.classifier_rsna(x4)
        classify_covid = self.classifier_covid(x4)
        # Decoder

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        xx = self.c(x1)
        x = self.up3(x, xx)
        x = self.up4(x)
        
     
        segment = self.outc(x)

        return classify_rsna,segment,classify_covid

    def use_checkpointing(self):
        self.encoder1 = torch.utils.checkpoint(self.encoder1)
        self.encoder2 = torch.utils.checkpoint(self.encoder2)
        self.encoder3 = torch.utils.checkpoint(self.encoder3)
        self.encoder4 = torch.utils.checkpoint(self.encoder4)
        self.up1 = torch.utils.checkpoint(self.up1)
        self.up2 = torch.utils.checkpoint(self.up2)
        self.up3 = torch.utils.checkpoint(self.up3)
        self.up4 = torch.utils.checkpoint(self.up4)
        self.outc = torch.utils.checkpoint(self.outc)

model = ResNet50UNet(n_channels=3, n_classes=2)
model.cuda()
if __name__ == '__main__':
    print(model)
