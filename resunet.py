import torch
import torch.nn as nn
from torchvision.models import resnet50

# 定义ResNet-50的主干部分
class ResNetBackbone(nn.Module):
    def __init__(self):
        super(ResNetBackbone, self).__init__()
        # 使用预训练的ResNet-50模型
        self.resnet = resnet50(pretrained=True)
        # 移除最后的全连接层
        self.resnet = nn.Sequential(*list(self.resnet.children())[:-2])

    def forward(self, x):
        return self.resnet(x)

# 定义UNet的解码部分
class UNetDecoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UNetDecoder, self).__init__()
        self.upconv1 = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1)
        self.upconv2 = nn.ConvTranspose2d(in_channels // 2, in_channels // 4, kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels // 2, in_channels // 4, kernel_size=3, padding=1)
        self.outconv = nn.Conv2d(in_channels // 4, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.upconv1(x)
        x = self.conv1(x)
        x = self.upconv2(x)
        x = self.conv2(x)
        x = self.outconv(x)
        return x

# 定义ResUNet模型
class ResUNet(nn.Module):
    def __init__(self, num_classes):
        super(ResUNet, self).__init__()
        self.backbone = ResNetBackbone()
        self.classifier_rsna= nn.Sequential(nn.AdaptiveAvgPool2d((7,7)),
                                        nn.Flatten(),
                                        nn.Linear(50176, num_classes))
        self.classifier_covid= nn.Sequential(nn.AdaptiveAvgPool2d((7,7)),
                                        nn.Flatten(),
                                        nn.Linear(50176, num_classes))
        self.decoder = UNetDecoder(2048, num_classes)

    def forward(self, x):
        x = self.backbone(x)
        classify_rsna = self.classifier_rsna(x)
        classify_covid = self.classifier_covid(x)

        segment = self.decoder(x)
        return classify_rsna,segment,classify_covid

# 创建ResUNet模型实例
model = ResUNet(num_classes=2)  # 假设num_classes为10
model.cuda()