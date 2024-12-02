from torchvision import models
from torchvision.models import ResNet50_Weights
from torch import nn
# 使用新的权重方式
resnet50 = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
num_ftrs = resnet50.fc.in_features

# 保持in_features不变，修改out_features=2
resnet50.fc = nn.Sequential(nn.Linear(num_ftrs, 2), nn.LogSoftmax(dim=1))

model = resnet50
model.cuda()

if __name__ == '__main__':
    print(model)