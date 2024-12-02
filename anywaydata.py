import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

class myData(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        
        self.normal_root=f'{root}/normal'
        self.abnormal_root=f'{root}/abnormal'
        self.normal_list=os.listdir(self.normal_root)
        self.abnormal_list=os.listdir(self.abnormal_root)

    def __len__(self):
        return len(self.normal_list)+len(self.abnormal_list)
    
    def __getitem__(self, index):
        if index < len(self.normal_list):
            img_path = os.path.join(self.normal_root, self.normal_list[index])
            img = Image.open(img_path).convert('RGB')
            label = 0
        else:
            img_path = os.path.join(self.abnormal_root, self.abnormal_list[index-len(self.normal_list)])
            img = Image.open(img_path).convert('RGB')
            label = 1
        img = self.transform(img)
        return img, label

root='dataset/rsna'
transform=transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),    
])

dataset=myData(root,transform)        
        
