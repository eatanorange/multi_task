import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

root='dataset/xray'
class mydataset(Dataset):
    def __init__(self, path, transform=None):
        self.root = root
        self.transform = transform
        self.img_root=f'{root}/img'
        self.mask_root=f'{root}/mask_masks_machine'
        self.img_list=os.listdir(self.img_root)
        self.mask_list=os.listdir(self.mask_root)

    def __len__(self):
        return len(self.img_list)
    def __getitem__(self, index):
        img_path=os.path.join(self.img_root,self.img_list[index])
        mask_path=os.path.join(self.mask_root,self.mask_list[index])
        img=Image.open(img_path)
        mask=Image.open(mask_path)
        

transform=transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),    
])

dataset=myData(root,transform)       
    
