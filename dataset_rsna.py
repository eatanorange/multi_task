import os
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from torchvision import transforms
import torch

root='dataset/rsna'
class myData(Dataset):
    def __init__(self, root, transform=None):
        self.root = root
        self.transform = transform
        self.normal_root=f'{root}/normal'
        self.abnormal_root=f'{root}/abnormal'
        self.mask_root=f'{root}/mask'
        self.normal_list=os.listdir(self.normal_root)
        self.abnormal_list=os.listdir(self.abnormal_root)
        self.mask_list=os.listdir(self.mask_root)
    def __len__(self):
        return len(self.normal_list)+len(self.abnormal_list)

    def __getitem__(self, index):
        if index < len(self.normal_list):
            img_path = os.path.join(self.normal_root, self.normal_list[index])
            img = Image.open(img_path).convert('RGB')
            mask = np.zeros([224,224])
            mask=np.array(mask)
            mask = torch.from_numpy(mask).long()
            label = 0
        else:
            img_path = os.path.join(self.abnormal_root, self.abnormal_list[index-len(self.normal_list)])
            img = Image.open(img_path).convert('RGB')
            mask = Image.open(self.mask_root+'/'+self.mask_list[index-len(self.normal_list)]).convert('L')
            mask = transforms.Resize((224,224))(mask)
            mask=np.array(mask)
            mask=np.where(mask>0,1,mask) 
            mask = torch.from_numpy(mask).long()
            label = 1
        img = self.transform(img)
        return img, label, mask     
transform=transforms.Compose([
    transforms.Resize((512,512)),
    transforms.ToTensor(),    
])

dataset=myData(root,transform)       
    


total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% 用于训练
val_size = total_size - train_size  # 剩余 20% 用于验证

rsna_train_dataset, rsna_val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])