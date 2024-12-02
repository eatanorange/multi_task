import math
import numpy as np
import cv2 as cv
import torch
import torchvision
from torch import nn
from torch.utils.data import DataLoader,Dataset
import os
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import torchvision.models as models
from tqdm import tqdm
from anywaydata import dataset
from resnet50 import model
import sys
from torch.utils.tensorboard import SummaryWriter
import torch.optim.lr_scheduler as lr_scheduler

import torch
import torch.nn.functional as F

total_size = len(dataset)
train_size = int(0.8 * total_size)  # 80% 用于训练
val_size = total_size - train_size  # 剩余 20% 用于验证
batch_size = 32
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
writer = SummaryWriter('runs/resnet50')
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
loss_function=nn.CrossEntropyLoss()
loss_function.cuda()

epochs = 10
total_train_step=0
total_val_step=0
for epoch in range(epochs):
     model.train()
     train_bar=tqdm(train_dataloader)
     for data in train_bar:
          imgs,labels=data
          imgs=imgs.cuda()
          labels=labels.cuda()
          output=model(imgs)
          loss=loss_function(output,labels)

          optimizer.zero_grad()#清零梯度
          loss.backward()#反向传播,给一个梯度
          optimizer.step()#优化器更新,就是根据梯度进行梯度下降

          total_train_step = total_train_step + 1
          if total_train_step % 100 == 0:
               writer.add_scalar("train_loss", loss.item(), total_train_step)
          train_bar.set_description("train_epoch:{}".format(epoch))
    
     model.eval()
     test_bar=tqdm(val_dataloader)
     total_loss=0
     total_accuracy=0
     for data in test_bar:
          imgs,labels=data
          imgs=imgs.cuda()
          labels=labels.cuda()
          output=model(imgs)
          #计算准确率
          accuracy = (output.argmax(1) == labels).sum()
          total_accuracy=total_accuracy+accuracy

          loss=loss_function(output,labels)
          total_loss=total_loss+loss.item()
          test_bar.set_description("val_epoch:{}".format(epoch))
          total_val_step = total_val_step + 1
     total_loss=total_loss/len(val_dataloader)
     total_accuracy=total_accuracy/len(val_dataset)

     writer.add_scalar("val_loss", total_loss,epoch)
     print("第{}轮训练结束，验证集loss为{},正确率为{}".format(epoch,total_loss,total_accuracy))
     torch.save(model.state_dict, "weight/model_{}.pth".format(epoch))
     print("模型已保存")
     epoch = epoch + 1