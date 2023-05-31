#!/usr/bin/env python
# coding: utf-8

# In[2]:

import torch
from torch import nn
from torchsummary import summary
import torch.nn.functional as F

class MLP2(nn.Module):

	def __init__(self):
		super().__init__()
		self.fc1 = nn.Linear(784, 256)
		self.fc2 = nn.Linear(256, 10)

	def forward(self, x):
		x = torch.flatten(x, 1)
		x = torch.sigmoid(self.fc1(x))
		x = self.fc2(x)
		return F.log_softmax(x, dim=1)

class CNN2(nn.Module): #70% test acc for cifar10, 30% test acc for cifar100 in center learning mode
    def __init__(self, hidden_dims, output_dim=10):
        super(CNN2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.conv3 = nn.Conv2d(64, 128, 3)

        #self.flatten = nn.Flatten()
        
        self.fc1 = nn.Linear(128*2*2, hidden_dims[0])
        self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
        self.fc3 = nn.Linear(hidden_dims[1], output_dim)

        self.dropout = nn.Dropout(p = 0.1)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        
        x = x.view(-1, 128*2*2)
        #x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(self.fc3(x), dim=1)
        return x


# In[ ]:


mpl = MLP2()
#self.model =  MLP2().to(device) #MNIST().to(device) #CNNModel().to(device) MNIST FEMNIST
summary(mpl, (1, 28, 28))

#==========================================================================================
#Layer (type:depth-idx)                   Output Shape              Param #
#==========================================================================================
#├─Linear: 1-1                            [-1, 256]                 200,960
#├─Linear: 1-2                            [-1, 10]                  2,570
#==========================================================================================
#Total params: 203,530
#Trainable params: 203,530
#Non-trainable params: 0
#Total mult-adds (M): 0.20
#==========================================================================================
#Input size (MB): 0.00
#Forward/backward pass size (MB): 0.00
#Params size (MB): 0.78
#Estimated Total Size (MB): 0.78
#==========================================================================================


cnn = CNN2(hidden_dims=[256, 128], output_dim=10)#cifar100, cifar10
summary(cnn, input_size=(3, 32, 32)) #channel, img_size, img_size

#=================================================================
#Layer (type:depth-idx)                   Param #
#=================================================================
#├─Conv2d: 1-1                            896
#├─MaxPool2d: 1-2                         --
#├─Conv2d: 1-3                            18,496
#├─Conv2d: 1-4                            73,856
#├─Linear: 1-5                            131,328
#├─Linear: 1-6                            32,896
#├─Linear: 1-7                            1,290
#├─Dropout: 1-8                           --
#=================================================================
#Total params: 258,762
#Trainable params: 258,762
#Non-trainable params: 0
#=================================================================
