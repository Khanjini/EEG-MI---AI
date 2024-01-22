from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import shutil
from torchvision import transforms
from torchvision import models
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
import time
%matplotlib inline
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import scipy.io
from sklearn.model_selection import train_test_split

def get_point():
    mat = scipy.io.loadmat('point.mat')

os.chdir('~\path')

from scipy import io
left = io.loadmat('left_alpha')
right = io.loadmat('right_alpha')

left['left_icoh_alpha_index_vec'].shape
right['right_icoh_alpha_index_vec'].shape

X = left['left_icoh_alpha_index_vec']
y = right['right_icoh_alpha_index_vec']

print(abs(X))

print(np.multiply(X,y))

left_list = []
for i in range(0,10):
    for j in range(0,100):
        left_list.append(X[i][j])

right_list = []
for i in range(0,10):
    for j in range(0,100):
        right_list.append(y[i][j]) 

for i in range(100,120):
    left_list.append(X[6][i])
    left_list.append(X[8][i])

for i in range(100,120):
    right_list.append(y[6][i])
    right_list.append(y[8][i])

dataset = []
for i in range(len(left_list)):
    dataset.append(left_list[i])
for i in range(len(right_list)):
    dataset.append(right_list[i])

dataset_array = np.array(dataset)

left_list[1].shape

len(dataset)

class CustomDataset(Dataset): 
    def __init__(self,x_data):
        self.x_data = x_data

  # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        if idx >= 1040:
            y = 1
        else: y = 0
        #y = torch.FloatTensor(1 if idx >= 1040 else 0)
        #y = torch.FloatTensor(self.y_data[idx])
        return x, y

Dataset = CustomDataset(dataset_array)
len(Dataset)

train, valid = train_test_split(Dataset, test_size=0.4, random_state=1)

print(len(train))
print(len(valid))
print(len(train)+len(valid))

train_data_gen = torch.utils.data.DataLoader(train,shuffle=True,batch_size=32,num_workers=2)
valid_data_gen = torch.utils.data.DataLoader(valid,shuffle=True,batch_size=32,num_workers=2)
batch_size=32

dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}
print(dataset_sizes)

dataloaders = {'train':train_data_gen,'valid':valid_data_gen}

class LogisticRegression(torch.nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegression, self).__init__()
        self.linear = torch.nn.Linear(input_dim, output_dim)
    def forward(self, x):
        outputs = torch.sigmoid(self.linear(x))
        return outputs

import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(2016, 4032)
        self.fc2 = nn.Linear(4032, 4032)
        self.fc3 = nn.Linear(4032, 2016)
        self.fc4 = nn.Linear(2016, 1008)
        self.fc5 = nn.Linear(1008, 1008)
        self.fc6 = nn.Linear(1008, 504)
        self.fc7 = nn.Linear(504, 250)
        self.fc8 = nn.Linear(250, 100)
        self.fc9 = nn.Linear(100, 50)
        self.fc10 = nn.Linear(50, 20)
        self.fc11 = nn.Linear(20, 10)
        self.fc12 = nn.Linear(10, 2)

    def forward(self, x):
        x = self.fc1(x).clamp(min=0)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = F.relu(self.fc6(x))
        x = F.relu(self.fc7(x))
        x = F.relu(self.fc8(x))
        x = F.relu(self.fc9(x))
        x = F.relu(self.fc10(x))
        x = F.relu(self.fc11(x))
        x = self.fc12(x)
        return x

import torch.nn.functional as F
class Net2(nn.Module):
    
    def __init__(self):
        super(Net2,self).__init__()
        self.fc1 = nn.Linear(2016, 3000)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(3000, 1500)
        self.dout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(1500, 700)
        self.dout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(700, 300)
        self.fc5 = nn.Linear(300, 1)
        
    def forward(self, input_):
        a1 = F.relu(self.fc1(input_))
        dout1 = self.dout(a1)
        a2 = F.relu(self.fc2(dout1))
        dout2 = self.dout2(a2)
        a3 = F.relu(self.fc3(dout2))
        dout3 = self.dout3(a3)
        a4 = F.relu(self.fc4(dout3))
        return torch.sigmoid(self.fc5(a4))

model_ft = Net2()#LogisticRegression(2016,1)

if torch.cuda.is_available():
    model_ft = model_ft.cuda()
    print('check')

criterion = nn.BCELoss()
optimizer_ft = torch.optim.Adam(model_ft.parameters(), lr=0.001) #, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=6, gamma=0.1)

from sklearn.preprocessing import OneHotEncoder

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    since = time.time()

    best_model_wts = model.state_dict()
    best_acc = 0.0
    
    for epoch in range(num_epochs):
        if epoch%10 == 0:
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
            print('-' * 10)
        
        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                scheduler.step()
                model.train(True)  # Set model to training mode
            else:
                model.train(False)  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for data in dataloaders[phase]:
                
                # get the inputs
                inputs, labels = data
                labels = labels.reshape(batch_size,1)
                onehot_encoder = OneHotEncoder()
                labels = onehot_encoder.fit_transform(labels)
                labels = labels.toarray()
                labels = torch.IntTensor(labels)
                #labels = labels.type(torch.float64)
                
                # wrap them in Variable
                if torch.cuda.is_available():
                    inputs = Variable(inputs.cuda(),requires_grad=True)
                    labels = Variable(labels.cuda())
                else:
                    inputs, labels = Variable(inputs,requires_grad=True), Variable(labels)
                
                # zero the parameter gradients
                optimizer.zero_grad()
                
                # forward
                outputs = model(inputs)
                outputs = outputs.squeeze(1)
                outputs = outputs.type(torch.float64)
                preds = outputs.round().detach()
                preds = preds.reshape(batch_size,1)
                preds = preds.type(torch.float64)
                labels = labels.cpu()
                labels = onehot_encoder.inverse_transform(labels)
                labels = torch.IntTensor(labels)
                labels = labels.type(torch.float64)
                labels = labels.cuda()
                loss = criterion(outputs, labels)
                
                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.requires_grad_(True)
                    loss.backward()
                    optimizer.step()
                # statistics
                running_loss += loss.data.type(torch.float64)
                running_corrects += torch.sum(preds == labels.data).type(torch.float64)
                
                
            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects / dataset_sizes[phase]
            
            if epoch%10 == 0:
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                        phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = model.state_dict()

        
    time_elapsed = time.time() - since
    
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=1000)

