from glob import glob
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import lr_scheduler
from torch import optim
%matplotlib inline
from torch.utils.data import DataLoader
from scipy import io
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
import time
from sklearn.model_selection import KFold

class Net(nn.Module):
    
    def __init__(self):
        super(Net,self).__init__()
        self.fc1 = nn.Linear(num_features, 128)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 128)
        self.dout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(128, 64)
        self.dout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(64, 32)
        self.fc5 = nn.Linear(32, 16)
        self.fc6 = nn.Linear(16, 1)
        
    def forward(self, input_):
        a1 = F.relu(self.fc1(input_))
        dout1 = self.dout(a1)
        a2 = F.relu(self.fc2(dout1))
        dout2 = self.dout2(a2)
        a3 = F.relu(self.fc3(dout2))
        dout3 = self.dout3(a3)
        a4 = F.relu(self.fc4(dout3))
        dout4 = self.dout3(a4)
        a5 = F.relu(self.fc5(dout4))
        return torch.sigmoid(self.fc6(a5))

def get_point():
    mat = scipy.io.loadmat('point.mat')
    
os.chdir('~\path')

test = io.loadmat('tt_csp_feat')
x_test_left = []
x_test_right = []
num_trials = test['x_test'][0][0].shape[2]
num_features = test['x_test'][0][0].shape[0]
num_subs = test['x_test'].shape[0]
for sub in range(0,num_subs):
    for trials in range(0,int(num_trials/2)):
        for features in range(0,num_features):
            x_test_left.append(test['x_test'][sub][0][features][0][trials])
for sub in range(0,num_subs):
    for trials in range(int(num_trials/2),num_trials):
        for features in range(0,num_features):
            x_test_right.append(test['x_test'][sub][0][features][0][trials])            
x_test_left = np.array(x_test_left)
x_test_right = np.array(x_test_right)
x_test = np.concatenate((x_test_left, x_test_right))
x_test = np.reshape(x_test,(num_subs*num_trials,num_features))

train = io.loadmat('tr_csp_feat')
x_train_left = []
x_train_right = []
num_trials = train['x_train'][0][0].shape[2]
num_features = train['x_train'][0][0].shape[0]
num_subs = test['x_test'].shape[0]
for sub in range(0,num_subs):
    for trials in range(0,int(num_trials/2)):
        for features in range(0,train['x_train'][0][0].shape[0]):
            x_train_left.append(train['x_train'][sub][0][features][0][trials])
for sub in range(0,num_subs):
    for trials in range(int(num_trials/2),num_trials):
        for features in range(0,train['x_train'][0][0].shape[0]):
            x_train_right.append(train['x_train'][sub][0][features][0][trials])
x_train_left = np.array(x_train_left)
x_train_right = np.array(x_train_right)
x_train = np.concatenate((x_train_left, x_train_right))
x_train = np.reshape(x_train,(num_subs*num_trials,num_features))

class CustomDataset(): 
    def __init__(self,x_data):
        self.x_data = x_data

  # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        if idx >= num_subs*num_trials/2:
            y = 1
        else: y = 0
        return x, y
    
train_set = CustomDataset(x_train)
test_set = CustomDataset(x_test)

batch_size=32
train_data_gen = torch.utils.data.DataLoader(train_set,shuffle=True,batch_size=batch_size,num_workers=0)
valid_data_gen = torch.utils.data.DataLoader(test_set,shuffle=True,batch_size=batch_size,num_workers=0)

dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}
print(dataset_sizes)

dataloaders = {'train':train_data_gen,'valid':valid_data_gen}

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):
    #since = time.time()
    save_path = f'./model-fold-{3-1}.pth'
    
    if os.path.isfile(save_path):
        model_ft.load_state_dict(torch.load(save_path), strict=False)
        model_ft.eval()
        best_model_wts = model_ft.eval() #model.state_dict()
        
    if len(record) == 0: 
        best_acc = 0.0
    else: best_acc = max(record)
    
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
                labels = labels.type(torch.float64)
                
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
            
#             if epoch%10 == 0:
#                 print('{} Loss: {:.4f} Acc: {:.4f}'.format(
#                         phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                if epoch_acc.item() < 0.8:
                    best_acc = epoch_acc
                    record.append(best_acc)
                    best_model_wts = model.state_dict()
                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                            phase, epoch_loss, epoch_acc))
                    best = best_model_wts
                    torch.save(best, save_path)
                    print('Best Model saved')
        
    #time_elapsed = time.time() - since
    
    #print('Training complete in {:.0f}m {:.0f}s'.format(
    #    time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    #model.load_state_dict(best_model_wts)
    

    #return model

record = []
# criterion = nn.BCELoss()
# optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
# exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=18, gamma=0.1)
# model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
#                        num_epochs=300)

model_ft = Net()
if torch.cuda.is_available():
    model_ft = model_ft.cuda()
    print('check')
criterion = nn.BCELoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.01, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=16, gamma=0.1)
model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                       num_epochs=50)

record

for i in range(int((x_test.shape[0]/2)/20)):
    globals()['sub{}_t'.format(i+1)] = x_test[20*i:20*(i+1)]
    
x_test_r = x_test[int(x_test.shape[0]/2):]

for i in range(int(x_test_r.shape[0]/20)):
    globals()['sub{}_t'.format(i+1)] = np.append(globals()['sub{}_t'.format(i+1)],x_test_r[20*i:20*(i+1)])

for i in range(52):
    globals()['sub{}_t'.format(i+1)] = globals()['sub{}_t'.format(i+1)].reshape(40,12)
    
class CustomDataset(): 
    def __init__(self,x_data):
        self.x_data = x_data

  # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        if idx >= 40/2:
            y = 1
        else: y = 0
        return x, y

for i in range(52):
    globals()['test_sub{}'.format(i+1)] = CustomDataset(globals()['sub{}_t'.format(i+1)])

for i in range(int((x_train.shape[0]/2)/80)):
    globals()['sub{}'.format(i+1)] = x_train[80*i:80*(i+1)]
    
x_train_r = x_train[int(x_train.shape[0]/2):]

for i in range(int(x_train_r.shape[0]/80)):
    globals()['sub{}'.format(i+1)] = np.append(globals()['sub{}'.format(i+1)],x_train_r[80*i:80*(i+1)])

for i in range(52):
    globals()['sub{}'.format(i+1)] = globals()['sub{}'.format(i+1)].reshape(160,12)

class CustomDataset(): 
    def __init__(self,x_data):
        self.x_data = x_data

  # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        if idx >= 160/2:
            y = 1
        else: y = 0
        return x, y

for i in range(52):
    globals()['train_sub{}'.format(i+1)] = CustomDataset(globals()['sub{}'.format(i+1)])

def train_model(model, criterion, optimizer, scheduler, num_epochs=5):

    save_path = f'./model-fold-{3-1}.pth'
    
    if os.path.isfile(save_path):
        model_ft.load_state_dict(torch.load(save_path), strict=False)
        model_ft.eval()
        best_model_wts = model_ft.eval() #model.state_dict()

    best_acc = 0.0
    
    for epoch in range(num_epochs):
        
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
                labels = labels.type(torch.float64)
                
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
              
            if epoch_acc.item() > best_acc:
                best_acc = epoch_acc.item()
                print('New best:',best_acc)

    print('Best val Acc:', best_acc)
    result.append(best_acc)

model_ft = Net()
if torch.cuda.is_available():
    model_ft = model_ft.cuda()
    print('check')
criterion = nn.BCELoss()
optimizer_ft = torch.optim.SGD(model_ft.parameters(), lr=0.00001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=1, gamma=0.1)

result = []
for i in range(52):
    model_ft = Net()
    if torch.cuda.is_available():
        model_ft = model_ft.cuda()
        print('check')
    train_set = globals()['train_sub{}'.format(i+1)]
    test_set = globals()['test_sub{}'.format(i+1)]
    batch_size=20
    train_data_gen = torch.utils.data.DataLoader(train_set,shuffle=True,batch_size=batch_size,num_workers=0)
    valid_data_gen = torch.utils.data.DataLoader(test_set,shuffle=True,batch_size=batch_size,num_workers=0)
    dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}
    dataloaders = {'train':train_data_gen,'valid':valid_data_gen}
    print('sub{}'.format(i+1))
    #save_path = f'./model-fold-{4}.pth'
    #model_ft.load_state_dict(torch.load(save_path), strict=False)
    #model_ft.eval()
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=50)

import statistics
result2 = []
for i in range(len(result)):
    result2.append(float(result[i]))
statistics.mean(result2)

result = []
for i in range(52):
    model_ft = Net()
    if torch.cuda.is_available():
        model_ft = model_ft.cuda()
        print('check')
    train_set = globals()['train_sub{}'.format(i+1)]
    test_set = globals()['test_sub{}'.format(i+1)]
    batch_size=20
    train_data_gen = torch.utils.data.DataLoader(train_set,shuffle=True,batch_size=batch_size,num_workers=0)
    valid_data_gen = torch.utils.data.DataLoader(test_set,shuffle=True,batch_size=batch_size,num_workers=0)
    dataset_sizes = {'train':len(train_data_gen.dataset),'valid':len(valid_data_gen.dataset)}
    dataloaders = {'train':train_data_gen,'valid':valid_data_gen}
    print('sub{}'.format(i+1))
    #save_path = f'./model-fold-{4}.pth'
    #model_ft.load_state_dict(torch.load(save_path), strict=False)
    #model_ft.eval()
    model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler,
                           num_epochs=100)

import statistics
result2 = []
for i in range(len(result)):
    result2.append(float(result[i]))
statistics.mean(result2)

result2
