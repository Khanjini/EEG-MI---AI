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

def get_point():
    mat = scipy.io.loadmat('point.mat')
    
os.chdir('~\path')

left_index = []
for s in range(52):
    left_mi = io.loadmat('left_mi_icoh_beta%02d'%(s+1))
    left_rest = io.loadmat('left_rest_icoh_beta%02d'%(s+1))
    for t in range(100):
        for r in range(64):
            for c in range(64):
                if r > c:
                    index = left_mi['left_mi_icoh_beta_mat'][s][t][r][c]/left_rest['left_rest_icoh_beta_mat'][s][t][r][c]
                    left_index.append(index)
left_index = np.array(left_index)
left_index = np.reshape(left_index,(52*100,2016))
left_index = np.tanh(np.log(left_index))

right_index = []
for s in range(52):
    right_mi = io.loadmat('right_mi_icoh_beta%02d'%(s+1))
    right_rest = io.loadmat('right_rest_icoh_beta%02d'%(s+1))
    for t in range(100):
        for r in range(64):
            for c in range(64):
                if r > c:
                    index = right_mi['right_mi_icoh_beta_mat'][s][t][r][c]/right_rest['right_rest_icoh_beta_mat'][s][t][r][c]
                    right_index.append(index)
right_index = np.array(right_index)
right_index = np.reshape(right_index,(52*100,2016))
right_index = np.tanh(np.log(right_index))

class CustomDataset(): 
    def __init__(self,x_data):
        self.x_data = x_data

  # 총 데이터의 개수를 리턴
    def __len__(self): 
        return len(self.x_data)

  # 인덱스를 입력받아 그에 맵핑되는 입출력 데이터를 파이토치의 Tensor 형태로 리턴
    def __getitem__(self, idx): 
        x = torch.FloatTensor(self.x_data[idx])
        if idx >= 52*100:
            y = 1
        else: y = 0
        return x, y

dataset_array = np.append(left_index,right_index)
dataset_array = np.reshape(dataset_array,(52*100*2,2016))
dataset = CustomDataset(dataset_array)
num_features = dataset_array[1].size

class Net2(nn.Module):
    
    def __init__(self):
        super(Net2,self).__init__()
        self.fc1 = nn.Linear(num_features, 1008)
        self.dout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(1008, 504)
        self.dout2 = nn.Dropout(0.2)
        self.fc3 = nn.Linear(504, 202)
        self.dout3 = nn.Dropout(0.2)
        self.fc4 = nn.Linear(202, 100)
        self.fc5 = nn.Linear(100, 30)
        self.fc6 = nn.Linear(30, 1)
        
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

def reset_weights(m):
    if isinstance(m, nn.Linear):
        m.reset_parameters()

# def train(fold, model, device, train_loader, optimizer, epoch):
#     model.train()
#     for batch_idx, (data, target) in enumerate(train_loader):
#         data, target = data.to(device), target.to(device)
#         target = target.reshape(batch_size,1)
#         target = target.type(torch.float64)
#         optimizer.zero_grad()
#         output = model(data)
#         output = output.type(torch.float64)
#         loss = criterion(output, target)
#         loss.backward()
#         optimizer.step()
#         if batch_idx % 500 == 0:
#             print('Train Fold/Epoch: {}/{} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
#                 fold,epoch, batch_idx * len(data), len(train_loader.dataset),
#                 100. * batch_idx / len(train_loader), loss.item()))

# def test(fold,model, device, test_loader):
#     model.eval()
#     test_loss = 0
#     correct = 0
#     with torch.no_grad():
#         for data, target in test_loader:
#             data, target = data.to(device), target.to(device)
#             target = target.reshape(batch_size,1)
#             target = target.type(torch.float64)
#             output = model(data)
#             output = output.type(torch.float64)
#             loss = criterion(output, target)
#             test_loss += loss.data.type(torch.float64)  # sum up batch loss
#             pred = output.round().detach()  # get the index of the max log-probability
#             correct += torch.sum(pred == target.data).type(torch.float64)
            
#     test_loss /= len(test_loader.dataset)
    
#     print(fold)
#     print(test_loss)
#     print(correct)
#     print(len(test_loader.dataset))
#     print('\nTest set for fold {}: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
#         fold,test_loss, correct, len(test_loader.dataset),
#         100. * correct / len(test_loader.dataset)))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

model = Net2()
model.apply(reset_weights)
if torch.cuda.is_available():
    model = model.cuda()
    print('check')

criterion = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

batch_size=40
k_folds=5
epochs=50
results = {}
kfold=KFold(n_splits=k_folds,shuffle=True)

# for fold,(train_idx,test_idx) in enumerate(kfold.split(dataset)):
#     print('------------fold no---------{}----------------------'.format(fold))
#     train_subsampler = torch.utils.data.SubsetRandomSampler(train_idx)
#     test_subsampler = torch.utils.data.SubsetRandomSampler(test_idx)

#     trainloader = torch.utils.data.DataLoader(
#                       dataset, 
#                       batch_size=batch_size, sampler=train_subsampler)
#     testloader = torch.utils.data.DataLoader(
#                       dataset,
#                       batch_size=batch_size, sampler=test_subsampler)

#     model.apply(reset_weights)

#     for epoch in range(1, epochs + 1):
#         train(fold, model, device, trainloader, optimizer, epoch)
#         test(fold,model, device, testloader)

for fold, (train_ids, test_ids) in enumerate(kfold.split(dataset)):
    
    # Print
    print(f'FOLD {fold}')
    print('--------------------------------')
    
    # Sample elements randomly from a given list of ids, no replacement.
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = torch.utils.data.DataLoader(
                      dataset, 
                      batch_size=40, sampler=train_subsampler)
    testloader = torch.utils.data.DataLoader(
                      dataset,
                      batch_size=40, sampler=test_subsampler)
    
    # Init the neural network
    network = Net2()
    network.apply(reset_weights)
#     if torch.cuda.is_available():
#         network = network.cuda()
#         print('check')
    
    # Initialize optimizer
    #optimizer = torch.optim.Adam(network.parameters(), lr=1e-4)
    
    # Run the training loop for defined number of epochs
    for epoch in range(0, epochs):

      # Print epoch
        if epoch % 10 == 0:
            print(f'Starting epoch {epoch+1}')

      # Set current loss value
        current_loss = 0.0

      # Iterate over the DataLoader for training data
        for i, data in enumerate(trainloader, 0):
        
        # Get inputs
            inputs, targets = data
        
            targets = targets.reshape(batch_size,1)
            targets = targets.type(torch.float64)
        
        # Zero the gradients
            optimizer.zero_grad()
        
        # Perform forward pass
            outputs = network(inputs)
            outputs = outputs.type(torch.float64)
        
        # Compute loss
            loss = criterion(outputs, targets)
        
        # Perform backward pass
            loss.backward()
        
        # Perform optimization
            optimizer.step()
        
        # Print statistics
            current_loss += loss.item()
#             if i == 207:
#                 print('Loss after mini-batch %5d: %.3f' %
#                   (i + 1, current_loss / 207))
#                 current_loss = 0.0
            
    # Process is complete.
    print('Training process has finished. Saving trained model.')

 # Print about testing
print('Starting testing')
    
    # Saving the model
save_path = f'./model-fold-{fold}.pth'
torch.save(network.state_dict(), save_path)

    # Evaluation for this fold
correct, total = 0, 0
with torch.no_grad():

      # Iterate over the test data and generate predictions
    for i, data in enumerate(testloader, 0):

        # Get inputs
        inputs, targets = data
        targets = targets.reshape(40,1)
        targets = targets.type(torch.float64)
        # Generate outputs
        outputs = network(inputs)
        outputs = outputs.type(torch.float64)
        # Set total and correct
        predicted = outputs.round().detach()
        predicted = predicted.reshape(batch_size,1)
        predicted = predicted.type(torch.float64)
        total += targets.size(0)
        correct += torch.sum(predicted == targets.data).type(torch.float64)

      # Print accuracy
    print('Accuracy for fold %d: %d %%' % (fold, 100.0 * correct / total))
    print('--------------------------------')
    results[fold] = 100.0 * (correct / total)

  # Print fold results
print(f'K-FOLD CROSS VALIDATION RESULTS FOR {k_folds} FOLDS')
print('--------------------------------')
sum = 0.0
for key, value in results.items():
    print(f'Fold {key}: {value} %')
    sum += value
    print(f'Average: {sum/len(results.items())} %')
