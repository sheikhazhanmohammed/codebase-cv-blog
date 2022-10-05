import numpy as np 
import pandas as pd
import os
from tqdm import tqdm_notebook as tqdm
from sklearn.preprocessing import LabelEncoder
from PIL import Image
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms

#uncomment the line below if using a CUDA device
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#uncomment the line below if using MAC device with MPS support
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print("Training on:",device)

images = []
labels = []

for folder in os.listdir("./data"):
    if folder=="dragonfly":
        for filename in os.listdir(os.path.join("./data", folder)):
            images.append(os.path.join("./data", folder, filename))
            labels.append("dragonfly")
    if folder=="grasshoppers":
        for filename in os.listdir(os.path.join("./data", folder)):
            images.append(os.path.join("./data", folder, filename))
            labels.append("grasshoppers")
    if folder=="prayingMantes":
        for filename in os.listdir(os.path.join("./data", folder)):
            images.append(os.path.join("./data", folder, filename))
            labels.append("prayingMantes")

print("Total images:",len(images))

data = {'ImagePath':images, 'Labels':labels} 
data = pd.DataFrame(data)
data = data.sample(frac=1).reset_index(drop=True)
lb = LabelEncoder()
data['EncodedLabels'] = lb.fit_transform(data['Labels'])

dataTrain = data[0:int(0.9*len(data))]
dataTrain = dataTrain.sample(frac=1).reset_index(drop=True)
dataTest = data[int(0.9*len(data)):]
dataTest = dataTest.sample(frac=1).reset_index(drop=True)
print("Size of training set:",len(dataTrain))
print("Size of testing set:",len(dataTest))

class InsectDataset(Dataset):
    def __init__(self, pathToImages, labels, transform=None):
        self.pathToImages = pathToImages
        self.transform = transform
        self.labels = labels
        
    def __len__(self):
        return len(self.pathToImages)
    
    def __getitem__(self, index):
        image = Image.open(self.pathToImages[index])
        #converts images to rgb
        image = image.convert('RGB')
        #resizing image to 256x256
        image = image.resize((256,256))
        label = torch.tensor(self.labels[index])
        if self.transform is not None:
            image = self.transform(image)
        return image, label

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

datasetTrain = InsectDataset(dataTrain['ImagePath'], dataTrain['EncodedLabels'], transform)
datasetTest = InsectDataset(dataTest['ImagePath'], dataTest['EncodedLabels'], transform)

trainLoader = torch.utils.data.DataLoader(datasetTrain, batch_size=2)
testLoader = torch.utils.data.DataLoader(datasetTest, batch_size=2)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(3, 32, kernel_size=3),
                                    nn.BatchNorm2d(32),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2,2)))
        self.conv2 = nn.Sequential(nn.Conv2d(32, 64, kernel_size=3),
                                    nn.BatchNorm2d(64),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2,2)))
        self.conv3 = nn.Sequential(nn.Conv2d(64, 128, kernel_size=3),
                                    nn.BatchNorm2d(128),
                                    nn.ReLU(),
                                    nn.MaxPool2d((2,2)))
        self.conv4 = nn.Sequential(nn.Conv2d(128, 256, kernel_size=3),
                                    nn.BatchNorm2d(256),
                                    nn.ReLU(),
                                    nn.Flatten())
        self.fc1 = nn.Linear(200704, 512)
        self.fc2 = nn.Linear(512, 256)
        self.classification = nn.Linear(256, 3)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.classification(x)
        x = F.log_softmax(x,dim = 1)
        return x

model = Net()
model.to(device)

def evaluateModel(model):
    totalT=0
    correctT=0
    model.eval()
    with torch.no_grad():
        for dataT, targetT in (testLoader):
            dataT, targetT = dataT.to(device), targetT.to(device)
            outputT = model(dataT)
            _, predictionT = torch.max(outputT, dim=1)
            correctT += torch.sum(predictionT==targetT).item()
            totalT += targetT.size(0)
        valiationAccuracy = 100 * (correctT / totalT)
    return valiationAccuracy

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)

numberOfEpochs = 10
summaryInterval = 100
validationAccuracyMax = 0.0
totalSteps = len(trainLoader)
for epoch in range(1, numberOfEpochs+1):
    print(f'Epoch {epoch}')
    for batch_idx, (data, target) in enumerate(trainLoader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()
        if(batch_idx)%summaryInterval==0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' .format(epoch, numberOfEpochs, batch_idx, totalSteps, loss.item()))
    validationAccuracy = evaluateModel(model)
    print("Completed training for first epoch")
    print("Accuracy on validation set:", validationAccuracy)
    if validationAccuracyMax<=validationAccuracy:
        validationAccuracyMax = validationAccuracy
        torch.save(model.state_dict(), 'classificationModel.pt')
        print('Detected network improvement, saving current model')
    model.train()