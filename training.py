import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import datasets, transforms, models
import cv2
import pandas as pd
import numpy as np
import torch.optim as optim


class DrivingDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.data_frame = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        img_name = self.data_frame.iloc[idx, 0]
        image = io.imread(img_name)
        # Reads throttle, brake, steering, reverse
        commands = self.data_frame.iloc[idx, 1:]
        commands = np.array([commands]).astype('float')
        sample = {'image': image, 'commands': commands}
        if self.transform:
            sample = self.transform(sample)

        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, commands = sample['image'], sample['commands']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'commands': torch.from_numpy(commands)}

class Normalize(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, commands = sample['image'], sample['commands']

        image = image/255.0
        return {'image': image,
                'commands': commands}


def train(epochs=1, dataset=None):
    if dataset is None:
        print('Missing dataset argument')
        return 0
    model = models.resnet50(pretrained=True)
    # Steering, Throttle, Brake 
    model.fc = nn.Linear(2048, 3)
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)


    for epoch in range(epochs):
        running_loss= 0.0

        for i, data in enumerate(dataset):
            image, labels = data['image'], data['commands']


            optimizer.zero_grad()

            outputs = model(image)

            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            running_loss += loss.item()
            if i % 10 == 0:    # print every 10 mini-batches
                print('[%d, %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 10))
                running_loss = 0.0
            
    
    images = torch.randn(1, 3, 224, 224)
    preds= model(images)
    print(preds)

    #loss = criterion(pr   eds, labels)

csv = 'C:\\Users\\cjrs2\\OneDrive\\Escritorio\\Ml\\ImitationLearningCarla\\data\\data.csv'
batch_size = 32
driving_dataset = DrivingDataset(csv_file=csv, transform=transforms.Compose( [Normalize() ,ToTensor()]))

train(epochs=1, dataset=driving_dataset)

