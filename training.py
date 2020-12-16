import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from skimage import io, transform
from torchvision import datasets, transforms, models
import cv2
import pandas as pd
import numpy as np
import torch.optim as optim
import time

class DrivingDataset(Dataset):
    ''' Dataloader for the dataset '''
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
        commands = np.array([commands]).astype('double')
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

'''Training loop'''
def train(epochs=1, dataset=None, model_save_path=None):
    if dataset is None:
        print('Missing dataset argument')
        return 0   
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # Creates the resnet
    model = models.resnet50(pretrained=True)
    # Steering, Throttle, Brake, reverse
    model.fc = nn.Linear(2048, 3)
    model.to(device)
    # Defines optimizer and loss criteria
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_losses = []


    # Iterates n epochs
    for epoch in range(epochs):
        start = time.time()
        # Iterates through the dataset
        for batch_idx, data in enumerate(dataset):
            # Gets the image data and the 3 first commands
            image, labels = data['image'], data['commands'][ :, :, :3]
            # From [batch, 1, 4] to [batch, 4]
            labels = labels.reshape(labels.shape[0], labels.shape[2])

            optimizer.zero_grad()
            
            # Forward pass
            outputs = model(image.float().to(device))


            # Calculates loss and backpropagate
            loss = criterion(outputs, labels.float().to(device))
            loss.backward()
            optimizer.step()
            
            if batch_idx % 100 == 0:
                loss_data = loss.item()
                train_losses.append(loss_data)
                print(
                    'Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.
                    format(epoch, batch_idx * len(data), len(dataset),
                        100. * batch_idx / len(dataset), loss_data))
        print(f'Epoch took {(time.time() - start )/60}  minutes')
        if epoch % 10 == 0:
            # Every 10th epochs
            print('Saving model')
            # Save the model
            torch.save(model.state_dict, model_save_path + f'driving_{epoch}.weights')
        

def main():
    # Path where the weights are saved
    model_save_path = 'C:\\Users\\cjrs2\\OneDrive\\Escritorio\\Ml\\ImitationLearningCarla\\'            
    # Path to the csv with the annotations
    csv = 'C:\\Users\\cjrs2\\OneDrive\\Escritorio\\Ml\\ImitationLearningCarla\\data\\data.csv'

    # Dataset set up
    driving_dataset = DrivingDataset(csv_file=csv, transform=transforms.Compose( [Normalize() ,ToTensor()]))
    batch_size = 4
    dataloader = DataLoader(driving_dataset, batch_size=batch_size, shuffle=True, num_workers=0)

    # Lets train :)
    train(epochs=11, dataset=dataloader,model_save_path=model_save_path)

if __name__ == "__main__":
    main()