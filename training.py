import torch
import torch.nn as nn
import torchvision

def train():
    model = torchvision.models.resnet50(pretrained=True)
    # Steering, Throttle, Brake 
    model.fc = nn.Linear(2048, 3)

    images = torch.randn(1, 3, 224, 224)
    criterion = nn.MSELoss()
    preds= model(images)
    print(preds)

    #loss = criterion(preds, labels)

train()
