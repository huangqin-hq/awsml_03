import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.datasets as datasets
import torchvision.transforms as transforms

import argparse
import json
import os

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

#Import dependencies for Debugging andd Profiling
import smdebug as smd
from smdebug.profiler.utils import str2bool
from smdebug.pytorch import get_hook


def test(model, test_loader, criterion, device, hook):
    '''
    Complete this function that can take a model and a 
    testing data loader and will get the test accuray/loss of the model
    '''
    print("Testing Model on Whole Testing Dataset")
    hook.set_mode(smd.modes.EVAL)
    model.eval()
    running_loss=0
    running_corrects=0
    
    for inputs, labels in test_loader:
        inputs=inputs.to(device)
        labels=labels.to(device)
        outputs=model(inputs)
        loss=criterion(outputs, labels)
        _, preds = torch.max(outputs, 1)
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data).item()

    print("Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n".format(running_loss / len(test_loader.dataset), 
                                                                               running_corrects, 
                                                                               len(test_loader.dataset), 
                                                                               100.0 * running_corrects / len(test_loader.dataset)))


def train(model, train_loader, criterion, optimizer, epoch, device, hook):
    '''
    Complete this function that can take a model and
    data loaders for training and will get train the model
    '''
    hook.set_mode(smd.modes.TRAIN)
    model.train()
    for e in range(1, epoch + 1):
        running_loss = 0
        running_corrects = 0
        for inputs, labels in train_loader:
            inputs=inputs.to(device)
            labels=labels.to(device)
            
            optimizer.zero_grad()
            preds = model(inputs)
            loss = criterion(preds, labels)
            running_loss += loss
            loss.backward()
            optimizer.step()
            
            _, preds = torch.max(preds, 1)
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data).item()
        print(f"Training Set: Epoch {e}: Loss {running_loss/len(train_loader.dataset)}, \
            Accuracy {100*(running_corrects/len(train_loader.dataset))}%")
    return model


def net():
    '''
    Complete this function that initializes your model
    Remember to use a pretrained model
    '''
    model = models.resnet50(pretrained=True)

    for param in model.parameters():
        param.requires_grad = False   

    num_features=model.fc.in_features
    model.fc = nn.Sequential(nn.Linear(num_features, 133))
    return model


def create_data_loaders(path, batch_size):
    '''
    This is an optional function that you may or may not need to implement
    depending on whether you need to use data loaders or not
    '''
    
    data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'test': transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    }
    
    print("Initializing Datasets and Dataloaders...")

    # Create training and test datasets
    image_datasets = {x: datasets.ImageFolder(os.path.join(path, x), 
                                              data_transforms[x]) 
                      for x in ['train', 'test']}
    
    # Create training and test dataloaders
    dataloaders_dict = {x: torch.utils.data.DataLoader(image_datasets[x], 
                                                       batch_size=batch_size, 
                                                       shuffle=True, num_workers=2) 
                        for x in ['train', 'test']}
    return dataloaders_dict


def main(args):
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on Device {device}")
        
    '''
    Initialize a model by calling the net function
    '''
    model=net()
    model=model.to(device)

    loaders = create_data_loaders(args.data_dir, args.batch_size)

    '''
    Create the loss and optimizer
    '''
    loss_criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=args.lr)
    
    '''
    Create the hook for debugger and profiler
    '''
    hook = get_hook(create_if_not_exists=True)
    hook.register_hook(model)
    
    '''
    Call the train function to start training your model
    Remember that you will need to set up a way to get training data from S3
    '''
    model = train(model, loaders['train'], loss_criterion, optimizer, args.epochs, device, hook)

    '''
    Test the model to see its accuracy
    '''
    test(model, loaders['test'], loss_criterion, device, hook)

    '''
    Save the trained model
    '''
    path = os.path.join(args.model_dir, "model.pth")
    torch.save(model.cpu().state_dict(), path)


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    '''
    Specify all the hyperparameters you need to use to train your model.
    '''
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        metavar="N",
        help="input batch size for training (default: 64)"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        metavar="N",
        help="number of epochs to train (default: 5)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.01,
        metavar="LR",
        help="learning rate (default: 0.01)"
    )

    # # Container environment
    parser.add_argument("--hosts", type=list, default=json.loads(os.environ["SM_HOSTS"]))
    parser.add_argument("--current-host", type=str, default=os.environ["SM_CURRENT_HOST"])
    parser.add_argument("--model-dir", type=str, default=os.environ["SM_MODEL_DIR"])
    parser.add_argument("--data-dir", type=str, default=os.environ["SM_CHANNEL_TRAINING"])
    parser.add_argument("--num-gpus", type=int, default=os.environ["SM_NUM_GPUS"])
    
    args = parser.parse_args()
        
    main(args)
