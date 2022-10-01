import os
import io
import torch
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
import torch
import torch.nn as nn
from PIL import Image
JPEG_CONTENT_TYPE = 'image/jpeg'


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


def model_fn(model_dir):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = net().to(device)

    with open(os.path.join(model_dir, 'model.pth'), 'rb') as f:
        model.load_state_dict(torch.load(f))
    model.eval()
    return model


def input_fn(request, content_type=JPEG_CONTENT_TYPE):
    if content_type == JPEG_CONTENT_TYPE:
        return Image.open(io.BytesIO(request))

    raise Exception('Unsupported input content type: {}'.format(content_type))


def predict_fn(input_data, model):
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    inf_data=data_transform(input_data)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    inf_data=inf_data.to(device)

    with torch.no_grad():
        prediction = model(inf_data.unsqueeze(0))
    return prediction
