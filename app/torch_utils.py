import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image

from torchvision.models.resnet import ResNet, BasicBlock


# model architecture used for quickdraw
class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2,2,2,2], num_classes=10)


model = ImageClassifier()

PATH = "app/resnet-18-finetuned.pt" # production
# PATH = "resnet-18-finetuned.pt" # dev

model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

# predict
def get_prediction(image_tensor):

    index_to_names = {
        3: 'car'
    }

    outputs = model(image_tensor)
        # max returns (value, index)
    _, label_index = torch.max(outputs.data, 1)

    return index_to_names[label_index.item()]
