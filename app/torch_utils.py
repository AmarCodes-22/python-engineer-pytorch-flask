import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image

from torchvision.models.resnet import ResNet, BasicBlock

# was from python engineer's video
# load model
# class NeuralNet(nn.Module):
#     def __init__(self, input_size, hidden_size, num_classes):
#         super(NeuralNet, self).__init__()
#         self.input_size = input_size
#         self.l1 = nn.Linear(input_size, hidden_size) 
#         self.relu = nn.ReLU()
#         self.l2 = nn.Linear(hidden_size, num_classes)  
    
#     def forward(self, x):
#         out = self.l1(x)
#         out = self.relu(out)
#         out = self.l2(out)
#         # no activation and no softmax at the end
#         return out


# model architecture used for quickdraw
class ImageClassifier(ResNet):
    def __init__(self):
        super(ImageClassifier, self).__init__(BasicBlock, [2,2,2,2], num_classes=10)


# input_size = 784 # 28x28
# hidden_size = 500 
# num_classes = 10
# model = NeuralNet(input_size, hidden_size, num_classes)

model = ImageClassifier()

# PATH = "app/mnist_ffn.pth"

# production
PATH = "app/resnet-18-finetuned.pt"
# dev
# PATH = "resnet-18-finetuned.pt"

# model.load_state_dict(torch.load(PATH))
model.load_state_dict(torch.load(PATH, map_location=torch.device('cpu')))
model.eval()

# image -> tensor
def transform_image(image_bytes):
    # transform = transforms.Compose([transforms.Grayscale(num_output_channels=1),
    #                                 transforms.Resize((28,28)),
    #                                 transforms.ToTensor(),
    #                                 transforms.Normalize((0.1307,),(0.3081,))])
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
    # images = image_tensor.reshape(-1, 28*28)
    outputs = model(image_tensor)
        # max returns (value, index)
    _, label_index = torch.max(outputs.data, 1)
    # print('inside get_prediction')
    # print(label_index)
    # print(index_to_names[label_index.item()])
    return index_to_names[label_index.item()]
    # return _, predicted
    # return outputs
