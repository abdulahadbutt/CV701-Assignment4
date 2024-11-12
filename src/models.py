import torch
from torch import nn

class ResNet_landmark(nn.Module):
    '''
    ResNet architecture modified for Landmark detection
    '''
    def __init__(self, model_structure:str='resnet18', num_keypoints:int=68):
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_structure, pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=num_keypoints*2, bias=True)


    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 68, 2)
        return x
    

if __name__ == '__main__':
    model = ResNet_landmark()
    print(model.model.fc)