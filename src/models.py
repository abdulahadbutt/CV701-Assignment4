import torch
from torch import nn

class ResNet_landmark(nn.Module):
    '''
    ResNet architecture modified for Landmark detection
    '''
    def __init__(self, model_structure:str='resnet18', num_keypoints:int=68, freeze:bool=False):
        super().__init__()

        self.model = torch.hub.load('pytorch/vision:v0.10.0', model_structure, pretrained=True)
        self.model.fc = nn.Linear(in_features=512, out_features=num_keypoints*2, bias=True)
        if freeze:
            self._freeze_initial_layers()


    def forward(self, x):
        x = self.model(x)
        return x.view(-1, 68, 2)
        return x
    

    def _freeze_initial_layers(self):
        print('[FREEZING INITIAL LAYERS]')
        for param in self.model.parameters():
            param.requires_grad = False

        for param in self.model.layer4.parameters():
            param.requires_grad = True
        
        for param in self.model.fc.parameters():
            param.requires_grad = True


        # for l in self.model.named_parameters():
        #     print(l)
        # block_counter = 0
        # for name, child in self.model.named_children():
        #     if block_counter < freeze_until:
        #         for param in child.parameters():
        #             param.requires_grad = False
        #         block_counter += 1
        #     else:
        #         break  # Stop freezing once the desired number of blocks is frozen

        # # Ensure the final fully connected layer remains trainable
        # for param in self.model.fc.parameters():
        #     param.requires_grad = True



if __name__ == '__main__':
    model = ResNet_landmark()
    print(model.model.fc)