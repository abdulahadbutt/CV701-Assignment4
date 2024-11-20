import torch 
import torchvision
from dataset import FacialKeypointsDataset
from models import ResNet_landmark
import glob 
import cv2 
import numpy as np
import matplotlib.pyplot as plt 
from dvclive import Live
import os 
import yaml 
from tqdm import tqdm
from torchvision import transforms
import os 


def load_checkpoint(path):
    model = ResNet_landmark()
    weights = torch.load(path)['model']
    model.load_state_dict(weights)
    return model 


os.makedirs('data/predictions', exist_ok=True)

params = yaml.safe_load(open('params.yaml'))
ROOT_DIR = params['ROOT_DIR']
IMG_SIZE = params['IMG_SIZE']
IMG_SIZE = int(IMG_SIZE)

image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
])
test_dataset = FacialKeypointsDataset(
    f'{ROOT_DIR}/test_frames_keypoints.csv',
    f'{ROOT_DIR}/test',
    image_transforms
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=1, shuffle=False
)



def draw_keypoints_and_save(img_path, keypoints):
    plt.clf()
    img_name = img_path.split('/')[-1]
    # print(img_path)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    x = keypoints[:, 0]
    y = keypoints[:, 1]

    plt.scatter(x, y)

    plt.savefig(f'data/predictions/{img_name}.png')



model = load_checkpoint('models/best_model.pth')
model.eval()
for sample in tqdm(test_dataloader):
    # keypoints = sample['keypoints']
    # keypoints[:, 0] = keypoints[:, 0] * height
    # keypoints[:, 1] = keypoints[:, 1] * width
    # print(keypoints)

    predicted_keypoints = model(sample['image'])
    height = sample['original_height']
    width = sample['original_width']
    img_path = sample['img_path'][0]
    predicted_keypoints = predicted_keypoints.squeeze()
    # print(predicted_keypoints.shape)
    # print(predicted_keypoints)
    predicted_keypoints[:, 0] = predicted_keypoints[:, 0] * height
    predicted_keypoints[:, 1] = predicted_keypoints[:, 1] * width

    predicted_keypoints = predicted_keypoints.detach().numpy()
    # print(predicted_keypoints)
    draw_keypoints_and_save(img_path, predicted_keypoints)





