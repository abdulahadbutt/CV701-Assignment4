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
# TODO: Need to pass through once
# TODO: Need to change accuracy measure to something like MSE or PCK (Percentage Correct Keypoints)


def compute_nme(predicted_landmarks, ground_truth_landmarks, normalization_factor):
    # TODO: Add this metric to track
    distances = torch.norm(predicted_landmarks - ground_truth_landmarks, dim=2)  # (batch_size, num_landmarks)

    # Normalize distances
    nme_per_sample = distances.sum(dim=1) / (predicted_landmarks.size(1) * normalization_factor)  # (batch_size,)
    return nme_per_sample.mean().item()




def compute_pck(predicted_landmarks, groundtruth_landmarks, normalization_factor, threshold=0.4):
    distances = torch.norm(predicted_landmarks - groundtruth_landmarks, dim=2)
    normalized_distances = distances / normalization_factor.unsqueeze(1)
    correct_keypoints = (normalized_distances < threshold).float()
    return correct_keypoints.mean().item()


def compute_interocular_distance(landmarks):
    # Indices for left and right eye landmarks (adjust based on dataset)
    left_eye_indices = [36, 37, 38, 39, 40, 41]  # Example for 68-point dataset
    right_eye_indices = [42, 43, 44, 45, 46, 47]

    # Compute the mean positions for left and right eyes
    left_eye_center = landmarks[:, left_eye_indices, :].mean(dim=1)  # (batch_size, 2)
    right_eye_center = landmarks[:, right_eye_indices, :].mean(dim=1)  # (batch_size, 2)

    # Compute Euclidean distance
    interocular_distances = torch.norm(left_eye_center - right_eye_center, dim=1)  # (batch_size,)
    # print(f'interocular distances: {interocular_distances}')
    return interocular_distances




def train_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim,
    data_loader: torch.utils.data.DataLoader,
    epoch_index: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:str='cpu'  
):
    model.train()
    with tqdm(data_loader, unit='batch') as data:
        batch_loss_list = []
        batch_pck_list = []
        for batch in data:
            data.set_description(f"Epoch {epoch_index}")

            # ? Feeding to CNN
            inputs = batch['image'].to(device)
            labels = batch['keypoints'].to(device)
    
            # inputs, labels = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()

            # * outputs.shape = [16, 68, 2] = [batch_size, # keypoints // 2, 2]
            outputs = model(inputs)

            
            # ? Getting Loss
            # print(outputs.shape, outputs.dtype)
            # print(labels.shape, labels.dtype)
            train_pck = compute_pck(outputs, labels, normalization_factor=compute_interocular_distance(labels))
            batch_pck_list.append(train_pck)
            # print(train_pck)
            # exit()
            batch_loss = criterion(outputs, labels)
            batch_loss_list.append(batch_loss.item())
            batch_loss.backward()

            # ? Gradient Descent
            optimizer.step()


            data.set_postfix(
                batch_loss=batch_loss.item(),
                batch_pck=train_pck
            )

    
    return {
        'epoch_idx': epoch_index,
        'batch_losses': batch_loss_list,
        'epoch_loss': np.mean(batch_loss_list),
        'epoch_average_pck': np.mean(batch_pck_list)
    }


def test_one_epoch(
    model: torch.nn.Module,
    optimizer: torch.optim,
    data_loader: torch.utils.data.DataLoader,
    epoch_index: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:str='cpu' 
):
    model.eval()
    num_correct = 0
    num_samples = 0
    with torch.no_grad():
        with tqdm(data_loader, unit='batch') as data:
            batch_loss_list = []
            batch_pck_list = []
            for batch in data:
                data.set_description(f"Testing after epoch{epoch_index}")

                # ? Feeding to CNN
                inputs = batch['image'].to(device)
                labels = batch['keypoints'].to(device)
                outputs = model(inputs)

                test_pck = compute_pck(outputs, labels, normalization_factor=compute_interocular_distance(labels))
                batch_pck_list.append(test_pck)
                
                # ? Getting Loss
                batch_loss = criterion(outputs, labels)
                batch_loss_list.append(batch_loss.item())

                data.set_postfix(
                    batch_loss=batch_loss.item(),
                    batch_pck=test_pck
                )

                # TODO: Change logic to account for PCK (Percentage Correct Keypoints)
                # _, predictions = outputs.max(1)
                # num_correct += (predictions == labels).sum()
                # num_samples += predictions.size(0)


    
    return {
        'epoch_idx': epoch_index,
        'batch_losses': batch_loss_list,
        'epoch_loss': np.mean(batch_loss_list),
        'epoch_average_pck': np.mean(batch_pck_list)
        # 'accuracy': (num_correct / num_samples).item()
    }



def train(
    model: torch.nn.Module,
    optimizer: torch.optim,
    train_dataloader: torch.utils.data.DataLoader,
    epochs: int,
    criterion: torch.nn.CrossEntropyLoss,
    device:torch.device,
    live: Live,
    test_dataloader: torch.utils.data.DataLoader,
    scheduler: torch.optim.lr_scheduler.LRScheduler
):

    
    best_acc = 0
    best_test_loss = np.inf
    train_statistics_list = []
    for epoch in range(epochs):
        # * Training Code
        train_epoch_statistics = train_one_epoch(
            model, optimizer, train_dataloader, epoch, criterion, device
        )
        train_statistics_list.append(train_epoch_statistics)
        live.log_metric('train/loss', train_epoch_statistics['epoch_loss'], plot=True)
        # live.log_metric('train/norm_loss', train_epoch_statistics['epoch_norm_loss'], plot=True)
        if scheduler:
            scheduler.step()
        
        # * Testing Code
        test_epoch_statistics = test_one_epoch(
            model, optimizer, test_dataloader, epoch, criterion, device
        )
        # TODO: Change code to account for PCK instead of accuracy
        # latest_test_acc = test_epoch_statistics['accuracy']
        # if latest_test_acc > best_acc:
        #     print(f'UPDATING BEST ACC [{best_acc}] -> [{latest_test_acc}]')
        #     best_acc = latest_test_acc
        #     save_checkpoint(model, epoch, optimizer, best_acc, 'models/best_model.pth')

        latest_test_loss = test_epoch_statistics['epoch_loss']
        if latest_test_loss < best_test_loss:
            print(f'UPDATING BEST LOSS [{best_test_loss}] -> [{latest_test_loss}]')
            best_test_loss = latest_test_loss
            save_checkpoint(model, epoch, optimizer, best_test_loss, 'models/best_model.pth')
            


        # print(latest_test_acc, type(latest_test_acc))
        live.log_metric('test/loss', test_epoch_statistics['epoch_loss'], plot=True)
        # live.log_metric('test/norm_loss', train_epoch_statistics['epoch_norm_loss'], plot=True)
        live.log_metric('test/pck', test_epoch_statistics['epoch_average_pck'], plot=True)
        # live.log_metric('test/accuracy', test_epoch_statistics['accuracy'], plot=True)



        live.next_step()

    return train_statistics_list


def save_checkpoint(
        model: torch.nn.Module, 
        epoch: int, 
        optimizer: torch.optim, 
        f1_score: int, 
        path: str):
    
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer,
            "f1_score": f1_score,
        },
        path,
    )



params = yaml.safe_load(open('params.yaml'))
IMG_SIZE = params['IMG_SIZE']
IMG_SIZE = int(IMG_SIZE)

ROOT_DIR = params['ROOT_DIR']
BATCH_SIZE = params['BATCH_SIZE']
LEARNING_RATE = params['LEARNING_RATE']
EPOCHS = params['EPOCHS']
OPTIMIZER = params['OPTIMIZER']
SCHEDULER = params['SCHEDULER']
MODEL_STRUCTURE = params['MODEL_STRUCTURE']

CRITERION = params['CRITERION']

# MAX_PARAMS = params['MAX_PARAMS']
# MAX_EPOCHS = params['MAX_EPOCHS']
# assert EPOCHS <= MAX_EPOCHS, "Too many epochs listed"

torch.manual_seed(1)

live = Live('metrics', dvcyaml=False, save_dvc_exp=True)
# live = None

os.makedirs('models', exist_ok=True)
os.makedirs('metrics', exist_ok=True)


image_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((IMG_SIZE,IMG_SIZE)),
    transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010],
        )
])

train_dataset = FacialKeypointsDataset(
    f'{ROOT_DIR}/training_frames_keypoints.csv',
    f'{ROOT_DIR}/training',
    image_transforms
)


test_dataset = FacialKeypointsDataset(
    f'{ROOT_DIR}/test_frames_keypoints.csv',
    f'{ROOT_DIR}/test',
    image_transforms
)

train_dataloader = torch.utils.data.DataLoader(
    train_dataset, batch_size=BATCH_SIZE, shuffle=True
)

test_dataloader = torch.utils.data.DataLoader(
    test_dataset, batch_size=BATCH_SIZE, shuffle=False
)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if 'resnet' in MODEL_STRUCTURE:
    model = ResNet_landmark(
        MODEL_STRUCTURE
    )
else:
    print('Invalid model structure listed')
    exit()

model.to(device)


if OPTIMIZER == 'sgd':
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER == 'adam':
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER == 'adamw':
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)
elif OPTIMIZER == 'rmsprop':
    optimizer = torch.optim.RMSprop(model.parameters(), lr=LEARNING_RATE)
else:
    print('Invalid optimizer listed')
    exit()

if SCHEDULER == 'constant':
    scheduler = torch.optim.lr_scheduler.ConstantLR(optimizer)
elif SCHEDULER == 'exponential':
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
elif SCHEDULER == 'linear':
    scheduler = torch.optim.lr_scheduler.LinearLR(optimizer)
elif SCHEDULER == 'step':
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
else:
    scheduler = None 

if CRITERION == 'cross-entropy':
    criterion = torch.nn.CrossEntropyLoss()
elif CRITERION == 'mse':
    criterion = torch.nn.MSELoss()
    

loss_statistics = train(
    model, optimizer, train_dataloader, EPOCHS, criterion, device, live, test_dataloader, scheduler
)


save_checkpoint(model, EPOCHS, optimizer, '00', 'models/last.pth')