import glob
import os
from torch.utils.data import Dataset
import matplotlib.image as mpimg
import pandas as pd
from torchvision import transforms

class FacialKeypointsDataset(Dataset):
    """Face Landmarks dataset."""

    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.key_pts_frame = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.key_pts_frame)

    def __getitem__(self, idx):
        image_name = os.path.join(self.root_dir, self.key_pts_frame.iloc[idx, 0])

        image = mpimg.imread(image_name)

        # if image has an alpha color channel, get rid of it
        if image.shape[2] == 4:
            image = image[:, :, 0:3]

        # print(f'image shape: {image.shape}')
        h, w, _ = image.shape 
        key_pts = self.key_pts_frame.iloc[idx, 1:].to_numpy()
        # ? Changed from astype("float") to astype("float32")
        key_pts = key_pts.astype("float32").reshape(-1, 2)

        key_pts = self.normalize_height_width(w, h, key_pts)
        # print(f'before cast: {key_pts.dtype}')
        if self.transform:
            image = self.transform(image)
        sample = {"image": image, "keypoints": key_pts}


        return sample


    def normalize_height_width(self, width, height, keypoints):
        keypoints[:, 0] = keypoints[:, 0] / height
        keypoints[:, 1] = keypoints[:, 1] / width
        return keypoints 


if __name__ == '__main__':
    ROOT_DIR = 'data'
    IMG_SIZE = 224

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

    for i in train_dataset:
        keypoints = i['keypoints']
        print(keypoints.shape, keypoints.dtype)
        print(keypoints[:5, :])
        break 