

import gradio as gr
import numpy as np
import cv2
import yaml 
import torch 
from models import ResNet_landmark
from torchvision import transforms


def decode_predictions(predicted_keypoints, img_shape):
    height, width, _ = img_shape
    predicted_keypoints = predicted_keypoints.detach().numpy()
    predicted_keypoints = predicted_keypoints.squeeze()
    predicted_keypoints[:, 0] = predicted_keypoints[:, 0] * height
    predicted_keypoints[:, 1] = predicted_keypoints[:, 1] * width

    return predicted_keypoints
    pass


def draw_keypoints(image, keypoints):
    image = np.copy(image)
    # image.setflags(write=1)
    for item in keypoints:
        print(item)
        y, x = item 
        x, y = int(x), int(y)
        output = cv2.circle(image, (x,y), radius=10, color=(0, 0, 255), thickness=-1)
    return output
    # output_image = cv2.drawKeypoints(img, keypoints, 0, (0, 0, 255), 
    #                              flags=cv2.DRAW_MATCHES_FLAGS_NOT_DRAW_SINGLE_POINTS) 
    return output_image

def load_checkpoint(path):
    model = ResNet_landmark()
    weights = torch.load(path, map_location='cpu')['model']
    model.load_state_dict(weights)
    return model 


def process_video(video_frame):
    # return draw_keypoints(video_frame, [[200, 200], [500, 500]])
    # print(type(video_frame))
    # frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
    frame = video_frame
    resized_frame = cv2.resize(frame, (224, 224))
    frame_tensor = t(resized_frame)
    frame_tensor = torch.unsqueeze(frame_tensor, 0)
    predictions = model(frame_tensor)
    predictions = decode_predictions(predictions, frame.shape)
    output = draw_keypoints(frame, predictions)
    return output

# def transform_cv2(frame):
    # return draw_keypoints(frame, [[200, 200], [500, 500]])
    
# #     # if transform == "cartoon":
# #     #     # prepare color
# #     #     img_color = cv2.pyrDown(cv2.pyrDown(frame))
# #     #     for _ in range(6):
# #     #         img_color = cv2.bilateralFilter(img_color, 9, 9, 7)
# #     #     img_color = cv2.pyrUp(cv2.pyrUp(img_color))

# #     #     # prepare edges
# #     #     img_edges = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
# #     #     img_edges = cv2.adaptiveThreshold(
# #     #         cv2.medianBlur(img_edges, 7),
# #     #         255,
# #     #         cv2.ADAPTIVE_THRESH_MEAN_C,
# #     #         cv2.THRESH_BINARY,
# #     #         9,
# #     #         2,
# #     #     )
# #     #     img_edges = cv2.cvtColor(img_edges, cv2.COLOR_GRAY2RGB)
# #     #     # combine color and edges
# #     #     img = cv2.bitwise_and(img_color, img_edges)
# #     #     return img
# #     # elif transform == "edges":
# #     #     # perform edge detection
# #     #     img = cv2.cvtColor(cv2.Canny(frame, 100, 200), cv2.COLOR_GRAY2BGR)
# #     #     return img
# #     # else:
# #     #     return np.flipud(frame)

def get_frame(frame):
    return frame 

if __name__ == '__main__':
    params = yaml.safe_load(open('params.yaml'))
    ROOT_DIR = params['ROOT_DIR']
    IMG_SIZE = params['IMG_SIZE']
    IMG_SIZE = int(IMG_SIZE)
    t = transforms.ToTensor()
    model = load_checkpoint('models/best_model.pth')
    model.eval()
    with gr.Blocks() as demo:
        with gr.Row():
            with gr.Column():
                # transform = gr.Dropdown(choices=["cartoon", "edges", "flip"],
                #                         value="flip", label="Transformation")
                input_img = gr.Image(sources=["webcam"], type="numpy")
            with gr.Column():
                output_img = gr.Image(streaming=True)
            dep = input_img.stream(process_video, [input_img], [output_img],
            # dep = input_img.stream(get_frame, [input_img], [output_img],
                                    time_limit=30, stream_every=0.1, concurrency_limit=30)

    demo.launch()

# # # import gradio as gr
# # # from models import ResNet_landmark
# # # import torch 
# # # import yaml 
# # # from torchvision import transforms
# # # import cv2 
# # # import warnings

# # # warnings.filterwarnings("ignore")








# # # def process_video(video_frame):
# # #     print(type(video_frame))
# # #     frame = cv2.cvtColor(video_frame, cv2.COLOR_RGB2BGR)
# # #     resized_frame = cv2.resize(frame, (224, 224))
# # #     predictions = model.predict(resized_frame)
# # #     predictions = decode_predictions(predictions)
# # #     output = draw_keypoints(frame, predictions)
# # #     return output







# # # #     image_transforms = transforms.Compose([
# # # #         transforms.ToTensor(),
# # # #         transforms.Resize((IMG_SIZE,IMG_SIZE)),
# # # # ])
# # #     model = load_checkpoint('models/best_model.pth')
# # #     interface = gr.Interface(
# # #         fn=process_video,
# # #         inputs=gr.Video(),
# # #         outputs="video", 
# # #         live=True
# # #     )

# # #     interface.launch()

# # # # import torch 
# # # # import torchvision
# # # # from dataset import FacialKeypointsDataset
# # # # from models import ResNet_landmark
# # # # import glob 
# # # # import cv2 
# # # # import numpy as np
# # # # import matplotlib.pyplot as plt 
# # # # from dvclive import Live
# # # # import os 
# # # # import yaml 
# # # # from tqdm import tqdm
# # # # from torchvision import transforms
# # # # import os 


# # # # def load_checkpoint(path):
# # # #     model = ResNet_landmark()
# # # #     weights = torch.load(path)['model']
# # # #     model.load_state_dict(weights)
# # # #     return model 


# # # # os.makedirs('data/predictions', exist_ok=True)

# # # # params = yaml.safe_load(open('params.yaml'))
# # # # ROOT_DIR = params['ROOT_DIR']
# # # # IMG_SIZE = params['IMG_SIZE']
# # # # IMG_SIZE = int(IMG_SIZE)

# # # # image_transforms = transforms.Compose([
# # # #     transforms.ToTensor(),
# # # #     transforms.Resize((IMG_SIZE,IMG_SIZE)),
# # # #     transforms.Normalize(
# # # #             mean=[0.4914, 0.4822, 0.4465],
# # # #             std=[0.2023, 0.1994, 0.2010],
# # # #         )
# # # # ])
# # # # test_dataset = FacialKeypointsDataset(
# # # #     f'{ROOT_DIR}/test_frames_keypoints.csv',
# # # #     f'{ROOT_DIR}/test',
# # # #     image_transforms
# # # # )

# # # # test_dataloader = torch.utils.data.DataLoader(
# # # #     test_dataset, batch_size=1, shuffle=False
# # # # )



# # # # def draw_keypoints_and_save(img_path, keypoints):
# # # #     plt.clf()
# # # #     img_name = img_path.split('/')[-1]
# # # #     # print(img_path)
# # # #     img = cv2.imread(img_path)
# # # #     img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
# # # #     plt.imshow(img)
# # # #     x = keypoints[:, 0]
# # # #     y = keypoints[:, 1]

# # # #     plt.scatter(x, y)

# # # #     plt.savefig(f'data/predictions/{img_name}.png')



# # # # model = load_checkpoint('models/best_model.pth')
# # # # model.eval()
# # # # for sample in tqdm(test_dataloader):
# # # #     # keypoints = sample['keypoints']
# # # #     # keypoints[:, 0] = keypoints[:, 0] * height
# # # #     # keypoints[:, 1] = keypoints[:, 1] * width
# # # #     # print(keypoints)

# # # #     predicted_keypoints = model(sample['image'])
# # # #     height = sample['original_height']
# # # #     width = sample['original_width']
# # # #     img_path = sample['img_path'][0]
# # # #     predicted_keypoints = predicted_keypoints.squeeze()
# # # #     # print(predicted_keypoints.shape)
# # # #     # print(predicted_keypoints)
# # # #     predicted_keypoints[:, 0] = predicted_keypoints[:, 0] * height
# # # #     predicted_keypoints[:, 1] = predicted_keypoints[:, 1] * width

# # # #     predicted_keypoints = predicted_keypoints.detach().numpy()
# # # #     # print(predicted_keypoints)
# # # #     draw_keypoints_and_save(img_path, predicted_keypoints)





