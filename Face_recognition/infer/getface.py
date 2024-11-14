import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision.transforms import transforms
import cv2
from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from supervision import Detections
from models.mtcnn import MTCNN
import torch

device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
mtcnn_inceptionresnetV1 = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)
mtcnn_inceptionresnetV2 = MTCNN(
    image_size=160, margin=0, min_face_size=20,
    thresholds=[0.6, 0.7, 0.7], factor=0.709, post_process=True,
    device=device
)

mtcnn_resnet = MTCNN(
    image_size=224, margin=0, min_face_size=20,
    device=device
)

model_path = hf_hub_download(repo_id="arnabdhar/YOLOv8-Face-Detection", filename="model.pt")
yolo = YOLO(model_path)



if __name__ == '__main__':
    
    image_path = 'testdata/sontung/002.jpg'
    image= Image.open(image_path).convert('RGB')
    cropped_images = mtcnn_inceptionresnetV1(image)

    print(cropped_images.shape)
    plt.imshow(cropped_images.permute(1,2,0).numpy())
    plt.show()


