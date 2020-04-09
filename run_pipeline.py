import cv2
import os
import torch
import time
import numpy as np
from torchvision.transforms import ToTensor
from utils.utils import ImageToTensor, xywh2xyxy, get_most_confident_bbox, transform_bbox_coords



PATH_TO_DETECTION_MODEL = os.path.join('log/detection/faced_model_lite')
PATH_TO_CLASSIFICATION_MODEL = os.path.join('log/classification/mini_xception')
DETECTION_SHAPE = (288, 288)
DETECTION_THRESHOLD = 0.4
GRID_SIZE = 9
CLASSIFICATION_SHAPE = (64, 64)
EMOTION_MAP = ['Anger', 'Happy', 'Neutral', 'Surprise']

# Initialising models weights
torch.backends.quantized.engine = 'qnnpack'
detection_model = torch.jit.load(os.path.join(PATH_TO_DETECTION_MODEL, 'model_quantized.pt'))
detection_model.to('cpu').eval()

classification_model = torch.jit.load(os.path.join(PATH_TO_CLASSIFICATION_MODEL, 'model_quantized.pt'))
classification_model.to('cpu').eval()

cap = cv2.VideoCapture(0)

# Start video capturing
while cap.isOpened():
    ret, image = cap.read()  # original image
    orig_shape = image.shape[:2]  # (H, W)
    start = time.time()

    with torch.no_grad():
        detection_image = cv2.resize(image, DETECTION_SHAPE)
        detection_image = ImageToTensor()(detection_image)
        detection_image = detection_image.unsqueeze(0)
        output = detection_model(detection_image)  # Prediction
        x, y, z = get_most_confident_bbox(output, 2)
        pred_xywh = transform_bbox_coords(output, x, y, z, DETECTION_SHAPE, GRID_SIZE)
        pred_xyxy = xywh2xyxy(pred_xywh)

        if output[0, z + 4, x, y].item() > DETECTION_THRESHOLD:  # prediction confidence threshold

            bbox_l_y = int((pred_xyxy[1]) * (orig_shape[0] / DETECTION_SHAPE[1]))  # Transform bbox coords
            bbox_r_y = int((pred_xyxy[3]) * (orig_shape[0] / DETECTION_SHAPE[1]))  # correspondingly to DETECTION_SHAPE -> orig_shape

            bbox_l_x = int((pred_xyxy[0]) * (orig_shape[1] / DETECTION_SHAPE[0]))
            bbox_r_x = int((pred_xyxy[2]) * (orig_shape[1] / DETECTION_SHAPE[0]))

            bbox_x_c = (bbox_l_x + bbox_r_x) // 2
            bbox_h = bbox_r_y - bbox_l_y

            bbox_l_x = bbox_x_c - bbox_h // 2  # Make bbox square with sides equal to bbox_h
            bbox_r_x = bbox_x_c + bbox_h // 2

            bbox_l_y = np.clip(bbox_l_y, 0, orig_shape[0])  # clip coordinates which limit image borders
            bbox_r_y = np.clip(bbox_r_y, 0, orig_shape[0])
            bbox_l_x = np.clip(bbox_l_x, 0, orig_shape[1])
            bbox_r_x = np.clip(bbox_r_x, 0, orig_shape[1])

            # Converting image to format and shape required by recognition model
            cl_image = image[bbox_l_y:bbox_r_y, bbox_l_x:bbox_r_x, :]

            cl_image = cv2.resize(cl_image, CLASSIFICATION_SHAPE, CLASSIFICATION_SHAPE)
            cl_image = cv2.cvtColor(cl_image, cv2.COLOR_BGR2GRAY)
            cl_image = ToTensor()(cl_image).unsqueeze(0)

            # Paint bbox and emotion prediction
            pred_emo = EMOTION_MAP[classification_model(cl_image).argmax(dim=1).item()]
            image = cv2.rectangle(image, (bbox_l_x, bbox_l_y), (bbox_r_x, bbox_r_y), color=(0, 255, 0), thickness=2)
            image = cv2.putText(image,
                                pred_emo,
                                (bbox_l_x, bbox_l_y + 10),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                color=(0, 0, 255),
                                fontScale=0.5,
                                thickness=2)

            fps = 1. / (time.time() - start)  # Count fps
            image = cv2.putText(image,'FPS: ' + str(fps), (20, 20), cv2.FONT_HERSHEY_SIMPLEX, color=(255, 255, 0), fontScale=0.5, thickness=2)

            cv2.imshow('image', image)
        else:
            cv2.imshow('image', image)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
