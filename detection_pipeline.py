import cv2
import torch
import numpy as np
import os
#from utils.utils import xywh2xyxy, from_yolo_target
#from data.detection.show_targets import show_rectangles
import time


def get_most_confident_bbox(output, num_bbox):
    shape = output.shape
    flatten_idx = torch.argmax(output[:, [i * 5 + 4 for i in range(num_bbox)],  :, :])
    z = 4 + (5 * (flatten_idx // (shape[2] * shape[3]))) 
    xy = flatten_idx % (shape[2] * shape[3])
    x = xy // shape[3]
    y = xy % shape[3]

    return x, y, z - 4


def transform_bbox_coords(output, x, y, z, image_size, grid_size):
    coords = output[:, z:z+4, x, y].squeeze(0)
    cell_size = image_size[0] / grid_size
    coords[0] = cell_size * x + cell_size * coords[0]
    coords[1] = cell_size * y + cell_size * coords[1]
    coords[2] = image_size[0] * coords[2]
    coords[3] = image_size[1] * coords[3]
    return coords


def xywh2xyxy(coords):
    """Args:
           coords: np.array([x_center, y_center, w, h])
    """
    new_coords = np.empty((4,), dtype=np.int)
    new_coords[0] = coords[0] - coords[2] // 2
    new_coords[1] = coords[1] - coords[3] // 2
    new_coords[2] = coords[0] + coords[2] // 2
    new_coords[3] = coords[1] + coords[3] // 2
    return new_coords


# Initialising detection model
PATH_TO_MODEL = 'models'
IMG_SIZE = (320, 320)
GRID_SIZE = 5

torch.backends.quantized.engine = 'qnnpack'
model = torch.jit.load(os.path.join(PATH_TO_MODEL, 'model_quantized.pt'))
model.eval()

cap = cv2.VideoCapture(0)

while cap.isOpened():  # Capturing video
    ret, image = cap.read()
    start = time.time()

    # Image preprocessing for format and shape required by model
    image = cv2.resize(image, IMG_SIZE)
    image_t = torch.from_numpy(image.transpose((2, 0, 1)))
    image_t = image_t.unsqueeze(0)
    output = model(image_t)  # Prediction
    x, y, z = get_most_confident_bbox(output, 2)
    pred = transform_bbox_coords(output, x, y, z, IMG_SIZE, GRID_SIZE)
    pred = xywh2xyxy(pred)
    #listed_output = from_yolo_target(output[:, :10, :, :], image.size(2), grid_size=5, num_bboxes=2)  # Converting from tensor format to list
    #pred_output = listed_output[:, np.argmax(listed_output[:, :, 4]).item(), :]  # Selecting most confident cell

    #show_rectangles(image.numpy().squeeze(0).transpose((1, 2, 0)),
    #                np.expand_dims(xywh2xyxy(pred_output[:, :4]), axis=0), str(pred_output[:, 4]))  # Painting bbox
    fps = 1. / (time.time() - start)
    print(fps)
    image = cv2.rectangle(image, (pred[0], pred[1]), (pred[2], pred[3]), color=(0, 255, 0), thickness=2)
    cv2.imshow('lol', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
'''
tn = torch.zeros((1, 11, 2, 2))
tn[0, 5:10, 1, 0] = torch.tensor([0.1, 0.2, 0.5, 0.6, 1])
x, y, z = get_most_confident_bbox(tn, 2)
print(x, y, z)

print(transform_bbox_coords(tn, x, y, z, (100, 100), 2))
'''

