import cv2
import torch
import numpy as np
import os
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
#PATH_TO_MODEL = 'log/detection/faced_model_lite'
PATH_TO_MODEL = 'models/'
IMG_SIZE = (288, 288)
GRID_SIZE = 9

torch.backends.quantized.engine = 'qnnpack'
model = torch.jit.load(os.path.join(PATH_TO_MODEL, 'light_model_quantized.pt'))
model.eval()

cap = cv2.VideoCapture(0)

while cap.isOpened():  # Capturing video
    ret, image = cap.read()
    start = time.time()

    # Image preprocessing for format and shape required by model
    image = cv2.resize(image, IMG_SIZE)
    with torch.no_grad():
        image_t = torch.from_numpy(image.transpose((2, 0, 1)))
        image_t = image_t.unsqueeze(0)
        output = model(image_t)  # Prediction
        x, y, z = get_most_confident_bbox(output, 2)
        print(output[0, z + 4, x, y])
        pred = transform_bbox_coords(output, x, y, z, IMG_SIZE, GRID_SIZE)
    pred = xywh2xyxy(pred)

    fps = 1. / (time.time() - start)
    print(fps)
    image = cv2.rectangle(image, (pred[0], pred[1]), (pred[2], pred[3]), color=(0, 255, 0), thickness=2)
    cv2.imshow('lol', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
