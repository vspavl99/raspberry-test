import torch
import os
import time
import sys
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import ToTensor
from torchvision import transforms
from models.detection.faced_model import FacedModelLite
from models.classification.mini_xception import MiniXception
from utils.utils import ImageToTensor


# POST TRAINING QUANTIZATION
def main(task, model_name, img_size, model_params):
    batch_size = 1
    num_calibration_batches = 10
    if task == 'detection':
        transform = transforms.Compose([transforms.Resize(img_size), ImageToTensor()])
    elif task == 'classification':
        transform = transforms.Compose([transforms.Resize(img_size), transforms.Grayscale(), ToTensor()])

    dataset = ImageFolder(PATH_TO_DATA, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size)

    if task == 'detection':
        model = FacedModelLite(*model_params)
    elif task == 'classification':
        model = MiniXception(['Anger', 'Happy', 'Neutral', 'Surprise'], *model_params)

    load = torch.load(os.path.join('..', PATH_TO_MODEL, model_name), map_location='cpu')
    model.load_state_dict(load['model_state_dict'])
    model.to('cpu')
    model.eval()

    model.fuse_model()
    torch.backends.quantized.engine = 'qnnpack'
    model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
    torch.quantization.prepare(model, inplace=True)

    with torch.no_grad():
        for i, (image, _) in enumerate(dataloader, 1):
            image = torch.tensor(image, dtype=torch.float)
            print(f'Image #{i}')
            start = time.time()
            model(image)
            print('Time passed (s):', time.time() - start)
            print('===========')
            if i >= num_calibration_batches:
                break

    torch.quantization.convert(model, inplace=True)
    torch.jit.save(torch.jit.trace(model, torch.ones_like(image)), os.path.join(PATH_TO_MODEL, model_name[:-3] + 'quantized.pt'))


if __name__ == '__main__':
    TASK = sys.argv[1]
    MODEL_NAME = sys.argv[2]
    IMG_SIZE = int(sys.argv[3]), int(sys.argv[3])
    MODEL_PARAMS = list(map(int, sys.argv[4].split()))

    PATH_TO_DATA = os.path.join('data', TASK, 'callibration_images')
    PATH_TO_MODEL = os.path.join('log', TASK)

    main(task=TASK, model_name=MODEL_NAME, img_size=IMG_SIZE, model_params=MODEL_PARAMS)
