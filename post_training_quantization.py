import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
from faced_model import FacedModel
import os
import numpy as np
import time

 
class ImageToTensor:
 	def __call__(self, img):
 		img = np.array(img).transpose((2, 0, 1))
 		return torch.from_numpy(img)

# POST TRAINING QUANTIZATION

PATH_TO_MODEL_FLOAT = 'models'
PATH_TO_DATA = 'callibration_images'

image_size = (320, 320)
batch_size = 1
num_calibration_batches = 10

transforms = transforms.Compose([transforms.Resize(image_size),
	                             ImageToTensor()])

dataset = ImageFolder(PATH_TO_DATA, transform=transforms)
dataloader = DataLoader(dataset, batch_size=batch_size)

model = FacedModel(5, 2)
load = torch.load(os.path.join(PATH_TO_MODEL_FLOAT, 'checkpoint.pt'), map_location='cpu')
model.load_state_dict(load['model_state_dict'])
model.to('cpu')
model.eval()

model.fuse_model()
torch.backends.quantized.engine = 'qnnpack'
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)

for i, (image, _) in enumerate(dataloader):
	if i < num_calibration_batches:
		print(f'Image #{i}')
		start = time.time()
		model(image)
		print('Time passed (s):', time.time() - start)
		print('===========')
	else:
		break


torch.quantization.convert(model, inplace=True)
torch.jit.save(torch.jit.script(model), os.path.join(PATH_TO_MODEL_FLOAT, 'model_quantized.pt'))
