import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision import transforms
import os

 
class ImageToTensor:
 	def __call__(self, img):
 		img = np.array(img).transpose((2, 0, 1))
 		return torch.from_numpy(img)

# POST TRAINING QUANTIZATION

PATH_TO_MODEL_FLOAT = ''
PATH_TO_DATA = ''

image_size = (320, 320)
batch_size = 1
num_calibration_batches = 5

transforms = transforms.Compose([transforms.Resize(image_size),
	                             ImageToTensor()])

dataset = ImageFolder(PATH_TO_DATA)
dataloader = DataLoader(dataset, batch_size=batch_size)

model = torch.load(os.path.join(PATH_TO_MODEL_FLOAT, 'model.pt'), map_location='cpu')
load = torch.load(os.path.join(PATH_TO_MODEL_FLOAT, 'checkpoint.pt'), map_location='cpu')
model.load_state_dict(load['model_state_dict'])
model.to('cpu')
model.eval()

model.fuse_model()
model.qconfig = torch.quantization.get_default_qconfig('qnnpack')
torch.quantization.prepare(model, inplace=True)

for i, (image, _) in enumerate(dataloader):
	if i > num_calibration_batches:
		break

	model(image)


torch.quantization.convert(model, inplace=True)
torch.jit.save(torch.jit.script(model), os.path.join(PATH_TO_MODEL_FLOAT, 'model_quantized.pt'))
