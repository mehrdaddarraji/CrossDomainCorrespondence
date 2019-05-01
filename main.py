import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V
from torchvision.models import vgg19
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import numpy as np
from PIL import Image

#torch.cuda.set_device(0)

vgg = vgg19(True).eval()

preprocess = transforms.Compose([transforms.Resize(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                              std=[0.229, 0.224, 0.225])])

im_a = Image.open("./imgs/dogs_to_morph/dog1.PNG")
# from https://gist.github.com/jkarimi91/d393688c4d4cdb9251e3f939f138876e
im_a_trans = Variable(preprocess(im_a).unsqueeze(0))
out = vgg(im_a_trans)
print('Prediction for dog1.png is: %s'%(np.argmax(out.cpu().detach().numpy())))

print(vgg)
# returns the feature tensors of each image, as they are forwarded through the trained model
# assumes that the images are already in Tensor form
def get_features(imgA, imgB, model):
    # load imgA into the model
    F_A = model.forward(imgA).data
    # load imgB into the model
    F_B = model.forward(imgB).data
    return [F_A, F_B]

# R5 = (F_A, F_B) --> the entire domain of F_A and F_B
# C_B5 and C_A5 = F_B and F_A (no style transfer needed)
# Then, go through other layers and...
# TODO: get P and Q from the feature tensors
# TODO: get p's and q's from each subset of P and Q