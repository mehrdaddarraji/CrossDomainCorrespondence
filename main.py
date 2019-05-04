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

torch.cuda.set_device(0)

class vgg19_model(nn.Module):
    def __init__(self):
        super(vgg19_model, self).__init__()
        self.model = vgg19(pretrained=True)
        # self.layers = self.forward(model)

#         self.batchSize = 1
#         self.num_channel = 3
#         self.img_size = 224
#         self.input = torch.Tensor(self.batchSize, self.num_channel, self.img_size, self.img_size)
        # print(self.layers)

    def forward(self, img):
        pyramid_layers = []
        def extract_feature(self, input, output):
            pyramid_layers.append(output)

        relu_idx = [1, 6, 11, 20, 29]
        for i in relu_idx:
            self.model.features[i].register_forward_hook(extract_feature)
        # Image preprocessing
        # VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406]
        # and std=[0.229, 0.224, 0.225].
        # We use the same normalization statistics here.
        transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.485, 0.456, 0.406),
                                 std=(0.229, 0.224, 0.225))])

        img_preproc = transform(img)
        img_preproc = torch.unsqueeze(img_preproc, 0)
        img_tens = V(img_preproc)
        self.model(img_tens)

        return pyramid_layers, img_tens

vgg = vgg19_model()
im_a = Image.open("../input/dog1.jpg")
im_b = Image.open("../input/dog.jpg")

feat_a, im_a_tens = vgg.forward(im_a)
feat_b, im_b_tens = vgg.forward(im_b)

# a, b = spatial domains
# feat_a is the feature of img a
# returns the common appearance C(a, b)
def common_appearance(a, b):
    #print(a.shape)
    mean_m = (a.mean() + b.mean())/2
    std_m = (a.var() + b.var())/2
    #print(feat_a.shape)
    common_app_a_b = (a - a.mean())/a.var()
    #print(common_app_a_b.shape)
    common_app_a_b = std_m * common_app_a_b
    common_app_a_b = common_app_a_b + mean_m
    return common_app_a_b

common_a_b = common_appearance(im_a_tens, im_b_tens)
common_b_a = common_appearance(im_b_tens, im_a_tens)
#print(common_a_b.squeeze().shape)
# we squeeze because we get a tensor of shape [1, 3, 224, 224] from common_appearance, and need [3, 224, 224] to use pil_image
# for ipython notebook:
# F.to_pil_image(common_a_b.squeeze())
# F.to_pil_image(common_b_a.squeeze())