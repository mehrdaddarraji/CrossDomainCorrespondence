import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn

class vgg19_model(nn.Module):
    def __init__(self, model):
        super(vgg19_model, self).__init__()
        self.layers = self.five_layers(model)

    def five_layers(self, model):

        return []