import torch
import numpy as np
import torchvision.models as models
import torch.nn as nn

class vgg19_model(nn.Module):
    def __init__(self):
        super(vgg19_model, self).__init__()
        model = models.vgg19(pretrained=True)
        self.layers = self.five_layers(model)

    def five_layers(self, model):
        five_layers = []

        vgg_features = next(model.children())

        model_layer = 0
        layer = []
        reset_layer = [2, 7, 12, 21]
        for m in vgg_features.children():
            if model_layer in reset_layer:
                five_layers += [nn.Sequential(*layer)]
                layer = []

            if model_layer >= 0 and model_layer < 2:
                layer += [m]
            elif model_layer >= 2 and model_layer < 7:
                layer += [m]
            elif model_layer >= 7 and model_layer < 12:
                layer += [m]
            elif model_layer >= 12 and model_layer < 21:
                layer += [m]
            elif model_layer >= 21 and model_layer < 29:
                layer += [m]
        five_layers += [nn.Sequential(*layer)]

        return five_layers





class inceptionv3_model(nn.Module):
    def __init__(self, model):
        super(inceptionv3_model, self).__init__()


class resnet50_model(nn.Module):
    def __init__(self, model):
        super(resnet50_model, self).__init__()
