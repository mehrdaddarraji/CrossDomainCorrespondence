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

# Neuron class, takes in row and col coordinates
class Neuron:
    def __init__(self, row, col):
        self.r = row
        self.c = col
    def __repr__(self):
        return "(" + str(self.r) + ", " + str(self.c) + ")"

# returns the index of the max arg of tensor
def feat_arg_max(feat):
    f = feat.clone().detach().numpy()
    idx = np.unravel_index(f.argmax(), f.shape)
    return idx

# function to do L2 normalization of a tensor
def L2_norm(A_tensor):
    A = A_tensor.clone().detach().numpy()
    pow_sum = np.power(A, 2).sum()
    A_sqrt = np.power(pow_sum, 0.5) 
    return torch.from_numpy(A / A_sqrt)

# returns nearest neighbor of neuron p ∈ P in the set Q under a similarity metric
# formula 1 from the paper
def nearest_neighbor(commapp_PQ, commapp_QP, P_region, Q_region):
    # region points to calculate 
#     top_left_p = P_region[0]
#     bottom_right_p = points_list[1]
#     top_left_q = Q_region[0]
#     bottom_right_q = Q_region[1]
    
    # TODO: change for loops so they only iterate through the region rather than whole map
    
    # common appearance
    commapp_PQ_norm = L2_norm(commapp_PQ)
    commapp_QP_norm = L2_norm(commapp_QP)
    
    # list of potential buddies
    buddies = []
    
    for p_i in range(commapp_PQ.shape[2]):
        for p_j in range(commapp_PQ.shape[3]):
            
            # calculating similarity metrix list
            # formula 3
            sim_met = torch.zeros((commapp_QP.shape[0], commapp_QP.shape[1], 
                                   commapp_QP.shape[2], commapp_QP.shape[3]))
            for q_x in range(commapp_QP.shape[2]):
                for q_y in range(commapp_QP.shape[3]):
                    sim_met[:, :, q_x, q_y] = commapp_PQ[:, :, p_x, p_y] * commapp_QP[:, :, q_x, q_y]
                    print(sim_met)
                    sim_met[:, :, q_x, q_y] /= commapp_QP_norm[:, :, p_x, p_x] * commapp_QP_norm[:, :, q_x, q_x]
                    print(sim_met)
            # saving the neuron with best similarity metric value as potential buddies
            # formula 2
            idx = feat_arg_max(sim_met)
            print(idx)
            q = Neuron(idx[2], idx[3])
            p = Neuron(p_x, p_y)
            buddies.append((p, q))
            
    return buddies

# Image preprocessing  
def img_preprocess(img):
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
    
    return img_tens

# vgg model
# takes in image a, image b, normalized tensor of image a, normalized tensor of img b
# returns 5 layered feature map of img a and img b
def vgg19_model(img_a, img_b, img_a_tens, img_b_tens):
    model = vgg19(pretrained=True).eval()
    pyramid_layers = []
    
    def extract_feature(module, input, output):
        pyramid_layers.append(output)

    relu_idx = [3, 8, 17, 26, 35]
    for j in relu_idx:
        model.features[j].register_forward_hook(extract_feature)
    
    model(img_a_tens)
    model(img_b_tens)
    
    return pyramid_layers[:5], pyramid_layers[5:]

def main():
    img_a = Image.open("../input/dog1.jpg")
    img_b = Image.open("../input/dog2.jpg")
    img_a_tens = img_preprocess(img_a)
    img_b_tens = img_preprocess(img_b)

    feat_a, feat_b = vgg19_model(img_a, img_b, img_a_tens, img_b_tens)

if __name__ == "__main__":
    main()
