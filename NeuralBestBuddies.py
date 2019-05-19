import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable as V
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.transforms.functional as F
import tensorflow as tf
import numpy as np
import math
from PIL import Image
from matplotlib import pyplot as plt
import math

# Neuron class, takes in row and col coordinates
class Neuron:
    def __init__(self, row, col, activation = 0):
        self.r = row
        self.c = col
        self.activation = activation
    def __repr__(self):
        return "(" + str(self.r) + ", " + str(self.c) + ")" + "- Activation: " + str(self.activation)
    def __eq__(self, other):
        return self.r == other.r and self.c == other.c
# returns the index of the max arg of tensor
# function 2
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


# input, feature tensors of the new regions from P and Q
# returns 2 lists of touples, one for P and one for Q
# each touple is a neuron p, with is corresponding NN q
# p_list, and q_list is a list of touples where each touple contains touples of coordinates [((x1, y1), (x2, y2))]
def NBB(C_A, C_B, R, neighbor_size):
    
    P_region = R[0]
    Q_region = R[1]

    # iterate through the p's in common_p_q to find its neighbors in Q, (p, q)
    # pass in list of neurons [p1, p2, q1, q2]
    # p1 - bottom left, p2 bottom right, same for q
    #     print("P_nearest")
    print("Nearest neighbor start")
    
#     qs_for_ps = nearest_neighbor(C_A, C_B, P_region, Q_region)
    # print(qs_for_ps)
    qs_for_ps = nearest_neighbor(C_A, C_B, P_region, neighbor_size)
    
    
    print("Nearest neighbor midpoint")
    
    
    # iterate through the q's in common_q_p to find its neighbors in P, (q, p)
     #     print("Q_nearest")
        
#     ps_for_qs = nearest_neighbor(C_B, C_A, Q_region, P_region)
    # print(ps_for_qs)
    ps_for_qs = nearest_neighbor(C_B, C_A, Q_region, neighbor_size)
    print("Nearest neighbor end")
    # returns in (p, q) format
    # get the candidates that are nearest neighbors to each other
    print("Candidates start")
    candidates = get_candidates(qs_for_ps, ps_for_qs)
    print("Candidates end")
    # check the activations and find the most meaningful buddies
    # must return in format p[], q[]

    feat_a_norm = normalize_feature_map(C_A)
    feat_b_norm = normalize_feature_map(C_B)

    return meaningful_NBBs(feat_a_norm, feat_b_norm, candidates, .05)

def normalize_feature_map(feat_map):
    """
    Assigns each neuron a value in the range [0, 1] to the
    given feature map

    Args:
        feat_map: feature map tensor
    Returns:
        norm_feat_map: normalized feature map

    """

    feat_min = feat_map.min()
    feat_max = feat_map.max()

    feat_map_norm = (feat_map - feat_min) / (feat_max - feat_min)

    return feat_map_norm

def meaningful_NBBs(C_A, C_B, candidates, act_threshold):
    """
    Use normalized activation maps to seek NNBS which have high activation
    values

    Args:
        feat_a: feature map tensor for image a
        feat_b: feature map tensor for image b
        candiates: list of neural best buddies candiates
        act_threshold: empirically determined activation threshold

    Returns:
        meanigful_buddies: list of neural best buddes with high activation
            values


    Returns:
        meanigful_buddies: list of neural best buddes with high activation
            values

    """
    feat_a = C_A.clone().squeeze().permute(1, 2, 0)
    feat_b = C_B.clone().squeeze().permute(1, 2, 0)
#     print(feat_a.shape)
#     print(feat_b.shape)

    num_candidate_pairs = len(candidates)

    meaningful_buddies = []

    for i in range (num_candidate_pairs):

        p_coords = candidates[i][0]
        q_coords = candidates[i][1]
        
#         if ((p_coords.r < feat_a.shape[0] and p_coords.c < feat_a.shape[0]) and (q_coords.r < feat_b.shape[0] and q_coords.r < feat_b.shape[0])):
        p_max_activation_indx = feat_arg_max(feat_a[p_coords.r, p_coords.c, :])
        p_max_activation = feat_a[p_coords.r][p_coords.c][p_max_activation_indx]


#             print("p idx: ", p_max_activation_indx)
#             print("p ac: ", p_max_activation)


        q_max_activation_indx = feat_arg_max(feat_b[q_coords.r, q_coords.c, :])
        q_max_activation = feat_b[q_coords.r][q_coords.c][q_max_activation_indx]


#             print("q idx: ", q_max_activation_indx)
#             print("q ac: ",q_max_activation)

        if (q_max_activation > act_threshold and p_max_activation > act_threshold):
            candidates[i][0].activation = p_max_activation
            candidates[i][1].activation = q_max_activation
            print("Candidate that is meaningful: ",  candidates[i])
            meaningful_buddies.append(candidates[i])

    return meaningful_buddies


def _get_neighborhood(P, i, j, neigh_rad):
    # 2
    P = P.permute(1, 2, 0)
    P_padded = torch.zeros((P.size()[0] + 2 * neigh_rad, P.size()[1] + 2 * neigh_rad, P.size()[2]))
    P_padded[neigh_rad: -neigh_rad, neigh_rad: -neigh_rad] = P
    return P_padded[i: i + 2 * neigh_rad + 1, j: j + 2 * neigh_rad + 1].permute(2, 0, 1)

def _NBB(Ps, Qs, neigh_rad, gamma=0.05):
    """
    args:
        P: 4D tensor of features in NCHW format
        Q: 4D tensor of features in NCHW format
        neigh_rad: int representing amount of surrounding neighbors to include in cross correlation.
                   so neigh_rad of 1 takes cross correlation of 3x3 patches of neurons
        gamma: (optional) activation threshold
    output:
        NBB pairs
    """

    height = Ps.size()[2]
    width = Ps.size()[3]
    n_channels = Ps.size()[1]

    best_buddies = []

    for P, Q in zip(Ps, Qs):
        #2
        P_L2 = P.clone().permute(1,2,0).norm(2, 2)
        Q_L2 = Q.clone().permute(1,2,0).norm(2, 2)

        P_over_L2 = P.div(P_L2)
        Q_over_L2 = Q.div(Q_L2)

        P_nearest = []
        Q_nearest = []
        for i in range(0, height):
            for j in range(0, width):
                p_neigh = _get_neighborhood(P_over_L2, i, j, neigh_rad)
                # 1
                conv = torch.nn.Conv2d(n_channels, 1, neigh_rad * 2 + 1, padding=neigh_rad)
                conv.train(False)
                conv.weight.data.copy_(p_neigh.unsqueeze(0))
                p_cross_corrs = conv(Q_over_L2.unsqueeze(0)).squeeze().view(-1)
                # 4
                P_nearest.append(p_cross_corrs.argmax())

                q_neigh = _get_neighborhood(Q_over_L2, i, j, neigh_rad)
                conv = torch.nn.Conv2d(n_channels, 1, neigh_rad * 2 + 1, padding=neigh_rad)
                conv.train(False)
                conv.weight.data.copy_(q_neigh.unsqueeze(0))
                q_cross_corrs = conv(P_over_L2.unsqueeze(0)).squeeze().view(-1)
                Q_nearest.append(q_cross_corrs.argmax())

      
        pq_size = int(math.sqrt(len(P_nearest)))
        
        for i in range(len(P_nearest)):
            if(i == Q_nearest[P_nearest[i]]):
                
                p_r = math.floor(1.0 * i / pq_size)
                p_c = i - (p_r * pq_size)
                p = Neuron(p_r, p_c)

                j = P_nearest[i]

                q_r = math.floor(1.0 * j / pq_size)
                q_c = j - (q_r * pq_size)
                q = Neuron(q_r, q_c)

                best_buddies.append([p, q])
                
    feat_a_norm = normalize_feature_map(Ps)
    feat_b_norm = normalize_feature_map(Qs)

    return meaningful_NBBs(feat_a_norm, feat_b_norm, best_buddies, .05)

# neighborhood function for P
def neighborhood(P_over_L2, p_i, p_j, neigh_size):

    neigh_rad = int((neigh_size - 1) / 2)
    P = P_over_L2.clone().permute(1, 2, 0)

    P_padded = torch.zeros((P.size()[0] + 2 * neigh_rad, P.size()[1] + 2 * neigh_rad, P.size()[2]))

    P_padded[neigh_rad: -neigh_rad, neigh_rad: -neigh_rad] = P
    P_padded = P_padded[p_i: p_i + 2 * neigh_rad + 1, p_j: p_j + 2 * neigh_rad + 1].permute(2, 0, 1)

    return P_padded


# takes in P and Q tensors, and the neighborhood size(5 or 3)
# returns list of nearest neighbors of P
def nearest_neighbor(P_tensor, Q_tensor, P_region, neigh_size):
  
    # info from P tensor
#     print(P_tensor.size())
    num_chan = P_tensor.shape[1]
    img_w = P_tensor.shape[2]
    img_h = P_tensor.shape[3]

    # list of nearest neighbors of P
    nearest_buddies = []

    # L2 of P and Q
    P = P_tensor.clone().squeeze()
    P_L2 = P.permute(1, 2, 0).norm(2, 2)

    Q = Q_tensor.clone().squeeze()
    Q_L2 = Q.permute(1, 2 ,0).norm(2, 2)

    # similarity metric
    P_over_L2 = P.div(P_L2)
    Q_over_L2 = Q.div(Q_L2)
    #print( P_over_L2.shape)

    neigh_rad = int((neigh_size - 1) / 2)
    
#     print("P_region: ", P_region)
    for r in range(len(P_region)): 
        # region points to calculate 
        top_left_p = P_region[r][0]
        bottom_right_p = P_region[r][1]

        
        for p_i in range(top_left_p.r, bottom_right_p.r):
            for p_j in range(top_left_p.c, bottom_right_p.c):
                conv = torch.nn.Conv2d(num_chan, 1, neigh_size, padding=neigh_rad)
                conv.train(False)

                p_neigh = neighborhood(P_over_L2, p_i, p_j, neigh_size)
                #print(" ", p_neigh.shape, neigh_size)
                conv.weight.data.copy_(p_neigh.unsqueeze(0))

                p_cross_corrs = conv(Q_over_L2.unsqueeze(0)).clone().squeeze().view(-1)

                #p_cross_corrs = conv(Q_over_L2.unsqueeze(0)).squeeze().detach().numpy()
                #q_idx = np.unravel_index(p_cross_corrs.argmax(), p_cross_corrs.shape)
                #p = Neuron(p_i, p_j)
                #q = Neuron(q_idx[0], q_idx[1])
                #nearest_buddies.append(q_idx)
                nearest_buddies.append(p_cross_corrs.argmax())
                
    return nearest_buddies


# P and Q should be feature maps for a given layer
# returns the common appearance C(P, Q)
def common_appearance(P, Q, region_p_list, region_q_list):
    # copy of the whole P - going to put common_app in specific region on this later
    p_to_q = P.clone()
    # changed to - [chann, height, width]

    P_copy = P.clone().squeeze()
    Q_copy = Q.clone().squeeze()

    
    for ind in range(len(region_p_list)):
        region_p = region_p_list[ind]
        region_q = region_q_list[ind]

        top_left_p = region_p[0]
        bottom_right_p = region_p[1]
        top_left_q = region_q[0]
        bottom_right_q = region_q[1]
        
        #         print(P_copy.shape)
        #         print(Q_copy.shape)
        #         print("tl p: ", top_left_p)
        #         print("tl: q: ", top_left_q)
        #         print("br p: ", bottom_right_p)
        #         print("br q: ", bottom_right_q)
        # these only represent P, Q in the region (AKA trimmed P, and Q)
        P_copy_reg = P_copy[:, int(top_left_p.r):int(bottom_right_p.r), int(top_left_p.c):int(bottom_right_p.c)]
        Q_copy_reg = Q_copy[:, int(top_left_q.r):int(bottom_right_q.r), int(top_left_q.c):int(bottom_right_q.c)]


        # have to squeeze to remove first dimension: [C, H, W]
        mean_p = P_copy_reg.mean(2).mean(1)
        mean_q = Q_copy_reg.mean(2).mean(1)
        mean_m = (mean_p + mean_q) / 2
        sig_p = P_copy_reg.std(2).std(1)
        sig_q = Q_copy_reg.std(2).std(1)
        sig_m = (sig_p + sig_q) / 2
        # have to permute, in order to be able to subtract the mean correctly
        temp = (P_copy_reg.permute(1,2,0) - mean_p)
        # common_app should be the size of the region we are doing style transfer on
        common_app = (temp/ sig_p * sig_m + mean_m).permute(2,0,1)

        p_to_q[:, :, top_left_p.r:bottom_right_p.r, top_left_p.c:bottom_right_p.c] = common_app
        
    return p_to_q

def refine_search_regions(prev_layer_nbbs, receptive_field_radius, feat_width, feat_height):
    """
    Return refined search regions for every p and q in the previous' layer nbbs
    Args:
        prev_layer_nbbs: Previous' layer (l-1) neural best buddies, represented
            as neurons using the Neuron class
        receptive_field_radius: radius of new search regions
            equal to 4 for l = 2,3 and equal to 6 for l = 4, 5
        feat_width: width of feature map for current layer
        feat_height: height of feature map for current layer
    Returns:
        Ps: List containing new P's
            P = ((r1, c1), (r2, c2))
            where (r1, c1) represent the top left of the search region
            and (r2, c2) represent the bottom right of the search region
        Qs: List containing new Q's
            Q = ((r1, c1), (r2, c2))
            where (r1, c1) represent the top left of the search region
            and (r2, c2) represent the bottom right of the search region
            
    """

    Ps = []
    Qs = []

    for p, q in prev_layer_nbbs:
        
        # Top left of search window for P
        P_r1 = max(int(2 * p.r - receptive_field_radius / 2), 0)
        P_c1 = max(int(2 * p.c - receptive_field_radius / 2), 0)
        P_top_left = Neuron(P_r1, P_c1)
        
        # Bottom right of search window for P
        P_r2 = min(int(2 * p.r + receptive_field_radius / 2), feat_width - 1)
        P_c2 = min(int(2 * p.c + receptive_field_radius / 2), feat_height - 1)
        P_bottom_right = Neuron(P_r2, P_c2)
        
        # Top left of search window for Q
        Q_r1 = max(int(2 * q.r - receptive_field_radius / 2), 0)
        Q_c1 = max(int(2 * q.c - receptive_field_radius / 2), 0)
        Q_top_left = Neuron(Q_c1, Q_r1)

        # Bottom right of search window for Q
        Q_r2 = min(int(2 * q.r + receptive_field_radius / 2), feat_width - 1)
        Q_c2 = min(int(2 * q.c + receptive_field_radius / 2), feat_height - 1)
        Q_bottom_right = Neuron(Q_c2, Q_r2)

        # Append P and Q to lists
        Ps.append((P_top_left, P_bottom_right))
        Qs.append((Q_top_left, Q_bottom_right))
    
    return (Ps, Qs)


# Image preprocessing
def img_preprocess_VGG(img):
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

def image_preprocess_resnet(img):
    # now lets do the same for resnet_18
    to_tensor = transforms.ToTensor()
    scaler = transforms.Scale((224, 224))
    normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])
    img_tens = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    return img_tens

def image_preprocess_alexnet(img):
    to_tensor = transforms.ToTensor()
    scaler = transforms.Scale((299, 299))
    normalize = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.485, 0.456, 0.406),
                             std=(0.229, 0.224, 0.225))])
    img_tens = Variable(normalize(to_tensor(scaler(img))).unsqueeze(0))
    return img_tens

# vgg model
# takes in image a, image b, normalized tensor of image a, normalized tensor of img b
# returns 5 layered feature map of img a and img b
def vgg19_model(img_a, img_b, img_a_tens, img_b_tens):
    print("**********************************V G G 1 9**********************************")
    print("img_a size: ", img_a.size)
    print("img a t-size: ", tf.shape(img_a_tens))
    model = models.vgg19(pretrained=True).eval()
    pyramid_layers = []

    def extract_feature(module, input, output):
        pyramid_layers.append(output)

    relu_idx = [3, 8, 17, 26, 35]
    #relu_idx = [35, 26, 17, 8, 3]
    print("vgg19: ", model.features[0] )
    for j in relu_idx:
        model.features[j].register_forward_hook(extract_feature)

    model(img_a_tens)
    model(img_b_tens)
    for layer in pyramid_layers:
        print("ith layer @ relu: ", layer.size())
        
    return pyramid_layers[:5], pyramid_layers[5:]

def resnet_18(img_a, img_b, img_a_tens, img_b_tens):
    print("**********************************R E S N E T 18**********************************")
    print("img_a size: ", img_a.size)
    print("img a t-size: ", tf.shape(img_a_tens))
    model = models.resnet18(pretrained=True).eval()
    # bb = list(model.layer1.children())[1] used to index into the right block of the layer, then add a .relu to get the relu
    layer_list = [list(model.layer1.children())[1].relu, list(model.layer2.children())[1].relu, list(model.layer3.children())[1].relu, list(model.layer4.children())[1].relu]
    pyramid_layers = []
    def extract_feature(module, input, output):
        pyramid_layers.append(output)
    # Attach that function to our selected layers
    for layer in layer_list:
        # print("layer type:", type(layer))
        layer.register_forward_hook(extract_feature)

    # Run the model on our transformed image
    model(img_a_tens)
    model(img_b_tens)
    # remove duplicates..... (not sure why we had them in the first place.)
    pyramid_layers = [ pyramid_layers[idx] for idx in range(0, len(pyramid_layers), 2)]
    # Return the feature vector
    for layer in pyramid_layers: # debug, check layers
        print("ith layer @ relu: ", layer.size())
    # print("first layer:",pyramid_layers[7].size())
    # print("2th layer:",pyramid_layers[4].size())
    return pyramid_layers[:4], pyramid_layers[4:]

def alexnet(img_a, img_b, img_a_tens, img_b_tens):
    print("**********************************A L E X N E T**********************************")
    print(type(img_a_tens) )
    model = models.alexnet(pretrained=True).eval()
    print(model)
    pyramid_layers = []
    layer_list = [model.features[1], model.features[4], model.features[7], model.features[9], model.features[11]]
    def extract_feature(module, input, output):
        pyramid_layers.append(output)

    for layer in layer_list:
        # print("layer type:", type(layer))
        layer.register_forward_hook(extract_feature)

    model(img_a_tens)
    model(img_b_tens)

    for layer in pyramid_layers: # debug, check layers
        print("ith layer @ relu: ", layer.size())
    return pyramid_layers[:5], pyramid_layers[5:]

def plot_neurons(n_list, i, img):
    # Mock data
    # n1 = Neuron(1, 2)
    # n2 = Neuron(150, 150)
    # n3 = Neuron(10, -2)
    # n4 = Neuron(-1, -3)
    # n_list = [[n1, n2], [n3, n4]]
    img_plot = plt.imshow(img)
    
    for pair in n_list:
        neuron = pair[i]
#         print("plotting", neuron.r, neuron.c)
        # figure(1)
        # plt.scatter(neuron.r, neuron.c)
        # figure(2)
        plt.scatter(neuron.r, neuron.c)

def scale_nbbs(nbbs, layer):
    
    scale_factor = int(math.pow(2, layer))
    scaled_nbbs = []
     
    for p, q in nbbs:
        
        scaled_p_r = p.r * scale_factor;
        scaled_p_c = p.c * scale_factor;
        scaled_p = Neuron(scaled_p_r, scaled_p_c, p.activation)

        scaled_q_r = q.r * scale_factor;
        scaled_q_c = q.c * scale_factor;
        scaled_q = Neuron(scaled_q_r, scaled_q_c, q.activation)
        
        scaled_nbbs.append((scaled_p, scaled_q))
    
    return scaled_nbbs

# given a list of meaningful BBs, we return a new list of NBB that contain the highest rank in their respective clusterss
def high_ranked_buddies(nbbs, k):
    
    print("hey du", nbbs)
    
    if k > len(nbbs):
        return nbbs
    # have buddies with act_sum
    # [(p, q), (p2, q2)]
    # make activation list ^^
    # [act = p.act_sum + q.act_sum, act = p2.act_sum + q2.act_sum]
    act_list = []
    p_coords = []
    p_neurons = []
    q_coords = []
    q_neurons = []
    for p, q in nbbs:
        act = p.activation + q.activation
        act_list.append(act)
        p_coords.append((p.r, p.c))
        p_neurons.append(p)
        q_coords.append((q.r, q.c))
        q_neurons.append(q)

    # creates k clusters for p coordinates
    kmeansp = KMeans(n_clusters=k)
    kmeansp.fit(p_coords)
    # a list of cluster # that corresponds to "p_coords"
    cluster_listp = kmeansp.labels_

    # creates k clusters for q coordinates
    kmeansq= KMeans(n_clusters=k)
    kmeansq.fit(q_coords)
    cluster_listq = kmeansq.labels_

    # list of lists, where each inner list is a list that corresponds to a cluster
    coords_per_clusterp = [[]] * k
    act_per_coordsp = [[]] * k

    coords_per_clusterq = [[]] * k
    act_per_coordsq = [[]] * k

    # iterate through cluster_listq (should be same size as cluster_listq)
    for i in range(len(cluster_listp)):
        # find cluster, coords, and activation that corresponds to i
        cluster_p = cluster_listp[i]
        coords_p = p_coords[i]
        ind_of_acts = []
        # ind_of_acts = [np.where(p_coords == coords_p)[0]]
        for ind, p in enumerate(p_coords):
            #print(p)
            if p[0] == coords_p[0] and p[1] == coords_p[1]:
                ind_of_acts.append(ind)
                
        act_p = 0
        for act in ind_of_acts:
            neuron = p_neurons[act]
            act_p += neuron.activation.item()

        # append to lists created, so each coordinates & activations are organized by cluster
        coords_per_clusterp[cluster_p].append(coords_p)
        act_per_coordsp[cluster_p].append(act_p)

        # do the same for q
        cluster_q = cluster_listq[i]
        coords_q = q_coords[i]
        ind_of_acts = np.where(q_coords == coords_q)[0]
        act_q = 0
        for ind in ind_of_acts:
            neuron = q_neurons[ind]
            act_q += neuron.activation

        coords_per_clusterq[cluster_q].append(coords_q)
        act_per_coordsq[cluster_q].append(act_q)

    true_buddies = []
    # find the final true buddies list
    # iterate through the list of activations of p and q
    for i in range(len(act_per_coordsp)):
        # find the argmax of the activation of the ith cluster and get the coordinates that correspond
        act_listp = act_per_coordsp[i]
        print("ACTIVATION LIST OF P: ", act_listp)
        max_act_indp = np.argmax(act_listp)
        print("------THE MAX: ", max_act_indp)
        buddy_coords_p = coords_per_clusterp[i][max_act_indp]
        # transform back to neuron
        neuronp = Neuron(buddy_coords_p[0],buddy_coords_p[1])

        act_listq = act_per_coordsq[i]
        max_act_indq = np.argmax(act_listq)
        buddy_coords_q = coords_per_clusterq[i][max_act_indq]
        neuronq = Neuron(buddy_coords_q[0],buddy_coords_q[1])

        # append both neurons to final list
        true_buddies.append((neuronp, neuronq))

    return true_buddies

def main():
    img_a = Image.open("../input/dog1.jpg")
    img_b = Image.open("../input/dog2.jpg")
    img_a_tens = img_preprocess_VGG(img_a)
    img_b_tens = img_preprocess_VGG(img_b)

    feat_a_19, feat_b_19 = vgg19_model(img_a, img_b, img_a_tens, img_b_tens)

    # print("vgg 19 types:", type(feat_a_19))

    # img_a_tens = image_preprocess_resnet(img_a)
    # img_b_tens = image_preprocess_resnet(img_b)
    # feat_a_18, feat_b_18 = resnet_18(img_a, img_b, img_a_tens, img_b_tens)
    # print(feat_a_18[0] )
    # img_a_tens = image_preprocess_alexnet(img_a)
    # img_b_tens = image_preprocess_alexnet(img_b)
    # feat_a_v3, feat_b_v3 = alexnet(img_a, img_b, img_a_tens, img_b_tens)
    
    layer = 4
    
    receptive_field_rs = [4, 4, 6, 6]
    # neigh_sizes = [5, 5, 5, 3, 3]
    neigh_sizes = [2, 2, 2, 1, 1]

    C_A = feat_a_19[layer]
    C_B = feat_b_19[layer]
    
    top_left_p = Neuron(0, 0)
    bottom_right_p = Neuron(C_A.shape[2], C_A.shape[2])
    
    top_left_q = Neuron(0, 0)
    bottom_right_q = Neuron(C_B.shape[2], C_B.shape[2])
    

    R = [[(top_left_p, bottom_right_p)], [(top_left_q, bottom_right_q)]]
    nbbs = []
    scaled_nbbs = []
    scaled_nbbs_high = []
    
    for l in range (layer, 3, -1):
        
        print ("------ Layer ", l + 1, " ------")

        feat_a = feat_a_19[l]
        feat_b = feat_b_19[l]
        
        print(feat_a.size())
        print(feat_b.size())
        
        layer_nbbs = _NBB(C_A, C_B, neigh_sizes[l])
#         layer_nbbs = NBB(C_A, C_B, R, neigh_sizes[l])
        nbbs.append(layer_nbbs)
        scaled_nbbs.append(scale_nbbs(layer_nbbs, l))
        plt.figure(1)
        plot_neurons(scaled_nbbs[layer - l], 0, img_a)
        plt.figure(2)
        plot_neurons(scaled_nbbs[layer - l], 1, img_b)
        plt.show()
        
#         print("nbbs: ", nbbs)
        
        nbbs_high = high_ranked_buddies(layer_nbbs, 40)
    
        print("NBBADSGSF:", len(nbbs_high))
        
        print(nbbs_high)
        
    
    
        scaled_nbbs_high.append(scale_nbbs(nbbs_high, l))
        
        plt.figure(1)
        plot_neurons(scaled_nbbs_high[layer - l], 0, img_a)
        plt.figure(2)
        plot_neurons(scaled_nbbs_high[layer - l], 1, img_b)
        plt.show()

        if l > 0:
            
#             print(feat_a_19[l - 1].size())
#             print(feat_b_19[l - 1].size())
            
            feat_width = feat_a_19[l - 1].shape[2]
            feat_height = feat_a_19[l - 1].shape[3]
            R = refine_search_regions(nbbs[len(nbbs) - 1], receptive_field_rs[l - 1], feat_width, feat_height)
            
            feat_a = feat_a_19[layer]
            feat_b = feat_b_19[layer]

            top_left_p = Neuron(0, 0)
            bottom_right_p = Neuron(feat_a.shape[2], feat_b.shape[2])

            top_left_q = Neuron(0, 0)
            bottom_right_q = Neuron(feat_a.shape[2], feat_b.shape[2])
            
            R = [[(top_left_p, bottom_right_p)], [(top_left_q, bottom_right_q)]]

#             top_left_p = Neuron(0, 0)
#             bottom_right_p = Neuron(feat_width, feat_height)

#             top_left_q = Neuron(0, 0)
#             bottom_right_q = Neuron(feat_width, feat_height)

#             R = [[(top_left_p, bottom_right_p)], [(top_left_q, bottom_right_q)]]
            
            C_A = common_appearance(feat_a_19[l - 1], feat_b_19[l - 1], R[0], R[1])
            C_B = common_appearance(feat_b_19[l - 1], feat_a_19[l - 1], R[1], R[0])
    
    print("Printing all nbbs") 
    plt.figure(1)
    for curr_nbb in scaled_nbbs: 
        plot_neurons(curr_nbb, 0, img_a)
        
    plt.figure(2)
    for curr_nbb in scaled_nbbs:
        plot_neurons(curr_nbb, 1, img_b)

    plt.show()

if __name__ == "__main__":
    main()