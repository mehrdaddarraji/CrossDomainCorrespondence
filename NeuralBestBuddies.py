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
from sklearn.cluster import KMeans
from matplotlib import colors as mcolors
from random import shuffle
import matplotlib.ticker as plticker

class Neuron:
    """
    Neuron class, takes in row and col coordinates
    
    """
    def __init__(self, row, col, activation = 0):
        self.r = row
        self.c = col
        self.activation = activation
    def __repr__(self):
        return "(" + str(self.r) + ", " + str(self.c) + ")"
    def __eq__(self, other):
        return self.r == other.r and self.c == other.c

def img_preprocess_VGG(img):
    """
    Image preprocessing
    VGGNet was trained on ImageNet where images are normalized by mean=[0.485, 0.456, 0.406]
    and std=[0.229, 0.224, 0.225].
    We use the same normalization statistics here.
    
    """
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

def vgg19_model(img_a, img_b, img_a_tens, img_b_tens):
    """
    vgg model
    takes in image a, image b, normalized tensor of image a, normalized tensor of img b
    returns 5 layered feature map of img a and img b
    
    """    
    print("********************************** V G G 1 9 **********************************")
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
    pyramid_layers.reverse()
    return pyramid_layers[:5], pyramid_layers[5:]

def resnet_18(img_a, img_b, img_a_tens, img_b_tens):
    print("********************************** R E S N E T 18 **********************************")
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
    print("********************************** A L E X N E T **********************************")
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

def feat_arg_max(feat):
    """
    Returns the index of the max arg of tensor
    
    Args:
        feat: feature map tensor
        
    Returns:
        idx: index of highest activation in feat
    
    """
    f = feat.clone().detach().numpy()
    idx = np.unravel_index(f.argmax(), f.shape)
    return idx

def L2_norm(A_tensor):
    """
    function to do L2 normalization of a tensor
    
    """

    A = A_tensor.clone().detach().numpy()
    pow_sum = np.power(A, 2).sum()
    A_sqrt = np.power(pow_sum, 0.5)
    return torch.from_numpy(A / A_sqrt)


def _NBB(Ps, Qs, neigh_rad, gamma=0.05):
    """
    input, feature tensors of the new regions from P and Q
    returns 2 lists of touples, one for P and one for Q
    each touple is a neuron p, with is corresponding NN q
    p_list, and q_list is a list of touples where each touple contains touples of coordinates [((x1, y1), (x2, y2))]
    
    """

    height = Ps.size()[2]
    width = Ps.size()[3]
    n_channels = Ps.size()[1]

    best_buddies = []

    for P, Q in zip(Ps, Qs):
        P_L2 = P.clone().permute(1,2,0).norm(2, 2)
        Q_L2 = Q.clone().permute(1,2,0).norm(2, 2)

        P_over_L2 = P.div(P_L2)
        Q_over_L2 = Q.div(Q_L2)

        P_nearest = []
        Q_nearest = []
        
        for i in range(0, height):
            for j in range(0, width):
                p_neigh = _get_neighborhood(P_over_L2, i, j, neigh_rad)
                conv = torch.nn.Conv2d(n_channels, 1, neigh_rad * 2 + 1, padding=neigh_rad)
                conv.train(False)
                conv.weight.data.copy_(p_neigh.unsqueeze(0))
                p_cross_corrs = conv(Q_over_L2.unsqueeze(0)).squeeze().view(-1)
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
                q = Neuron(q_r, q_c.item())

                best_buddies.append([p, q])
                
    feat_a_norm = normalize_feature_map(Ps)
    feat_b_norm = normalize_feature_map(Qs)

    return meaningful_NBBs(feat_a_norm, feat_b_norm, best_buddies, .05)

def _get_neighborhood(P, i, j, neigh_rad):
    P = P.permute(1, 2, 0)
    P_padded = torch.zeros((P.size()[0] + 2 * neigh_rad, P.size()[1] + 2 * neigh_rad, P.size()[2]))
    P_padded[neigh_rad: -neigh_rad, neigh_rad: -neigh_rad] = P
    return P_padded[i: i + 2 * neigh_rad + 1, j: j + 2 * neigh_rad + 1].permute(2, 0, 1)

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

    num_candidate_pairs = len(candidates)

    meaningful_buddies = []

    for i in range (num_candidate_pairs):

        p_coords = candidates[i][0]
        q_coords = candidates[i][1]
        
        p_max_activation_indx = feat_arg_max(feat_a[p_coords.r, p_coords.c, :])
        p_max_activation = feat_a[p_coords.r][p_coords.c][p_max_activation_indx]

        q_max_activation_indx = feat_arg_max(feat_b[q_coords.r, q_coords.c, :])
        q_max_activation = feat_b[q_coords.r][q_coords.c][q_max_activation_indx]

        if (q_max_activation > act_threshold and p_max_activation > act_threshold):
            candidates[i][0].activation = p_max_activation
            candidates[i][1].activation = q_max_activation
            meaningful_buddies.append(candidates[i])

    return meaningful_buddies

def scale_nbbs(nbbs, width):
    
    scale_factor = int(224 / width)
    scaled_nbbs = []
     
    for p, q in nbbs:
        
        scaled_p_r = p.r * scale_factor;
        scaled_p_c = p.c * scale_factor;
        scaled_p = Neuron(scaled_p_r, scaled_p_c, p.activation)

        scaled_q_r = q.r * scale_factor;
        scaled_q_c = q.c * scale_factor;
        scaled_q = Neuron(scaled_q_r, scaled_q_c, q.activation)
        
        scaled_nbbs.append([scaled_p, scaled_q])
    
    return scaled_nbbs

def plot_with_grid(subplt, img, n_cells, nbbs, a_or_b, colors, my_dpi=60):
    ax = plt.subplot(*subplt)
    ax.imshow(img)
    plt.axis('off')
    
    nbb_index = 0 if a_or_b == 'a' else 1
    for index, coords in enumerate(nbbs):
        j = coords[nbb_index].c
        i = coords[nbb_index].r
        ax.add_artist(plt.Circle((i, j), 3, color=colors[index], alpha=0.9))

def plot_neurons(n_list, i, img):
    
    plt.axis('off')
    img_plot = plt.imshow(img)
    
    for pair in n_list:
        neuron = pair[i]
        plt.scatter(neuron.r, neuron.c)

# given a list of meaningful BBs, we return a new list of NBB that contain the highest rank in their respective clusterss
def high_ranked_buddies(nbbs, k):

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
    coords_per_clusterp = []
    act_per_coordsp = []
    coords_per_clusterq = []
    act_per_coordsq = []
    for i in range(k):
        coords_per_clusterp.append([])
        act_per_coordsp.append([])
        coords_per_clusterq.append([])
        act_per_coordsq.append([])

    # iterate through cluster_listq (should be same size as cluster_listq)
    for i, val in enumerate(cluster_listp):
        # find cluster, coords, and activation that corresponds to i
        #cluster_p = cluster_listp[i]
        cluster_p = val
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
        max_act_indp = np.argmax(act_listp)
        activp = act_listp[max_act_indp]
        buddy_coords_p = coords_per_clusterp[i][max_act_indp]
        # transform back to neuron
        neuronp = Neuron(buddy_coords_p[0],buddy_coords_p[1], activp)

        act_listq = act_per_coordsq[i]
        max_act_indq = np.argmax(act_listq)
        activq = act_listp[max_act_indp]
        buddy_coords_q = coords_per_clusterq[i][max_act_indq]
        neuronq = Neuron(buddy_coords_q[0],buddy_coords_q[1], activq)

        # append both neurons to final list
        true_buddies.append((neuronp, neuronq))

    return true_buddies

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

        # these only represent P, Q in the region (AKA trimmed P, and Q)
        P_copy_reg = P_copy[:, int(top_left_p.r):int(bottom_right_p.r), int(top_left_p.c):int(bottom_right_p.c)]
        Q_copy_reg = Q_copy[:, int(top_left_q.r):int(bottom_right_q.r), int(top_left_q.c):int(bottom_right_q.c)]

        # have to squeeze to remove first dimension: [C, H, W]
        mean_p = P_copy_reg.mean()
        mean_q = Q_copy_reg.mean()
        mean_m = (mean_p + mean_q) / 2
        sig_p = P_copy_reg.std()
        sig_q = Q_copy_reg.std()
        sig_m = (sig_p + sig_q) / 2
        # have to permute, in order to be able to subtract the mean correctly
        temp = (P_copy_reg - mean_p)
        # common_app should be the size of the region we are doing style transfer on
        common_app = (temp/ sig_p * sig_m + mean_m)

        p_to_q[:, :, top_left_p.r:bottom_right_p.r, top_left_p.c:bottom_right_p.c] = common_app
        
    return p_to_q

def main():
    img_a = Image.open("../input/cat.jpg")
    img_b = Image.open("../input/dog.jpg")
    img_a_tens = img_preprocess_VGG(img_a)
    img_b_tens = img_preprocess_VGG(img_b)

    feat_a_19, feat_b_19 = vgg19_model(img_a, img_b, img_a_tens, img_b_tens)
    
    colors = list(dict(mcolors.BASE_COLORS, **mcolors.CSS4_COLORS).keys())
    shuffle(colors)

    # img_a_tens = image_preprocess_resnet(img_a)
    # img_b_tens = image_preprocess_resnet(img_b)
    # feat_a_18, feat_b_18 = resnet_18(img_a, img_b, img_a_tens, img_b_tens)
    
    # img_a_tens = image_preprocess_alexnet(img_a)
    # img_b_tens = image_preprocess_alexnet(img_b)
    # feat_a_v3, feat_b_v3 = alexnet(img_a, img_b, img_a_tens, img_b_tens)
    
    layer = 1
    
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
    
    for l in range (layer, -1, -1):
        
        print ("------ Layer ", l + 1, " ------")

        feat_a = feat_a_19[l]
        feat_b = feat_b_19[l]
        
        print(feat_a.size())
        print(feat_b.size())
        
        layer_nbbs = _NBB(C_A, C_B, neigh_sizes[l])
        nbbs.append(layer_nbbs)
        scaled_nbbs.append(scale_nbbs(layer_nbbs, feat_a.shape[2]))
        

        nbbs_high = high_ranked_buddies(layer_nbbs, 80)
        scaled_nbbs_high.append(scale_nbbs(nbbs_high, feat_a.shape[2]))
        
        print("nbbs: ", scaled_nbbs_high[layer - 1])
        
        plot_with_grid((1,2,1), img_a, feat_a.size()[2], scaled_nbbs_high[layer - l], 'a', colors)
        plot_with_grid((1,2,2), img_b, feat_a.size()[2], scaled_nbbs_high[layer - l], 'b', colors)
        plt.show()

        if l > 0:
                 
            feat_width = feat_a_19[l - 1].shape[2]
            feat_height = feat_a_19[l - 1].shape[3]
            R = refine_search_regions(nbbs[len(nbbs) - 1], receptive_field_rs[l - 1], feat_width, feat_height)
            
            r = [[Neuron(0, 0), Neuron(feat_width, feat_height)]]
            
            C_A = common_appearance(feat_a_19[l - 1], feat_b_19[l - 1], r, r)# R[0], R[1])
            C_B = common_appearance(feat_b_19[l - 1], feat_a_19[l - 1], r, r)#R[1], R[0]), 
            

    print("Printing all nbbs") 
    for curr_nbb in scaled_nbbs_high: 
        plot_with_grid((1,2,1), img_a, feat_a.size()[2], curr_nbb, 'a', colors)
        plot_with_grid((1,2,2), img_b, feat_a.size()[2], curr_nbb, 'b', colors)
    plt.show()


if __name__ == "__main__":
    main()
