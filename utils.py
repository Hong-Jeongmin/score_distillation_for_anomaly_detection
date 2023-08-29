import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import faiss, time, random
import torchvision.transforms as T
import dataset, argparse
from sklearn.metrics import roc_auc_score


from model.model import get_model, freeze_parameters
from model.CSI import ResNet18
from memory_profiler import profile

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")




def precompute_feature(dataset, label):
    print('Pre-compute global statistics...')
    feats_train = {}
    feats_train['simclr'] = torch.load(f'models/CSI/{dataset}/one_class_{label}/train_simclr_{label}.pth')
    feats_train['shift'] = torch.load(f'models/CSI/{dataset}/one_class_{label}/train_shift_{label}.pth')
    N = feats_train['simclr'].shape[0]
    feats_train['simclr'] = feats_train['simclr']
    feats_train['shift'] = feats_train['shift']

    print("Train NUM", feats_train['simclr'].shape)
    axiss = [] #(4, 5000, 128)
    for f in feats_train['simclr'].chunk(4, dim=1):
        axis = f.mean(dim=1)  # (M, d)
        axiss.append(normalize(axis, dim=1).to(device))
    print('axis size: ' + ' '.join(map(lambda x: str(len(x)), axiss)))

    f_sim = [f.mean(dim=1) for f in feats_train['simclr'].chunk(4, dim=1)]  # list of (M, d)
    f_shi = [f.mean(dim=1) for f in feats_train['shift'].chunk(4, dim=1)]  # list of (M, 4)

    weight_sim = []
    weight_shi = []
    for shi in range(4):
        sim_norm = f_sim[shi].norm(dim=1)  # (M)
        shi_mean = f_shi[shi][:, shi]  # (M)
        weight_sim.append(1 / sim_norm.mean().item())
        weight_shi.append(1 / shi_mean.mean().item())
    return {"axis" : axiss, "weight_sim" : weight_sim, "weight_shi" : weight_shi}


def csi_score(dataset, x, model, features):
    axis, weight_sim, weight_shi = features["axis"], features["weight_sim"], features["weight_shi"]
    shift_trans = Rotation().to(device)
    size = (28, 28) if dataset == "fashion" else (32, 32)
    simclr_aug = T.Compose([
        T.RandomApply([T.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.RandomResizedCrop(scale=(0.54,0.54), size= size),
    ])
    x_t = torch.cat([shift_trans(x, k) for k in range(4)])
    if dataset == "fashion":
        x_t = simclr_aug(x_t)
    _, output_aux = model(x_t, simclr=True, shift=True, penultimate=True)
    layers = ('simclr', 'shift')
    feats_dict= {layer:[] for layer in layers}
    for layer in layers:
        feats_dict[layer] += output_aux[layer].chunk(4)
    for key, val in feats_dict.items():
        feats_dict[key] = torch.stack(val, dim=1)
        
    # convert to gpu tensor
    feats_sim = feats_dict['simclr'].to(device)
    feats_shi = feats_dict['shift'].to(device)
    N = feats_sim.size(0)

    # compute scores
    scores = []
    
    for f_sim, f_shi in zip(feats_sim, feats_shi):
        f_sim = [f.mean(dim=0, keepdim=True) for f in f_sim.chunk(4)]  # list of (1, d)
        f_shi = [f.mean(dim=0, keepdim=True) for f in f_shi.chunk(4)]  # list of (1, 4)
        score = 0
        for shi in range(4):
            neighbor, idx = (f_sim[shi] * axis[shi]).sum(dim=1).sort(descending=True)
            score += neighbor[1].item() * weight_sim[shi]
            score += f_shi[shi][:, shi].item() * weight_shi[shi]
        score = score/4
        scores.append(score)
    return scores, _




def knn_score(train_set, test_set, n_neighbours=3):
    """
    Calculates the KNN distance
    """
    index = faiss.IndexFlatL2(train_set.shape[1])
    index.add(train_set)
    D, _ = index.search(test_set, n_neighbours)
    return np.sum(D[:, 1:], axis=1)

def get_score(args, x, model, train_feature_space):
    if args.teacher == "CSI":
        scores = csi_score(args.dataset, x, model, train_feature_space)
    
    elif args.teacher == "MSCL":
        _, features = model(x)
        features = F.normalize(features, dim=1)
        scores = knn_score(train_feature_space, features.cpu().detach().numpy())

    else:
        _, features = model(x)
        scores = knn_score(train_feature_space, features.cpu().detach().numpy())
    return torch.Tensor(scores).cuda()


def normalize(x, dim=1, eps=1e-8):
    return x / (x.norm(dim=dim, keepdim=True) + eps)
    
class Rotation(nn.Module):
    def __init__(self, max_range = 4):
        super(Rotation, self).__init__()
        self.max_range = max_range
        self.prob = 0.5

    def forward(self, input, aug_index=None):
        _device = input.device

        _, _, H, W = input.size()

        if aug_index is None:
            aug_index = np.random.randint(4)

            output = torch.rot90(input, aug_index, (2, 3))

            _prob = input.new_full((input.size(0),), self.prob)
            _mask = torch.bernoulli(_prob).view(-1, 1, 1, 1)
            output = _mask * input + (1-_mask) * output

        else:
            aug_index = aug_index % self.max_range
            output = torch.rot90(input, aug_index, (2, 3))

        return output


