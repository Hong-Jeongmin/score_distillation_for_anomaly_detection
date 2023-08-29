import os, random
import csv, pickle
from copy import deepcopy
from tqdm import tqdm
import argparse
import pandas as pd


import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
import torchvision

import torchvision.transforms as transforms
import albumentations as A
import albumentations.pytorch
import random
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
import utils, dataset
from model.model import get_model, freeze_parameters
from model.CSI import ResNet18
import time
from memory_profiler import profile
from torchvision.transforms import InterpolationMode
BICUBIC = InterpolationMode.BICUBIC

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.cuda.empty_cache()
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])




def score_distribution(args, base_path, tnet, feats_train, train_loader):
    
    l = args.label
    teacher = args.teacher
    
    tnet.eval()
    print(f"Calculate score distribution")


    score_list = []
    for (x,  _) in (train_loader):
        x= x.cuda()
        with torch.no_grad():
            score_t = utils.get_score(args, x, tnet, feats_train)
            score_list.append(score_t)

    score = torch.cat(score_list, dim=0).detach().cpu().numpy()

    with open(base_path+f'/statistics_{l}.pickle', 'wb') as f:
        pickle.dump([np.mean(score), np.std(score)], f)

@profile
def measure_auroc(args, model, test_loader):
    scores = []
    times = []
    with torch.no_grad():
        for (imgs, _) in tqdm(test_loader, desc='Test set feature extracting', leave=False):
            imgs = imgs.to(device)
            start = time.time()
            score, _ = model(imgs)
            times.append(time.time()-start)

            scores.append(score.cpu().numpy().flatten())
            
        test_labels = test_loader.dataset.targets
        scores = np.hstack(scores)
    auc = roc_auc_score(test_labels, scores)
    print("Elapsed Time of Score computation:", np.sum(times))
    return auc*100


def teacher_auroc(args, tnet, test_loader, statis, feats_train):
    l = args.label
    teacher = args.teacher
    
    tnet.eval()

    score_list = []
    for (x,  _) in (test_loader):
        x= x.cuda()
        with torch.no_grad():
                    
            score_t = utils.get_score(args, x, tnet, feats_train)

            score_t = -(score_t-statis[0])/statis[1]

            score_list.append(score_t)
    test_labels = test_loader.dataset.targets
    scores = torch.cat(score_list, dim=0).detach().cpu().numpy()
    auc = roc_auc_score(test_labels, scores)
    print(f"AUROC: {auc*100:.2f}")
    return auc

def distillation(args, model_t, model_s, train_feature_space, statis, loader, ood_loader, optimizer):
    model_t.eval()
    model_s.train()
    total_num, total_loss = 0, 0
    for (x,  _) in (loader):
        x= x.cuda()
        neg, _ = next(iter(ood_loader))
        neg = neg.cuda()
        x_t = torch.cat([x, neg], dim=0)            

        with torch.no_grad():
            _, features = model_t(x)
            score_t = utils.get_score(args, x, model_t, train_feature_space)

        if args.teacher == "CSI":
            x_t = transform(x_t)          
            pos_t = -(score_t-statis[0])/statis[1]

        else :
            pos_t = (score_t-statis[0])/statis[1]

        score_s, feature_s = model_s(x_t)

        pos_s, neg_s = torch.flatten(score_s).chunk(2, 0)

        #score distillation
        sd_loss = F.mse_loss(pos_s, pos_t)

        #MarginRankingloss
        margin_loss = F.margin_ranking_loss(neg_s, pos_t, torch.ones(args.batch_size).cuda(), args.margin)

        
        loss = sd_loss  + args.oe_lambda * margin_loss 
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_num += x_t.size(0)
        total_loss += loss.item() * x_t.size(0)

        return total_loss/total_num
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--teacher', default="MSCL", type=str, choices=["CSI", "PANDA", "MSCL"])
    parser.add_argument('--mode', default='train', type=str)
    parser.add_argument('--epochs', default=500, type=int, help='number of epochs')
    parser.add_argument('--label', default=1, type=int, help='The normal class')
    parser.add_argument('--exp_id', default=0, type=int, help='The normal class')
    parser.add_argument('--resnet_type', default=152, type=int, help='which resnet to use')
    parser.add_argument('--student_resnet', default=18, type=int, help='which resnet to use')
    parser.add_argument('--pretrained', default=True, type=bool)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--model_name', default='student', type=str)
    parser.add_argument('--scheduler', default='steplr', type=str)
    parser.add_argument('--lr', help='Initial learning rate',default=1e-3, type=float)
    parser.add_argument('--weight_decay', help='Weight decay',default=1e-6, type=float)
    parser.add_argument('--oe_lambda', help='hyperparameter for OE',default=1, type=float)
    parser.add_argument('--margin', default=0.5, type=float)
    parser.add_argument('--dataset', default='cifar10', type=str)



    args = parser.parse_args()
    model_name = args.model_name
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ###Model###
    
    snet = get_model(resnet_type=args.student_resnet, pretrained=args.pretrained).cuda()
    # freeze_parameters(snet)

    #### NN-based method####
    
    base_path = f'models/{args.teacher}/{args.dataset}/one_class_{args.label}'
    if not os.path.isdir(base_path):
        os.makedirs(base_path)
    
    if args.teacher == "CSI":
        classes = 100 if args.dataset=="cifar100" else 10
        tnet = ResNet18(classes).cuda()
        feats_train = utils.precompute_feature(args.dataset, args.label)
    else:
        tnet = get_model(resnet_type=args.resnet_type).cuda()
        feats_train = torch.load(base_path+f'/train_{args.label}.pth')
    
    tnet.load_state_dict(torch.load(base_path+f"/teacher_{args.label}.pth"), strict=False)
    tnet.eval()

    print(f"Model : {args.teacher} {model_name}")

    
    ood_train_loader = dataset.get_outliers_loader(args)
    print(f"Normal Class : {args.label}")

    tnet_param = sum(map(torch.numel, tnet.parameters()))
    snet_param = sum(map(torch.numel, snet.parameters()))
    print(f"Number of Parameters | Teacher : {tnet_param}, Student : {snet_param}")


    optimizer = optim.Adam(snet.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if args.scheduler=='steplr':
        milestones = [350]
        scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=0.1, milestones=milestones)
    elif args.scheduler=='cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, args.epochs, 0)

    
    train_loader, test_loader = dataset.get_loaders(args)

    if not os.path.isfile(base_path+f'/statistics_{args.label}.pickle'):
        score_distribution(args, base_path, tnet, feats_train, train_loader)

    with open(base_path+f'/statistics_{args.label}.pickle', 'rb') as f:
        statis = pickle.load(f)
    

    if args.mode == 'train':
        for epoch in range(1, args.epochs+1):
            loss= distillation(args, tnet, snet, feats_train, statis, train_loader, ood_train_loader, optimizer)

            scheduler.step()
            print(f'Epoch : {epoch}/{args.epochs}, train loss : {loss}')
            
        torch.save(snet.state_dict(), base_path + f'/{model_name}_{args.exp_id}.pth')

    elif args.mode == 'test':
        snet.load_state_dict(torch.load(base_path + f'/{model_name}_{args.exp_id}.pth'), strict=False)

        snet.eval()


        auroc = measure_auroc(args, snet, test_loader)
        print(f"AUROC : {auroc}")

        if not os.path.isdir("results/"):
            os.mkdir("results/")
            
        if not os.path.isfile(f'results/{args.dataset}_{model_name}.csv'):
            with open(f'results/{args.dataset}_{model_name}.csv', 'a', newline='') as f:
                w = csv.writer(f)
                w.writerow(['teacher', 'label', 'exp_id','AUROC'])
        
        with open(f'results/{args.dataset}_{model_name}.csv', 'a', newline='') as f:
            w = csv.writer(f)
            w.writerow([args.teacher, args.label, args.exp_id, f"{auroc:.2f}"])
    



torch.cuda.empty_cache()