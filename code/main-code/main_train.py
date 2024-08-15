import os
import torch
from torch.utils.data import DataLoader
from torchvision import models, datasets, transforms
import torch.nn as nn
import torch.optim as optim
import time
import copy
import sys
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_auc_score
from scripts.model import train_model, train_model2
import scripts.dataset as DATA
import scripts.config as config
import argparse
from efficientnet_pytorch import EfficientNet
from scripts.fc import LabelPredictor, DomainClassifier

sys.path.append("../../code/")
from MAE.model import MAE_Encoder, MAE_Classifier


def main(args):
    modelname = args.modelname
    imagepath = args.imagepath

    label_num, subgroup_num = config.ISIC2019_Age()
    Datasets = DATA.ISIC_Age_Datasets

    encoder = MAE_Encoder(image_size=224, mask_ratio=0, depth=10)
    net = MAE_Classifier(encoder=encoder, num_classes=100)

    if os.path.exists('./modelsaved/%s' % modelname) == False:  
        os.makedirs('./modelsaved/%s' % modelname)
    if os.path.exists('./result/%s' % modelname) == False:  
        os.makedirs('./result/%s' % modelname)


    data_transforms = config.Transforms(modelname)

    print("%s Initializing Datasets and Dataloaders..." % modelname)
    
    transformed_datasets = {}
    transformed_datasets['train'] = Datasets(
        path_to_images=imagepath,
        fold=args.train_data,
        PRED_LABEL=label_num,
        transform=data_transforms['train'])
    transformed_datasets['valid'] = Datasets(
        path_to_images=imagepath,
        fold=args.valid_data,
        PRED_LABEL=label_num,
        transform=data_transforms['valid'])
    transformed_datasets['test'] = Datasets(
        path_to_images=imagepath,
        fold=args.test_data,
        PRED_LABEL=label_num,
        transform=data_transforms['valid'])

    dataloaders = {}
    dataloaders['train'] = torch.utils.data.DataLoader(
        transformed_datasets['train'],
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=24)
    dataloaders['valid'] = torch.utils.data.DataLoader(
        transformed_datasets['valid'],
        batch_size=64,
        shuffle=False,
        num_workers=24)
    dataloaders['test'] = torch.utils.data.DataLoader(
        transformed_datasets['test'],
        batch_size=64,
        shuffle=False,
        num_workers=24)


    if args.modelload_path:
        net.load_state_dict(torch.load('%s' % args.modelload_path , map_location=lambda storage, loc: storage),strict=False)
   
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    net = net.to(device)
    fc_label= LabelPredictor(len(label_num)).to(device)
    fc_domain= DomainClassifier(len(subgroup_num)).to(device)
    fc_label.initialize()
    fc_domain.initialize()

    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=args.learning_rate, betas=(0.9, 0.99),weight_decay=0.03)
    optimizer_label = optim.Adam(filter(lambda p: p.requires_grad, fc_label.parameters()), lr=args.learning_rate, betas=(0.9, 0.99),weight_decay=0.03)
    optimizer_domain = optim.Adam(filter(lambda p: p.requires_grad, fc_domain.parameters()), lr=args.learning_rate, betas=(0.9, 0.99),weight_decay=0.03)


    if len(label_num)>2:
        criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.CrossEntropyLoss()

    gam_d = args.gamma_D
    gam_mmd = args.gamma_MMD
    train_model(net, fc_label, fc_domain,
                dataloaders, label_num, subgroup_num, criterion, 
                optimizer, optimizer_label, optimizer_domain,
                gam_d, gam_mmd, 
                args.num_epochs, modelname, device)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--modelname", type=str, choices=["ISIC2019_Age"], default="ISIC2019_Age")
    parser.add_argument("--architecture", type=str, choices= ["resnet","efficientnet","vmamba"], default="resnet")
    parser.add_argument("--modelload_path", type=str,  default= None)
    parser.add_argument("--imagepath", type=str,  default="./dataset/")
    parser.add_argument("--train_data", type=str, default='isic_train')
    parser.add_argument("--valid_data", type=str, default='isic_valid')
    parser.add_argument("--test_data", type=str, default='isic_test')
    parser.add_argument("--learning_rate", type=float, default=0.0001)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_epochs", type=int, default=1000)
    parser.add_argument("--gamma_D", type=float, default=0.2)
    parser.add_argument("--gamma_MMD", type=float, default=0.8)
    args = parser.parse_args()
    main(args)
