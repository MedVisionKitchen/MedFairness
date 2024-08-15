from cmath import phase
from doctest import FAIL_FAST
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
from torch.autograd import Variable
from scripts.multiAUC import Metric
import numpy
from tqdm import tqdm
from random import sample
from scripts.plot import bootstrap_auc,result_csv,plotimage
from scripts.QuasiPareto_loss import MMDLoss, BatchSpectralShrinkage, Divide
import pynvml
pynvml.nvmlInit()
from prettytable import PrettyTable

def train_model(model, LabelPredictor, DomainClassifier,
                dataloaders, label_num, subgroup_num, criterion, 
                optimizer, optimizer_label, optimizer_domain,
                gam_d, gam_mmd, 
                num_epochs, modelname, device):
    VAL_auc,TEST_auc = 0, 0
    since = time.time()
    train_loss_history, valid_loss_history, test_loss_history= [], [], []
    train_maj_history, train_min_history, test_maj_history, test_min_history,train_auc_history, val_auc_history, test_auc_history = [], [], [], [], [], [], []
    best_model_wts = copy.deepcopy(model.state_dict())
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    V_AUC, T_AUC, G1  = [], [], []
    MMD = MMDLoss()
    BSS = BatchSpectralShrinkage()

    epoch_log_path = "/home/tom/result/CXP_Age/epoch_accuracy.txt"
    everySample_log_path = "/home/tom/result/CXP_Age/everySampleLog.txt"
    epoch_file = open(epoch_log_path, "w")
    everySample_file = open(everySample_log_path, "w")

    for epoch in range(num_epochs):
        start = time.time()
        print() 
        print('{}  Epoch {}/{}  {}'.format('-' * 40, epoch, num_epochs - 1, '-' * 40))
        for phase in ['train','valid', 'test']:
            data_auc, Data_auc_maj, Data_auc_min = 0, 0, 0
            G =[]
            if phase == 'train' and epoch != 0:
                model.train()
            else:
                model.eval()

            accuracy = 0
            correct = 0
            total = 0
            running_loss, Batch, gamma = 0.0, 0, 0.5
            with tqdm(range(len(dataloaders[phase])),desc='%s' % phase, ncols=100) as t:
                if Batch == 0 :
                    t.set_postfix(L = 0.000, G = 0.5, n = 0, S = "0")

                for data in dataloaders[phase]:
                    inputs, labels, subg, imgsName = data
                    inputs = Variable(inputs.to(device))
                    labels = Variable(labels.to(device)).float()
                    # print(labels)
                    subg = Variable(subg.to(device)).float()
                    optimizer.zero_grad(set_to_none=True) 
                    batch_size = inputs.shape[0]
                    with torch.set_grad_enabled(phase == 'train'):
                        # CNN feature extractor
                        outputs_512 = model(inputs)
                        outputs_out = LabelPredictor(outputs_512)
                        outputs_domain = DomainClassifier(outputs_512)
                        try:
                            if phase =='train':
                                outputs_maj_img, outputs_maj_lab, sub_maj, outputs_min_img, outputs_min_lab, sub_min = Divide(outputs_out, labels, subg)

                                myMetic_maj = Metric(outputs_maj_img.cpu().detach().numpy(),outputs_maj_lab.cpu().detach().numpy())
                                data_auc_maj,auc = myMetic_maj.auROC()
                                myMetic_min = Metric(outputs_min_img.cpu().detach().numpy(),outputs_min_lab.cpu().detach().numpy())
                                data_auc_min,auc = myMetic_min.auROC()

                                #my loss
                                loss_y_maj = criterion(outputs_maj_img, outputs_maj_lab).cuda()
                                loss_y_min = criterion(outputs_min_img, outputs_min_lab).cuda()
                                x =  (float(data_auc_maj) - float(data_auc_min)) / 0.05
                                gamma = 1. / (1 + np.exp(x))
                                loss_y = (gamma * loss_y_maj + (1-gamma) * loss_y_min).cuda()
                                
                                loss_d = criterion(outputs_domain, subg).cuda()
                                Loss = loss_y + gam_d*loss_d
                                Loss_state = "my"

                                optimizer.zero_grad(), optimizer_label.zero_grad(), optimizer_domain.zero_grad()
                                Loss.backward()
                                optimizer.step(), optimizer_label.step(), optimizer_domain.step()

                            else:
                                with torch.no_grad():
                                    Loss = criterion(outputs_out, labels).cuda()
                                    Loss_state = "0"
                        except Exception as e:
                            print(f"An error occurred: {e}")
                            print(f"Exception type: {type(e)}")
                            Loss = torch.tensor(0).cuda()
                            Loss_state = "0"
                            print("Error and loss = 0")
                    running_loss += Loss.detach()

                    """
                    B:batch 
                    L:Loss
                    maj: Maj group AUC
                    min: Min group AUC
                    n: NVIDIA Memory used
                    """   
                    gpu_device = pynvml.nvmlDeviceGetHandleByIndex(0)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).total
                    usedMemory = pynvml.nvmlDeviceGetMemoryInfo(gpu_device).used
                    n = usedMemory/meminfo              
                    t.set_postfix(L = Loss.data.item(), G = gamma, n = n, S = Loss_state)  # 
                    t.update()

                    if phase in ["valid", "test"]:
                        outputs_maj_img, outputs_maj_lab, sub_maj, outputs_min_img, outputs_min_lab, sub_min = Divide(outputs_out.cpu(), labels.cpu(), subg.cpu())
                        _, predicted = torch.max(outputs_out.data, 1)
                        
                        predicted_labels = torch.argmax(labels, dim=1)
                        total += labels.size(0)
                        predicted_true = (predicted == predicted_labels)
                        correct += predicted_true.sum().item()
                        for index, imgN in enumerate(imgsName):
                            everySample_file.write(f"Epoch {epoch + 1}, State=={phase}, imageName=={imgN}, predict=={predicted_true[index]}\n")

                    
                    if Batch == 0 :
                        Output = outputs_out.cpu().detach().numpy()
                        Output_maj = outputs_maj_img.cpu().detach().numpy()
                        Output_min = outputs_min_img.cpu().detach().numpy()
                        Label = labels.cpu().detach().numpy()
                        Label_maj = outputs_maj_lab.cpu().detach().numpy()
                        Label_min = outputs_min_lab.cpu().detach().numpy()
                    else:
                        Output = np.concatenate((Output, outputs_out.cpu().detach().numpy()),axis=0)
                        Output_maj = np.concatenate((Output_maj, outputs_maj_img.cpu().detach().numpy()),axis=0)
                        Output_min = np.concatenate((Output_min, outputs_min_img.cpu().detach().numpy()),axis=0)
                        Label = np.concatenate((Label, labels.cpu().detach().numpy()),axis=0)
                        Label_maj = np.concatenate((Label_maj, outputs_maj_lab.cpu().detach().numpy()),axis=0)
                        Label_min = np.concatenate((Label_min, outputs_min_lab.cpu().detach().numpy()),axis=0)

                    Batch += 1


            if phase in ['valid', 'test']:
                accuracy = 100 * correct / total
                # 将 epoch 和对应的 accuracy 写入文件
                epoch_file.write(f"Epoch {epoch + 1}, State=={phase}, Accuracy: {accuracy:.2f}%\n")
                epoch_file.flush()

            myMetic = Metric(Output,Label)
            data_auc,auc = myMetic.auROC()
            data_auc_maj = Metric(Output_maj,Label_maj).auROC()
            data_auc_min = Metric(Output_min,Label_min).auROC()
            epoch_loss = running_loss / Batch
            statistics = bootstrap_auc(Label, Output, [0,1,2,3,4])
            max_auc = np.max(statistics, axis=1).max()
            min_auc = np.min(statistics, axis=1).max()
            if G == [] and phase == "train":
                G1.append(0)
            elif phase == "train":
                G1.append(sum(G)/len(G))

            print('{} --> Num: {} Loss: {:.4f}  Gamma: {:.4f} AUROC: {:.4f} ({:.2f} ~ {:.2f}) (Maj {:.4f}, Min {:.4f})'.format(
            phase, len(outputs_out), epoch_loss, G1[-1], data_auc, min_auc, max_auc, data_auc_maj[0], data_auc_min[0]))

            if phase == 'train':
                train_loss_history.append(epoch_loss)
                train_auc_history.append(data_auc)
                train_maj_history.append(data_auc_maj[0])
                train_min_history.append(data_auc_min[0])

            if phase == 'valid':
                valid_loss_history.append(epoch_loss)
                val_auc_history.append(data_auc)

            if phase == 'test':
                test_loss_history.append(epoch_loss)
                test_auc_history.append(data_auc)
                test_maj_history.append(data_auc_maj[0])
                test_min_history.append(data_auc_min[0])
                best_index = test_auc_history.index(max(test_auc_history))
                print()
                print("Best result : train : {:.4f}, valid : {:.4f}, test : {:.4f}, test_maj : {:.4f}, test_min : {:.4f}".format(train_auc_history[best_index], 
                                                                                                                   val_auc_history[best_index],
                                                                                                                    test_auc_history[best_index],
                                                                                                                    test_maj_history[best_index], 
                                                                                                                    test_min_history[best_index]))

        scheduler.step()
        print("learning rate = %.6f  time: %.1f sec" % (optimizer.param_groups[-1]['lr'], time.time() - start))
        result_csv(train_auc_history, val_auc_history, test_auc_history, test_maj_history, test_min_history, modelname, gam_d, gam_mmd)

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    model.load_state_dict(best_model_wts)

