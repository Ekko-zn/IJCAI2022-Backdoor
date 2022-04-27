from config import opt
from utils import *
import torch
import torch.nn as nn
import Models
import numpy as np
import copy
from tqdm import tqdm
import os
from time import time
from EmbedModule import Embbed
import torchvision

opt.model_type = 'resnet18'

if opt.model_type == 'resnet18':
    Classifer = Models.resnet18()
    Classifer.fc = nn.Linear(512, opt.num_class)

if opt.dataset == 'GTSRB':
    traindataset = TrainDatasetGTSRB
    valdataset = TestDatasetGTSRB
    
elif opt.dataset == 'CelebA':
    traindataset = TrainDatasetCelebA
    valdataset = TestDatasetCelebA

valdataloader = DataLoader(valdataset,batch_size=opt.batch_size,shuffle=False,num_workers=opt.num_workers,drop_last=True)

EmbbedNet = Embbed()
EmbbedNet = EmbbedNet.cuda()
TriggerNet = Models.U_Net()

Target_labels = 0*torch.ones(1,1).squeeze().to(dtype=torch.long, device='cuda').unsqueeze(0)
if opt.dataset == 'GTSRB':
    checkpoiont = torch.load('./well_trained/{}.pth'.format(opt.dataset, opt.model_type))
elif opt.dataset == 'CelebA':
    checkpoiont = torch.load('./well_trained/{}.pth'.format(opt.dataset, opt.model_type))
Classifer.load_state_dict(checkpoiont['netC'])
TriggerNet.load_state_dict(checkpoiont['netP'])
Classifer = Classifer.cuda()
TriggerNet = TriggerNet.cuda()




def Eval_poison(dataset,ii):
    Classifer.eval()
    TriggerNet.eval()
    Correct = 0
    Loss = 0
    Tot = 0
    L1 = 0
    LF = 0
    count = 0
    for fs, labels in tqdm(dataset):
        # fs = fs.to(dtype=torch.float).cuda().unsqueeze(0)
        fs = fs.to(dtype=torch.float).cuda()
        Triggers = TriggerNet(fs)
        Triggers = EmbbedNet(Triggers[:, 0:3*opt.num_class, :, :],
                             Triggers[:, 3*opt.num_class:6*opt.num_class, :, :])
        Triggers = torch.round(Triggers)/255
        # Triggers = Triggers[:,0:3,:,:]
        Triggers = Triggers[:,ii*3:(ii+1)*3,:,:]
        fs_zn = fs + Triggers
        fs_zn = torch.clip(fs_zn, min=0, max=1)
        out, f = Classifer(fs_zn)
        loss = criterion(out, Target_labels)
        _, predicts = out.max(1)
        Correct += predicts.eq(Target_labels).sum().item()
        Loss += loss.item()
        Tot += fs.shape[0]
        L1 += torch.sum(torch.abs(Triggers*255)).item()
    Acc = 100*Correct/Tot
    l1_norm = L1/(Tot*3*opt.image_size*opt.image_size)
    print('Attack success rate : {:.2f}'.format(Acc))
    print('L1-norm:{:.5f}'.format(l1_norm))
    
if __name__=='__main__':
    for ii in range(opt.num_class):
        Target_labels = ii*torch.ones(opt.batch_size).to(dtype=torch.long, device='cuda')
        with torch.no_grad():
            Eval_poison(valdataloader,ii)
