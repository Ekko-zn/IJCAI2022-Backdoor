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


if opt.model_type == 'resnet18':
    Classifer = Models.resnet18()
    Classifer.fc = nn.Linear(512, opt.num_class)

Classifer = Classifer.cuda()


if opt.dataset == 'GTSRB':
    traindataloader = TrainDataloaderGTSRB
    valdataloader = TestDataloaderGTSRB
elif opt.dataset == 'CelebA':
    traindataloader = TrainDataloaderCelebA
    valdataloader = TestDataloaderCelebA


EmbbedNet = Embbed()
EmbbedNet = EmbbedNet.cuda()
TriggerNet = Models.U_Net()
TriggerNet = TriggerNet.cuda()
Target_labels = torch.stack([i*torch.ones(1) for i in range(opt.num_class)]).expand(
    opt.num_class, opt.batch_size).permute(1, 0).reshape(-1, 1).squeeze().to(dtype=torch.long, device='cuda')
optimizer_net = torch.optim.Adam(
    Classifer.parameters(), lr=opt.lr_optimizer_for_c,weight_decay=opt.weight_decay)
optimizer_map = torch.optim.Adam(
    TriggerNet.parameters(), lr=opt.lr_optimizer_for_t)


recoder_train = Recorder()
recoder_val = Recorder()
epoch_start = 0


def Train(dataset, feature_r, recoder):
    Classifer.train()
    TriggerNet.train()
    for fs, labels in tqdm(dataset):
        fs = fs.to(dtype=torch.float).cuda()
        fs_copy = copy.deepcopy(fs)
        Triggers = TriggerNet(fs)
        Triggersl2norm = torch.mean(torch.abs(Triggers))
        Triggers = EmbbedNet(Triggers[:, 0:3*opt.num_class, :, :],
                             Triggers[:, 3*opt.num_class:6*opt.num_class, :, :])
        Triggers = (Triggers)/255
        Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
        fs = fs.unsqueeze(1).expand(fs.shape[0], opt.num_class, 3, opt.image_size,
                                    opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        fs_poison = fs + Triggers
        labels = labels.to(dtype=torch.long).cuda().squeeze()
        imgs_input = torch.cat((fs_copy, fs_poison), 0)
        optimizer_net.zero_grad()
        optimizer_map.zero_grad()
        out, f = Classifer(imgs_input)
        loss_f = MAE(f[fs_copy.shape[0]::,:],feature_r)
        loss_ori = criterion(out[0:labels.shape[0], :], labels)
        loss_p = criterion(out[labels.shape[0]::], Target_labels)
        loss = loss_ori + loss_p + loss_f * opt.a + Triggersl2norm * opt.b
        loss.backward()
        optimizer_net.step()
        optimizer_map.step()
        out_ori = out[0:labels.shape[0], :]
        out_p = out[labels.shape[0]::, :]
        _, predicts_ori = out_ori.max(1)
        recoder.train_acc[0] += predicts_ori.eq(labels).sum().item()
        _, predicts_p = out_p.max(1)
        recoder.train_acc[1] += predicts_p.eq(Target_labels).sum().item()
        recoder.train_loss[0] += loss_ori.item()
        recoder.train_loss[1] += loss_p.item()
        recoder.count[0] += labels.shape[0]
        recoder.count[1] += Target_labels.shape[0]
    if opt.to_print == 'True':
        print('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}'.format(
            recoder.train_loss[0]/len(dataset), recoder.train_acc[0] / recoder.count[0], recoder.train_loss[1]/len(
                dataset), recoder.train_acc[1] / recoder.count[1]
        ))
    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            f.write('Train model: Clean loss:{:.4f} Clean acc:{:.2f} Poison loss:{:.4f} Poison acc:{:.2f}\n'.format(
                recoder.train_loss[0]/len(dataset), recoder.train_acc[0] / recoder.count[0], recoder.train_loss[1]/len(
                    dataset), recoder.train_acc[1] / recoder.count[1]
            ))
    recoder.ac()


def Eval_normal(dataset, recoder):
    Classifer.eval()
    TriggerNet.eval()
    Correct = 0
    Loss = 0
    Tot = 0
    for fs, labels in dataset:
        fs = fs.to(dtype=torch.float).cuda()
        labels = labels.to(dtype=torch.long).cuda(
        ).view(-1, 1).squeeze().squeeze()
        out, _ = Classifer(fs)
        loss = criterion(out, labels)
        _, predicts = out.max(1)
        Correct += predicts.eq(labels).sum().item()
        Loss += loss.item()
        Tot += fs.shape[0]
    recoder.currect_val_acc = 100*Correct/Tot
    recoder.moving_normal_acc.append(recoder.currect_val_acc)
    if 100*Correct/Tot > recoder.best_acc:
        recoder.best_acc = 100*Correct/Tot
    if opt.to_print == 'True':
        print('Eval-normal Loss:{:.3f} Train Acc:{:.2f} Best Acc:{:.2f}'.format(
            Loss/len(dataset), 100*Correct/Tot, recoder.best_acc))
    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            f.write('Eval-normal Loss:{:.3f} Train Acc:{:.2f} Best Acc:{:.2f}\n'.format(
                Loss/len(dataset), 100*Correct/Tot, recoder.best_acc)
            )


def Eval_poison(dataset,feature_r, recoder):
    Classifer.eval()
    TriggerNet.eval()
    Correct = 0
    Loss = 0
    Tot = 0
    L1 = 0
    LF = 0
    for fs, labels in dataset:
        fs = fs.to(dtype=torch.float).cuda()
        Triggers = TriggerNet(fs)
        Triggers = EmbbedNet(Triggers[:, 0:3*opt.num_class, :, :],
                             Triggers[:, 3*opt.num_class:6*opt.num_class, :, :])
        Triggers = torch.round(Triggers)/255
        fs = fs.unsqueeze(1).expand(fs.shape[0], opt.num_class, 3, opt.image_size,
                                    opt.image_size).reshape(-1, 3, opt.image_size, opt.image_size)
        Triggers = Triggers.reshape(-1, 3, opt.image_size, opt.image_size)
        fs = fs + Triggers
        fs = torch.clip(fs, min=0, max=1)
        out, f = Classifer(fs)
        loss_f = MAE(f,feature_r)
        loss = criterion(out, Target_labels)
        _, predicts = out.max(1)
        Correct += predicts.eq(Target_labels).sum().item()
        Loss += loss.item()
        Tot += fs.shape[0]
        L1 += torch.sum(torch.abs(Triggers*255)).item()
        LF += loss_f.item()
    Acc = 100*Correct/Tot
    l1_norm = L1/(Tot*3*opt.image_size*opt.image_size)
    LF = LF / len(dataset)
    recoder.moving_poison_acc.append(Acc)
    recoder.moving_l1norm.append(l1_norm)
    if len(recoder.moving_l1norm) > 5:
        recoder.moving_poison_acc.pop(0)
        recoder.moving_normal_acc.pop(0)
        recoder.moving_l1norm.pop(0)
    if opt.to_print == 'True':
        print('Eval-poison Loss:{:.3f} Train Acc:{:.2f} L1 norm:{:.4f} HyperPara a:{} HyperPara b:{} L-f:{:.4f}  Moving Normal Acc:{:.2f} Moving Poison Acc:{:.2f} Moving L1 norm:{:.2f}'.format(
            Loss/len(dataset), Acc, l1_norm, opt.a, opt.b, LF,np.mean(recoder.moving_normal_acc), np.mean(recoder.moving_poison_acc), np.mean(recoder.moving_l1norm)))
    if opt.to_save == 'True':
        with open(opt.logname, 'a+') as f:
            f.write('Eval-poison Loss:{:.3f} Train Acc:{:.2f} L1 norm:{:.4f} HyperPara a:{} HyperPara b:{}  L-f:{:.4f} Moving Normal Acc:{:.2f} Moving Poison Acc:{:.2f} Moving L1 norm:{:.2f}\n'.format(
                Loss/len(dataset), Acc, l1_norm, opt.a, opt.b, LF, np.mean(recoder.moving_normal_acc), np.mean(recoder.moving_poison_acc), np.mean(recoder.moving_l1norm)))


def ref_f(dataset):
    Classifer.eval()
    F = {}
    F_out = []
    for ii in range(opt.num_class):
        F[ii] = []
    for fs,labels in (dataset):
        fs = fs.to(dtype=torch.float).cuda()
        labels = labels.to(dtype=torch.long).cuda(
        ).view(-1, 1).squeeze().squeeze()
        out, features = Classifer(fs)
        for ii in (range(fs.shape[0])):
            label = labels[ii].item()
            F[label].append(features[ii,:].detach().cpu()) 
    for ii in range(opt.num_class):
        F[ii] = torch.stack(F[ii]).mean(dim=0).unsqueeze(0)
        dim_f = F[ii].shape[1]
        F[ii] = F[ii].expand(opt.batch_size,dim_f)
        F_out.append(F[ii])
    F_out = torch.stack(F_out)
    F_out = F_out.permute(1,0,2).reshape(-1,dim_f)
    return F_out.cuda()




if __name__ == '__main__':
    opt.logpath = './log/{}_{}_{}/'.format(opt.dataset,
                                           opt.model_type, opt.tag)
    if not os.path.exists(opt.logpath):
        os.makedirs(opt.logpath)
    opt.logname = opt.logpath+'log.txt'
    with open(opt.logname, 'w+') as f:
        f.write('start \n')
    for epoch in range(1, 120+1):
        if epoch % 20 == 0:
            opt.b *= 2
        start = time()
        print('epoch:{}'.format(epoch))
        with open(opt.logname, 'a+') as f:
            f.write('epoch:{}\n'.format(epoch))
        with torch.no_grad():
            feature_r = ref_f(traindataloader)
        Train(traindataloader, feature_r, recoder_train)

        with torch.no_grad():
            Eval_normal(valdataloader, recoder_val)
            Eval_poison(valdataloader, feature_r, recoder_val)
        paras = {
            'netC':Classifer.state_dict(),
            'netP':TriggerNet.state_dict()
        }
        torch.save(paras,opt.logpath+str(epoch)+'.pth')
        end = time()
        print('cost time:{:.2f}s'.format(end-start))
