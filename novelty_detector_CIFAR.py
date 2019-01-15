# Copyright 2018 Stanislav Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

from __future__ import print_function
import torch.utils.data
from torch import optim
from torchvision.utils import save_image
from net import *
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import json
import pickle
import time
import random
from torch.autograd.gradcheck import zero_gradients
import matplotlib
matplotlib.use('agg')
import matplotlib.pyplot as plt
import scipy.stats
import os
from sklearn.metrics import roc_auc_score
from evaluate import evaluate
from cifar_data import load_cifar_data
power = 2.0

device = torch.device("cuda")
use_cuda = torch.cuda.is_available()

FloatTensor = torch.FloatTensor
IntTensor = torch.IntTensor
LongTensor = torch.LongTensor
torch.set_default_tensor_type('torch.FloatTensor')

if use_cuda:
    device = torch.cuda.current_device()
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
    FloatTensor = torch.cuda.FloatTensor
    IntTensor = torch.cuda.IntTensor
    LongTensor = torch.cuda.LongTensor
    print("Running on ", torch.cuda.get_device_name(device))


def setup(x):
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()


def numpy2torch(x):
    return setup(torch.from_numpy(x))


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size]) / 255.0
    #x.sub_(0.5).div_(0.5)
    return Variable(x)
	
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
		
def main():
    batch_size = 64
    z_size = 100
    anomaly_class = 0
    test_epoch = 25
    isize = 64
    workers = 8
    classes = ['plane', 'car', 'bird', 'cat', 'deer','dog', 'frog', 'horse', 'ship', 'truck']
    for i in range(0,1):
        anomaly_class = classes[i]
        print(anomaly_class)
        print('start loading test data')
        dataloader = load_cifar_data(anomaly_class,batch_size,isize,workers)
        length = len(dataloader['test'])
        print("Test set batch number:", length)

        best_roc_auc = 0
        best_prc_auc = 0
        best_f1_score = 0
        best_recall_score = 0
        best_precision_score = 0   
        noise_factor =.25
        print("start testing")
        
        for j in range(0,test_epoch):
            epoch = j
            print("testing epoch is "+ str(epoch))
            G = Generator(z_size)
            E = Encoder(z_size)
            D = Discriminator()
            ZD = ZDiscriminator(z_size, batch_size)

            setup(E)
            setup(G)
            setup(D)
            setup(ZD)

            G.eval()
            E.eval()
            D.eval()
            ZD.eval()

            BCE_loss = nn.BCELoss()
            MSE_loss = nn.MSELoss()
            y_real_ = torch.ones(batch_size)
            y_fake_ = torch.zeros(batch_size)
            
            y_real_z = torch.ones(batch_size)
            y_fake_z = torch.zeros(batch_size)
            model_dir = os.path.join('cifar', str(anomaly_class), 'model')
            G.load_state_dict(torch.load(model_dir+'/Gmodel_epoch{}.pkl'.format(str(epoch))))
            E.load_state_dict(torch.load(model_dir+'/Emodel_epoch{}.pkl'.format(str(epoch))))
            D.load_state_dict(torch.load(model_dir+'/Dmodel_epoch{}.pkl'.format(str(epoch))))
            ZD.load_state_dict(torch.load(model_dir+'/ZDmodel_epoch{}.pkl'.format(str(epoch))))

            directory = 'cifar/'+str(anomaly_class)+'/test'
            if not os.path.exists(directory):
                os.makedirs(directory)
            #sample = setup(torch.randn(64, z_size))
            #sample = setup(G(sample.view(-1, z_size, 1, 1)))
            #save_image(sample.view(64, 1, 32, 32), 'results/{}/test/sample_{}.png'.format(str(anomaly_class), str(epoch)))
            
            
            X_score = []
            Z_score = []
            recons_error = []
            y_label = []
            # length = len(mnist_test_x) // batch_size
            
            for it,data in enumerate(dataloader['test']):
                x, labels = data
                x = Variable(x)
                z = E(x)
                recon_batch = G(z)

                D_result = D(x).squeeze().detach().cpu().numpy()
                X_score.append(D_result)
                
                z = z.view.view(-1, zsize, 1, 1)
                recon_batch = G(z)
            
                comparison = torch.cat([x[:4], recon_batch[:4]])
                save_image(comparison,'cifar/'+str(anomaly_class)+'/test/recon_'+ str(epoch)+'_'+ str(it) + '.png', nrow=4)
                
                ZD_result = ZD(z.squeeze()).squeeze().detach().cpu().numpy()
                Z_score.append(ZD_result)
                
                recon_batch = recon_batch.squeeze().detach().cpu().numpy()
                x = x.squeeze().detach().cpu().numpy()
                z = z.detach().cpu().numpy()
                y_label = y_label.append(labels)
                for i in range(batch_size):
                    distance = np.sum(np.power(recon_batch[i].flatten() - x[i].flatten(), power))
                    recons_error.append(distance)
            print("start calculating")
            X_score = np.array(X_score).reshape(length*batch_size)
            Z_score = np.array(Z_score).reshape(length*batch_size)
            y_label = np.array(y_label).reshape(length*batch_size)
            recons_error = np.array(recons_error).reshape(length*batch_size)
            anomaly_score = recons_error-100*X_score-1*Z_score
            anomaly_score = np.array(anomaly_score)
            anomaly_score = torch.from_numpy(anomaly_score)
            # print(anomaly_score)
            # break
            anomaly_score = (anomaly_score - torch.min(anomaly_score)) / (torch.max(anomaly_score) - torch.min(anomaly_score))
            anomaly_score = anomaly_score.detach().cpu().numpy()
            path = "./cifar/{}/test/".format(str(anomaly_class))
            roc_auc = evaluate(labels= y_label, scores= anomaly_score, directory=path, epoch = epoch, anomaly_class = anomaly_class,metric='roc')
            prc_auc = evaluate(labels= y_label, scores= anomaly_score, directory=path, epoch = epoch, anomaly_class = anomaly_class,metric='auprc')
            f1_score = evaluate(labels= y_label, scores= anomaly_score, directory=path, epoch = epoch, anomaly_class = anomaly_class , metric='f1_score')
            recall = evaluate(labels= y_label, scores= anomaly_score, directory=path, epoch = epoch, anomaly_class = anomaly_class , metric='recall')
            precision = evaluate(labels= y_label, scores= anomaly_score, directory=path, epoch = epoch, anomaly_class = anomaly_class , metric='precision')
            print(roc_auc)
            print(prc_auc)
            print(f1_score)
            print(recall)
            print(precision)

            roc_file_name = os.path.join(path, 'roc_auc.txt')
            with open(roc_file_name, 'a+') as opt_file:
                opt_file.write('---epoch----\n')
                opt_file.write('%s' %(str(epoch))+'\n')
                opt_file.write('----auc---\n')
                opt_file.write('%s' %(str(roc_auc))+'\n')
                opt_file.write('----------\n')

            if roc_auc > best_roc_auc:
                best_roc_auc = roc_auc
                roc_file_name = os.path.join(path, 'best_roc_auc.txt')
                with open(roc_file_name, 'a+') as opt_file:
                    opt_file.write('---epoch----\n')
                    opt_file.write('%s' %(str(epoch))+'\n')
                    opt_file.write('----auc---\n')
                    opt_file.write('%s' %(str(best_roc_auc))+'\n')
                    opt_file.write('----------\n')

            prc_file_name = os.path.join(path, 'prc_auc.txt')
            with open(prc_file_name, 'a+') as opt_file:
                opt_file.write('---epoch----\n')
                opt_file.write('%s' %(str(epoch))+'\n')
                opt_file.write('----prc---\n')
                opt_file.write('%s' %(str(prc_auc))+'\n')
                opt_file.write('----------\n')
            if prc_auc > best_prc_auc:
                best_prc_auc = prc_auc
                prc_file_name = os.path.join(path, 'best_prc_auc.txt')
                with open(prc_file_name, 'a+') as opt_file:
                    opt_file.write('---epoch----\n')
                    opt_file.write('%s' %(str(epoch))+'\n')
                    opt_file.write('----prc---\n')
                    opt_file.write('%s' %(str(best_prc_auc))+'\n')
                    opt_file.write('----------\n')


            f1_file_name = os.path.join(path, 'f1_score.txt')
            with open(f1_file_name, 'a+') as opt_file:
                opt_file.write('---epoch----\n')
                opt_file.write('%s' %(str(epoch))+'\n')
                opt_file.write('----f1---\n')
                opt_file.write('%s' %(str(f1_score))+'\n')
                opt_file.write('----------\n')
            if f1_score > best_f1_score:
                best_f1_score = f1_score
                f1_file_name = os.path.join(path, 'best_f1_score.txt')
                with open(f1_file_name, 'a+') as opt_file:
                    opt_file.write('---epoch----\n')
                    opt_file.write('%s' %(str(epoch))+'\n')
                    opt_file.write('----f1---\n')
                    opt_file.write('%s' %(str(best_f1_score))+'\n')
                    opt_file.write('----------\n')
            
            recall_file_name = os.path.join(path, 'recall.txt')
            with open(recall_file_name, 'a+') as opt_file:
                opt_file.write('---epoch----\n')
                opt_file.write('%s' %(str(epoch))+'\n')
                opt_file.write('----recall---\n')
                opt_file.write('%s' %(str(recall))+'\n')
                opt_file.write('----------\n')

            if recall > best_recall_score:
                best_recall_score = recall
                recall_file_name = os.path.join(path, 'best_recall.txt')
                with open(recall_file_name, 'a+') as opt_file:
                    opt_file.write('---epoch----\n')
                    opt_file.write('%s' %(str(epoch))+'\n')
                    opt_file.write('----recall---\n')
                    opt_file.write('%s' %(str(best_recall_score))+'\n')
                    opt_file.write('----------\n')
            
            precision_file_name = os.path.join(path, 'precision.txt')
            with open(precision_file_name, 'a+') as opt_file:
                opt_file.write('---epoch----\n')
                opt_file.write('%s' %(str(epoch))+'\n')
                opt_file.write('----precision---\n')
                opt_file.write('%s' %(str(precision))+'\n')
                opt_file.write('----------\n')

            if precision > best_precision_score:
                best_precision_score = precision
                precision_file_name = os.path.join(path, 'best_precision.txt')
                with open(precision_file_name, 'a+') as opt_file:
                    opt_file.write('---epoch----\n')
                    opt_file.write('%s' %(str(epoch))+'\n')
                    opt_file.write('----precision---\n')
                    opt_file.write('%s' %(str(best_precision_score))+'\n')
                    opt_file.write('----------\n')
            
    # print(anomaly_score)
if __name__ == '__main__':
    main()
