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
from net_cifar import *
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
from OC256 import load_OC_train_data, load_OC_test_data

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
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size])
    #x.sub_(0.5).div_(0.5)
    return Variable(x)

def next_batch(data, batch_size):
    length = len(data)
    for i in range(length // batch_size):
        start = i * batch_size
        end = (i + 1) * batch_size
        yield data[start:end]

def extract_batch_(data, it, batch_size):
    x = data[it * batch_size:(it + 1) * batch_size]
    return x

def test():
    # percentage = p
    # print("outlier percentage: "+ str(percentage) )
    for i in range(1):
        #######################################################
        inliner_data = load_OC_train_data(1, 32, load_flag=True)
        outliner_data = load_OC_test_data(32)[:len(inliner_data)]
        test_x = inliner_data + outliner_data
        test_y = [1 for i in range(len(inliner_data))]\
                +[0 for i in range(len(outliner_data))]
        #######################################################
        # print("start testing")
        batch_size = 128
        z_size = 100
        test_epoch = 500
        
        best_roc_auc = 0
        best_prc_auc = 0
        best_f1_score = 0
        best_R_f1_score = 0
        best_R_roc_auc = 0
        for j in range(0,test_epoch):
            epoch = j
            
            G = Generator(z_size, channels=3)
            E = Encoder(z_size, channels=3)
            D = Discriminator(channels=3)
            ZD = ZDiscriminator(z_size, batch_size)
            P = PDense(1, 128, 2)
            C = Dense(1, 128, 1, batch_size)

            setup(E)
            setup(G)
            setup(D)
            setup(ZD)
            setup(P)
            setup(C)

            G.eval()
            E.eval()
            D.eval()
            ZD.eval()
            P.eval()
            C.eval()

            BCE_loss = nn.BCELoss()
            y_real_ = torch.ones(batch_size)
            y_fake_ = torch.zeros(batch_size)
            
            y_real_z = torch.ones(batch_size)
            y_fake_z = torch.zeros(batch_size)
            
            model_dir = os.path.join('Model', 'Caltech256')
            G.load_state_dict(torch.load(model_dir+'/Gmodel_epoch{}.pkl'.format(str(epoch))))
            E.load_state_dict(torch.load(model_dir+'/Emodel_epoch{}.pkl'.format(str(epoch))))
            D.load_state_dict(torch.load(model_dir+'/Dmodel_epoch{}.pkl'.format(str(epoch))))
            ZD.load_state_dict(torch.load(model_dir+'/ZDmodel_epoch{}.pkl'.format(str(epoch))))
            P.load_state_dict(torch.load(model_dir+'/Pmodel_epoch{}.pkl'.format(str(epoch))))
            C.load_state_dict(torch.load(model_dir+'/Cmodel_epoch{}.pkl'.format(str(epoch))))

            directory = 'Test/Caltech'
            if not os.path.exists(directory):
                os.makedirs(directory)
            X_score = []
            X_R_score = []
            length = len(test_x) // batch_size
            
            for batch in next_batch(test_x, batch_size):
                x = Variable(numpy2torch(np.array(batch))).view(-1, 3, 32, 32)
                ###########D(x)
                D_score, _ = D(x)
                D_score = D_score.reshape((batch_size, 1))
                _, D_score = P(D_score)
                D_result = D_score.squeeze().detach().cpu().numpy()
                X_score.append(D_result)

                #############D(R(X))
                z = E(x).view(-1, z_size, 1, 1)
                x_fake = G(z).detach()
                D_fake, _ = D(x_fake)
                D_fake = D_fake.reshape((batch_size, 1))
                _, D_fake = P(D_fake)
                D_fake = D_fake.squeeze().detach().cpu().numpy()
                X_R_score.append(D_fake)
            # print("start calculating")
            X_score = np.array(X_score).reshape(length*batch_size, 2)
            X_R_score = np.array(X_R_score).reshape(length*batch_size, 2)
            #print(X_score)
            anomaly_score = X_score
            labels = test_y[0:length*batch_size]
            binary_class_labels = np.zeros((length*batch_size, 2))
            for i in range(len(labels)):
                binary_class_labels[i, 1-labels[i]] = 1.
            path = "./Test/Caltech/"
            roc_auc = evaluate(labels= binary_class_labels, scores= anomaly_score, directory=path, metric='roc')
            f1_score = evaluate(labels= labels, scores= anomaly_score, directory=path, metric='f1_score')
            R_roc_auc = evaluate(labels= binary_class_labels, scores= X_R_score, directory=path, metric='roc')
            R_f1_score = evaluate(labels= labels, scores= X_R_score, directory=path, metric='f1_score')
            # print(roc_auc, f1_score)
            if f1_score > best_f1_score:
              best_f1_score = f1_score
            if roc_auc > best_roc_auc:
              best_roc_auc = roc_auc
            if R_f1_score > best_R_f1_score:
              best_R_f1_score = R_f1_score
            if R_roc_auc > best_R_roc_auc:
              best_R_roc_auc = R_roc_auc
        print(best_roc_auc)
        print(best_f1_score)
        print(best_R_roc_auc)
        print(best_R_f1_score)
        
        
        path = "./Test/Caltech/"
        f1_file_name = os.path.join(path, 'result.txt')
        with open(f1_file_name, 'a+') as opt_file:
            opt_file.write('---best_roc_auc----\n')
            opt_file.write('%s' %(str(best_roc_auc))+'\n')
            opt_file.write('---best_f1_score----\n')
            opt_file.write('%s' %(str(best_f1_score))+'\n')
            opt_file.write('---best_R_roc_auc----\n')
            opt_file.write('%s' %(str(best_R_roc_auc))+'\n')
            opt_file.write('---best_R_f1_score----\n')
            opt_file.write('%s' %(str(best_R_f1_score))+'\n')
            opt_file.write('----------\n')
    return None
def main():  
    test()
if __name__ == '__main__':
    main()
