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
from net_mnist import *
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


def extract_batch_(data, it, batch_size):
    x = data[it * batch_size:(it + 1) * batch_size]
    return x
def test(p):
    percentage = p
    print("outlier percentage: "+ str(percentage) )
    for i in range(0,10):
        #######################################################
        inlier_class = i
        print('inlier_class is '+str(inlier_class))
        total_classes = 10
        outlier_classes = []
        for i in range(total_classes):
            if i != inlier_class:
                outlier_classes.append(i)
        #print('outlier_classes is')
        #print(outlier_classes)
        #######################################################
        # print("start loading test data")
        ## load inlier
        with open('data/mnist_{}_{}.pkl'.format(str(inlier_class),'train'), 'rb') as pkl:
            mnist_train = pickle.load(pkl)
        mnist_train = [x for x in mnist_train if x[0] == inlier_class ]
        # random.seed(0)
        random.shuffle(mnist_train)

        ##load outlier
        with open('data/mnist_{}_{}.pkl'.format(str(inlier_class),'test'), 'rb') as pkl:
            mnist_test = pickle.load(pkl)
        # random.shuffle(mnist_test)

        ## combine them into test dataset
        inliner_count = len(mnist_train)
        outlier_count = inliner_count * percentage // (100 - percentage)
        mnist_test_outlier = mnist_test[:outlier_count]
        
        mnist_test = mnist_test_outlier + mnist_train
        
        # random.shuffle(mnist_test)

        def list_of_pairs_to_numpy(l):
            return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)
        mnist_test_x, mnist_test_y = list_of_pairs_to_numpy(mnist_test)
        # print(len(mnist_test_y))
        #print(mnist_test_y)
        #print("data loaded success")
        #######################################################
        # print("Test set size:", len(mnist_test_x))
        #print("label dealing")
        for i in range(mnist_test_y.shape[0]):
            if mnist_test_y[i] == inlier_class:
                mnist_test_y[i] = 10
            else:
                mnist_test_y[i] = 0
        for i in range(mnist_test_y.shape[0]):
            if mnist_test_y[i] == 10:
                mnist_test_y[i] = 1
        # print(mnist_test_y)
        #######################################################
        # print("start testing")
        batch_size = 128
        mnist_train = []
        z_size = 32
        test_epoch = 15
        
        best_roc_auc = 0
        best_prc_auc = 0
        best_f1_score = 0

        for j in range(0,test_epoch):
            epoch = j
            #print("testing epoch is "+ str(epoch))
            G = Generator(z_size)
            E = Encoder(z_size)
            D = Discriminator()
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
            model_dir = os.path.join('Model/MNIST', str(inlier_class))
            G.load_state_dict(torch.load(model_dir+'/Gmodel_epoch{}.pkl'.format(str(epoch))))
            E.load_state_dict(torch.load(model_dir+'/Emodel_epoch{}.pkl'.format(str(epoch))))
            D.load_state_dict(torch.load(model_dir+'/Dmodel_epoch{}.pkl'.format(str(epoch))))
            ZD.load_state_dict(torch.load(model_dir+'/ZDmodel_epoch{}.pkl'.format(str(epoch))))
            P.load_state_dict(torch.load(model_dir+'/Pmodel_epoch{}.pkl'.format(str(epoch))))
            C.load_state_dict(torch.load(model_dir+'/Cmodel_epoch{}.pkl'.format(str(epoch))))
            
            X_score = []
            length = len(mnist_test_x) // batch_size
            
            for it in range(length):
                x = Variable(extract_batch(mnist_test_x, it, batch_size).view(-1, 1, 32, 32))
                ###########D(x)
                # D_score, _ = D(x)
                # D_score = D_score.reshape((batch_size, 1))
                # _, D_score = P(D_score)
                # D_result = D_score.squeeze().detach().cpu().numpy()
                # X_score.append(D_result)
                #############D(R(X))
                z = E(x)
                x_fake = G(z).detach()
                D_fake, _ = D(x_fake)
                D_fake = D_fake.reshape((batch_size, 1))
                _, D_fake = P(D_fake)
                D_fake = D_fake.squeeze().detach().cpu().numpy()
                X_score.append(D_fake)


            X_score = np.array(X_score).reshape(length*batch_size, 2)
            anomaly_score = X_score
            
            labels = mnist_test_y[0:length*batch_size]
            binary_class_labels = np.zeros((length*batch_size, 2))
            for i in range(len(labels)):
                binary_class_labels[i, 1-labels[i]] = 1.
            path = "./Test/MNIST/{}/".format(str(inlier_class))
            f1_score = evaluate(labels= labels, scores= anomaly_score, directory=path, metric='f1_score')
            if f1_score>best_f1_score:
              best_f1_score = f1_score
        print(best_f1_score)
    return None
def main():  
    percentages = [10, 20, 30, 40, 50]
    # percentages = [50]
    results = {}

    for p in percentages:
        results[p] = test(p)

if __name__ == '__main__':
    main()
