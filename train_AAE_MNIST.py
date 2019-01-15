# Copyright 2018 Stanis/av Pidhorskyi
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#  http://www.apache.org/licenses/LICENSE-2.0
# 

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
import os
# from data import load_data
from skimage.util import random_noise

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

def get_noisy_data(data):
    lst_noisy = []
    sigma = 0.155
    for image in data:
        noisy = random_noise(image, var=sigma ** 2)
        lst_noisy.append(noisy)
    return np.array(lst_noisy)

def setup(x):
    if use_cuda:
        return x.cuda()
    else:
        return x.cpu()


def numpy2torch(x):
    return setup(torch.from_numpy(x))


def extract_batch(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size, :, :]) / 255.0
    #x.sub_(0.5).div_(0.5)
    return Variable(x)


def extract_batch_label(data, it, batch_size):
    x = numpy2torch(data[it * batch_size:(it + 1) * batch_size])
    #x.sub_(0.5).div_(0.5)
    return Variable(x)
# def train(batch_size,zsize,anomaly_class):
def Cutout(n_holes, length,train_set,train_label):
    # print(len(train_set))
    train = []
    label = []

    for i,img in enumerate(train_set):
        p = random.random()
        # print(p)
        if p > 0.5:
            h = img.shape[0]
            w = img.shape[1]
            mask = np.ones((h, w), np.float32)

            for n in range(n_holes):
                y = np.random.randint(h)
                x = np.random.randint(w)
                y1 = np.clip(y - length // 2, 0, h)
                y2 = np.clip(y + length // 2, 0, h)
                x1 = np.clip(x - length // 2, 0, w)
                x2 = np.clip(x + length // 2, 0, w)
                mask[y1: y2, x1: x2] = 0.

            mask = torch.from_numpy(mask)
            img = torch.from_numpy(img)
            mask = mask.expand_as(img)
            img = img * mask
            img = img.numpy()
            train.append(img)
            ## cutout label = 0 false label
            label.append(1.)
        else: 
            train.append(img)
            ## not cutout label = 1 true label
            label.append(1.)
    train = np.array(train)
    label = np.array(label)
    # print(label)
    return train,label

def main():
    # new hyperparameter
    lambd = 0.5
    batch_size = 128
    zsize = 32
    anomaly_class = 0
    cutout = True # whether cover the img
    n_holes = 1 # number of holes to cut out from image
    length = 32 # length of the holes
    TINY = 1e-15
    for i in range(0,10):
        anomaly_class = i
        print('start loading data')

        with open('data/mnist/mnist_{}_{}.pkl'.format(str(anomaly_class),'train'), 'rb') as pkl:
            fold = pickle.load(pkl)
        mnist_train = fold
        
        def list_of_pairs_to_numpy(l):
            return np.asarray([x[1] for x in l], np.float32), np.asarray([x[0] for x in l], np.int)

        print("Train set size:", len(mnist_train))

        mnist_train_x, mnist_train_y = list_of_pairs_to_numpy(mnist_train)
        # print(mnist_train_x)
        #cutout
        if cutout:
           print("start randomly cutout")
           mnist_train_x, mnist_train_y = Cutout(n_holes, length, mnist_train_x, mnist_train_y)
        
        train_epoch = 15

        G = Generator(zsize)
        setup(G)
        G.weight_init(mean=0, std=0.02)

        D = Discriminator()
        setup(D)
        D.weight_init(mean=0, std=0.02)

        P = PDense(1, 128, 2)
        setup(P)
        P.weight_init(mean=0, std=0.01)

        C = Dense(1, 128, 1, batch_size)
        setup(C)
        C.weight_init(mean=0, std=0.01)

        E = Encoder(zsize)
        setup(E)
        E.weight_init(mean=0, std=0.02)

        ZD = ZDiscriminator(zsize, batch_size)
        setup(ZD)
        ZD.weight_init(mean=0, std=0.02)

        lr = 0.002

        G_optimizer = optim.Adam(G.parameters(), lr=lr, betas=(0.5, 0.9))
        D_optimizer = optim.Adam(list(D.parameters())\
                                +list(P.parameters())\
                                +list(C.parameters()), lr=lr, betas=(0.5, 0.9))
        E_optimizer = optim.Adam(E.parameters(), lr=lr, betas=(0.5, 0.9))
        GE_optimizer = optim.Adam(list(E.parameters()) + list(G.parameters()), lr=lr, betas=(0.5, 0.9))
        ZD_optimizer = optim.Adam(ZD.parameters(), lr=lr, betas=(0.5, 0.9))

        BCE_loss = nn.BCELoss()
        y_real_ = torch.ones(batch_size)
        y_fake_ = torch.zeros(batch_size)
        
        y_real_z = torch.ones(batch_size)
        y_fake_z = torch.zeros(batch_size)

        sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)
        
        for epoch in range(train_epoch):
            G.train()
            D.train()
            P.train()
            C.train()
            E.train()
            ZD.train()
            epoch_start_time = time.time()
            def shuffle(X):
                np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)

            shuffle(mnist_train_x)
            
            length = len(mnist_train_x) // batch_size
            for it in range(length):
                x = extract_batch(mnist_train_x, it, batch_size).view(-1, 1, 32, 32)
                y = extract_batch_label(mnist_train_y, it, batch_size).view(-1)             
                binary_class_y_real = torch.zeros(batch_size, 2)
                for i in range(batch_size):
                    binary_class_y_real[i][0] = 0.8
                    binary_class_y_real[i][1] = 0.2
                
                binary_class_y_fake = torch.zeros(batch_size, 2)
                for i in range(batch_size):
                    binary_class_y_fake[i][0] = 0.2
                    binary_class_y_fake[i][1] = 0.8
                #############################################

                ZD.zero_grad()
                E.zero_grad()

                z = torch.randn((batch_size, zsize)).view(-1, zsize)
                z = Variable(z)

                ZD_result = ZD(z).squeeze()
                ZD_real_loss = BCE_loss(ZD_result, y_real_z )

                z = E(x).squeeze().detach()

                ZD_result = ZD(z).squeeze()
                ZD_fake_loss = BCE_loss(ZD_result, y_fake_z)

                ZD_train_loss = ZD_real_loss + ZD_fake_loss
                ZD_train_loss.backward()

                ZD_optimizer.step()

                #############################################

                E.zero_grad()
                G.zero_grad()

                z = E(x)
                x_d = G(z)

                ZD_result = ZD(z.squeeze()).squeeze()

                E_loss = BCE_loss(ZD_result, y_real_z ) * 2.0

                Recon_loss = F.binary_cross_entropy(x_d,x)

                (Recon_loss + E_loss).backward()

                GE_optimizer.step()
                #############################################
                for i in range(1):
                    D.zero_grad()
                    G.zero_grad()
                    E.zero_grad()
                    # TODO
                    D_real, _ = D(x)
                    D_real = D_real.reshape((batch_size, 1))
                    z = E(x)
                    x_fake = G(z).detach()
                    D_fake, _ = D(x_fake)
                    D_fake = D_fake.reshape((batch_size, 1))
                        
                    _, P_real = P(D_real)
                    _, C_real = C(D_real)
                    eps = 1e-12
                    P_real = torch.clamp(P_real, 0. + eps, 1. - eps)
                    C_real = torch.clamp(C_real, 0. + eps, 1. - eps)
                    # binary_class_y_real = torch.zeros(batch_size, 2)
                    # binary_class_y_real[:, 1] = 1.
                    P_real_prim = P_real * C_real + (1. - C_real) * binary_class_y_real
                    D_real_loss = torch.sum(- torch.log(P_real_prim + TINY) * binary_class_y_real, 1).reshape((batch_size, 1))\
                                  - lambd * torch.log(C_real + TINY)
                    
                    _, P_fake = P(D_fake)
                    _, C_fake = C(D_fake)
                    P_fake = torch.clamp(P_fake, 0. + eps, 1. - eps)
                    C_fake = torch.clamp(C_fake, 0. + eps, 1. - eps)
                    # binary_class_y_fake = torch.zeros(batch_size, 2)

                    # ### 
                    # binary_class_y_fake[:, 0] = 1.
                    P_fake_prim = P_fake * C_real + (1. - C_fake) * binary_class_y_fake
                    D_fake_loss = torch.sum(- torch.log(P_fake_prim + TINY) * binary_class_y_fake, 1).reshape((batch_size, 1))\
                                  - lambd * torch.log(C_fake + TINY)

                    D_train_loss = torch.mean(D_real_loss + TINY) \
                                   + torch.mean(D_fake_loss + TINY) 
                    D_train_loss.backward()

                    D_optimizer.step()

                #############################################
                for i in range(1):
                    G.zero_grad()
                    E.zero_grad()
                    
                    z = E(x)
                    x_fake = G(z).detach()
                    D_result, _ = D(x_fake)
                    D_result = D_result.reshape((batch_size, 1))

                    _, P_result = P(D_result)
                    _, C_result = C(D_result)
                    ########
                    P_result_prim = P_result * C_result + (1. - C_result) * binary_class_y_real 
                    D_result_loss = torch.sum(- torch.log(P_result_prim + TINY) * binary_class_y_real, 1) \
                                    - lambd * torch.log(C_result + TINY)

                    G_train_loss = torch.mean(D_result_loss + TINY)#BCE_loss(D_result, y_real_)

                    G_train_loss.backward()
                    G_optimizer.step()

                z = E(x)
                x_d = G(z)

                if it % 30 ==0:
                    directory = 'Train/MNIST/'+str(anomaly_class)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    #comparison = torch.cat([x[:4], x_d[:4]])
                    #save_image(comparison.cpu(),
                                #'Train/MNIST/'+str(anomaly_class)+'/reconstruction_'+ str(epoch)+'_'+ str(it) + '.png', nrow=4)
                    iter_end_time = time.time()
                    per_iter_ptime = iter_end_time - epoch_start_time

                    print('[%d]-[%d/%d]-[%d/%d] - ptime: %.2f, Gloss: %.3f, Dloss: %.3f, ZDloss: %.3f, GEloss: %.3f, Eloss: %.3f' \
                          % ((anomaly_class),(epoch + 1), train_epoch,it, length, per_iter_ptime,\
                              G_train_loss, D_train_loss, ZD_train_loss, Recon_loss, E_loss))
            print("Training finish!... save training results")
            model_dir = os.path.join('Model/MNIST', str(anomaly_class))
            if not os.path.isdir(model_dir):
                os.makedirs(model_dir)
            torch.save(G.state_dict(), '{}/Gmodel_epoch{}.pkl'.format(model_dir,str(epoch)))
            torch.save(E.state_dict(), '{}/Emodel_epoch{}.pkl'.format(model_dir,str(epoch)))
            torch.save(D.state_dict(), '{}/Dmodel_epoch{}.pkl'.format(model_dir,str(epoch)))
            torch.save(ZD.state_dict(), '{}/ZDmodel_epoch{}.pkl'.format(model_dir,str(epoch)))
            torch.save(P.state_dict(), '{}/Pmodel_epoch{}.pkl'.format(model_dir,str(epoch)))
            torch.save(C.state_dict(), '{}/Cmodel_epoch{}.pkl'.format(model_dir,str(epoch)))
if __name__ == '__main__':
    main()
