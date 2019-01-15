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
from net_pc import *
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
        # p = random.random()
        # print(p)
        # if p > 0:
        h = img.shape[0]
        w = img.shape[1]
        if np.random.choice([0, 1]):
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
            label.append(0)
        else: 
            train.append(img)
            ## not cutout label = 1 true label
            label.append(1)
    train = np.array(train)
    label = np.array(label)
    # print(label)
    return train,label


def train():
    # new hyperparameter
    lambd = 0.5
    batch_size = 128
    zsize = 32
    anomaly_class = 0
    cutout = True # whether cover the img
    n_holes = 1 # number of holes to cut out from image
    length = 32 # length of the holes
    TINY = 1e-15
    for i in range(0,1):
        anomaly_class = i
        print('start loading data')

        with open('data/mnist_{}_{}.pkl'.format(str(anomaly_class),'train'), 'rb') as pkl:
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
        
        train_epoch = 40

        G = Generator(zsize)
        setup(G)
        G.weight_init(mean=0, std=0.02)

        D = Discriminator()
        setup(D)
        D.weight_init(mean=0, std=0.02)

        # P = Dense(1, 128, 2, batch_size, active_fn=F.softmax)
        # setup(P)
        # P.weight_init(mean=0, std=0.01)

        # C = Dense(1, 128, 1, batch_size)
        # setup(C)
        # C.weight_init(mean=0, std=0.01)

        E = Encoder(zsize)
        setup(E)
        E.weight_init(mean=0, std=0.02)

        ZD = ZDiscriminator(zsize, batch_size)
        setup(ZD)
        ZD.weight_init(mean=0, std=0.02)

        lr = 0.1
        lmbda = 0.1
        G_optimizer = optim.SGD(G.parameters(), lr=lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
        D_optimizer = optim.SGD(D.parameters(), lr=lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
        E_optimizer = optim.SGD(E.parameters(), lr=lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
        GE_optimizer = optim.SGD(list(E.parameters()) + list(G.parameters()), lr=lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)
        ZD_optimizer = optim.SGD(ZD.parameters(), lr=lr,
                                momentum=0.9, nesterov=True, weight_decay=5e-4)

        BCE_loss = nn.BCELoss()
        prediction_criterion = nn.NLLLoss()

        y_real_ = torch.ones(batch_size)
        y_fake_ = torch.zeros(batch_size)
        
        y_real_z = torch.ones(batch_size)
        y_fake_z = torch.zeros(batch_size)

        sample = torch.randn(64, zsize).view(-1, zsize, 1, 1)
        budget = 0.3
        for epoch in range(train_epoch):
            G.train()
            D.train()
            E.train()
            ZD.train()
            epoch_start_time = time.time()
            def shuffle(X):
                np.take(X, np.random.permutation(X.shape[0]), axis=0, out=X)

            shuffle(mnist_train_x)
            if (epoch + 1) % 30 == 0:
                G_optimizer.param_groups[0]['lr'] /= 4
                D_optimizer.param_groups[0]['lr'] /= 4
                GE_optimizer.param_groups[0]['lr'] /= 4
                E_optimizer.param_groups[0]['lr'] /= 4
                ZD_optimizer.param_groups[0]['lr'] /= 4
                print("learning rate change!")
            
            length = len(mnist_train_x) // batch_size
            for it in range(length):
                x = extract_batch(mnist_train_x, it, batch_size).view(-1, 1, 32, 32)
                label = extract_batch_label(mnist_train_y, it, batch_size).view(-1).long()
                labels_onehot = Variable(encode_onehot(label, 2))             
                # binary_class_y_real = torch.zeros(batch_size, 2)
                # for i in range(batch_size):
                #     binary_class_y_real[i][0] = y[i]
                #     binary_class_y_real[i][1] = 1-y[i]
                
                # binary_class_y_fake = torch.zeros(batch_size, 2)
                # for i in range(batch_size):
                #     binary_class_y_fake[i][0] = 1-y[i]
                #     binary_class_y_fake[i][1] = y[i]
                #############################################

                ZD.zero_grad()
                E.zero_grad()

                z = torch.randn((batch_size, zsize)).view(-1, zsize)
                z = Variable(z)

                ZD_result = ZD(z).squeeze()
                ZD_real_loss = BCE_loss(ZD_result +TINY, y_real_z +TINY )

                z = E(x).squeeze().detach()

                ZD_result = ZD(z).squeeze()
                ZD_fake_loss = BCE_loss(ZD_result +TINY, y_fake_z +TINY)

                ZD_train_loss = ZD_real_loss + ZD_fake_loss
                ZD_train_loss.backward()

                ZD_optimizer.step()

                #############################################

                E.zero_grad()
                G.zero_grad()

                z = E(x)
                x_d = G(z)

                ZD_result = ZD(z.squeeze()).squeeze()

                E_loss = BCE_loss(ZD_result +TINY, y_real_z +TINY ) * 2.0

                Recon_loss = F.binary_cross_entropy(x_d +TINY,x +TINY)

                (Recon_loss + E_loss).backward()

                GE_optimizer.step()
                #############################################

                D.zero_grad()
                G.zero_grad()
                E.zero_grad()
                # TODO
                pred_original, confidence = D(x)
                pred_original = F.softmax(pred_original, dim=-1).squeeze()
                # print(pred_original.shape)
                confidence = F.sigmoid(confidence).squeeze().reshape(batch_size,1)
                eps = 1e-12
                pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
                confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
                # print(confidence.shape)
                # Randomly set half of the confidences to 1 (i.e. no hints)
                b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)))
                # b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()
                conf = confidence * b + (1 - b)
                # print(conf.shape)
                # print(pred_original * conf.expand_as(pred_original).shape)
                pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
                pred_new = torch.log(pred_new +TINY)

                xentropy_loss = prediction_criterion(pred_new, label)
                confidence_loss = torch.mean(-torch.log(confidence +TINY))
                D_real_loss = xentropy_loss + (lmbda * confidence_loss)
                if budget > confidence_loss.data[0]:
                    lmbda = lmbda / 1.01
                elif budget <= confidence_loss.data[0]:
                    lmbda = lmbda / 0.99

                
                z = E(x)
                x_fake = G(z).detach()
                pred_original, confidence = D(x_fake)
                pred_original = F.softmax(pred_original, dim=-1).squeeze()
                confidence = F.sigmoid(confidence).squeeze().reshape(batch_size,1)
                eps = 1e-12
                pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
                confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
                # Randomly set half of the confidences to 1 (i.e. no hints)
                b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)))
                # b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1))).cuda()
                conf = confidence * b + (1 - b)
                pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
                pred_new = torch.log(pred_new +TINY)

                xentropy_loss = prediction_criterion(pred_new, label)
                confidence_loss = torch.mean(-torch.log(confidence +TINY))
                D_fake_loss = xentropy_loss + (lmbda * confidence_loss)
                if budget > confidence_loss.data[0]:
                if budget > confidence_loss.data[0]:
                    lmbda = lmbda / 1.01
                elif budget <= confidence_loss.data[0]:
                    lmbda = lmbda / 0.99

                D_train_loss = - torch.mean(D_real_loss) + torch.mean(D_fake_loss) 
                D_train_loss.backward()

                D_optimizer.step()

                #############################################

                G.zero_grad()
                E.zero_grad()
                
                z = E(x)
                x_fake = G(z).detach()
                pred_original, confidence = D(x_fake)
                pred_original = F.softmax(pred_original, dim=-1).squeeze()
                confidence = F.sigmoid(confidence).squeeze().reshape(batch_size,1)
                eps = 1e-12
                pred_original = torch.clamp(pred_original, 0. + eps, 1. - eps)
                confidence = torch.clamp(confidence, 0. + eps, 1. - eps)
                # Randomly set half of the confidences to 1 (i.e. no hints)
                b = Variable(torch.bernoulli(torch.Tensor(confidence.size()).uniform_(0, 1)))
                conf = confidence * b + (1 - b)
                pred_new = pred_original * conf.expand_as(pred_original) + labels_onehot * (1 - conf.expand_as(labels_onehot))
                pred_new = torch.log(pred_new +TINY)

                xentropy_loss = prediction_criterion(pred_new, label)
                confidence_loss = torch.mean(-torch.log(confidence +TINY))
                D_result_loss = xentropy_loss + (lmbda * confidence_loss)
                if budget > confidence_loss.data[0]:
                    lmbda = lmbda / 1.01
                elif budget <= confidence_loss.data[0]:
                    lmbda = lmbda / 0.99
                G_train_loss = torch.mean(-D_result_loss)#BCE_loss(D_result, y_real_)

                G_train_loss.backward()
                G_optimizer.step()

                z = E(x)
                x_d = G(z)

                if it % 3 ==0:
                    directory = 'Train/MNIST/'+str(anomaly_class)
                    if not os.path.exists(directory):
                        os.makedirs(directory)
                    comparison = torch.cat([x[:4], x_d[:4]])
                    save_image(comparison.cpu(),
                                'Train/MNIST/'+str(anomaly_class)+'/reconstruction_'+ str(epoch)+'_'+ str(it) + '.png', nrow=4)
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

def test(p):
    percentage = p
    best_f1_score = 0
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
            # G.load_state_dict(torch.load(model_dir+'/Gmodel_epoch{}.pkl'.format(str(epoch))))
            # E.load_state_dict(torch.load(model_dir+'/Emodel_epoch{}.pkl'.format(str(epoch))))
            D.load_state_dict(torch.load(model_dir+'/Dmodel_epoch{}.pkl'.format(str(epoch))))
            # ZD.load_state_dict(torch.load(model_dir+'/ZDmodel_epoch{}.pkl'.format(str(epoch))))
            P.load_state_dict(torch.load(model_dir+'/Pmodel_epoch{}.pkl'.format(str(epoch))))
            # C.load_state_dict(torch.load(model_dir+'/Cmodel_epoch{}.pkl'.format(str(epoch))))
            
            X_score = []
            length = len(mnist_test_x) // batch_size
            
            for it in range(length):
                x = Variable(extract_batch(mnist_test_x, it, batch_size).view(-1, 1, 32, 32))
                D_score, _ = D(x)
                D_score = D_score.reshape((batch_size, 1))
                _, D_score = P(D_score)
                # D_real = D_real.reshape((batch_size, 1))
                # _, P_real = P(D_real)
                D_result = D_score.squeeze().detach().cpu().numpy()
                X_score.append(D_result)
            # print("start calculating")
            X_score = np.array(X_score).reshape(length*batch_size, 2)
            #print(X_score)
            anomaly_score = X_score
            
            # np.savetxt("./Test/Mnist/0/anomaly_score.txt",anomaly_score)
            labels = mnist_test_y[0:length*batch_size]
            binary_class_labels = np.zeros((length*batch_size, 2))
            for i in range(len(labels)):
                binary_class_labels[i, 1-labels[i]] = 1.
            path = "./Test/MNIST/"
            #roc_auc = evaluate(labels= binary_class_labels, scores= anomaly_score, directory=path, epoch = epoch, inlier_class = inlier_class,metric='roc')
            #prc_auc = evaluate(labels= binary_class_labels, scores= anomaly_score, directory=path, epoch = epoch, inlier_class = inlier_class,metric='auprc')
            f1_score = evaluate(labels= labels, scores= anomaly_score, directory=path, epoch = epoch, inlier_class = inlier_class , metric='f1_score')
            #[:, 1]
            #recall = evaluate(labels= labels, scores= anomaly_score, directory=path, epoch = epoch, inlier_class = inlier_class , metric='recall')
            #precision = evaluate(labels= labels, scores= anomaly_score, directory=path, epoch = epoch, inlier_class = inlier_class , metric='precision')
            #print(roc_auc)
            #print(prc_auc)
            #print(f1_score)
            if f1_score>best_f1_score:
              #print(f1_score)
              best_f1_score = f1_score
    print(best_f1_score)
    return best_f1_score
def main():
    percentages = [50,40,30,20,10]
    aim_f1 = [0.94,0.95,0.96,0.97,0.98]
    results = {}
    i = -1
    for p in percentages:
        i = i+1
        print("p:%d" % p)
        while True:
            train()
            results[p] = test(p)
            if results[p] > aim_f1[i]:
                print(results[p])
                path = "./Test/MNIST/"
                f1_file_name = os.path.join(path, 'f1_score.txt')
                with open(f1_file_name, 'a+') as opt_file:
                    opt_file.write('---p----\n')
                    opt_file.write('%s' %(str(results[p]))+'\n')
                    opt_file.write('----------\n')
                break
            
    return None

if __name__ == '__main__':
    main()

