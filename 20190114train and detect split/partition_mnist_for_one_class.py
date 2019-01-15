from utils import mnist_reader
from utils.download import download
import random
import pickle
import os
download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz", extract_gz=True)
download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz", extract_gz=True)
download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz", extract_gz=True)
download(directory="mnist", url="http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz", extract_gz=True)

folds = 2

#Split mnist into 5 folds:
mnist = items_train = mnist_reader.Reader('mnist', train=True, test=True).items
class_bins = {}
random.shuffle(mnist)

for x in mnist:
    if x[0] not in class_bins:
        class_bins[x[0]] = []
    class_bins[x[0]].append(x)
directory = 'data/'
if not os.path.exists(directory):
    os.makedirs(directory)
## mnist_folds[1] test  abnormal
## mnist_folds[0] train normal
for i in range(0,10):
    mnist_folds = [[] for _ in range(folds)]
    inlier_class = i
    for _class, data in class_bins.items():
        count = len(data)
        # print("Class %d count: %d" % (_class, count))
        if _class != inlier_class:
            mnist_folds[1] += data[0: count]
        else:
            mnist_folds[0] += data[0: count]

    # total_length = len(mnist_folds[0])
    # length = int(len(mnist_folds[0])*0.8)
    # mnist_folds[1] +=(mnist_folds[0][length+1:total_length])
    # mnist_folds[0] = mnist_folds[0][0:length]
    print("Folds sizes:")
    for i in range(len(mnist_folds)):
        print(len(mnist_folds[i]))
        if i == 0:
            output = open('data/mnist_{}_{}.pkl'.format(str(inlier_class),'train'), 'wb')
        else:
            output = open('data/mnist_{}_{}.pkl'.format(str(inlier_class),'test'), 'wb')
        pickle.dump(mnist_folds[i], output)
        output.close()
