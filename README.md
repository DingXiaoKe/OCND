

### Content

* **partition_mnist_for_one_class.py** - code for preparing MNIST dataset.
* **train_AAE.py** - code for training the autoencoder.
* **novelty_detector.py** - code for running novelty detector
* **net.py** - contains definitions of network architectures. 

### How to run

You will need to run **partition_mnist_for_one_class.py** first.

Then from **train_AAE.py**, you need to call *main* function:

train_AAE.main()
  
After autoencoder was trained, from **novelty_detector.py**, you need to call *main* function:

novelty_detector.main()


### MNIST:

You will need to run **partition_mnist_for_one_class.py** first.

Then from **train_AAE_MNISY.py**, you need to call *main* function:

train_AAE.main()
  
After autoencoder was trained, from **novelty_detector_mnist.py**, you need to call *main* function:

novelty_detector.main()

it uses **net_mnist**

training: use one class as inlier

testing: use training class and other class as inlier and different proportion

lr = 0.002 

### fashion MNIST:
the same as MNIST
**partition_mnist_for_one_class.py**

**train_AAE_fashion-mnist.py**

**novelty_detector_fashion-mnist.py**

**net_mnist**
control different proportion

###caltech:

use **OC256.py** to load data

train: **train_AAE_Caltech.py**  

n_class controls random choose some classed from dataset inlier {1,3,5} ,outlier proportion 50%, 25%, 15%

lr:{2e-4,4.5e-4,1e-4,}

test: **novelty_detector_Caltech**

net: **net_cifar**  

###cifar:



###coil100
n_class 1,4,7
1:
lr = 2e-3 lambd = 0.1
4:
lr = 0.002 lambd = 0.1
7:
lr = 0.001 lambd = 0.01
