

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

