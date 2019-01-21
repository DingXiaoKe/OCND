import os
import numpy as np
from PIL import Image
import random
BASE_PATH = 'data/coil-100'
RECORD_FILE = 'Coil100-7.txt'

def load_Coil_train_data(n_class, input_size, load_flag=False):
    datas = []
    # class_paths = os.listdir(BASE_PATH)
    # print(class_paths)
    class_number = 100
    if load_flag:
        with open(RECORD_FILE, "r") as fp:
            for line in fp.readlines():
                # print(line.strip())
                full_class_path = '{}/obj{}'.format(BASE_PATH, line.strip())
                for j in range(0,356,5):
                # for img_path in np.array((class_paths)):
                    # if img_path
                    img = Image.open('{}__{}.png'.format(full_class_path,j))
                    img = np.array(img.resize((input_size, input_size)), dtype=np.float32).swapaxes(0,-1) / 255.0
                    # img = img * 2. - 1.
                    if len(img.shape) == 3:
                        datas.append(img)
            random.shuffle(datas)
            # print(len(data))
            return datas      
    else:
        if os.path.exists(RECORD_FILE):
            os.remove(RECORD_FILE)

    assert n_class <= class_number, "class numbers selected is too larger than the dataset"

    selected = np.random.choice(class_number, n_class, replace=False)

    for i in selected:
        # print(i)
        full_class_path = '{}/obj{}'.format(BASE_PATH, i)
        if not load_flag:
            with open(RECORD_FILE, "a+") as fp:
                fp.write("{}\n".format(i))
        for j in range(0,356,5):
        # for img_path in np.array((class_paths)):
            # if img_path
            img = Image.open('{}__{}.png'.format(full_class_path,j))
            # img.resize((input_size, input_size)).show()
            # print(img)
            # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            img = np.array(img.resize((input_size, input_size)), dtype=np.float32).swapaxes(0,-1) / 255.
            # img = img * 2. - 1.
            # print(img)
            if len(img.shape) == 3:
                datas.append(img)
        random.shuffle(datas)
    return datas

def load_Coil_test_data(input_size,length):
    
    datas = []
    # class_paths = os.listdir(BASE_PATH)
    # path = '{}/{}/'.format(BASE_PATH, TEST_PATH)
    train_class = []
    with open(RECORD_FILE, "r") as fp:
        for line in fp.readlines():
            # print(line.strip())
            train_class.append(line.strip())
    for m in range(length):
        # print(m)
        i = random.randint(1,100)
        if i not in train_class:
            j = random.randrange(0,356,5)
            full_class_path = '{}/obj{}'.format(BASE_PATH, i)
            # for j in range(0,356,5):
            img = Image.open('{}__{}.png'.format(full_class_path,j))
            img = np.array(img.resize((input_size, input_size)), dtype=np.float32).swapaxes(0,-1) / 255.
            # img = img * 2. - 1.
            if len(img.shape) == 3:
                datas.append(img)
        else:
            i = random.randint(1,100)
            j = random.randrange(0,356,5)
            full_class_path = '{}/obj{}'.format(BASE_PATH, i)
            # for j in range(0,356,5):
            img = Image.open('{}__{}.png'.format(full_class_path,j))
            img = np.array(img.resize((input_size, input_size)), dtype=np.float32).swapaxes(0,-1) / 255.
            # img = img * 2. - 1.
            if len(img.shape) == 3:
                datas.append(img)
    random.shuffle(datas)
    # print(len(datas))
    return datas

if __name__ == '__main__':
    data = load_Coil_train_data(2, 32)
    # data = load_Coil_train_data(1, 32, load_flag=True)
    # data = load_Coil_test_data(32)
