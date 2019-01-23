import os
import numpy as np
from PIL import Image

BASE_PATH = 'data/256_ObjectCategories'
TEST_PATH = '257.clutter'
RECORD_FILE = 'Caltech.txt'

def load_OC_train_data(n_class, input_size, load_flag=False):

    class_paths = os.listdir(BASE_PATH)
    # print(class_paths)
    if load_flag:
        class_paths = []
        with open(RECORD_FILE, "r") as fp:
            for line in fp.readlines():
                print(line.strip())
                class_paths.append(line.strip())
    else:
        if os.path.exists(RECORD_FILE):
            os.remove(RECORD_FILE)

    assert n_class <= len(class_paths), "class numbers selected is too larger than the dataset"
    # selected = []
    selected = np.random.choice(len(class_paths), n_class, replace=False)
    # selected.append(24)
    # print(selected)
    datas = []
    for class_path in np.array(class_paths)[selected]:
        # full_class_path = '{}/{}'.format(BASE_PATH, class_path)
        full_class_path = 'data/256_ObjectCategories/024.butterfly'
        class_path = '024.butterfly'
        print(full_class_path)
        if not load_flag:
            with open(RECORD_FILE, "a+") as fp:
                fp.write("{}\n".format(class_path))
        for img_path in os.listdir(full_class_path):
            img = Image.open('{}/{}'.format(full_class_path, img_path))
            img = np.array(img.resize((input_size, input_size)), dtype=np.float32).swapaxes(0,-1) / 255.
            img = img * 2. - 1.
            if len(img.shape) == 3:
                datas.append(img)
    return datas

def load_OC_test_data(input_size):
    
    datas = []
    path = '{}/{}/'.format(BASE_PATH, TEST_PATH)
    for img_path in os.listdir(path):
        img = Image.open('{}{}'.format(path, img_path))
        img = np.array(img.resize((input_size, input_size)), dtype=np.float32).swapaxes(0,-1) / 255.
        img = img * 2. - 1.
        if len(img.shape) == 3:
            datas.append(img)
    return datas

if __name__ == '__main__':
    data = load_OC_train_data(2, 32)
    data = load_OC_train_data(1, 32, load_flag=True)
