import os.path
import mindspore.dataset as ds
from mindspore import nn, ops, context
import mindspore.dataset.vision as vision
from PIL import Image
import numpy as np


class AbstractDataset:
    def __init__(self, cfg):
        self.data = list()
        self.labels = list()
        # self.data_dir = cfg['root']  # "/root/autodl-tmp/Datasets/WildDeepfake"
        # self.classes = cfg['classes']  # ["real", "fake"]
        self.classes = ['original', 'fake']
        self.split = cfg['split']
        self.class_to_idx = {class_name: i for i, class_name in enumerate(self.classes)}  # {"real": 0, "fake": 1}
        # "/root/autodl-tmp/Datasets/WildDeepfake/real_train/1"
        # for class_name in self.classes:
        #     class_dir = os.path.join(self.data.dir, class_name)
        #     for img_name in os.listdir(class_dir):
        #         self.data.append(os.path.join(class_dir, img_name))
        #         self.labels.append(self.class_to_idx[class_name])

    def __getitem__(self, index):
        img = Image.open(self.data[index]).covert('RGB')
        label = self.labels[index]

        return np.array(img), np.array(label, dtype=np.int32)

    def __len__(self):
        return len(self.data)
