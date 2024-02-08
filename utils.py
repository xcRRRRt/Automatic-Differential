"""
工具类
内含计算图拓扑排序
"""
from typing import List

import numpy as np


# 拓扑排序
def topological_sort(nodes: List['Node']) -> List['Node']:
    visited = set()
    sorted_nodes = []

    def find(_node: 'Node'):
        if _node in visited:
            return
        visited.add(_node)
        for _input in _node.inputs:
            find(_input)
        sorted_nodes.append(_node)

    for n in nodes:
        find(n)

    return sorted_nodes


def read_file(filename, file_type):
    with open(filename, 'rb') as f:
        if file_type == 'image':
            f.seek(4)
            num_images = int(f.read(4).hex(), 16)
            img_height = int(f.read(4).hex(), 16)
            img_width = int(f.read(4).hex(), 16)
            imgs_all = np.zeros((num_images, img_height, img_width))
            for i in range(num_images):
                for h in range(img_height):
                    for w in range(img_width):
                        imgs_all[i, h, w] = int(f.read(1).hex(), 16)
            return imgs_all
        elif file_type == 'label':
            f.seek(4)
            num_labels = int(f.read(4).hex(), 16)
            labels_all = np.zeros(num_labels, dtype=int)
            for i in range(num_labels):
                labels_all[i] = int(f.read(1).hex(), 16)
            return labels_all


def pre_process(imgs: np.ndarray, label: np.ndarray):
    imgs = imgs.reshape(imgs.shape[0], -1) / 255
    one_hot_label = np.eye(10)[label]   # TODO 不需要One-hot，直接用vector
    return imgs, one_hot_label


def init_weight(input_shape, output_shape):
    # np.random.seed(0)
    limit = np.sqrt(6.0 / (input_shape + output_shape))
    return np.random.uniform(-limit, limit, size=(input_shape, output_shape))
