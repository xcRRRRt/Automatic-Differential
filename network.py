import numpy as np
from utils import read_file, pre_process, init_weight
from reverseAD import *


class Sigmoid:
    def __call__(self, X):
        return 1 / (1 + exp(-X))

    def __str__(self):
        return "Sigmoid()"


class Linear:
    def __init__(self, in_size: int, out_size: int, bias=True, name: str = ""):
        self.in_size = in_size
        self.out_size = out_size
        self.bias = bias
        self.name = name
        self.weight = init_weight(self.in_size, self.out_size)
        if self.bias:
            self.weight = np.insert(self.weight, 0, 1, axis=0)
        else:
            pass

    def __call__(self, X_var):
        self.weight_var = var(self.name)
        return X_var @ self.weight_var

    def __str__(self):
        return f"Linear({self.in_size}, {self.out_size})"


class NeuralNetwork:
    def __init__(self, *args):
        self.weight_grads_var = None
        self.executor = None
        self.layers = [layer for layer in args]

    # TODO 这部分未完成
    def __call__(self, X):
        res = self.executor.run({self.layers[0]: X})
        y_pred = res[0]
        grads = res[1:]
        return y_pred

    # TODO 这部分未测试
    def _get_weight_vars(self, X_var):
        for layer in self.layers:
            X_var = layer(X_var)
        y_var = X_var
        weight_vars = [layer.weight_var for layer in self.layers if isinstance(layer, Linear)]
        self.executor = Executor([y_var] + weight_vars)
        self.weight_grads_var = gradient(y_var, weight_vars)

    def __str__(self):
        s = ""
        for layer in self.layers:
            s += str(layer) + "\n"
        return s


net = NeuralNetwork(
    Linear(784, 25, name="w1"),
    Sigmoid(),
    Linear(25, 10, name="w2"),
    Sigmoid()
)

print(net)

train_images = read_file("dataset/train-images-idx3-ubyte", 'image')
train_label = read_file("dataset/train-labels-idx1-ubyte", 'label')
test_images = read_file("dataset/t10k-images-idx3-ubyte", 'image')
test_label = read_file("dataset/t10k-labels-idx1-ubyte", 'label')

train_data, train_labels = pre_process(train_images, train_label)
test_data, test_labels = pre_process(test_images, test_label)

x = var("x")
