import numpy as np

from reverseAD import *
from utils import read_file, pre_process, init_weight


class Sigmoid:
    def __call__(self, X) -> Node:
        return 1 / (1 + exp(-X))

    def __str__(self):
        return "Sigmoid()"


class Linear:
    def __init__(self, in_size: int, out_size: int, bias=True, name: str = ""):
        self.w_grad = None
        self.b_grad = None
        self._bias = None
        self.in_size = in_size
        self.out_size = out_size
        self.has_bias = bias
        self.name = name
        self._weight = init_weight(self.in_size, self.out_size)

    def __call__(self, X_var) -> Node:
        if self.has_bias:
            self.bias_var = var(self.name+"bias")
            self._bias = init_weight(1, self.out_size)
        self.weight_var = var(self.name)
        return X_var @ self.weight_var + self.bias_var

    @property
    def weight(self):
        return self._weight

    @weight.setter
    def weight(self, value):
        self._weight = value

    @property
    def bias(self):
        return self._bias

    @bias.setter
    def bias(self, value):
        self._bias = value

    def __str__(self):
        return f"Linear({self.in_size}, {self.out_size})"


class NeuralNetwork:
    def __init__(self, *args):
        self.layers = [layer for layer in args]
        self._get_weight_vars()

    def __call__(self, input_data):
        """
        把self全部返回，以便损失函数能够调用所有的var来进行计算
        :param input_data 网络的输入数值
        :return self
        """
        self.input_data = input_data
        return self

    @property
    def data(self):
        executor = Executor([self.y_var])
        value = {self.input_var: self.input_data}
        for layer in self.layers:
            if isinstance(layer, Linear):
                value.update({layer.weight_var: layer.weight, layer.bias_var: layer.has_bias})
        return executor.run(value)[0]

    def _get_weight_vars(self):
        """
        获得计算y的pipeline和计算weight的pipline
        """
        X_var = var("x")
        self.input_var = X_var
        for layer in self.layers:
            X_var = layer(X_var)
        self.y_var = X_var
        self.weight_vars = [layer.weight_var for layer in self.layers if isinstance(layer, Linear)]
        self.bias_vars = [layer.bias_var for layer in self.layers if isinstance(layer, Linear)]

    def parameters(self):
        paras = []
        for layer in self.layers:
            if isinstance(layer, Linear):
                paras.append(layer.weight)
                if layer.has_bias:
                    paras.append(layer.bias)
        return paras

    def __str__(self):
        s = ""
        for layer in self.layers:
            s += str(layer) + "\n"
        return s


class CrossEntropyLoss:
    def __call__(self, network: NeuralNetwork, y_real: np.ndarray):
        """
        交叉熵损失函数
        :param network 神经网络对象，其包含神经网络所有的属性
        :param y_real one-hot真实标签
        :return 损失
        """
        y_real_var = var("y_real")   # 真实值
        value = {layer.weight_var: layer.weight for layer in network.layers if isinstance(layer, Linear)}   # 权重
        value.update({layer.bias_var: layer.bias for layer in network.layers if isinstance(layer, Linear) and layer.has_bias})   # 偏置
        value.update({y_real_var: y_real, network.input_var: network.input_data})   # 真实标签，神经网络输入值
        L = -sumOp(y_real_var * log(network.y_var) + (1 - y_real_var) * log(1 - network.y_var)) / len(y_real)      # 交叉熵损失
        weight_grads = gradient(L, network.weight_vars + network.bias_vars)     # L是损失函数, weight_vars和bias_vars是需要计算的梯度
        executor = Executor([L] + weight_grads)
        results = executor.run(value)   # result的第一个是网络输出值，后面几个的前一半是权重，另一半是偏置
        _loss, _grads = results[0], results[1:]
        j = 0
        for i in range(len(network.layers)):
            if isinstance(network.layers[i], Linear):
                network.layers[i].w_grad = _grads[j]
                if network.layers[i].has_bias:
                    network.layers[i].b_grad = np.sum(_grads[j + int(len(_grads) / 2)], axis=0)
                j += 1
        return _loss


class Optimizer:
    def __init__(self, network, lr):
        self.network = network
        self.lr = lr

    def step(self):
        for layer in self.network.layers:
            if isinstance(layer, Linear):
                layer.weight -= self.lr * layer.w_grad
                if layer.has_bias:
                    layer.bias -= self.lr * layer.b_grad


criterion = CrossEntropyLoss()
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

optim = Optimizer(net, lr=1)

for i in range(1000):
    output = net(train_data)
    loss = criterion(output, train_labels)
    optim.step()

    pred_train_label = np.argmax(output.data, axis=1)
    correct_train = np.sum(pred_train_label == train_label)

    output_test = net(test_data)
    pred_label = np.argmax(output_test.data, axis=1)
    correct = np.sum(pred_label == test_label)
    print(f"iter: {i + 1}, loss: {loss}, train acc: {correct_train / len(train_labels)}, test acc: {correct / len(test_labels)}")
