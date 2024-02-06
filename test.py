"""测试用例"""
import numpy as np

from reverseAD import *

# x = var('x')
# y = var('y')
#
# z = log(x) + x * y

# z = x + y
# z = x + 1 + y
# z = 1 + x + y
# z = x - y
# z = 1 - x - y
# z = x - y - 1
# z = x * y
# z = 3 * x * y
# z = x / y
# z = x / 3 + y
# z = 3 / x + y
# z = log(x) + y

# z = x @ y
# values = {x: np.array([[1, 2], [3, 4], [5, 6]]), y: np.array([[7, 8, 9], [10, 11, 12]])}

# z = sumOp(x) + sumOp(y)
# values = {x: np.asarray([1, 2, 3]), y: np.asarray([4, 5, 6])}

# z = 1 / (1 + exp(-x)) + y     # sigmoid

# values = {x: np.asarray(2), y: np.asarray(5)}
#
# print(f"z = {z}")
#
# grads = gradient(z, [x, y])
# print(f"grad_x: {grads[0]}\ngrad_y: {grads[1]}")
#
# executor = Executor([z] + grads)
# res = executor.run(values)
# print(f"z = {res[0]}")
# print(f"grad_x: {res[1]}\ngrad_y: {res[2]}")
# print(res)


# # softmax
# def test_softmax():
#
#     x = var("x")
#     z = exp(x) / sumOp(exp(x))
#     print(f"z = {z}")
#     grads = gradient(z, [x])
#     print(f"grad_x: {grads[0]}")
#     executor = Executor([z] + grads)
#     values = {x: np.asarray([2, 3, 5])}
#     res = executor.run(values)
#     """
#     softmax求导方法
#     对于i==j  ∂softmax(x)/∂xi = softmax(xi)(1-softmax(xi)
#     对于i!=j  ∂softmax(x)/∂xi = -softmax(xj)softmax(xi)
#     """
#     res[1][np.argmax(res[0])] = 1 - res[1][np.argmax(res[0])]
#     print(f"z = {res[0]}")
#     print(f"grad_x: {res[1]}")
#
# test_softmax()


class CrossEntropyLoss:
    def __call__(self, y_pred: np.ndarray, y_real: np.ndarray):
        pred_label = np.argmax(y_pred, axis=1)
        equality_vector = (pred_label == np.argmax(y_real, axis=1)).astype(int)
        # L = 0
        # for i in range(np.max(y_real) + 1):
        #     equality_class_vector = (equality_vector == i).astype(int)
        #     prob_vector = y_pred[equality_class_vector]
        #     Li = -sumOp(equality_class_vector * log(prob_vector))
        #     L = L + Li
        # L = L / len(y_real)
        L = 0
        inputs = {}
        for i in range(len(y_real[0])):
            equality_class_vector = (pred_label == i).astype(int)
            x = var(f"pred{i}")
            inputs[x] = y_pred * equality_class_vector[:, np.newaxis]
            Li = -sumOp(log(x) * inputs[x])
            L = L + Li
        L = L / len(y_real)
        print(L)
        grads = gradient(L, list(inputs.keys()))

        executor = Executor([L] + grads)
        res = executor.run(inputs)
        print(res[0])
        print(res[1:])
        pass


criterion = CrossEntropyLoss()
criterion(np.asarray([[0.3, 0.3, 0.4], [0.3, 0.4, 0.3], [0.1, 0.2, 0.7]]),
          np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]]))
