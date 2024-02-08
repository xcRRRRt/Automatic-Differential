"""
Reverse Auto Differential 反向自动微分
参考 https://blog.csdn.net/aws3217150/article/details/70214422
整体思路
1. 先构建好input到output的pipeline
2. 然后再从output继续构建，根据拓扑倒序，继续构建从output到input的pipeline
3. 把input的真实值输进去，自动计算

# 可以在test.py中测试该例
例：y = ln(x1) + x1 * x2
1.  v-1 = x1    v0 = x2
2.  v1 = ln(v-1)
3.  v2 = v-1 * v0
4.  v3 = v1 + v2
5.  y = v3
这里对应上方的第一步，从x1和x2构建到y。
对于v1,v2,v3.etc来说，每个节点都是一个Node对象
每个Node对象都包含一个Operator运算符，和1个或2个inputs节点(3 * x就是1个，x * y就是两个，3是常量，也包含在Node对象里)，比如v3的inputs节点是[v1, v2]
(当然，对于v-1和v0来说，它们已经是最前面的节点了，不含inputs和Operator)
于是现在的计算图是这样的(从上往下)
    v-1   v0
   /   \ /
 v1     v2
   \    /
     v3
   计算图（一）

5.  ∂y/∂v3 = ∂y/∂y = 1
4.  ∂y/v1 = ∂y/∂v3 * ∂v3/∂v1 = 1 * 1 = 1                ∂y/∂v2 = ∂y/∂v3 * ∂v3/∂v2 = 1 * 1 = 1
3.  ∂y/∂v-1 = ∂y/∂v2 * ∂v2/∂v-1 = 1 * v0 = v0           ∂/v0 = ∂y/∂v2 * ∂y/∂v0 = 1 * v-1 = v-1
2.  ∂y/∂v-1 = ∂y/∂v-1 + ∂y/v1 * ∂v1/v-1 = v0 + 1/v-1
1.  ∂y/∂x1 = ∂y/∂v-1 = v0 + 1/v-1 = x2 + 1/x1           ∂y/∂x2 = ∂y/∂v0 = v-1 = x1
# 横式有点看不懂是不是，可以在纸上写成分式哦
这里对应上方的第二步
首先要明白链式法则
从5开始是求y对v3的偏导，然后是y对v2，y对v1......etc.
通过∂y/∂v3可以求出什么？
看4，∂y/v1 = ∂y/∂v3 * ∂v3/∂v1     ∂y/∂v2 = ∂y/∂v3 * ∂v3/∂v2
得到了∂y/v1和∂y/∂v2
再继续，通过∂y/v1可以求出什么？
得到了∂y/∂v-1 = ∂v-1 + ∂v1/v-1
......这样一层层下来，最终可以得到y对v-1和v0节点的偏导(实际上对所有节点的偏导都得到了)
现在来看看求导的计算图(还是从上往下)
    v3
   /  \
 v1    v2
  \   /  \
    v-1   v0
   计算图（二）
注意，求导的计算图与求值的计算图完全对称

那么现在完整的计算图就有了
    v-1   v0    这两个节点是输入
   /   \ /
 v1    v2
   \   /
     v3         这个节点是输出，也就是y
     |
    v3‘         这个节点一定是1，因为∂y/∂v3 = ∂y/∂y = 1
   /  \
 v1’   v2‘
  \   /  \
   v-1’  v0‘    这两个节点分别是∂y/∂v-1 = ∂y/∂x1, ∂y/∂v0 = ∂y/∂x2, 也就是y对x1和x2的偏导

那么我们现在从v-1’和v0‘节点开始拓扑排序（还记得每个Node都有inputs吗，通过这个进行拓扑排序），得到拓扑排序的结果
然后reverse一下，得到pipeline：v-1, v0, v1, v2, v3, v3', v1', v2', v-1', v0'

对应上方第三步
那么，把x1=v-1=2, y=v0=5输进去会得到什么呢？(还记得每个节点包含一个Operator运算符吗，调用node.operator计算node.inputs的结果)
v-1=2
v0=5
v1=ln(2)
v2=10
v3=ln(2)+10                        这步就把y求出来了
v3'=1
v1'= v3' * ∂v3/∂v1 = 1 * 1 = 1     v1到v3的操作是v1+v2=v3, 所以加法的偏导是1
v2'= v3' * ∂v3/∂v2 = 1 * 1 = 1     v2到v3的操作是...balabalabala
v-1'= v2'*∂v2/∂v-1 + v1'*∂v1/v-1 = 1*v0 + 1/v-1 = 1*5 + 1/2 = 5.5   这步就把grad_x1 = v-1’求出来了
                                                                    这是因为v1和v2的inputs里都包含了v-1，所以要加起来
                                                                    v2=v-1*v0,那么∂v2/∂v-1=v0, v1=ln(v-1),那么∂v1/∂v-1=1/v-1
v0'=v2'*∂v2/∂v0 = 1 * 2 = 2         v0到v2的操作是v2=v-1*v0，把v-1看成常量，乘法的偏导是v-1=2



运算符重载
__add__
__radd__
__sub__
__rsub__
__mul__
__rmul__
__truediv__
__rtruediv__
__matmul__
__neg__
可调用
sumOp = Sum()           # sum
log = Log()             # ln
exp = Exp()             # e^x
oneslike = OnesLike()
zerolike = ZeroLike()
"""

from typing import List, Union
import numpy as np
from utils import topological_sort


# TODO 解决与numpy运算符重载的冲突问题
class Node:
    """
    计算图的节点
    """
    def __init__(self, inputs: List['Node'] = None, op: 'Operator' = None,
                 name: str = None, const_value=None):
        """
        :params inputs 该节点的输入节点
        :params op 运算符
        :params name 用于调试的名字
        :params const_value 如果该节点对常量进行运算，则存储该常量，例：x+1；3*y；oneslike(1)
        """
        self.inputs = inputs if inputs is not None else []
        self.op = op
        self.name = name
        self.const_value = const_value
        self.bias = False

    # 重载运算符
    def __add__(self, other: Union['Node', np.ndarray, float, int]):
        if isinstance(other, Node):
            return add(self, other)
        else:
            return add_const(self, other)

    __radd__ = __add__

    def __sub__(self, other: Union['Node', np.ndarray, float, int]):
        if isinstance(other, Node):
            return subtract(self, other)
        else:
            return subtract_const(self, other)

    def __rsub__(self, other: Union['Node', np.ndarray, float, int]):
        if isinstance(other, Node):
            return subtract(self, other)
        else:
            return subtract_constR(other, self)

    def __mul__(self, other: Union['Node', np.ndarray, float, int]):
        if isinstance(other, Node):
            return multiply(self, other)
        else:
            return multiply_const(self, other)

    __rmul__ = __mul__

    def __truediv__(self, other: Union['Node', np.ndarray, float, int]):
        if isinstance(other, Node):
            return divide(self, other)
        else:
            return divide_const(self, other)
        pass

    def __rtruediv__(self, other: Union['Node', np.ndarray, float, int]):
        if isinstance(other, Node):
            return divide(self, other)
        else:
            return divide_constR(other, self)

    def __matmul__(self, other: Union['Node', np.ndarray, float, int]):
        return matmul(self, other)

    def __pow__(self, other: Union['Node', np.ndarray, float, int]):
        return NotImplemented

    def __rpow__(self, other: Union['Node', np.ndarray, float, int]):
        return NotImplemented

    def __neg__(self):
        return negative(self)

    def __str__(self) -> str:
        return self.name


class Operator:
    """运算符的基类"""
    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        """
        通过out节点的inputs节点，计算出使用该operator的out节点的实际值

        :params out 要计算的节点
        :params inputs out的输入值
        :return 计算后的值
        """
        pass

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        """
        从out节点构建对应operator的梯度节点
        例:
        对于 y = x1 + x2, 返回的梯度节点为 [grad_x1 = grad_y*∂y/∂x1, grad_x2 = grad_y*∂y/∂x2]
        其中grad_y为上一级节点的梯度, 即grad_y对应变量grad，y对应变量out，x1, x2对应out.inputs

        :params grad 上一级节点的梯度
        :params out 需要构建梯度的节点
        :return out节点的梯度节点
        """
        pass


class Placeholder(Operator):
    def __call__(self) -> Node:
        return Node(op=self)

    def compute(self, out: Node, inputs: List[np.ndarray]):
        pass

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        pass


def var(name, **kwargs):
    """构建输入节点，不需要输入实际的值"""
    input_node = placeholder()
    input_node.name = name
    return input_node


class Add(Operator):
    def __call__(self, a: Node, b: Node) -> Node:
        return Node(inputs=[a, b], op=self, name=f'({a.name} + {b.name})')

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(inputs[0] + inputs[1])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad, grad]


class AddConst(Operator):
    def __call__(self, a: Node, b: np.ndarray) -> Node:
        return Node(inputs=[a], op=self, name=f"({a.name} + {b})", const_value=b)

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(inputs[0] + out.const_value)

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad]


class Subtract(Operator):
    def __call__(self, a: Node, b: Node) -> Node:
        return Node(inputs=[a, b], op=self, name=f"({a.name} - {b.name})")

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(inputs[0] - inputs[1])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad, -grad]


class SubtractConst(Operator):
    def __call__(self, a: Node, b: np.ndarray) -> Node:
        return Node(inputs=[a], op=self, name=f"({a.name} - {b})", const_value=b)

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(inputs[0] - out.const_value)

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad]


class SubtractConstR(Operator):
    def __call__(self, a: np.ndarray, b: Node) -> Node:
        return Node(inputs=[b], op=self, name=f"({a} - {b.name})", const_value=a)

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(out.const_value - inputs[0])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [-grad]


class Negative(Operator):
    def __call__(self, a: Node) -> Node:
        return Node(inputs=[a], op=self, name=f"(-{a.name})")

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(-inputs[0])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [-grad]


class Multiply(Operator):
    def __call__(self, a: Node, b: Node) -> Node:
        return Node(inputs=[a, b], op=self, name=f'{a.name} * {b.name}')

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(inputs[0] * inputs[1])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad * out.inputs[1], grad * out.inputs[0]]


class MultiplyConst(Operator):
    def __call__(self, a: Node, b) -> Node:
        return Node(inputs=[a], op=self, name=f'{a.name} * {b}', const_value=b)

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(inputs[0] * out.const_value)

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad * out.const_value]


class Divide(Operator):
    def __call__(self, a: Node, b: Node) -> Node:
        return Node(inputs=[a, b], op=self, name=f'{a.name} / {b.name}')

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(inputs[0] / inputs[1])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad / out.inputs[1], -grad * out.inputs[0] / (out.inputs[1] * out.inputs[1])]


class DivideConst(Operator):
    def __call__(self, a: Node, b) -> Node:
        return Node(inputs=[a], op=self, name=f'{a.name} / {b}', const_value=b)

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(inputs[0] / out.const_value)

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad / out.const_value]


class DivideConstR(Operator):
    def __call__(self, a, b: Node) -> Node:
        return Node(inputs=[b], op=self, name=f'{a} / {b.name}', const_value=a)

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(out.const_value / inputs[0])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [-grad * out.const_value / (out.inputs[0] * out.inputs[0])]


class MatMul(Operator):
    def __call__(self, a: Node, b: Node, a_t: bool = False, b_t: bool = False) -> Node:
        name = a.name
        if a_t:
            name += ".T"
        name += " @ "
        name += b.name
        if b_t:
            name += ".T"
        out_node = Node(inputs=[a, b], op=self, name=name)
        out_node.a_t = a_t
        out_node.b_t = b_t
        return out_node

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        a, b = inputs
        if out.a_t:
            a = a.T
        if out.b_t:
            b = b.T
        return np.asarray(np.matmul(a, b))

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [matmul(grad, out.inputs[1], a_t=False, b_t=True),
                matmul(out.inputs[0], grad, a_t=True, b_t=False)]


class Log(Operator):
    """ln"""

    def __call__(self, a: Node) -> Node:
        return Node(inputs=[a], op=self, name=f'log({a.name})')

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.log(inputs[0])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad / out.inputs[0]]


class Exp(Operator):
    """e^x"""

    def __call__(self, a: Node) -> Node:
        return Node(inputs=[a], op=self, name=f'exp({a.name})')

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.exp(inputs[0])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad * exp(out.inputs[0])]


class OnesLike(Operator):
    def __call__(self, a: Node) -> Node:
        return Node(inputs=[a], op=self, name=f"1")

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.ones_like(inputs[0])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [zerolike(out.inputs[0])]


class ZeroLike(Operator):
    def __call__(self, a: Node) -> Node:
        return Node(inputs=[a], op=self, name=f"0")

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.zeros_like(inputs[0])

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [zerolike(out.inputs[0])]


class Sum(Operator):
    def __call__(self, a: Node) -> Node:
        return Node(inputs=[a], op=self, name=f"sum({a.name})")

    def compute(self, out: Node, inputs: List[np.ndarray]) -> np.ndarray:
        return np.asarray(np.sum(inputs[0]))

    def gradient(self, grad: Node, out: Node) -> List[Node]:
        return [grad * oneslike(out.inputs[0])]


placeholder = Placeholder()

add = Add()
add_const = AddConst()

subtract = Subtract()
subtract_const = SubtractConst()
subtract_constR = SubtractConstR()

negative = Negative()

multiply = Multiply()
multiply_const = MultiplyConst()

divide = Divide()
divide_const = DivideConst()
divide_constR = DivideConstR()

matmul = MatMul()

log = Log()
exp = Exp()

oneslike = OnesLike()
zerolike = ZeroLike()
sumOp = Sum()


class Executor:
    """给定节点及其实际值，计算指定节点的梯度值"""
    def __init__(self, to_compute: List[Node]):
        """
        :param to_compute 需要计算的节点，可以是梯度节点、也可以是前馈计算的节点
        """
        self.to_compute = to_compute
        self.topological_graph_sorted = topological_sort(to_compute)  # 拓扑排序，从输入到输出，为了前向计算

    def run(self, inputs_dict: dict[Node, np.ndarray]) -> List[np.ndarray]:
        """
        计算指定节点的值
        :param inputs_dict 输入节点的值字典 dict[Node, np.ndarray]
        :return to_compute节点的值
        """
        node_value_map: dict[Node, np.ndarray] = inputs_dict
        for node in self.topological_graph_sorted:
            node: Node
            if isinstance(node.op, Placeholder):  # 输入节点不用计算
                continue
            # 计算该节点的值，存储到{Node:value}中
            node_value_map[node] = node.op.compute(
                node,
                [node_value_map[n] for n in node.inputs],  # [node_value_map[n] for n in node.inputs] 节点的输入值
            )
        return [node_value_map[node] for node in self.to_compute]  # 返回要计算的节点的值


def gradient(output: Node, to_grad: List[Node]):
    """
    给定输出节点和需要构建梯度的节点，构建梯度计算图，返回需要构建梯度节点对应的计算图
    :param output 输出节点
    :param to_grad 需要构建梯度的节点
    :return to_grad节点对应的梯度节点
    """
    # 节点和对应梯度节点的映射，如例中所示，∂y/∂v-1对应v-1，∂y/∂v0对应v0，∂y/∂v1对应v1，etc.
    node_grad_map: dict[Node, Node] = {output: oneslike(output)}

    for node in reversed(topological_sort([output])):  # 反转拓扑排序，从输出到输入，从输出节点开始构建计算图
        node: Node
        grads = node.op.gradient(node_grad_map[node], node)  # grads是node的梯度
        if grads is None:  # 说明已经到了placeholder输入节点
            continue
        for i, grad in enumerate(grads):
            grad: Node
            if node.inputs[i] in node_grad_map:
                # 如例中倒数第2行 ∂y/∂v-1 = ∂v-1 + ∂v1/v-1 = v0 + 1/v-1 所示，梯度需要累加
                node_grad_map[node.inputs[i]] = node_grad_map[node.inputs[i]] + grad
            else:
                node_grad_map[node.inputs[i]] = grad
    return [node_grad_map[node] for node in to_grad]
