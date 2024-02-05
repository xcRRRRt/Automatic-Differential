"""
工具类
内含计算图拓扑排序
"""
from typing import List


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
