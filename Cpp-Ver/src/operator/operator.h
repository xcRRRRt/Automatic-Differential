//
// Created by 47925 on 2024/2/12.
//

#ifndef NEW_DL_OPERATOR_H
#define NEW_DL_OPERATOR_H

#include "Eigen/Eigen"
#include "../node/node.h"

template<typename T>
class __Data_Node__;

// 运算符基类
template<typename T>
class __Operator__ {
public:
    /*
     * 前馈计算，使用out_node节点的inputs节点的data进行前馈计算，并修改out_node节点的data
     *
     * @param out_node 当前要进行前馈计算的节点
     */
    virtual void forward(__Data_Node__<T> *out_node) = 0;

    /*
     * 反向传播，使用out_node节点的grad计算out_node节点的inputs节点的grad，并修改inputs节点的grads
     *
     * @param out_node 当前要为inputs计算反向传播的数据节点
     */
    virtual void backward(__Data_Node__<T> *out_node) = 0;

    virtual __Data_Node__<T> *operator()(__Data_Node__<T> &input) {};

    virtual void update(double lr = 1.) {};
};

#endif //NEW_DL_OPERATOR_H
