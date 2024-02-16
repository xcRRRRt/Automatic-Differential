//
// Created by 47925 on 2024/2/13.
//

#ifndef NEW_DL_OTHER_OPERATOR_H
#define NEW_DL_OTHER_OPERATOR_H

#include "Eigen/Eigen"
#include "../node/node.h"
#include "operator.h"

template<typename T>
class __Ln__ : public __Operator__<T> {
public:
    void forward(__Data_Node__<T> *out_node) override {
        out_node->data = out_node->inputs[0]->data.array().log();
        out_node->inputs[0]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
    }

    void backward(__Data_Node__<T> *out_node) override {
        out_node->inputs[0]->grad.array() += out_node->grad.array() / out_node->inputs[0]->data.array();
    }
};


template<typename T>
class __Exp__ : public __Operator__<T> {
public:
    void forward(__Data_Node__<T> *out_node) override {
        out_node->data = out_node->inputs[0]->data.array().exp();
        out_node->inputs[0]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
    }

    void backward(__Data_Node__<T> *out_node) override {
        out_node->inputs[0]->grad.array() += out_node->grad.array() * out_node->inputs[0]->data.array().exp();
    }
};


template<typename T> static __Ln__<T> ln_instance;
template<typename T> static __Exp__<T> exp_instance;

template<typename T> static __Ln__<T> *_ln = &ln_instance<T>;
template<typename T> static __Exp__<T> *_exp = &exp_instance<T>;

#endif //NEW_DL_OTHER_OPERATOR_H
