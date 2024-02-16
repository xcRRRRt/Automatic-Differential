//
// Created by 47925 on 2024/2/12.
//

#ifndef NEW_DL_BASIC_OPERATOR_H
#define NEW_DL_BASIC_OPERATOR_H

#include "Eigen/Eigen"
#include "../node/node.h"
#include "operator.h"

template<typename T>
class __Add__ : public __Operator__<T> {
public:
    void forward(__Data_Node__<T> *out_node) override {
        out_node->data = out_node->inputs[0]->data + out_node->inputs[1]->data;
        out_node->inputs[0]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
        out_node->inputs[1]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
    }

    void backward(__Data_Node__<T> *out_node) override {
        out_node->inputs[0]->grad.array() += out_node->grad.array();
        out_node->inputs[1]->grad.array() += out_node->grad.array();
    }
};

template<typename T>
class __Sub__ : public __Operator__<T> {
public:
    void forward(__Data_Node__<T> *out_node) override {
        out_node->data = out_node->inputs[0]->data - out_node->inputs[1]->data;
        out_node->inputs[0]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
        out_node->inputs[1]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
    }

    void backward(__Data_Node__<T> *out_node) override {
        out_node->inputs[0]->grad.array() += out_node->grad.array();
        out_node->inputs[1]->grad.array() += -out_node->grad.array();
    }
};

template<typename T>
class __Mul__ : public __Operator__<T> {
public:
    void forward(__Data_Node__<T> *out_node) override {
        out_node->data = out_node->inputs[0]->data.array() * out_node->inputs[1]->data.array();
        out_node->inputs[0]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
        out_node->inputs[1]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
    }

    void backward(__Data_Node__<T> *out_node) override {
        out_node->inputs[0]->grad.array() += out_node->grad.array() * out_node->inputs[1]->data.array();
        out_node->inputs[1]->grad.array() += out_node->grad.array() * out_node->inputs[0]->data.array();
    }
};

template<typename T>
class __Div__ : public __Operator__<T> {
public:
    void forward(__Data_Node__<T> *out_node) override {
        out_node->data = out_node->inputs[0]->data / out_node->inputs[1]->data;
        out_node->inputs[0]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
        out_node->inputs[1]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
    }

    void backward(__Data_Node__<T> *out_node) override {
        out_node->inputs[0]->grad.array() += out_node->grad / out_node->inputs[1]->data.array();
        out_node->inputs[1]->grad.array() +=
                out_node->grad.array() * out_node->inputs[0]->data.array() /
                (out_node->inputs[0]->data.array() * out_node->inputs[0]->data.array());
    }
};

template<typename T>
class __Neg__ : public __Operator__<T> {
public:
    void forward(__Data_Node__<T> *out_node) override {
        out_node->data = -out_node->inputs[0]->data;
        out_node->inputs[0]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
        out_node->inputs[1]->grad = Eigen::MatrixX<T>::Zero(out_node->data.rows(), out_node->data.cols());
    }

    void backward(__Data_Node__<T> *out_node) override {
        out_node->inputs[0]->grad.array() += -out_node->grad.array();
    }
};

template<typename T> static __Add__<T> add_instance;
template<typename T> static __Sub__<T> subtract_instance;
template<typename T> static __Mul__<T> multiply_instance;
template<typename T> static __Div__<T> divide_instance;
template<typename T> static __Neg__<T> negative_instance;

template<typename T> static __Add__<T> *add = &add_instance<T>;
template<typename T> static __Sub__<T> *subtract = &subtract_instance<T>;
template<typename T> static __Mul__<T> *multiply = &multiply_instance<T>;
template<typename T> static __Div__<T> *divide = &divide_instance<T>;
template<typename T> static __Neg__<T> *negtive = &negative_instance<T>;

#endif //NEW_DL_BASIC_OPERATOR_H
