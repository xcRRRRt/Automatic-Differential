//
// Created by 47925 on 2024/2/13.
//

#ifndef NEW_DL_SIGMOID_H
#define NEW_DL_SIGMOID_H

#include "Eigen/Eigen"
#include "../node/node.h"
#include "../operator/operator.h"

template<typename T>
class Sigmoid : public __Operator__<T> {
public:
    void forward(__Data_Node__<T> *out_node) override {
        out_node->data = this->__sigmoid(out_node->inputs[0]->data);
    }

    void backward(__Data_Node__<T> *out_node) override {
        Eigen::MatrixX<T> temp = __sigmoid(out_node->inputs[0]->data);
        out_node->inputs[0]->grad = out_node->grad.array() * temp.array() * (1 - temp.array());
    }

    Eigen::MatrixX<T> __sigmoid(Eigen::MatrixX<T> &data) {
        return 1 / (1 + (-data.array()).exp());
    }

    __Data_Node__<T> *operator()(__Data_Node__<T> &input) override {
        auto new_node = new __Data_Node__<T>({&input}, this);
        pipeline<T>.cal_list.push_back(new_node);
        return new_node;
    }
};


#endif //NEW_DL_SIGMOID_H
