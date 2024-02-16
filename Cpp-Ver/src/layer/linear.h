//
// Created by 47925 on 2024/2/15.
//

#ifndef NEW_DL_LINEAR_H
#define NEW_DL_LINEAR_H

#include "Eigen/Eigen"
#include "../node/node.h"
#include "../operator/operator.h"
#include "../utils/init_weights.h"

template<typename T>
class Linear : public __Operator__<T> {
public:
    Eigen::MatrixX<T> weight;
    Eigen::MatrixX<T> weight_grad;
    Eigen::VectorX<T> bias;
    Eigen::VectorX<T> bias_grad;
    int input_size;
    int output_size;

    Linear(int input_size, int output_size) : weight(init_weight<T>(input_size, output_size)),
                                              bias(init_weight<T>(1, output_size).row(0)),
                                              weight_grad(Eigen::MatrixX<T>::Zero(input_size, output_size)),
                                              bias_grad(Eigen::VectorX<T>::Zero(output_size)),
                                              input_size(input_size), output_size(output_size) {}

    void forward(__Data_Node__<T> *out_node) override {
        out_node->data = (out_node->inputs[0]->data * this->weight).rowwise() + this->bias.transpose();
    }

    void backward(__Data_Node__<T> *out_node) override {
        this->weight_grad = out_node->inputs[0]->data.transpose() * out_node->grad;
        this->bias_grad = out_node->grad.colwise().sum();
//        std::cout << this->weight.sum() << " " << this->weight_grad.sum() << std::endl;
        out_node->inputs[0]->grad = out_node->grad * this->weight.transpose();
//        std::cout << this->weight << std::endl << std::endl;
//        std::cout << this->weight_grad << std::endl << std::endl;
    }

    void update(double lr = 1.) {
        this->weight = this->weight.array() - lr * this->weight_grad.array();
        this->bias = this->bias.array() - lr * this->bias_grad.array();
    }

    __Data_Node__<T> *operator()(__Data_Node__<T> &input) override {
        auto new_node = new __Data_Node__<T>({&input}, this);
        pipeline<T>.cal_list.push_back(new_node);
        return new_node;
    }
};

#endif //NEW_DL_LINEAR_H
