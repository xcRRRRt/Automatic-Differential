//
// Created by 47925 on 2024/2/15.
//

#ifndef NEW_DL_NN_H
#define NEW_DL_NN_H

#include <vector>
#include "operator/operator.h"

template<typename T>
class NeuralNetwork {
public:
    std::vector<__Operator__<T> *> layers;
    __Data_Node__<T> *pred_node;

    template<typename ...Args>
    explicit NeuralNetwork(Args... args) {
        _add_layer(args...);
        this->_construct_pipeline();
    }

    ~NeuralNetwork() {
        for (auto layer: this->layers)
            delete layer;
    }

    template<typename L, typename ...Args>
    void _add_layer(L layer, Args...args) {
        this->layers.push_back(layer);
        _add_layer(args...);
    }

    template<typename L>
    void _add_layer(L layer) {
        this->layers.push_back(layer);
    }

    void update(double lr = 1.) {
        pipeline<T>.update(lr);
    }

    Eigen::MatrixX<T> data() {
        return this->pred_node->data;
    }

    void operator()(Eigen::MatrixX<T> &data) {
        pipeline<T>.cal_list[0]->inputs[0]->data = data;
//        pipeline<T>.forward();
    }

    void _construct_pipeline() {
        auto temp = new __Data_Node__<T>;
        for (auto it = this->layers.begin(); it < this->layers.end(); it++) {
            auto layer = *it;
            temp = (*layer)(*temp);
        }
        this->pred_node = temp;
    }
};

#endif //NEW_DL_NN_H
