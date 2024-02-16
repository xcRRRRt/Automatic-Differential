//
// Created by 47925 on 2024/2/16.
//

#ifndef NEW_DL_CROSS_ENTROPY_H
#define NEW_DL_CROSS_ENTROPY_H

#include "Eigen/Eigen"
#include "../node/node.h"
#include "../operator/operator.h"

template<typename T>
class CrossEntropyLoss : public __Operator__<T> {
public:
    Eigen::MatrixX<T> real_label;

    CrossEntropyLoss() {
        auto new_node = new __Data_Node__<T>({pipeline<T>.cal_list.back()}, this);
        pipeline<T>.cal_list.push_back(new_node);
    }

    void forward(__Data_Node__<T> *out_node) override {
        auto input_pred = out_node->inputs[0];
        Eigen::MatrixX<T> t1 = this->real_label.array() * input_pred->data.array().log();
        Eigen::MatrixX<T> t2 = (1 - this->real_label.array()) * ((1 - input_pred->data.array()).log());
//        std::cout << t1.sum() << "   " <<  t2.sum() << std::endl;
        auto temp = -((t1 + t2).sum()) / input_pred->data.rows();
        out_node->data = (Eigen::MatrixX<T>(1, 1) << temp).finished();
    }

    void backward(__Data_Node__<T> *out_node) override {
        auto input_pred = out_node->inputs[0];
        out_node->inputs[0]->grad = (-this->real_label.array() / input_pred->data.array() +
                                    (1 - this->real_label.array()) / (1 - input_pred->data.array())) / input_pred->data.rows();
//        std::cout << out_node->inputs[0]->grad.sum() << std::endl;
    }

    Eigen::MatrixX<T> operator()(Eigen::MatrixX<T> &real_l) {
        this->real_label = real_l;
        pipeline<T>.forward();
        pipeline<T>.backward();
        return pipeline<T>.cal_list.back()->data;
    }
};

#endif //NEW_DL_CROSS_ENTROPY_H
