//
// Created by 47925 on 2024/2/12.
//

#ifndef NEW_DL_CAL_MAP_H
#define NEW_DL_CAL_MAP_H

#include <vector>
#include <unordered_map>
#include <algorithm>
#include <set>
#include <Eigen/Eigen>
#include "../node/node.h"

template<typename T>
class Pipeline {
public:
    std::vector<__Data_Node__<T> *> cal_list;

    Pipeline() = default;

    ~Pipeline() {
        for (int i = 0; i < this->cal_list.size(); ++i)
            delete this->cal_list[i];
    }

    void forward() {
        for (int i = 0; i < this->cal_list.size(); ++i){
            cal_list[i]->op->forward(cal_list[i]);
        }
    }

    void backward() {
        auto out_node = this->cal_list.back();
        out_node->grad = Eigen::MatrixX<T>::Ones(out_node->data.rows(), out_node->data.cols());
        for (int i = this->cal_list.size() - 1; i >= 0; --i) {
            cal_list[i]->op->backward(cal_list[i]);
        }
    }

    void update(double lr = 1.) {
        for (int i = 0; i < this->cal_list.size(); ++i)
            cal_list[i]->op->update(lr);
    }
};

template<typename T>
Pipeline<T> pipeline;

#endif //NEW_DL_CAL_MAP_H
