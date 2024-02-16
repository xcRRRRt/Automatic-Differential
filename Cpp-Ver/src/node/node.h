//
// Created by 47925 on 2024/2/12.
//

#ifndef NEW_DL_NODE_H
#define NEW_DL_NODE_H

#include <vector>
#include <string>
#include <memory>
#include "Eigen/Eigen"
#include "../operator/operator.h"
#include "../operator/basic_operator.h"
#include "../operator/other_operator.h"
#include "../calculate/cal_map.h"
#include "../active_func/sigmoid.h"

template<typename T>
extern Pipeline<T> pipeline;

// 数据节点
template<typename T>
class __Data_Node__ {
public:
    std::vector<__Data_Node__<T> *> inputs;
    __Operator__<T> *op = nullptr;    // 运算符
    Eigen::MatrixX<T> data = Eigen::MatrixX<T>::Zero(1, 1); // 数据
    Eigen::MatrixX<T> grad = Eigen::MatrixX<T>::Zero(1, 1); // 梯度
//    static Pipeline<T> pipeline;
    std::string name;

    __Data_Node__() = default;

    explicit __Data_Node__(Eigen::MatrixX<T> &data) : data(data) {}

    explicit __Data_Node__(const std::vector<__Data_Node__<T> *> &inputs, __Operator__<T> *op)
            : inputs(inputs), op(op) {}

    __Data_Node__<T> &operator+(__Data_Node__<T> &dn) {
        auto *new_node = new __Data_Node__<T>(std::vector({this, &dn}), add<T>);
        pipeline<T>.cal_list.push_back(new_node);
        return *new_node;
    }

    __Data_Node__<T> &operator-(__Data_Node__<T> &dn) {
        auto *new_node = new __Data_Node__<T>(std::vector({this, &dn}), subtract<T>);
        pipeline<T>.cal_list.push_back(new_node);
        return *new_node;
    }

    __Data_Node__<T> &operator-() {
        auto *new_node = new __Data_Node__<T>(std::vector({this}), negtive<T>);
        pipeline<T>.cal_list.push_back(new_node);
        return *new_node;
    }

    __Data_Node__<T> &operator*(__Data_Node__<T> &dn) {
        auto *new_node = new __Data_Node__<T>(std::vector({this, &dn}), multiply<T>);
        pipeline<T>.cal_list.push_back(new_node);
        return *new_node;
    }

    __Data_Node__<T> &operator/(__Data_Node__<T> &dn) {
        auto *new_node = new __Data_Node__<T>(std::vector({this, &dn}), divide<T>);
        pipeline<T>.cal_list.push_back(new_node);
        return *new_node;
    }

    friend __Data_Node__<T> &ln(__Data_Node__<T> &dn) {
        auto *new_node = new __Data_Node__<T>(std::vector({&dn}), _ln<T>);
        pipeline<T>.cal_list.push_back(new_node);
        return *new_node;
    }

    friend __Data_Node__<T> &exp(__Data_Node__<T> &dn) {
        auto *new_node = new __Data_Node__<T>(std::vector({&dn}), _exp<T>);
        pipeline<T>.cal_list.push_back(new_node);
        return *new_node;
    }

//    friend __Data_Node__<T> &sigmoid(__Data_Node__<T> &dn) {
//        auto *new_node = new __Data_Node__<T>(std::vector({&dn}),_sigmoid<T>);
//        pipeline<T>.cal_list.push_back(new_node);
//        return *new_node;
//    }

//    bool operator==(const __Data_Node__<T> &dn) { return this == &dn; }

    size_t operator()(const __Data_Node__<T> &dn) {
        std::size_t h1 = std::hash<Eigen::MatrixX<T>>{}(this->data);
        std::size_t h2 = std::hash<Eigen::MatrixX<T>>{}(this->grad);
        return h1 ^ (h2 << 1);
    }
};
#endif //NEW_DL_NODE_H
