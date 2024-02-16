#include <iostream>
#include "src/node/node.h"
#include "src/datatype.h"
#include "src/utils/init_weights.h"
#include "src/utils/read_data.h"
#include "src/nn.h"
#include "src/layer/linear.h"
#include "src/loss/cross_entropy.h"

template<typename T>
void show_grad(const T &node) { std::cout << node.grad << std::endl; }

template<typename T, typename ...Args>
void show_grad(const T &node, const Args &... args) {
    std::cout << node.grad << std::endl;
    show_grad(args...);
}

template<typename T>
double accuracy(Eigen::MatrixX<T> &real_label, Eigen::MatrixX<T> &pred_prob) {
    // 获取每行的最大值
    int a, b, corr = 0;
    for (int i = 0; i < real_label.rows(); ++i) {
        real_label.row(i).maxCoeff(&a);
        pred_prob.row(i).maxCoeff(&b);
        if (a == b)
            corr++;
    }
    return double(corr) / real_label.rows();
}

int main() {

    Eigen::MatrixX<double> m1(2, 2);
    Eigen::MatrixX<double> m2(2, 2);
    Eigen::MatrixX<double> m3(2, 2);
    Eigen::MatrixX<double> m4(2, 2);
    m1 << 2, 3, 4, 5;
    m2 << 3, 4, 5, 6;
    m3 << 5, 5, 6, 6;
    m4 << 7, 7, 7, 7;
//    __Data_Node__<double> t1(m1);
//    __Data_Node__<double> t2(m2);
//    __Data_Node__<double> t3(m3);
//    __Data_Node__<double> t4(m4);
//    t1.name = "t1";
//    t2.name = "t2";
//    t3.name = "t3";
//    t4.name = "t4";
//    __Data_Node__<double> _ = exp(t1) - t2 + t3 * t4;
//    pipeline<double>.forward();
//    pipeline<double>.backward();
//    show_grad(t1, t2, t3, t4);

    Eigen::MatrixX<double> train_imgs;
    Eigen::MatrixX<double> train_labels;
    Eigen::MatrixX<double> test_imgs;
    Eigen::MatrixX<double> test_labels;
    read_images<double>(R"(C:\Users\47925\Desktop\new_dl\dataset\train-images-idx3-ubyte)", train_imgs);
    read_label(R"(C:\Users\47925\Desktop\new_dl\dataset\train-labels-idx1-ubyte)", train_labels);
//    read_images<double>(R"(C:\Users\47925\Desktop\new_dl\dataset\t10k-images-idx3-ubyte)", test_imgs);
//    read_label(R"(C:\Users\47925\Desktop\new_dl\dataset\t10k-labels-idx1-ubyte)", test_labels);

    // 我规划的pipeline是全局变量，所以要按照神经网络向前传播的顺序堆积，损失函数放在网络后面

    auto net = NeuralNetwork<double>(
            new Linear<double>(784, 25),
            new Sigmoid<double>(),
            new Linear<double>(25, 10),
            new Sigmoid<double>()
    );
    auto criterion = CrossEntropyLoss<double>();

    for (int i = 0; i < 1000; ++i) {
        net(train_imgs);
        auto loss = criterion(train_labels);
        net.update(1);
        auto pred_prob = net.data();

        std::cout << "Epoch: " << i + 1 << "  Loss: " << loss << "  Train Acc: " << accuracy(train_labels, pred_prob)
                  << std::endl;
    }

}
