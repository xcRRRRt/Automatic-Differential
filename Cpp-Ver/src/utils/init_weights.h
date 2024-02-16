//
// Created by 47925 on 2024/2/16.
//

#ifndef NEW_DL_INIT_WEIGHTS_H
#define NEW_DL_INIT_WEIGHTS_H

#include <random>
#include <cmath>
#include <ctime>
#include <Eigen/Eigen>

template<typename T>
Eigen::MatrixX<T> init_weight(int input_size, int output_size, bool uniform = false) {
    if (!uniform) {
        srand(time(nullptr));
        double v = sqrt(6. / (input_size + output_size));
        Eigen::MatrixX<T> temp(input_size, output_size);
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < output_size; ++j)
                temp(i, j) = (rand() % 200) * 0.01 - 1;
        }
        return v * temp;
    } else {
        double sigma = sqrt(2. / (input_size + output_size));
        std::default_random_engine engine;
        engine.seed(time(nullptr));
        std::uniform_real_distribution<T> uniformRealDistribution(0, sigma);
        Eigen::MatrixX<T> temp(input_size, output_size);
        for (int i = 0; i < input_size; ++i) {
            for (int j = 0; j < output_size; ++j)
                temp(i, j) = uniformRealDistribution(engine);
        }
        return sigma * temp;
    }
}

#endif //NEW_DL_INIT_WEIGHTS_H
