//
// Created by 47925 on 2024/2/16.
//

#ifndef NEW_DL_READ_DATA_H
#define NEW_DL_READ_DATA_H

#include <string>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include "Eigen/Eigen"

int ReverseInt(int i) {
    unsigned char ch1, ch2, ch3, ch4;
    ch1 = i & 255;
    ch2 = (i >> 8) & 255;
    ch3 = (i >> 16) & 255;
    ch4 = (i >> 24) & 255;
    return ((int) ch1 << 24) + ((int) ch2 << 16) + ((int) ch3 << 8) + ch4;
}

void read_label(const std::string &filename, Eigen::MatrixX<double> &labels) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        file.read((char *) &magic_number, sizeof(magic_number));
        file.read((char *) &number_of_images, sizeof(number_of_images));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        labels = Eigen::MatrixX<double>::Zero(number_of_images, 10);

        for (int i = 0; i < number_of_images; i++) {
            unsigned char label = 0;
            file.read((char *) &label, sizeof(label));
            labels(i, int(label)) = 1;
        }

    }
}

template<typename T>
void read_images(const std::string &filename, Eigen::MatrixX<T> &images) {
    std::ifstream file(filename, std::ios::binary);
    if (file.is_open()) {
        int magic_number = 0;
        int number_of_images = 0;
        int n_rows = 0;
        int n_cols = 0;
        file.read((char *) &magic_number, sizeof(magic_number));
        file.read((char *) &number_of_images, sizeof(number_of_images));
        file.read((char *) &n_rows, sizeof(n_rows));
        file.read((char *) &n_cols, sizeof(n_cols));
        magic_number = ReverseInt(magic_number);
        number_of_images = ReverseInt(number_of_images);
        n_rows = ReverseInt(n_rows);
        n_cols = ReverseInt(n_cols);
        images = Eigen::MatrixX<T>(number_of_images, n_rows * n_cols);

        for (int i = 0; i < number_of_images; i++) {
            for (int r = 0; r < n_rows; r++) {
                for (int c = 0; c < n_cols; c++) {
                    unsigned char image = 0;
                    file.read((char *) &image, sizeof(image));
                    images(i, r * n_rows + c) = image;
                }
            }
        }
        images /= 255;
    }
}

#endif //NEW_DL_READ_DATA_H
