//
// Created by 47925 on 2024/2/14.
//

#ifndef NEW_DL_DATATYPE_H
#define NEW_DL_DATATYPE_H
#include <Eigen/Eigen>

#define INT16 short int
#define INT32 int
#define INT64 long long int

#define UNSIGNED_INT16 unsigned short int
#define UNSIGNED_INT32 unsigned int
#define UNSIGNED_INT64 unsigned long long

#define FLOAT32 float
#define FLOAT64 double
#define FLOAT128 long double

#define DEFAULT_MATRIX_ONES_INT16 Eigen::MatrixX<INT16>::Ones(1,1)
#define DEFAULT_MATRIX_ONES_INT32 Eigen::MatrixX<INT32>::Ones(1,1)
#define DEFAULT_MATRIX_ONES_INT64 Eigen::MatrixX<INT64>::Ones(1,1)

#define DEFAULT_MATRIX_ONES_UNSIGNED_INT16 Eigen::MatrixX<UNSIGNED_INT16>::Ones(1,1)
#define DEFAULT_MATRIX_ONES_UNSIGNED_INT32 Eigen::MatrixX<UNSIGNED_INT32>::Ones(1,1)
#define DEFAULT_MATRIX_ONES_UNSIGNED_INT64 Eigen::MatrixX<UNSIGNED_INT64>::Ones(1,1)

#define DEFAULT_MATRIX_ONES_FLOAT32 Eigen::MatrixX<FLOAT32>::Ones(1,1)
#define DEFAULT_MATRIX_ONES_FLOAT64 Eigen::MatrixX<FLOAT64>::Ones(1,1)
#define DEFAULT_MATRIX_ONES_FLOAT128 Eigen::MatrixX<FLOAT128>::Ones(1,1)

#define DEFAULT_MATRIX_ZEROS_INT16 Eigen::MatrixX<INT16>::Zero(1,1)
#define DEFAULT_MATRIX_ZEROS_INT32 Eigen::MatrixX<INT32>::Zero(1,1)
#define DEFAULT_MATRIX_ZEROS_INT64 Eigen::MatrixX<INT64>::Zero(1,1)

#define DEFAULT_MATRIX_ZEROS_UNSIGNED_INT16 Eigen::MatrixX<UNSIGNED_INT16>::Zero(1,1)
#define DEFAULT_MATRIX_ZEROS_UNSIGNED_INT32 Eigen::MatrixX<UNSIGNED_INT32>::Zero(1,1)
#define DEFAULT_MATRIX_ZEROS_UNSIGNED_INT64 Eigen::MatrixX<UNSIGNED_INT64>::Zero(1,1)

#define DEFAULT_MATRIX_ZEROS_FLOAT32 Eigen::MatrixX<FLOAT32>::Zero(1,1)
#define DEFAULT_MATRIX_ZEROS_FLOAT64 Eigen::MatrixX<FLOAT64>::Zero(1,1)
#define DEFAULT_MATRIX_ZEROS_FLOAT128 Eigen::MatrixX<FLOAT128>::Zero(1,1)

#endif //NEW_DL_DATATYPE_H
