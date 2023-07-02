//
// Created by H on 2023/6/1.
//

#ifndef MNIST_BLAS_UTILS_H
#define MNIST_BLAS_UTILS_H

#include "matrix.h"
Matrix<float> create_normal_matrix(int row, int col, float mean=0.0, float stddev=1.0);

Matrix<float> normalize_image_to_matrix(const Matrix<unsigned char>& images);

Matrix<float> label_to_onehot(const Matrix<unsigned char>& labels, int num_classes=10);

vector<int> range(int end);

template<typename DataType>
Matrix<DataType> choice_rows(const Matrix<DataType>& m, const vector<int>& indexs, int begin=0, int size=-1){
    if(size == -1) size = (int)indexs.size();
    Matrix<DataType> out(size, m.col_);
    for(int i = 0; i < size; i++){
        for(int j = 0; j < m.col_; j++){
            *out.get_data(i * m.col_ + j) = *m.get_data(indexs[i + begin] * m.col_ + j);
        }
    }
    return out;
}


float compute_loss(const Matrix<float>& probability, const Matrix<float>& onehot_labels);

Matrix<float> row_sum(const Matrix<float>& value);

Matrix<float> delta_sigmoid(const Matrix<float>& sigmoid_value);

Matrix<float> delta_relu(const Matrix<float>& relu_value);

float eval_test_accuracy(const Matrix<float>& probability, const Matrix<unsigned char>& labels);

Matrix<float> sqrt_matrix(const Matrix<float>& value);

#endif //MNIST_BLAS_UTILS_H
