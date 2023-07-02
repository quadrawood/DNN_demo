//
// Created by H on 2023/6/1.
//

#include "utils.h"

Matrix<float> create_normal_matrix(int row, int col, float mean, float stddev){
    normal_distribution<float> norm(mean, stddev);
    Matrix<float> out(row, col);
    for(int i = 0; i < row; ++i){
        for(int j = 0; j < col; ++j){
            *out.get_data(i * col + j) = norm(global_random_engine);
        }
    }
    return out;
}

Matrix<float> normalize_image_to_matrix(const Matrix<unsigned char>& images){
    Matrix<float> out(images.row_, images.col_);
    for(int i = 0; i < images.data_size_; i++)
        *out.get_data(i) = ((float)*images.get_data(i) / 255.0f - 0.1307f) / 0.3081f;
    return out;
}

Matrix<float> label_to_onehot(const Matrix<unsigned char>& labels, int num_classes){
    Matrix<float> out(labels.row_, 10);
    for(int i = 0; i < out.row_; ++i)
        *out.get_data(i * 10 + *labels.get_data(i)) = 1;
    return out;
}

vector<int> range(int end){
    vector<int> out(end);
    for(int i = 0; i < end; ++i)
        out[i] = i;
    return out;
}
//
////template<typename DataType>
////Matrix<DataType> choice_rows(const Matrix<DataType>& m, const vector<int>& indexs, int begin, int size){
////    if(size == -1) size = (int)indexs.size();
////    Matrix<DataType> out(size, m.col_);
////    for(int i = 0; i < size; i++){
////        for(int j = 0; j < m.col_; j++){
////            out[i * m.col_ + j] = out[indexs[i + begin] * m.col_ + j];
////        }
////    }
////    return out;
////}
//
//
float compute_loss(const Matrix<float>& probability, const Matrix<float>& onehot_labels){
    float eps = 1e-5;
    float sum_loss  = 0;
    for(int i = 0; i < probability.data_size_; i++){
        auto y = *onehot_labels.get_data(i);
        auto p = *probability.get_data(i);
        p = max(min(p, 1 - eps), eps);
        sum_loss += -(y * log(p) + (1 - y) * log(1 - p));
    }
    return sum_loss / (float)probability.row_;
}

Matrix<float> row_sum(const Matrix<float>& value){
    Matrix<float> out(1, value.col_);
    for(int i = 0; i < value.data_size_; ++i)
        *out.get_data(i % value.col_) += *value.get_data(i);
    return out;
}

Matrix<float> delta_sigmoid(const Matrix<float>& sigmoid_value){
    auto out = sigmoid_value.copy();
    for(int i = 0; i < out.data_size_; i++)
        *out.get_data(i) = *out.get_data(i) * (1 - *out.get_data(i));
    return out;
}

Matrix<float> delta_relu(const Matrix<float>& relu_value){
    auto out = relu_value.copy();
    for(int i = 0; i < out.data_size_; i++)
        *out.get_data(i) = *out.get_data(i) <= 0 ? 0.0 : 1.0;
    return out;
}

////Matrix<float> delta_relu(const Matrix<float>& grad, const Matrix<float>& x){
////    auto out = grad.copy();
////    auto optr = out.ptr();
////    auto xptr = x.ptr();
////    for(int i = 0; i < out.numel(); ++i, ++optr, ++xptr){
////        if(*xptr <= 0)
////            *optr = 0;
////    }
////    return out;
////}
//
float eval_test_accuracy(const Matrix<float>& probability, const Matrix<unsigned char>& labels){
    int success = 0;
    for(int i = 0; i < probability.row_; ++i){
        int predict_label = 0;
        float m = -1;
        for(int j = 0; j < probability.col_; j++){

            if(m < *probability.get_data(i * probability.col_ + j)){
                m = *probability.get_data(i * probability.col_ + j);
                predict_label = j;
            }
        }
        if(predict_label == (int)*labels.get_data(i))
            success++;
    }
    return (float)success / (float)probability.row_;
}


Matrix<float> sqrt_matrix(const Matrix<float>& value){
    auto out = value.copy();
    for(int i = 0; i < out.data_size_; i++)
        *out.get_data(i) = (float)sqrt(*out.get_data(i) + 1e-7);
    return out;
}