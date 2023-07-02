//
// Created by H on 2023/6/1.
//

#ifndef MNIST_BLAS_MATRIX_H
#define MNIST_BLAS_MATRIX_H

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <random>
#include <memory>
#include <iostream>
#include <cblas.h>
#include <cmath>
using namespace std;
static default_random_engine global_random_engine;

template<typename DataType>
class Matrix{
public:
    Matrix() = default;
    Matrix(int row, int col, const DataType* pdata = nullptr){
        row_ = row;
        col_ = col;
        data_size_ = row * col;
        if(pdata)
            memcpy(data_->data(), pdata, row_ * col_ * sizeof(DataType));
        else{
            data_ = make_shared<vector<DataType>>();
            data_->resize(data_size_);
        }
    }
//    ~Matrix(){delete [] data_;}

    void resize(int row, int col){
        row_ = row;
        col_ = col;
        data_size_ = row_ * col_;

        data_ = make_shared<vector<DataType>>();
        data_->resize(data_size_);
    }

    Matrix<DataType> copy() const {
        Matrix<DataType> out = *this;
        out.data_ = make_shared<vector<DataType>>(*this->data_);
        return out;
    }

    DataType* get_data(int index=0){
        return data_->data() + index;
    }

    const DataType* get_data(int index = 0) const{
        return data_->data() + index;
    }


    Matrix<DataType> T(){
        Matrix<DataType> out(col_, row_);
        for(int i = 0; i < row_; ++i){
            int index_1 = i * col_;
            int index_2 = i;
            for(int j = 0; j < col_; ++j, index_2 += out.col_)
                *out.get_data(index_2) = *get_data(index_1 + j);
        }
        return out;
    }

    Matrix<DataType>& sigmoid(){
        for(int i = 0; i < data_size_; i++){
            if(*get_data(i) < 0){
                *get_data(i) = exp(*get_data(i)) / (1 + exp(*get_data(i)));
            }else{
                *get_data(i) = 1 / (1 + exp(-*get_data(i)));
            }
        }
        return *this;
    }

    Matrix<DataType>& relu(){
        for(int i = 0; i < data_size_; ++i)
            *get_data(i) = max<DataType>(DataType(0), *get_data(i));
        return *this;
    }

    Matrix<DataType>& softmax(){
        for(int i = 0; i < row_; i++){
            DataType maxNum = *max_element(data_->data() + i * col_, data_->data() + (i + 1) * col_);
            DataType total = 0;
            for(int j = 0; j < col_; j++){
                total += exp(*get_data(i * col_ + j) - maxNum);
            }
            for(int j = 0; j < col_; j++){
                *get_data(i * col_ + j) = exp(*get_data(i * col_ + j) - maxNum) / total;
            }
        }
        return *this;
    }


public:
    int row_;
    int col_;
    int data_size_;
    shared_ptr<vector<DataType>> data_;
//    DataType* data_;
};


template<typename T>
void print_matrix(const Matrix<T>& a, bool show_data=true){
    cout<<a.row_<<" * "<<a.col_<<endl;
    if(!show_data)
        return;
    for(int i = 0; i < a.row_; ++i){
        for(int j = 0; j < a.col_; ++j){
            cout<<*a.get_data(i * a.col_ + j)<<" ";
        }
        cout<<endl;
    }
}

template<typename _TA, typename _TB>
Matrix<_TA> gemm_mul(const Matrix<_TA>& a, bool ta, const Matrix<_TB>& b, bool tb){
    // NMK,  a.shape = N x M, b.shape = M x K.  c.shape = N x K
    // 定义a和b，经过ta、tb后的shape
    int a_elastic_rows = ta ? a.col_ : a.row_;
    int a_elastic_cols = ta ? a.row_ : a.col_;
    int b_elastic_rows = tb ? b.col_ : b.row_;
    int b_elastic_cols = tb ? b.row_ : b.col_;
    Matrix<_TA> c(a_elastic_rows, b_elastic_cols);
    cblas_sgemm(
            CblasRowMajor,
            ta ? CblasTrans : CblasNoTrans,
            tb ? CblasTrans : CblasNoTrans,
            a_elastic_rows,
            b_elastic_cols,
            a_elastic_cols,
            1.0f,
            a.get_data(),
            a.col_,
            b.get_data(),
            b.col_,
            0.0f,
            c.get_data(),
            c.col_
    );
    return c;
}

template<typename TA, typename TB>
Matrix<TA> broadcast_mul (const Matrix<TA>& a, const Matrix<TB>& b){
    auto c = a.copy();
    for(int i = 0; i < a.data_size_; i++){
        *c.get_data(i) = (*a.get_data(i)) * (*b.get_data(i % b.col_));
    }
    return c;
}

template<typename TA, typename TB>
Matrix<TA> broadcast_div (const Matrix<TA>& a, const Matrix<TB>& b){
    auto c = a.copy();
    for(int i = 0; i < a.data_size_; i++){
        *c.get_data(i) = (*a.get_data(i)) / (*b.get_data(i % b.col_));
    }
    return c;
}

template<typename TA, typename TB>
Matrix<TA> broadcast_add (const Matrix<TA>& a, const Matrix<TB>& b){
    auto c = a.copy();
    for(int i = 0; i < a.data_size_; i++){
        *c.get_data(i) = (*a.get_data(i)) + (*b.get_data(i % b.col_));
    }
    return c;
}

template<typename TA, typename TB>
Matrix<TA> broadcast_sub (const Matrix<TA>& a, const Matrix<TB>& b){
    auto c = a.copy();
    for(int i = 0; i < a.data_size_; i++){
        *c.get_data(i) = (*a.get_data(i)) - (*b.get_data(i % b.col_));
    }
    return c;
}

template<typename TA, typename TB>
Matrix<TA> operator * (const Matrix<TA>& a, const Matrix<TB>& b){
    if(b.row_ == 1 && a.data_size_ != b.data_size_){
        return broadcast_mul(a, b);
    }
    if(a.row_ == 1 && a.data_size_ != b.data_size_){
        return broadcast_mul(b, a);
    }

    auto c = a.copy();
    for(int i = 0; i < a.data_size_; i++){
        *c.get_data(i) = (*a.get_data(i)) * (*b.get_data(i));
    }
    return c;
}

template<typename TA, typename TB>
Matrix<TA> operator / (const Matrix<TA>& a, const Matrix<TB>& b){
    if(b.row_ == 1 && a.data_size_ != b.data_size_){
        return broadcast_div(a, b);
    }
    if(a.row_ == 1 && a.data_size_ != b.data_size_){
        return broadcast_div(b, a);
    }
    auto c = a.copy();
    for(int i = 0; i < a.data_size_; i++){
        *c.get_data(i) = (*a.get_data(i)) / (*b.get_data(i));
    }
    return c;
}

template<typename TA, typename TB>
Matrix<TA> operator + (const Matrix<TA>& a, const Matrix<TB>& b){
    if(b.row_ == 1 && a.data_size_ != b.data_size_){
        return broadcast_add(a, b);
    }
    if(a.row_ == 1 && a.data_size_ != b.data_size_){
        return broadcast_add(b, a);
    }
    auto c = a.copy();
    for(int i = 0; i < a.data_size_; i++){
        *c.get_data(i) = (*a.get_data(i)) + (*b.get_data(i));
    }
    return c;
}

template<typename TA, typename TB>
Matrix<TA> operator - (const Matrix<TA>& a, const Matrix<TB>& b){
    if(b.row_ == 1 && a.data_size_ != b.data_size_){
        return broadcast_sub(a, b);
    }
    if(a.row_ == 1 && a.data_size_ != b.data_size_){
        return broadcast_sub(b, a);
    }
    auto c = a.copy();
    for(int i = 0; i < a.data_size_; i++){
        *c.get_data(i) = (*a.get_data(i)) - (*b.get_data(i));
    }
    return c;
}

template<typename TA, typename TB>
Matrix<TB> operator * (TA a, const Matrix<TB>& b){
    auto out = b.copy();
    for(int i = 0; i < out.data_size_; ++i)
        *out.get_data(i) = a * *b.get_data(i);
    return out;
}

template<typename TA, typename TB>
Matrix<TB> operator / (TA a, const Matrix<TB>& b){
    auto out = b.copy();
    for(int i = 0; i < out.data_size_; ++i)
        *out.get_data(i) = a / *b.get_data(i);
    return out;
}

template<typename TA, typename TB>
Matrix<TB> operator + (TA a, const Matrix<TB>& b){
    auto out = b.copy();
    for(int i = 0; i < out.data_size_; ++i)
        *out.get_data(i) = a + *b.get_data(i);
    return out;
}

template<typename TA, typename TB>
Matrix<TB> operator - (TA a, const Matrix<TB>& b){
    auto out = b.copy();
    for(int i = 0; i < out.data_size_; ++i)
        *out.get_data(i) = a * *b.get_data(i);
    return out;
}

template<typename TA, typename TB>
Matrix<TA> operator * (const Matrix<TA>& a, TB b){
    auto out = a.copy();
    for(int i = 0; i < out.data_size_; ++i)
        *out.get_data(i) = *a.get_data(i) * b;
    return out;
}

template<typename TA, typename TB>
Matrix<TA> operator / (const Matrix<TA>& a, TB b){
    auto out = a.copy();
    for(int i = 0; i < out.data_size_; ++i)
        *out.get_data(i) = *a.get_data(i) / b;
    return out;
}

template<typename TA, typename TB>
Matrix<TA> operator + (const Matrix<TA>& a, TB b){
    auto out = a.copy();
    for(int i = 0; i < out.data_size_; ++i)
        *out.get_data(i) = *a.get_data(i) + b;
    return out;
}

template<typename TA, typename TB>
Matrix<TA> operator - (const Matrix<TA>& a, TB b){
    auto out = a.copy();
    for(int i = 0; i < out.data_size_; ++i)
        *out.get_data(i) = *a.get_data(i) - b;
    return out;
}

#endif //MNIST_BLAS_MATRIX_H