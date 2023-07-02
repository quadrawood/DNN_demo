//
// Created by H on 2023/6/4.
//

#ifndef MNIST_BLAS_OPTIMIZER_H
#define MNIST_BLAS_OPTIMIZER_H

#include "matrix.h"
#include "utils.h"


struct SGDMomentum{
    float momentum = 0.9;
    vector<Matrix<float>> delta_momentums;
    void update_params(const vector<Matrix<float>*>& params, const vector<Matrix<float>*>& grads, float lr){

        if(delta_momentums.size() != params.size())
            delta_momentums.resize(params.size());

        for(int i =0 ; i < params.size(); ++i){
            auto& delta_momentum = delta_momentums[i];
            auto& param          = *params[i];
            auto& grad           = *grads[i];

            if(delta_momentum.data_size_ == 0)
                delta_momentum.resize(param.row_, param.col_);

            delta_momentum = momentum * delta_momentum + (1 - momentum) * grad;
            param          = param - lr * delta_momentum;
        }
    }
};

struct AdaGrad{
    vector<Matrix<float>> accumulative_grids;
    void update_params(const vector<Matrix<float>*>& params, const vector<Matrix<float>*>& grads, float lr){

        if(accumulative_grids.size() != params.size())
            accumulative_grids.resize(params.size());

        for(int i =0 ; i < params.size(); ++i){
            auto& accumulative_grid = accumulative_grids[i];
            auto& param          = *params[i];
            auto& grad           = *grads[i];

            if(accumulative_grid.data_size_ == 0)
                accumulative_grid.resize(param.row_, param.col_);

            accumulative_grid = accumulative_grid + grad * grad;
            param = param - lr * (1 / sqrt_matrix(accumulative_grid)) * grad;
        }
    }
};

struct RMSProp{
    float decay = 0.8;
    vector<Matrix<float>> accumulative_grids;
    void update_params(const vector<Matrix<float>*>& params, const vector<Matrix<float>*>& grads, float lr){

        if(accumulative_grids.size() != params.size())
            accumulative_grids.resize(params.size());

        for(int i =0 ; i < params.size(); ++i){
            auto& accumulative_grid = accumulative_grids[i];
            auto& param          = *params[i];
            auto& grad           = *grads[i];

            if(accumulative_grid.data_size_ == 0)
                accumulative_grid.resize(param.row_, param.col_);

            accumulative_grid = decay * accumulative_grid + (1 - decay) * grad * grad;
            param = param - lr * (1 / sqrt_matrix(accumulative_grid)) * grad;
        }
    }
};

struct Adam{
    int t = 0;
    float momentum = 0.9;
    float decay = 0.8;
    vector<Matrix<float>> delta_momentums;
    vector<Matrix<float>> accumulative_grids;
    void update_params(const vector<Matrix<float>*>& params, const vector<Matrix<float>*>& grads, float lr){
        t += 1;

        if(delta_momentums.size() != params.size())
            delta_momentums.resize(params.size());

        if(accumulative_grids.size() != params.size())
            accumulative_grids.resize(params.size());

        for(int i =0 ; i < params.size(); ++i){
            auto& delta_momentum = delta_momentums[i];
            auto& accumulative_grid = accumulative_grids[i];
            auto& param          = *params[i];
            auto& grad           = *grads[i];

            if(delta_momentum.data_size_ == 0){
                delta_momentum.resize(param.row_, param.col_);
            }

            if(accumulative_grid.data_size_ == 0){
                accumulative_grid.resize(param.row_, param.col_);
            }

            delta_momentum = momentum * delta_momentum + (1 - momentum) * grad;
            accumulative_grid = decay * accumulative_grid + (1 - decay) * grad * grad;

//            auto delta_momentum_ = delta_momentum / (1 - pow(momentum, t));
//            auto accumulative_grid_ = accumulative_grid / (1 - pow(decay, t));

            lr = lr * sqrt((1 - pow(decay, t)) / (1 - pow(momentum, t)));
            param = param - lr * delta_momentum * (1 / sqrt_matrix(accumulative_grid));
        }
    }
};


#endif //MNIST_BLAS_OPTIMIZER_H
