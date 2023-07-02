//
// Created by H on 2023/5/29.
//

#ifndef MNIST_LAB_READ_MNIST_H
#define MNIST_LAB_READ_MNIST_H

#include <iostream>
#include <fstream>
#include <string>
#include <Windows.h>
#include "matrix.h"
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;


uint32_t swap_endian(uint32_t val);

void Read_image_label(const string& mnist_img_path, const string& mnist_label_path, const string& saved_path);

Matrix<unsigned char> load_mnist_images(const string& file);

Matrix<unsigned char> load_mnist_labels(const string& file);


#endif //MNIST_LAB_READ_MNIST_H
