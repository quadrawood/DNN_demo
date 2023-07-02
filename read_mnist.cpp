//
// Created by H on 2023/5/29.
//

#include "read_mnist.h"

using namespace std;
using namespace cv;
#define InvertBit(number)           (((number & 0x000000FF) << 24) | ((number & 0x0000FF00) << 8) | ((number & 0x00FF0000) >> 8) | ((number & 0xFF000000) >> 24))


uint32_t swap_endian(uint32_t val)
{
    val = ((val << 8) & 0xFF00FF00) | ((val >> 8) & 0xFF00FF);
    return (val << 16) | (val >> 16);
}


void Read_image_label(const string& mnist_img_path, const string& mnist_label_path, const string& saved_path)
{
    CreateDirectory(saved_path.c_str(), NULL);
    ofstream saveLabel;
    string labelSaveFile = saved_path + "label.txt";
    saveLabel.open(labelSaveFile, ios_base::out);
    ifstream mnist_image(mnist_img_path, ios::in | ios::binary);
    ifstream mnist_label(mnist_label_path, ios::in | ios::binary);
    if (!mnist_image.is_open())
    {
        cout << "open mnist image file error!" << endl;
        return;
    }
    if (!mnist_label.is_open())
    {
        cout << "open mnist label file error!" << endl;
        return;
    }

    uint32_t magic;//文件中的魔术数(magic number)
    uint32_t num_items;//mnist图像集文件中的图像数目
    uint32_t num_label;//mnist标签集文件中的标签数目
    uint32_t rows;//图像的行数
    uint32_t cols;//图像的列数
    //读魔术数
    mnist_image.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2051)
    {
        cout << "this is not the mnist image file" << endl;
        return;
    }
    mnist_label.read(reinterpret_cast<char*>(&magic), 4);
    magic = swap_endian(magic);
    if (magic != 2049)
    {
        cout << "this is not the mnist label file" << endl;
        return;
    }
    mnist_image.read(reinterpret_cast<char*>(&num_items), 4);
    num_items = swap_endian(num_items);
    mnist_label.read(reinterpret_cast<char*>(&num_label), 4);
    num_label = swap_endian(num_label);
    cout << num_items << endl;
    if (num_items != num_label)
    {
        cout << "the image file and label file are not a pair" << endl;
    }
    for (int i = 0; i < num_label; i++)
    {
        unsigned char label = 0;
        mnist_label.read((char*)&label, sizeof(label));
        saveLabel<<(unsigned int)label<<endl;
    }

    mnist_image.read(reinterpret_cast<char*>(&rows), 4);
    rows = swap_endian(rows);
    mnist_image.read(reinterpret_cast<char*>(&cols), 4);
    cols = swap_endian(cols);
    for (int i = 0; i < num_items; i++)
    {
        char* pixels = new char[rows * cols];
        mnist_image.read(pixels, rows * cols);
        char label;
        mnist_label.read(&label, 1);
        Mat image(rows, cols, CV_8UC1);
        for (int m = 0; m != rows; m++)
        {
            uchar* ptr = image.ptr<uchar>(m);
            for (int n = 0; n != cols; n++)
            {
                if (pixels[m * cols + n] == 0)
                    ptr[n] = 0;
                else
                    ptr[n] = 255;
            }
        }
        string imageSaveFile = saved_path + to_string(i) + ".jpg";
        imwrite(imageSaveFile, image);
    }
}

Matrix<unsigned char> load_mnist_labels(const string& file){
    ifstream in(file, ios::binary | ios::in);
    int header[2];
    in.read((char*)header, sizeof(header));
    int num_labels = InvertBit(header[1]);
    Matrix<unsigned char> out(num_labels, 1);
    in.read((char*)out.get_data(), out.data_size_);
    return out;
}


Matrix<unsigned char> load_mnist_images(const string& file){
    ifstream in(file, ios::binary | ios::in);
    int header[4];
    in.read((char*)header, sizeof(header));
    int num_images  = InvertBit(header[1]);
    int rows        = InvertBit(header[2]);
    int cols        = InvertBit(header[3]);
    printf("%d\n%d\n", header[1], num_images);
    Matrix<unsigned char> out(num_images, rows * cols);
    in.read((char*)out.get_data(), out.data_size_);
    return out;
}


