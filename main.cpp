#include "read_mnist.h"
#include "matrix.h"
#include "utils.h"
#include "optimizer.h"





void train(){
    auto train_images = load_mnist_images("..\\mnist_data\\train-images.idx3-ubyte");
    auto train_labels = load_mnist_labels("..\\mnist_data\\train-labels.idx1-ubyte");
    auto train_norm_images    = normalize_image_to_matrix(train_images);
    auto train_onehot_labels  = label_to_onehot(train_labels);

    auto test_images  = load_mnist_images("..\\mnist_data\\t10k-images.idx3-ubyte");
    auto test_labels  = load_mnist_labels("..\\mnist_data\\t10k-labels.idx1-ubyte");
    auto test_norm_images     = normalize_image_to_matrix(test_images);
    auto test_onehot_labels   = label_to_onehot(test_labels);



    int num_images  = train_norm_images.row_;
    int num_input   = train_norm_images.col_;
    int num_hidden  = 1024;
    int num_output  = 10;
    int num_epoch   = 10;
    float lr        = 1e-3;
    int batch_size  = 256;
    float momentum  = 0.9f;
    float decay     = 0.99f;
    int num_batch_per_epoch = num_images / batch_size;
    auto image_indexs = range(num_images);

    Matrix<float> input_to_hidden  = create_normal_matrix(num_input,  num_hidden, 0, 2.0f / sqrt((float)(num_input + num_hidden)));
    Matrix<float> hidden_bias(1, num_hidden);
    Matrix<float> hidden_to_output = create_normal_matrix(num_hidden, num_output, 0, 1.0f / sqrt((float)(num_hidden + num_output)));
    Matrix<float> output_bias(1, num_output);
//    SGDMomentum optimizer;
//    optimizer.momentum = momentum;

//    AdaGrad optimizer;

//    RMSProp optimizer;
//    optimizer.decay = decay;

    Adam optimizer;
    optimizer.t = 0;
    optimizer.momentum = momentum;
    optimizer.decay = decay;



    cout<<"train start"<<endl;
    for(int epoch = 0; epoch < num_epoch; ++epoch){

        if(epoch == 8){
            lr *= 0.1;
        }

        // 打乱索引
        shuffle(image_indexs.begin(), image_indexs.end(), global_random_engine);

        // 开始循环所有的batch
        for(int ibatch = 0; ibatch < num_batch_per_epoch; ++ibatch){

            // 前向过程
            auto x           = choice_rows(train_norm_images,   image_indexs, ibatch * batch_size, batch_size);
            auto y           = choice_rows(train_onehot_labels, image_indexs, ibatch * batch_size, batch_size);
            auto hidden      = gemm_mul(x,          false, input_to_hidden,  false) + hidden_bias;
            auto hidden_act  = hidden.relu();
            auto output      = gemm_mul(hidden_act, false, hidden_to_output, false) + output_bias;
//            auto probability = output.sigmoid();
            auto probability = output.softmax();
            float loss       = compute_loss(probability, y);
            if(ibatch % 50 == 0){
                cout<<"Epoch: "<<epoch <<"  Loss:"<<loss<<endl;
            }

            // 反向过程
            // C = AB
            // dA = G * BT
            // dB = AT * G
            // loss部分求导，loss对output求导
            auto doutput           = (probability - y) / batch_size;
//            print_matrix(doutput);

            // 第二个Linear求导
            auto doutput_bias      = row_sum(doutput);
            auto dhidden_to_output = gemm_mul(hidden_act, true, doutput, false);
            auto dhidden_act       = gemm_mul(doutput, false, hidden_to_output, true);

            // 第一个Linear输出求导

            auto drelu             = delta_relu(hidden);
            auto dhidden           = drelu * dhidden_act;

            // 第一个Linear求导
            auto dinput_to_hidden  = gemm_mul(x, true, dhidden, false);
            auto dhidden_bias      = row_sum(dhidden);

            // 调用优化器来调整更新参数
            optimizer.update_params(
                    {&input_to_hidden,  &hidden_bias,  &hidden_to_output,  &output_bias},
                    {&dinput_to_hidden, &dhidden_bias, &dhidden_to_output, &doutput_bias},
                    lr);
        }

        // 模型对测试集进行测试，并打印精度
        auto test_hidden      = (gemm_mul(test_norm_images, false, input_to_hidden, false) + hidden_bias).relu();
        auto test_probability = (gemm_mul(test_hidden,false, hidden_to_output, false)     + output_bias).sigmoid();
        float accuracy        = eval_test_accuracy(test_probability, test_labels);
        float test_loss       = compute_loss(test_probability, test_onehot_labels);
        cout<<"Test Accuracy: "<<accuracy * 100<<endl;
//        INFO("Test Accuracy: %.2f %%, Loss: %f", accuracy * 100, test_loss);
    }




}



int main() {

//    Read_image_label("..\\mnist_data\\train-images.idx3-ubyte", "..\\mnist_data\\train-labels.idx1-ubyte", "C:\\HMS\\_CODE\\C++\\mnist_dataset\\trainData");
//    Read_image_label("..\\mnist_data\\t10k-images.idx3-ubyte", "..\\mnist_data\\t10k-labels.idx1-ubyte", "C:\\HMS\\_CODE\\C++\\mnist_dataset\\testData");


    auto m1 = create_normal_matrix(2, 3);
    auto m2 = create_normal_matrix(3, 2);
    auto m3 = gemm_mul(m1, false, m2, false);

    print_matrix(m1);
    print_matrix(m2);
    print_matrix(m3);
    m3.softmax();
    print_matrix(m3);


    train();



    return 0;
}

