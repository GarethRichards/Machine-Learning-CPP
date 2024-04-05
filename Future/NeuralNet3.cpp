// NeuralNet3.cpp : Defines the entry point for the console application.
//
// An example written to implement the stochastic gradient descent learning
// algorithm for a feed forward neural network. Gradients are calculated using
// back propagation.
//
// Code is written to be a C++ version of network2.py from
// http://neuralnetworksanddeeplearning.com/chap3.html Variable and functions
// names follow the names used in the original Python
//
// This implementation aims to be slight better C++ rather than Python code
// ported to C++
//
// Uses the experimental/linalg extension for linear algebra operations

#include "NeuralNet.h"
#include "mnist_loader.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace NeuralNet;
using namespace std::filesystem;
using NeuralNet1 = NeuralNet::Network<float, NeuralNet::CrossEntropyCost<float>, NeuralNet::ReLUActivation<float>>;
using NetData = std::vector<NeuralNet1::TrainingData>;
using NetDataMem = std::vector<std::pair<std::vector<float>, std::vector<float>>>;

void test_net(const NeuralNet1 &net, const NetData &testData, int test_case);

int main() {
    NetData td;
    NetData testData;
    NetDataMem training_data_mem;
    NetDataMem test_data_mem;

    try {
        // Load training data
        const path train_images = path("..") / path("Data") / path("train-images.idx3-ubyte");
        const path train_labels = path("..") / path("Data") / path("train-labels.idx1-ubyte");
        mnist_loader<float> loader(train_images, train_labels, training_data_mem, td);
        // Load test data
        const path t10K_images = path("..") / path("Data") / path("t10k-images.idx3-ubyte");
        const path t10K_labels = path("..") / path("Data") / path("t10k-labels.idx1-ubyte");
        mnist_loader<float> loader2(t10K_images, t10K_labels, test_data_mem, testData);
    } catch (const char *Error) {
        std::cout << "Error: " << Error << "\n";
        return 0;
    }

    float Lmbda = 0.1f; // try 5.0;
    float eta = 0.03f;  // try 0.5
                        /*
                        auto start = std::chrono::high_resolution_clock::now();
                        auto periodStart = std::chrono::high_resolution_clock::now();
                        NeuralNet1 net({ 784, 100, 10 });
                        net.SGD(td.begin(), td.end(), 20, 100, eta, Lmbda,
                                [&periodStart, &Lmbda, &testData, &td](const NeuralNet1 &network, int Epoch, float &currenctEta) {
                                // eta can be manipulated in the feed back function
                                auto end = std::chrono::high_resolution_clock::now();
                                std::chrono::duration<double> diff = end - periodStart;
                                std::cout << "Epoch " << Epoch << " time taken: " << diff.count() << "\n";
                                std::cout << "Test accuracy     : " << network.accuracy(testData.begin(), testData.end()) << " / "
                                        << testData.size() << "\n";
                                std::cout << "Training accuracy : " << network.accuracy(td.begin(), td.end()) << " / " << td.size()
                                        << "\n";
                                std::cout << "Cost Training: " << network.total_cost(td.begin(), td.end(), Lmbda) << "\n";
                                std::cout << "Cost Test    : " << network.total_cost(testData.begin(), testData.end(), Lmbda)
                                        << std::endl;
                        currenctEta *= .95f;
                                periodStart = std::chrono::high_resolution_clock::now();
                        });
                        auto end = std::chrono::high_resolution_clock::now();
                        std::chrono::duration<double> diff = end - start;
                        std::cout << "Total time: " << diff.count() << "\n";
                        // write out net
                    {
                        std::ofstream f("./net-save.txt", std::ios::binary | std::ios::out);
                        f << net;
                        f.close();
                    }*/
                        // read it and use it
    NeuralNet1 net2;
    {
        std::fstream f("./net-save.txt", std::ios::binary | std::ios::in);
        if (!f.is_open()) {
            std::cout << "failed to open ./net-save.txt\n";
            return 0;
        } else {
            f >> net2;
        }
        // test total cost should be same as before
        std::cout << "Cost Test    : " << net2.total_cost(testData.begin(), testData.end(), Lmbda) << "\n";
        test_net(net2, testData, 0);
    }
    // test assigments

    NeuralNet1 net3(std::move(net2));
    test_net(net3, testData, 1);

    NeuralNet1 net4{};
    net4 = std::move(net3);
    test_net(net4, testData, 2);

    NeuralNet1 net5(net4);
    net4 = net5;
    test_net(net4, testData, 3);
    test_net(net5, testData, 4);

    return 0;
}

void test_net(const NeuralNet1 &net, const NetData &testData, int test_case) {
    std::vector<float> res(10);
    mdspan<float, std::dextents<size_t, 2>> nres(res.data(), (size_t)res.size(), 1);

    net.feedforward(testData[test_case].first, nres);
    auto x = net.result(nres);
    auto y = net.result(testData[test_case].second);
    std::cout << "looks like a " << x << " is a " << y << "\n";
    for (auto i = 0; i < 29; ++i) {
        std::string data("");
        for (auto j = 0; j < 29; ++j) {
            if (testData[test_case].first(i * 28 + j, 0) > .5)
                data += (char)254u;
            else
                data += " ";
        }
        std::cout << data << std::endl;
    }
}
