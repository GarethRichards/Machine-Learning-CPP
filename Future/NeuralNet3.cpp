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
// Uses the boost ublas library for linear algebra operations

#include "stdafx.h"

#include "NeuralNet.h"
#include "boost\numeric\ublas\matrix.hpp"
#include "boost\numeric\ublas\vector.hpp"
#include "mnist_loader.h"
#include <chrono>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using namespace boost::numeric;
using namespace NeuralNet;

int main() {
	using NeuralNet1 = NeuralNet::Network<float, NeuralNet::CrossEntropyCost<float>, NeuralNet::ReLUActivation<float>>;
	std::vector<NeuralNet1::TrainingData> td, testData;
	try {
		// Load training data
		mnist_loader<float> loader("..\\Data\\train-images.idx3-ubyte", "..\\Data\\train-labels.idx1-ubyte", td);
		// Load test data
		mnist_loader<float> loader2("..\\Data\\t10k-images.idx3-ubyte", "..\\Data\\t10k-labels.idx1-ubyte", testData);
	}
	catch (const char *Error) {
		std::cout << "Error: " << Error << "\n";
		return 0;
	}
	float Lmbda = 0.1f; // 5.0;
	float eta = 0.03f;  // 0.5

	auto start = std::chrono::high_resolution_clock::now();
	auto periodStart = std::chrono::high_resolution_clock::now();
	NeuralNet1 net({ 784, 80, 20, 10 });
	net.SGD(td.begin(), td.end(), 20, 100, eta, Lmbda,
		[&periodStart, &Lmbda, &testData, &td](const NeuralNet1 &network, int Epoch, float &eta) {
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
		eta *= .95f;
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
    }
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
		auto x = net2.result(net2.feedforward(testData[0].first));
        auto y = net2.result(testData[2].second);
        std::cout << "looks like a " << x << " is a " << y << "\n";
	}
	return 0;
}
