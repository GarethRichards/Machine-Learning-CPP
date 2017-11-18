// NeuralNet1.cpp : Defines the entry point for the console application.
//
// An example written to implement the stochastic gradient descent learning algorithm
// NeuralNet1.cpp : Defines the entry point for the console application.
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
	using NeuralNet1 = NeuralNet::Network<double, NeuralNet::CrossEntropyCost<double>>;
	std::vector<NeuralNet1::TrainingData> td, testData;
	try {
		// Load training data
		mnist_loader<double> loader("..\\Data\\train-images.idx3-ubyte", "..\\Data\\train-labels.idx1-ubyte", td);
		// Load test data
		mnist_loader<double> loader2("..\\Data\\t10k-images.idx3-ubyte", "..\\Data\\t10k-labels.idx1-ubyte", testData);
	}
	catch (const char *ex) {
		std::cout << "Error " << ex << std::endl;
		return 0;
	}
	double Lmbda = 5.0;
	NeuralNet1 net({ 784, 30, 10 });
	net.SGD(td.begin(), td.begin() + 1000, 400, 10, 0.5, Lmbda,
		[&testData, &td, Lmbda](const NeuralNet1 &network, int Epoch) {
		std::cout << "Epoch " << Epoch << " : " << network.accuracy(testData.begin(), testData.end()) << " / "
			<< testData.size() << std::endl;
		std::cout << "Epoch " << Epoch << " : " << network.accuracy(td.begin(), td.begin() + 1000) << " / "
			<< 1000 << std::endl;
		std::cout << "Cost : " << network.total_cost(td.begin(), td.begin() + 1000, Lmbda) << std::endl;
		std::cout << "Cost : " << network.total_cost(testData.begin(), testData.end(), Lmbda) << std::endl;
	});

	net.SGD(td.begin(), td.end(), 30, 10, 0.5, Lmbda, [&testData, &td, Lmbda](const NeuralNet1 &network, int Epoch) {
		std::cout << "Epoch " << Epoch << " : " << network.accuracy(testData.begin(), testData.end()) << " / "
			<< testData.size() << std::endl;
		std::cout << "Epoch " << Epoch << " : " << network.accuracy(td.begin(), td.end()) << " / " << td.size()
			<< std::endl;
		std::cout << "Cost : " << network.total_cost(td.begin(), td.end(), Lmbda) << std::endl;
		std::cout << "Cost : " << network.total_cost(testData.begin(), testData.end(), Lmbda) << std::endl;
	});

	return 0;
}
