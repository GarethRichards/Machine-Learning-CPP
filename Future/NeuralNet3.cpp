// NeuralNet1.cpp : Defines the entry point for the console application.
//
// An example written to implement the stochastic gradient descent learning algorithm 
// for a feed forward neural network. Gradients are calculated using back propagation.
// 
// Code is written to be a C++ version of network2.py from http://neuralnetworksanddeeplearning.com/chap3.html
// Variable and functions names follow the names used in the original Python
//
// This implementation aims to be slight better C++ rather than Python code ported to C++
//
// Uses the boost ublas library for linear algebra operations

#include "stdafx.h"

#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include "boost\numeric\ublas\vector.hpp"
#include "boost\numeric\ublas\matrix.hpp"
#include "mnist_loader.h"
#include <chrono>
#include "NeuralNet.h"

using namespace boost::numeric;
using namespace NeuralNet;

int main()
{
	using NeuralNet1 = NeuralNet::Network<double, 
						NeuralNet::CrossEntropyCost<double>,
						NeuralNet::ReLUActivation<double>>;
	std::vector<NeuralNet1::TrainingData> td, testData;
	// Load training data
	mnist_loader<double> loader("..\\Data\\train-images.idx3-ubyte",
		"..\\Data\\train-labels.idx1-ubyte", td);
	// Load test data
	mnist_loader<double> loader2("..\\Data\\t10k-images.idx3-ubyte",
		"..\\Data\\t10k-labels.idx1-ubyte", testData);

	double Lmbda = 0.1; //5.0;
	double eta = 0.03;  //0.5
	NeuralNet1 net({ 784, 30, 10 });
	/*
	net.SGD(td.begin(), td.begin() + 1000, 100, 10, eta, Lmbda, [&testData, &td, Lmbda](const NeuralNet1 &network, int Epoch) {
		std::cout << "Epoch " << Epoch << " : " << network.accuracy(testData.begin(), testData.end()) << " / " << testData.size() << std::endl;
		std::cout << "Epoch " << Epoch << " : " << network.accuracy(td.begin(), td.begin() + 1000) << " / " << 1000 << std::endl;
		std::cout << "Cost : " << network.total_cost(td.begin(), td.begin() + 1000, Lmbda) << std::endl;
		std::cout << "Cost : " << network.total_cost(testData.begin(), testData.end(), Lmbda) << std::endl;
	});
	*/
	NeuralNet1 net2({ 784, 30, 10 });
	net2.SGD(td.begin(), td.end(), 30, 10, eta, Lmbda, [&Lmbda,&testData,&td](const NeuralNet1 &network, int Epoch, double &eta) {
		// Lmbda can be manipulated in the feed back function
		std::cout << "Epoch " << Epoch << std::endl;
		std::cout << "Test accuracy     : " << network.accuracy(testData.begin(), testData.end()) << " / " << testData.size() << std::endl;
		std::cout << "Training accuracy : " << network.accuracy(td.begin(), td.end()) << " / " << td.size() << std::endl;
		std::cout << "Cost Training: " << network.total_cost(td.begin(), td.end(), Lmbda) << std::endl;
		std::cout << "Cost Test    : " << network.total_cost(testData.begin(), testData.end(), Lmbda) << std::endl;
	});

	return 0;
}
