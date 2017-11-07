#pragma once

#pragma once

#include <cmath>
#include <iostream>
#include <vector>
#include <random>
#include "boost\numeric\ublas\vector.hpp"
#include "boost\numeric\ublas\matrix.hpp"
#include <numeric> 

using namespace boost::numeric;

namespace NeuralNet {
	// Set up the random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Randomize as ublas vector
	template<typename T>
	void Randomize(ublas::vector<T> &vec)
	{
		std::normal_distribution<> d(0, 1);
		for (auto &e : vec) { e = d(gen); }
	}
	// Randomize as ublas matrix
	template<typename T>
	void Randomize(ublas::matrix<T> &m)
	{
		std::normal_distribution<> d(0, 1);
		T sx = sqrt(m.size2());
		for (auto &e : m.data()) { e = d(gen) / sx; }
	}
	// The sigmoid function.
	template<typename T>
	void sigmoid(ublas::vector<T> &v)
	{
		for (auto &iv : v) { iv = 1.0 / (1.0 + exp(-iv)); }
	}
	// Derivative of the sigmoid function.
	template<typename T>
	void sigmoid_prime(ublas::vector<T> &v)
	{
		for (auto &iv : v) {
			iv = 1.0 / (1.0 + exp(-iv));
			iv = iv*(1.0 - iv);
		}
	}

	template<typename T>
	class QuadraticCost {
	public:
		T cost_fn(const ublas::vector<T>& a,
			const ublas::vector<T>& y) const
		{
			return 0.5 *pow(norm_2(a - y));;
		}
		ublas::vector<T> cost_delta(const ublas::vector<T>& z, const ublas::vector<T>& a,
			const ublas::vector<T>& y) const
		{
			auto zp = z;
			sigmoid_prime(zp);
			return element_prod(a - y, zp);
		}

	};

	template<typename T>
	class CrossEntropyCost {
	public:
		// Return the cost associated with an output ``a`` and desired output
		// ``y``.  Note that np.nan_to_num is used to ensure numerical
		// stability.In particular, if both ``a`` and ``y`` have a 1.0
		// in the same slot, then the expression(1 - y)*np.log(1 - a)
		// returns nan.The np.nan_to_num ensures that that is converted
		// to the correct value(0.0).
		T cost_fn(const ublas::vector<T>& a,
			const ublas::vector<T>& y) const
		{
			T total(0);
			for (auto i = 0; i < a.size(); ++i)
			{
				total += -y(i)*log(a(i)) - (1 - y(i))*log(1 - a(i));
			}
			return total;
		}
		// Return the error delta from the output layer.  Note that the
		// parameter ``z`` is not used by the method.It is included in
		// the method's parameters in order to make the interface
		// consistent with the delta method for other cost classes.
		ublas::vector<T> cost_delta(const ublas::vector<T>& z, const ublas::vector<T>& a,
			const ublas::vector<T>& y) const
		{
			return a - y;
		}

	};

	template<typename T, typename CostPolicy>
	class Network : private CostPolicy {
	private:
		using BiasesVector = std::vector<ublas::vector<T>>;
		using WeightsVector = std::vector<ublas::matrix<T>>;
		std::vector<int> m_sizes;
		BiasesVector biases;
		WeightsVector weights;
	public:
		Network(std::vector<int> sizes)
			: m_sizes(sizes)
		{
			PopulateZeroWeightsAndBiases(biases, weights);
			for (auto &b : biases) Randomize(b);
			for (auto &w : weights) Randomize(w);
		}
		// Initalize the array of Baises and Matrix of weights
		void PopulateZeroWeightsAndBiases(BiasesVector &b, WeightsVector &w)  const
		{
			for (auto i = 1; i < m_sizes.size(); ++i)
			{
				b.push_back(ublas::zero_vector<T>(m_sizes[i]));
				w.push_back(ublas::zero_matrix<T>(m_sizes[i], m_sizes[i - 1]));
			}
		}
		// Returns the output of the network if the input is a
		ublas::vector<T> feedforward(ublas::vector<T> a) const
		{
			for (auto i = 0; i < biases.size(); ++i)
			{
				ublas::vector<T> c = prod(weights[i], a) + biases[i];
				sigmoid(c);
				a = c;
			}
			return a;
		}
		// Type definition of the Training data
		using TrainingData = std::pair<ublas::vector<T>, ublas::vector<T>>;
		//	Train the neural network using mini-batch stochastic
		//	gradient descent.The training_data is a vector of pairs
		// representing the training inputs and the desired
		//	outputs.The other non - optional parameters are
		//	self - explanatory.If test_data is provided then the
		//	network will be evaluated against the test data after each
		//	epoch, and partial progress printed out.This is useful for
		//	tracking progress, but slows things down substantially.
		void SGD(typename std::vector<TrainingData>::iterator td_begin,
			typename std::vector<TrainingData>::iterator td_end,
			int epochs, int mini_batch_size, T eta, T lmbda,
			std::function<void(const Network &,int Epoc)> feedback)
		{
			for (auto j = 0; j < epochs; j++)
			{
				std::random_shuffle(td_begin, td_end);
				for (auto td_i = td_begin; td_i < td_end; td_i += mini_batch_size) {
					update_mini_batch(td_i, mini_batch_size, eta, lmbda, std::distance(td_begin, td_end));
				}
				feedback(*this, j);
			}
		}
		// Update the network's weights and biases by applying
		//	gradient descent using backpropagation to a single mini batch.
		//	The "mini_batch" is a list of tuples "(x, y)", and "eta"
		//	is the learning rate."""
		void update_mini_batch(typename std::vector<TrainingData>::iterator td, int mini_batch_size, T eta,
			T lmbda, int n)
		{
			std::vector<ublas::vector<T>> nabla_b;
			std::vector<ublas::matrix<T>> nabla_w;
			PopulateZeroWeightsAndBiases(nabla_b, nabla_w);
			for (auto i = 0; i < mini_batch_size; ++i, td++) {
				ublas::vector<T> x = td->first; // test data
				ublas::vector<T> y = td->second; // expected result
				std::vector<ublas::vector<T>> delta_nabla_b;
				std::vector<ublas::matrix<T>> delta_nabla_w;
				PopulateZeroWeightsAndBiases(delta_nabla_b, delta_nabla_w);
				backprop(x, y, delta_nabla_b, delta_nabla_w);
				for (auto i = 0; i < biases.size(); ++i)
				{
					nabla_b[i] += delta_nabla_b[i];
					nabla_w[i] += delta_nabla_w[i];
				}
			}
			for (auto i = 0; i < biases.size(); ++i)
			{
				biases[i] -= eta / mini_batch_size * nabla_b[i];
				weights[i] = (1 - eta * (lmbda / n)) * weights[i] - (eta / mini_batch_size) * nabla_w[i];
			}
		}
		// Populates the gradent for the cost function for the biases in the vector nabla_b 
		// and the weights in nabla_w
		void backprop(const ublas::vector<T> &x, const ublas::vector<T> &y,
			std::vector<ublas::vector<T>> &nabla_b,
			std::vector<ublas::matrix<T>> &nabla_w)
		{
			auto activation = x;
			std::vector<ublas::vector<T>> activations; // Stores the activations of each layer
			activations.push_back(x);
			std::vector<ublas::vector<T>> zs; // The z vectors layer by layer
			for (auto i = 0; i < biases.size(); ++i) {
				ublas::vector<T> z = prod(weights[i], activation) + biases[i];
				zs.push_back(z);
				activation = z;
				sigmoid(activation);
				activations.push_back(activation);
			}
			// backward pass
			auto iActivations = activations.end() - 1;
			auto izs = zs.end() - 1;
			sigmoid_prime(*izs);
			ublas::vector<T> delta = cost_delta(*izs, *iActivations, y);
			auto ib = nabla_b.end() - 1;
			auto iw = nabla_w.end() - 1;
			*ib = delta;
			iActivations--;
			*iw = outer_prod(delta, trans(*iActivations));
			auto iWeights = weights.end();
			while (iActivations != activations.begin())
			{
				izs--; iWeights--; iActivations--; ib--; iw--;
				sigmoid_prime(*izs);
				delta = element_prod(prod(trans(*iWeights), delta), *izs);
				*ib = delta;
				*iw = outer_prod(delta, trans(*iActivations));
			}
		}
		// Return the vector of partial derivatives \partial C_x /
		//	\partial a for the output activations.
		int accuracy(typename std::vector<TrainingData>::iterator td_begin,
			typename std::vector<TrainingData>::iterator td_end) const
		{
			return count_if(td_begin, td_end, [this](const TrainingData &testElement) {
				auto res = feedforward(testElement.first);
				return (std::distance(res.begin(), max_element(res.begin(), res.end()))
					== std::distance(testElement.second.begin(), max_element(testElement.second.begin(), testElement.second.end())));
			});
		}
		// Return the total cost for the data set ``data``.  

		double total_cost(typename std::vector<TrainingData>::iterator td_begin,
			typename std::vector<TrainingData>::iterator td_end,
			T lmbda) const
		{
			T cost(0);
			cost = std::accumulate(td_begin, td_end, cost, [this](T cost,const TrainingData &td)
			{
				auto res = feedforward(td.first);
				return cost + cost_fn(res, td.second);
			});
			size_t count = std::distance(td_begin, td_end);
			cost /= static_cast<double>(count);
			T reg = std::accumulate(weights.begin(), weights.end(), 0.0, [this, lmbda, count](T reg,const ublas::matrix<T> &w)
			{
				return reg + .5 * (lmbda * pow(norm_frobenius(w), 2)) / static_cast<T>(count);
			});
			return cost + reg;
		}
	};

}
