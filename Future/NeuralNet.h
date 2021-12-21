#pragma once
//
//  Copyright (c) 2017
//  Gareth Richards
//
// NeuralNet.h Definition for NeuralNet namespace contains the following classes
// Network - main class containing the implemention of the NeuralNet
// The following Cost policies can be applied to this class.
// Cost Policies:
//		QuadraticCost
//		CrossEntropyCost
// Activation Policies:
//		SigmoidActivation
//		TanhActivation
//		ReLUActivation

#include "boost\numeric\ublas\matrix.hpp"
#include "boost\numeric\ublas\vector.hpp"
#include <cmath>
#include <execution>
#include <mutex>
#include <numeric>
#include <random>
#include <vector>
#include <functional>
#include <ranges>

using namespace boost::numeric;

namespace NeuralNet {
	// Set up the random number generator
	std::random_device rd;
	std::mt19937 gen(rd());

	// Randomize as ublas vector
	template <typename T> 
	void Randomize(ublas::vector<T> &vec) {
		std::normal_distribution<T> d(0, 1);
        std::ranges::for_each(vec, [&](T& e) { e = d(gen); });
	}
	// Randomize as ublas matrix
	template <typename T> 
	void Randomize(ublas::matrix<T> &m) {
		std::normal_distribution<T> d(0, 1);
		T sx = sqrt(static_cast<T>(m.size2()));
        std::ranges::for_each(m.data(), [&](auto &e) { e = d(gen) / sx; });
	}

	// The sigmoid function.
	template <typename T> 
	class SigmoidActivation {
	public:
		void Activation(ublas::vector<T> &v) const {
			constexpr T one = 1.0;
            std::ranges::for_each(v, [](T &iv) { iv = one / (one + exp(-iv)); });
		}
		void ActivationPrime(ublas::vector<T> &v) const {
			constexpr T one = 1.0;
            std::ranges::for_each(v, [](T &iv) { 
				iv = one / (one + exp(-iv)); 
				iv = iv * (one - iv);
				});
		}
	};

	// The tanh function.
	template <typename T>
	class TanhActivation {
	public:
		void Activation(ublas::vector<T> &v) const {
			constexpr T one = 1.0;
			constexpr T two = 2.0;
			for (auto &iv : v) {
				iv = (one + tanh(iv)) / two;
			}
		}
		void ActivationPrime(ublas::vector<T> &v) const {
			constexpr T two = 2.0;
			for (auto &iv : v) {
				iv = pow(two / (exp(-iv) + exp(iv)), two) / two;
			}
		}
	};

	// The ReLU function.
	template <typename T> 
	class ReLUActivation {
	public:
		void Activation(ublas::vector<T> &v) const {
			constexpr T zero = 0.0;
			for (auto &iv : v) {
				iv = std::max(zero, iv);
			}
		}
		void ActivationPrime(ublas::vector<T> &v) const {
			constexpr T zero = 0.0;
			constexpr T one = 1.0;
			for (auto &iv : v) {
				iv = iv < zero ? zero : one;
			}
		}
	};

	template <typename T>
	class QuadraticCost {
	public:
		T cost_fn(const ublas::vector<T> &a, const ublas::vector<T> &y) const {
			return 0.5 * pow(norm_2(a - y));
			;
		}
		ublas::vector<T> cost_delta(const ublas::vector<T> &z, const ublas::vector<T> &a, const ublas::vector<T> &y) const {
			auto zp = z;
			this->ActivationPrime(zp);
			return element_prod(a - y, zp);
		}
	};

	template <typename T> 
	class CrossEntropyCost {
	public:
		// Return the cost associated with an output ``a`` and desired output
		// ``y``.  Note that np.nan_to_num is used to ensure numerical
		// stability.In particular, if both ``a`` and ``y`` have a 1.0
		// in the same slot, then the expression(1 - y)*np.log(1 - a)
		// returns nan.The np.nan_to_num ensures that that is converted
		// to the correct value(0.0).
		T cost_fn(const ublas::vector<T> &a, const ublas::vector<T> &y) const {
			constexpr T zero = 0.0;
			constexpr T one = 1.0;
			T total(zero);
			for (auto i = 0; i < a.size(); ++i) {
				total += a(i) == zero ? zero : -y(i) * log(a(i));
				total += a(i) >= one ? zero : -(one - y(i)) * log(one - a(i));
			}
			return total;
		}
		// Return the error delta from the output layer.  Note that the
		// parameter ``z`` is not used by the method.It is included in
		// the method's parameters in order to make the interface
		// consistent with the delta method for other cost classes.
		ublas::vector<T> cost_delta(const ublas::vector<T> &z, const ublas::vector<T> &a, const ublas::vector<T> &y) const {
			(void)z; // not used by design
			return a - y;
		}
	};

	template <typename T, typename CostPolicy, typename ActivationPolicy>
    requires std::floating_point<T>
	class Network : private CostPolicy, private ActivationPolicy {
	private:
		using BiasesVector = std::vector<ublas::vector<T>>;
		using WeightsVector = std::vector<ublas::matrix<T>>;

	public:
		// Type definition of the Training data
		using TrainingData = std::pair<ublas::vector<T>, ublas::vector<T>>;
		using TrainingDataIterator = typename std::vector<TrainingData>::iterator;
		using TrainingDataVector = std::vector<TrainingData>;

	protected:
		class NetworkData {
		public:
			std::vector<int> m_sizes;
			BiasesVector biases;
			WeightsVector weights;
			NetworkData() {}
			NetworkData(const std::vector<int> &m_sizes) : m_sizes(m_sizes) { PopulateZeroWeightsAndBiases(); }
			NetworkData(const NetworkData &other) {
				biases = other.biases;
				weights = other.weights;
			}
			void PopulateZeroWeightsAndBiases() {
				for (auto i = 1; i < m_sizes.size(); ++i) {
					biases.push_back(ublas::zero_vector<T>(m_sizes[i]));
					weights.push_back(ublas::zero_matrix<T>(m_sizes[i], m_sizes[i - 1]));
				}
			}
			NetworkData &operator+=(const NetworkData &rhs) {
				for (auto j = 0; j < biases.size(); ++j) {
					biases[j] += rhs.biases[j];
					weights[j] += rhs.weights[j];
				}
				return *this;
			}
            friend NetworkData operator+(NetworkData lhs, const NetworkData &rhs) {
                lhs += rhs; // reuse compound assignment
                return lhs;
            }
			void Randomize() {
				for (auto &b : biases)
					NeuralNet::Randomize(b);
				for (auto &w : weights)
					NeuralNet::Randomize(w);
			}
		};

		NetworkData nd;
	public:
      Network() {};
      Network(const std::vector<int> &sizes) : nd(sizes) { nd.Randomize(); }
		// Initalize the array of Biases and Matrix of weights

		// Returns the output of the network if the input is a
		ublas::vector<T> feedforward(ublas::vector<T> a) const {
			for (auto i = 0; i < nd.biases.size(); ++i) {
				ublas::vector<T> c = prod(nd.weights[i], a) + nd.biases[i];
				this->Activation(c);
				a = c;
			}
			return a;
		}
		//	Train the neural network using mini-batch stochastic
		//	gradient descent.The training_data is a vector of pairs
		// representing the training inputs and the desired
		//	outputs.The other non - optional parameters are
		//	self - explanatory.If test_data is provided then the
		//	network will be evaluated against the test data after each
		//	epoch, and partial progress printed out.This is useful for
		//	tracking progress, but slows things down substantially.
		//	The lmbda parameter can be altered in the feedback function
		void SGD(TrainingDataIterator td_begin, TrainingDataIterator td_end, int epochs, int mini_batch_size, T eta,
			T lmbda, std::function<void(const Network &, int Epoc, T &lmbda)> feedback) {
			for (auto j = 0; j < epochs; j++) {
				std::shuffle(td_begin, td_end, gen);
				for (auto td_i = td_begin; td_i < td_end; td_i += mini_batch_size) {
					update_mini_batch(td_i, mini_batch_size, eta, lmbda, std::distance(td_begin, td_end));
				}
				feedback(*this, j, eta);
			}
		}
		// Update the network's weights and biases by applying
		//	gradient descent using backpropagation to a single mini batch.
		//	The "mini_batch" is a list of tuples "(x, y)", and "eta"
		//	is the learning rate."""
		void update_mini_batch(TrainingDataIterator td, int mini_batch_size, T eta, T lmbda, auto n) {
			NetworkData nabla(nd.m_sizes);
			nabla = std::transform_reduce(std::execution::par, td, td + mini_batch_size, nabla, std::plus<NetworkData>(), [=](const TrainingData& td) {
				const auto& [x, y] = td; // test data x, expected result y
				NetworkData delta_nabla(this->nd.m_sizes);
				backprop(x, y, delta_nabla);
				return delta_nabla;
				});
            constexpr T one = 1.0;
			for (auto i = 0; i < nd.biases.size(); ++i) {
				nd.biases[i] -= eta / mini_batch_size * nabla.biases[i];
				nd.weights[i] = (one - eta * (lmbda / n)) * nd.weights[i] - (eta / mini_batch_size) * nabla.weights[i];
			}
		}
		// Populates the gradient for the cost function for the biases in the vector
		// nabla_b and the weights in nabla_w
		void backprop(const ublas::vector<T> &x, const ublas::vector<T> &y, NetworkData &nabla) {
			auto activation = x;
			std::vector<ublas::vector<T>> activations; // Stores the activations of each layer
			activations.push_back(x);
			std::vector<ublas::vector<T>> zs; // The z vectors layer by layer
			for (auto i = 0; i < nd.biases.size(); ++i) {
				ublas::vector<T> z = prod(nd.weights[i], activation) + nd.biases[i];
				zs.push_back(z);
				activation = z;
				this->Activation(activation);
				activations.push_back(activation);
			}
			// backward pass
			auto iActivations = activations.end() - 1;
			auto izs = zs.end() - 1;

			this->ActivationPrime(*izs);
			ublas::vector<T> delta = this->cost_delta(*izs, *iActivations, y);
			auto ib = nabla.biases.end() - 1;
			auto iw = nabla.weights.end() - 1;
			*ib = delta;
			iActivations--;
			*iw = outer_prod(delta, trans(*iActivations));
			auto iWeights = nd.weights.end();
			while (iActivations != activations.begin()) {
				izs--;
				iWeights--;
				iActivations--;
				ib--;
				iw--;
				this->ActivationPrime(*izs);
				delta = element_prod(prod(trans(*iWeights), delta), *izs);
				*ib = delta;
				*iw = outer_prod(delta, trans(*iActivations));
			}
		}
		auto result(const ublas::vector<T> &res) const {
			return std::distance(res.begin(), max_element(res.begin(), res.end()));
		}
		// Return the vector of partial derivatives \partial C_x /
		//	\partial a for the output activations.
		auto accuracy(TrainingDataIterator td_begin, TrainingDataIterator td_end) const {
			return count_if(std::execution::par, td_begin, td_end, [=](const TrainingData& testElement) {
				const auto& [x, y] = testElement; // test data x, expected result y
				auto res = feedforward(x);
                return result(res) == result(y);				
				});
		}
		// Return the total cost for the data set ``data``.

		T total_cost(TrainingDataIterator td_begin, TrainingDataIterator td_end, T lmbda) const {
			auto count = std::distance(td_begin, td_end);
			T cost(0);
			cost = std::transform_reduce(std::execution::par, td_begin, td_end, cost, std::plus<>(), [=](const TrainingData& td) {
				const auto& [testData, expectedResult] = td;
				auto res = feedforward(testData);
				return this->cost_fn(res, expectedResult);
				});

			cost /= static_cast<T>(count);
			constexpr T zero = 0.0;
            constexpr T half = 0.5;
			T reg = std::accumulate(nd.weights.begin(), nd.weights.end(), zero,
				[lmbda, count](T reg, const ublas::matrix<T> &w) {
                return reg + half * (lmbda * pow(norm_frobenius(w), 2)) / static_cast<T>(count);
			});
			return cost + reg;
		}

        friend std::ostream &operator<<(std::ostream &os, const Network &net) {
            os << net.nd.m_sizes.size() << " ";
            std::ranges::for_each(net.nd.m_sizes, [&](int x) { os << x << " "; });
            for (auto x = 0; x < net.nd.m_sizes.size() - 1; ++x) {
                std::ranges::for_each(net.nd.biases[x], [&](T y) { os << y << " "; });
                std::ranges::for_each(net.nd.weights[x].data(), [&](T y) { os << y << " "; });			
			};

            return os;
        }

		friend std::istream &operator>>(std::istream &is, Network &obj) {
            int netSize;
            is >> netSize;
            for (int i = 0; i < netSize; ++i) {
                int size;
                is >> size;
                obj.nd.m_sizes.push_back(size);
			}
            obj.nd.PopulateZeroWeightsAndBiases();
            T a;
            for (auto x = 1; x < obj.nd.m_sizes.size(); ++x) {
                for (auto y = 0; y < obj.nd.m_sizes[x]; ++y) {
                    is >> a;
                    obj.nd.biases[x - 1][y] = a;
                }
                for (auto y = 0; y < obj.nd.m_sizes[x] * obj.nd.m_sizes[x - 1]; ++y) {
                    is >> a;
                    obj.nd.weights[x - 1].data()[y] = a;
                }
            };
            return is;
        }
	};

} // namespace NeuralNet