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

#include <cmath>
#include <execution>
#include <experimental/linalg>
#include <functional>
#include <iostream>
#include <mutex>
#include <numeric>
#include <random>
#include <ranges>
#include <vector>

using std::experimental::extents;
using std::experimental::mdspan;

namespace linalg = std::experimental::linalg;

#if defined(__cpp_lib_span)
#include <span>
using std::dynamic_extent;
#else
using std::experimental::dynamic_extent;
#endif

namespace NeuralNet {

// The sigmoid function.
template <typename T> class SigmoidActivation {
    using nvec = mdspan<T, std::dextents<size_t, 2>>;

  public:
    void Activation(nvec &v) const {
        constexpr T one = 1.0;
        std::ranges::for_each(v, [](T &iv) { iv = one / (one + exp(-iv)); });
    }
    void ActivationPrime(nvec &v) const {
        constexpr T one = 1.0;
        std::ranges::for_each(v, [](T &iv) {
            iv = one / (one + exp(-iv));
            iv = iv * (one - iv);
        });
    }
};

// The tanh function.
template <typename T> class TanhActivation {
    using nvec = mdspan<T, std::dextents<size_t, 2>>;

  public:
    void Activation(nvec &v) const {
        constexpr T one = 1.0;
        constexpr T two = 2.0;
        for (auto &iv : v) {
            iv = (one + tanh(iv)) / two;
        }
    }
    void ActivationPrime(nvec &v) const {
        constexpr T two = 2.0;
        for (auto &iv : v) {
            iv = pow(two / (exp(-iv) + exp(iv)), two) / two;
        }
    }
};

// The ReLU function.
template <typename T> class ReLUActivation {
    using nvec = mdspan<T, std::dextents<size_t, 2>>;

  public:
    void Activation(nvec &v) const {
        constexpr T zero = 0.0;
        for (int i = 0; i < v.extent(0); ++i)
            v(i, 0) = std::max(zero, v(i, 0));
    }
    void ActivationPrime(nvec &v) const {
        constexpr T zero = 0.0;
        constexpr T one = 1.0;
        for (int i = 0; i < v.extent(0); ++i) {
            v(i, 0) = v(i, 0) < zero ? zero : one;
        }
    }
};

template <typename T> class QuadraticCost {
    using nvec = mdspan<T, std::dextents<size_t, 2>>;

  public:
    T cost_fn(nvec &a, const nvec &y) const {
        return 0.5 * pow(linalg::vector_norm2(a - y));
        ;
    }
    void cost_delta(const nvec &z, const nvec &a, const nvec &y, nvec &result) const {
        auto zp = z;
        this->ActivationPrime(zp);
        std::vector<T> sy_data(y.extent(0));
        nvec sy(sy_data.data(), y.extent(0), 1);
        for (int i = 0; i < y.extent(0); ++i)
            sy(i, 0) = y(i, 0) * -1.0;
        linalg::add(a, sy, sy);
        linalg::dot(sy, zp, result);
    }
};

template <typename T> class CrossEntropyCost {
    using nvec = mdspan<T, std::dextents<size_t, 2>>;

  public:
    // Return the cost associated with an output ``a`` and desired output
    // ``y``.  Note that np.nan_to_num is used to ensure numerical
    // stability.In particular, if both ``a`` and ``y`` have a 1.0
    // in the same slot, then the expression(1 - y)*np.log(1 - a)
    // returns nan.The np.nan_to_num ensures that that is converted
    // to the correct value(0.0).
    T cost_fn(const nvec &a, const nvec &y) const {
        constexpr T zero = 0.0;
        constexpr T one = 1.0;
        T total(zero);
        for (auto i = 0; i < a.extent(0); ++i) {
            total += a(i, 0) == zero ? zero : -y(i, 0) * log(a(i, 0));
            total += a(i, 0) >= one ? zero : -(one - y(i, 0)) * log(one - a(i, 0));
        }
        return total;
    }
    // Return the error delta from the output layer.  Note that the
    // parameter ``z`` is not used by the method.It is included in
    // the method's parameters in order to make the interface
    // consistent with the delta method for other cost classes.
    void cost_delta(const nvec &z, const nvec &a, const nvec &y, nvec &result) const {
        (void)z; // not used by design
        std::vector<T> sy_data(y.extent(0));
        nvec sy(sy_data.data(), y.extent(0), 1);
        for (int i = 0; i < y.extent(0); ++i)
            sy(i, 0) = y(i, 0) * -1.0;
        linalg::add(a, sy, result);
    }
};

template <typename T, typename CostPolicy, typename ActivationPolicy>
    requires std::floating_point<T>
class Network : private CostPolicy, private ActivationPolicy {
  private:
    using nvec = mdspan<T, std::dextents<size_t, 2>>;
    using nmatrix = std::mdspan<T, std::dextents<size_t, 2>>;
    using BiasesVector = std::vector<nvec>;
    using WeightsVector = std::vector<nmatrix>;
    // using WeightsVector = std::vector<ublas::matrix<T>>;

  public:
    // Type definition of the Training data
    using TrainingData = std::pair<nvec, nvec>;
    using TrainingDataIterator = typename std::vector<TrainingData>::iterator;
    using TrainingDataVector = std::vector<TrainingData>;

  protected:
    class NetworkData {
      public:
        std::vector<size_t> m_sizes;
        size_t max_vec_size = 0;
        size_t tot_vec_size = 0;
        BiasesVector biases;
        std::vector<T> biases_data;
        WeightsVector weights;
        std::vector<T> weights_data;
        std::mt19937 gen;

        NetworkData() = default;
        explicit NetworkData(const std::vector<size_t> &m_sizes) : m_sizes(m_sizes) {
            std::random_device r;
            gen = std::mt19937(r());
            PopulateZeroWeightsAndBiases();
        }
        ~NetworkData() = default;
        void PopulateZeroWeightsAndBiases() {
            size_t biases_data_size = 0;
            size_t weights_data_size = 0;
            for (auto i = 1; i < m_sizes.size(); ++i) {
                biases_data_size += m_sizes[i];
                weights_data_size += m_sizes[i] * m_sizes[i - 1];
            }
            biases_data.resize(biases_data_size, 0);
            weights_data.resize(weights_data_size, 0);
            biases_data_size = 0;
            weights_data_size = 0;
            for (auto i = 1; i < m_sizes.size(); ++i) {
                biases.push_back(nvec(&biases_data[biases_data_size], m_sizes[i], 1));
                weights.push_back(nmatrix(&weights_data[weights_data_size], m_sizes[i], m_sizes[i - 1]));
                biases_data_size += m_sizes[i];
                weights_data_size += m_sizes[i] * m_sizes[i - 1];
            }
            max_vec_size = *std::max_element(m_sizes.begin(), m_sizes.end());
            tot_vec_size = 0;
            for (auto s : m_sizes)
                tot_vec_size += s;
        }
        NetworkData &operator+=(const NetworkData &rhs) {
            for (auto j = 0; j < biases.size(); ++j) {
                linalg::add(biases[j], rhs.biases[j], biases[j]);
                linalg::add(weights[j], rhs.weights[j], weights[j]);
            }
            return *this;
        }
        NetworkData &operator+(const NetworkData &rhs) {
            *this += rhs; // reuse compound assignment
            return *this;
        }

        void Randomize() {
            for (auto &b : biases)
                RandomizeVector(b);
            for (auto &w : weights)
                RandomizeMatrix(w);
        }
        void RandomizeVector(nvec &vec) {
            std::normal_distribution<T> d(0, 1);
            for (int i = 0; i < vec.extent(0); ++i)
                vec(i, 0) = d(gen);
        }
        // Randomize as ublas matrix
        void RandomizeMatrix(nmatrix &m) {
            std::normal_distribution<T> d(0, 1);
            T sx = sqrt(static_cast<T>(m.extent(1)));
            for (int i = 0; i < m.extent(0); ++i)
                for (int j = 0; j < m.extent(1); ++j)
                    m(i, j) = d(gen) / sx;
        }
    };

  private:
    NetworkData nd;

  public:
    Network() = default;
    explicit Network(const std::vector<size_t> &sizes) : nd(sizes) { nd.Randomize(); }
    // Initalize the array of Biases and Matrix of weights

    // Returns the output of the network if the input is a
    void feedforward(nvec in_vec, nvec result) const {
        std::vector<T> res_data(nd.tot_vec_size);
        size_t start_index = nd.m_sizes[0];
        size_t prev_index = 0;
        for (auto i = 0; i < nd.biases.size(); ++i) {
            nvec resd(&res_data[start_index], nd.m_sizes[i + 1], 1);
            nvec &res = resd;
            if (i == nd.biases.size() - 1)
                res = result;
            nvec res_ind(&res_data[prev_index], nd.m_sizes[i], 1);
            nvec &res_in = res_ind;
            if (i == 0)
                res_in = in_vec;
            linalg::matrix_product(nd.weights[i], res_in, res);
            linalg::add(res, nd.biases[i], res);
            this->Activation(res);
            prev_index = start_index;
            start_index += nd.m_sizes[i + 1];
        }
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
            std::shuffle(td_begin, td_end, nd.gen);
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
        for (auto i = 0; i < mini_batch_size; ++i, td++) {
            const auto &[x, y] = *td; // test data x, expected result y
            NetworkData delta_nabla(this->nd.m_sizes);
            backprop(x, y, delta_nabla);
            nabla += delta_nabla;
        }
        constexpr T one = 1.0;
        for (auto i = 0; i < nd.biases.size(); ++i) {
            linalg::add(nd.biases[i], linalg::scaled(-eta / mini_batch_size, nabla.biases[i]), nd.biases[i]);
            // nd.biases[i] -= eta / mini_batch_size * nabla.biases[i];
            linalg::add(linalg::scaled(one - eta * (lmbda / n), nd.weights[i]),
                        linalg::scaled(-eta / mini_batch_size, nabla.weights[i]), nd.weights[i]);
            // nd.weights[i] = (one - eta * (lmbda / n)) * nd.weights[i] - (eta / mini_batch_size) * nabla.weights[i];
        }
    }
    // Populates the gradient for the cost function for the biases in the vector
    // nabla_b and the weights in nabla_w
    void backprop(const nvec &x, const nvec &y, NetworkData &nabla) {
        auto activation = x;
        std::vector<T> a_data(nd.tot_vec_size);
        std::vector<nvec> activations; // Stores the activations of each layer
        activations.push_back(x);
        std::vector<T> zs_data(nd.tot_vec_size);
        std::vector<nvec> zs; // The z vectors layer by layer
        size_t start_index = 0;
        for (auto i = 0; i < nd.biases.size(); ++i) {
            nvec z(&zs_data[start_index], nd.m_sizes[i + 1], 1);
            linalg::matrix_product(nd.weights[i], activation, z);
            linalg::add(z, nd.biases[i], z);
            zs.push_back(z);
            std::copy(std::next(zs_data.cbegin(), start_index),
                      std::next(zs_data.cbegin(), start_index + nd.m_sizes[i + 1]),
                      std::next(a_data.begin(), start_index));
            activation = nvec(&a_data[start_index], nd.m_sizes[i + 1], 1);
            this->Activation(activation);
            activations.push_back(activation);
            start_index += nd.m_sizes[i + 1];
        }
        // backward pass
        auto iActivations = activations.end() - 1;
        auto izs = zs.end() - 1;

        this->ActivationPrime(*izs);
        auto ib = nabla.biases.end() - 1;
        auto iw = nabla.weights.end() - 1;
        this->cost_delta(*izs, *iActivations, y, *ib);
        iActivations--;
        linalg::matrix_product(*ib, linalg::transposed(*iActivations), *iw);

        auto iWeights = nd.weights.end();
        while (iActivations != activations.begin()) {
            auto previous_delta = *ib;
            izs--;
            iWeights--;
            iActivations--;
            ib--;
            iw--;
            this->ActivationPrime(*izs);
            linalg::matrix_product(linalg::transposed(*iWeights), previous_delta, *ib);
            for (int i = 0; i < (*ib).extent(0); i++) {
                (*ib)(i, 0) = (*ib)(i, 0) * (*izs)(i, 0);
            }
            linalg::matrix_product(*ib, linalg::transposed(*iActivations), *iw);
        }
    }
    auto result(const nvec &res) const {
        T maxR = res(0, 0);
        int v = 0;
        for (int i = 1; i < res.extent(0); ++i) {
            if (res(i, 0) > maxR) {
                v = i;
                maxR = res(i, 0);
            }
        }
        return v;
    }
    // Return the vector of partial derivatives \partial C_x /
    //	\partial a for the output activations.
    auto accuracy(TrainingDataIterator td_begin, TrainingDataIterator td_end) const {
        return count_if(std::execution::seq, td_begin, td_end, [this](const TrainingData &testElement) {
            const auto &[x, y] = testElement; // test data x, expected result y
            std::vector<T> res(nd.m_sizes[nd.m_sizes.size() - 1]);
            nvec nres(res.data(), (size_t)res.size(), 1);
            feedforward(x, nres);
            return result(nres) == result(y);
        });
    }
    // Return the total cost for the data set ``data``.

    T total_cost(TrainingDataIterator td_begin, TrainingDataIterator td_end, T lmbda) const {
        auto count = std::distance(td_begin, td_end);
        T cost(0);
        cost = std::transform_reduce(std::execution::seq, td_begin, td_end, cost, std::plus<>(),
                                     [this](const TrainingData &td) {
                                         const auto &[testData, expectedResult] = td;
                                         std::vector<T> res(nd.m_sizes[nd.m_sizes.size() - 1]);
                                         nvec nres(res.data(), (size_t)res.size(), 1);
                                         feedforward(testData, nres);
                                         return this->cost_fn(nres, expectedResult);
                                     });

        cost /= static_cast<T>(count);
        constexpr T zero = 0.0;
        constexpr T half = 0.5;
        T reg = std::accumulate(nd.weights.begin(), nd.weights.end(), zero, [lmbda, count](T regC, const nmatrix &w) {
            return regC + half * (lmbda * pow(linalg::matrix_frob_norm(w), 2)) / static_cast<T>(count);
        });
        return cost; // +reg;
    }

    friend std::ostream &operator<<(std::ostream &os, const Network &net) {
        os << net.nd.m_sizes.size() << " ";
        std::ranges::for_each(net.nd.m_sizes, [&](size_t x) { os << x << " "; });
        std::ranges::for_each(net.nd.biases_data, [&](T y) { os << y << " "; });
        std::ranges::for_each(net.nd.weights_data, [&](T y) { os << y << " "; });
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
        for (auto x = 0; x < obj.nd.biases_data.size(); ++x) {
            is >> a;
            obj.nd.biases_data[x] = a;
        }
        for (auto x = 0; x < obj.nd.weights_data.size(); ++x) {
            is >> a;
            obj.nd.weights_data[x] = a;
        }
        return is;
    }
};

} // namespace NeuralNet
