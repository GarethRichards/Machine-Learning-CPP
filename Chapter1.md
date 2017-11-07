# Chapter 1
The C++ in the Chapter1 directory is unsurprisingly a port of the code of [chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html)
of the online book [Neural networks and deep learning](http://neuralnetworksanddeeplearning.com).

## The Network
As explained in the book a neural network can be modelled by a list of biases, 1D vectors 
and a set of weights, 2D matrices. These can be defined in C++ as follows:
```c++
 	using BiasesVector = std::vector<ublas::vector<double>>;
	using WeightsVector = std::vector<ublas::matrix<double>>;
	BiasesVector biases;
	WeightsVector weights;
```
The weights and biases can be initialised from a single vector<int> which we store in sizes:
```C++
	void PopulateZeroWeightsAndBiases(BiasesVector &b, WeightsVector &w)  const
	{
		for (size_t i = 1; i < m_sizes.size(); ++i)
		{
			b.push_back(ublas::zero_vector<double>(m_sizes[i]));
			w.push_back(ublas::zero_matrix<double>(m_sizes[i], m_sizes[i - 1]));
		}
	}
```
This code is equivalent to the python code:
```python
        self.biases = [np.random.randn(y, 1) for y in sizes[1:]]
        self.weights = [np.random.randn(y, x)
                        for x, y in zip(sizes[:-1], sizes[1:])]
```
But not quite equivalent as the C++ biases and weights are zero at this point.

## The Training Data
The training data is a vector of pair or uBLAS vectors where the first vector is the array of inputs (x’s in the Python) the second vector is the expected result, a uBLAS vector which contains 1 in the index of the expected answer.
```c++
using TrainingData = std::pair<ublas::vector<T>, ublas::vector<T>>;
```
By creating the alias TrainingData we can add greater readability to the class definition. The TrainingData definition is not actually a type so the class which loads the training data can fully define the TrainingData as:
```c++
std::vector<std::pair<ublas::vector<T>, ublas::vector<T>>> 
```
And this interoperates with the TrainingData definition.

## Feed Forward

The equivalent of the feedforward function in C++ is as follows:
```c++
	ublas::vector<double> feedforward(ublas::vector<double> a) const
	{
		for (size_t i = 0; i < biases.size(); ++i)
		{
			ublas::vector<double> c = prod(weights[i], a) + biases[i];
			sigmoid(c);
			a = c;
		}
		return a;
	}
```
This is of course the only function your application needs if you just need to process inputs and classify using a pre-learned set of weights.
In this function the prod function of the ublas library is doing the matrix multiplication and the result is a 1D vector which we add the biases to. 
The ublas library is an extensible library and with a bit of effort I'm sure you could achieve:
``` c++
		a = sigmoid(prod(weights[i], a) + biases[i]);
```
Which would by no more verbose than the python:
``` python
	sigmoid(np.dot(w, a)+b)
```
By now if your unfamiliar with python you may be puzzled by the some of the python constructs used in network.py. This construct:
``` python
	for b, w in zip(self.biases, self.weights):
```
Is a pythonic way of iterating through the weights and biases lists together. In C++ I've fallen back to writing a simple loop for this expression.

## Back propogation

Let's have a brief look at the backprop function and to explore the differences between the python and C++ code. The first loop over the biases and weights
runs the same algorithm as the feedforward function, but storing the intermediate working which we use later in the function.

The second part of the function calculated the derivatives using the back prop method. The first calculation involves calculating the derivative of the cost and the
Remaining loops propagate this back through the rest of net. In the python we can access and array with [-1] to obtain the last element of the array. In the 
C++ version of the code we set up the iterators we need for the calculation and move them backwards for each iteration.
```c++
	// Populates the gradent for the cost function for the biases in the vector nabla_b 
	// and the weights in nabla_w
	void backprop(const ublas::vector<double> &x, const ublas::vector<double> &y,
		BiasesVector &nabla_b, WeightsVector &nabla_w)
	{
		auto activation = x;
		std::vector<ublas::vector<double>> activations; // Stores the activations of each layer
		activations.push_back(x);
		std::vector<ublas::vector<double>> zs; // The z vectors layer by layer
		for (size_t i = 0; i < biases.size(); ++i) {
			ublas::vector<double> z = prod(weights[i], activation) + biases[i];
			zs.push_back(z);
			activation = z;
			sigmoid(activation);
			activations.push_back(activation);
		}
		// backward pass
		auto iActivations = activations.end() - 1;
		auto izs = zs.end() - 1;
		sigmoid_prime(*izs);
		ublas::vector<double> delta = element_prod(cost_derivative(*iActivations, y), *izs);
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
```
In the numpy library to multiply two vectors together you can use a*b and the result multiplies the elements of each of them. In ublas 
element_prod(a,b) is equivalent. In python the code np.dot(delta, activations[-l-1].transpose()) creates a matrix which the m[i,j]=a[i]*b[j]
the equivalent function in the ublas library is the outer_prod function. 

## Evaluate

In C++ version of the evaluate function I've used the max element and distance functions to determine the location of the maximum 
signal.
```c++
	int evaluate(const std::vector<TrainingData> &td) const
	{
		return count_if(td.begin(), td.end(), [this](const TrainingData &testElement) {
			auto res = feedforward(testElement.first);
			return (std::distance(res.begin(), max_element(res.begin(), res.end()))
				== std::distance(testElement.second.begin(), max_element(testElement.second.begin(), testElement.second.end()))
				);
		});
	}
```

## Loading the MNIST data
The data used by this project was obtained from the [MNIST Database](http://yann.lecun.com/exdb/mnist/). The code to load the data is contained in
the Loader directory.

## Running the code
So you've downloaded the code compiled it and you wait, and wait, go make a cup tea come back and still nothing. The debug version of this code 
is almost certainly not using the vectorising abilities of your computer. If this is the case please build an optimized version of the code 
and if your PC is similar to mine you should start to see output like this
```
Epoch 0: 9137 / 10000
Epoch 1: 9294 / 10000
Epoch 2: 9306 / 10000
Epoch 3: 9396 / 10000
Epoch 4: 9420 / 10000
Epoch 5: 9428 / 10000
.
.
.
Epoch 26: 9522 / 10000
Epoch 27: 9530 / 10000
Epoch 28: 9518 / 10000
Epoch 29: 9527 / 10000
```
Similar accuracy to Network.py a result!
