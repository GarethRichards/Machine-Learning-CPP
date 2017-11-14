# Future
In this section I'll look at how I've evolved the code from the Chapter 3 version and plans to improve it in the future. The Machine learning ideas 
implemented here are expounded on in greater depth in the [Deep Learning](http://neuralnetworksanddeeplearning.com/chap6.html). There are some
great new features in C++ 17 and I've tried to showcase some of them here. I'll start with some new C++ features.

## C++ 17
### Structured bindings
Structured Bindings give us the ability to declare multiple variables from a pair or a tuple. The training data in the NeuralNet library is 
a vector containg pairs of uBLAS vectors std::vector<std::pair<ublas::vector<T>,ublas::vector<T>>> before C++ 17 to reference this data as follows:
``` c++
			auto nabla=std::accumulate(td,td+mini_batch_size,nabla0,[=](NetworkData &nabla,const TrainingData &td)
			{
				const auto &x=td.first;
				const auto &y=td.second;
```
With C++ 17 This can be simplefied to:
```c++
			auto nabla=std::accumulate(td,td+mini_batch_size,nabla0,[=](NetworkData &nabla,const TrainingData &td)
			{
				const auto& [ x, y ] = td; // test data x, expected result y
```
A definate win for readability.

## Parallel STL
When writing modern C++ a good practice is lookout for possible places where you can replace a hand written loop with an algorithm from the standard 
library. The processing of the mini batch looked like an ideal case:
```c++
			std::vector<ublas::vector<T>> nabla_b;
			std::vector<ublas::matrix<T>> nabla_w;
			PopulateZeroWeightsAndBiases(nabla_b, nabla_w);
			for (auto i = 0; i < mini_batch_size; ++i, td++) {
                const auto& [ x, y ] = td; // test data x, expected result y
				std::vector<ublas::vector<T>> delta_nabla_b;
				std::vector<ublas::matrix<T>> delta_nabla_w;
				PopulateZeroWeightsAndBiases(delta_nabla_b, delta_nabla_w);
				backprop(x, y, delta_nabla_b, delta_nabla_w);
				for (auto j = 0; j< biases.size(); ++j)
				{
					nabla_b[j] += delta_nabla_b[j];
					nabla_w[j] += delta_nabla_w[j];
				}
			}
```
All I have written here is my very own version of the accumulate algorithm. I will have to pass the matrix and the vector together of course after a 
little rearranging:
```c++
			NetworkData nabla0(nd.m_sizes);
			auto nabla=std::accumulate(td,td+mini_batch_size,nabla0,[=](NetworkData &nabla,const TrainingData &td)
			{
                const auto& [ x, y ] = td; // test data x, expected result y
				NetworkData delta_nabla(this->nd.m_sizes);
				backprop(x, y, delta_nabla);
				nabla += delta_nabla;
				return nabla;
			});
```
Much nice but apart from the aesthetic value of the cleaner code are we to be rewarded for our hard work. The answer is yes but not yet!
For associative addition operations we will soon be able to replace accumulate with [reduce](http://en.cppreference.com/w/cpp/algorithm/reduce) and 
the reduce algorithm takes an [execution policy](http://en.cppreference.com/w/cpp/algorithm/execution_policy_tag_t) which should execute this algorithm 
over all your CPU's and possible your GPU. Sadly, this functionality is not yet available for either MSVC or GCC, hopefully we will see this 
Implemented soon.

## Activation functions
The previous two versions of this code used the Sigmoid activation function. Clearly a Neural net library should provide a set of activation functions 
and way to add activation functions. Again, one can use policy classes for this purpose and thus one can create Neural nets with many different features:
```c++
	using NeuralNet1 = NeuralNet::Network<double, 
						NeuralNet::CrossEntropyCost<double>,
						NeuralNet::ReLUActivation<double>>;

	using NeuralNet2 = NeuralNet::Network<double, 
  					NeuralNet::CrossEntropyCost<double>,
						NeuralNet::TanhActivation<double>>;
```
Where in this example 2 possible networks have been defined one using rectified linear units and the other the Tanh activation function. If you fancy 
Modifying the library to add a new activation function, Wikipedia has a list [here](https://en.wikipedia.org/wiki/Activation_function).

## To Do
There are a number of improvements which could be made to the Library. If anyone wishes to extend their understanding of Machine learning concepts 
and C++ I've created a list in what I believe is in order of difficulty of interesting changes which could be made:
1. Different regularization schemes
2. Dropout
3. Convolutional neural networks 

