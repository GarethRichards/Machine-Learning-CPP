# Machine-Learning-C++

The C++ code in this repository is a hopfully accurate port of the python code in Michael Nielsen's book 
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/). Just as the original Python
code requires a library to do linear algebra, the C++ uses the ublas library from [boost](http://www.boost.org)
to do the matrix manipulation.

## Chapter 1
The C++ in the Chapter1 directory is unsurprisingly a port of the code of [chapter 1](http://neuralnetworksanddeeplearning.com/chap1.html)
of the book.

As explained in the book a neural network can be modeled by a list of baises, 1D vectors 
and a set of weights, 2D matracies. These can be defined in C++ as follows:
```c++
 	using BiasesVector = std::vector<ublas::vector<double>>;
	using WeightsVector = std::vector<ublas::matrix<double>>;
	BiasesVector biases;
	WeightsVector weights;
```
The weights and baises can be initialised from a single vector<int> 