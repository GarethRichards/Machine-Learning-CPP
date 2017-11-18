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
All I have written here is my very own version of the accumulate algorithm. But a little refactoring is required before I'm able to use
std::accumulate. First, I refactored the code to create a class NetworkData which contains the matrix of weights and the vector of 
biases. This class can also hold the randomization functions and handle its own creation. The finished product is here:
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
Much nicer and more readable. C++11 introduced threads to the standard library, but if you needed to run a loop over many threads it 
was easier to relay on soultions such as the [Parallel Patterns Library](https://msdn.microsoft.com/en-us/library/dd492418.aspx) rather
than a standard library C++ soultion. C++17 gives us the [Extensions for parallelism](http://en.cppreference.com/w/cpp/experimental/parallelism)
which allow you to set execution policies to your STL algoriths.
Accumulate is not the simplist algorithm to parallize as you require guarded access to the item which keeping
the running total. The go to algorithm for simple parallelism is for_each as I'm accumulating the result I need a mutex to
avoid race conditions.
```c++
			NetworkData nabla(nd.m_sizes);
			std::mutex mtx;           // mutex for critical section
			for_each(std::execution::par, td, td + mini_batch_size, [=, &nabla, &mtx](const TrainingData &td)
			{
				const auto& [ x, y ] = td; // test data x, expected result y
				NetworkData delta_nabla(this->nd.m_sizes);				
				backprop(x, y, delta_nabla);
				// critical section 
				std::lock_guard<std::mutex> guard(mtx);
				nabla += delta_nabla;
			});

```
An important note: In this example the code in the loop before the critical section will takes a lot more clock cycles than the 
addition after the critical section and many more cycles than the critical section code. It's important to remember the critical
section is an expensive operation and the constant execution of the critical section will cause your application to run slower
than the single threaded version. The golden rule here is test and measure.

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
There are many improvements which could be made to the Library. If anyone wishes to extend their understanding of Machine learning concepts 
and C++ I've created a list in what I believe is in order of difficulty of interesting changes which could be made:
1. Add code to load and save a Network.
2. Add different regularization schemes.
3. Add a Dropout scheme to the code.
5. Add Convolutional neural networks scheme to the library.