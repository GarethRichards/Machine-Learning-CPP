# Chapter 3
The code written for chapter 1 was passable C++ but obviously not good enough for library code. You may ask why did I choose double rather than float as type to do the processing on? Perhaps float would be more efficient on your hardware? In this logic we have the justification for templates. Here is how we convert the Network class to a template:
```c++
	template<typename T>
	class Network {
  	private:
		using BiasesVector = std::vector<ublas::vector<T>>;
		using WeightsVector = std::vector<ublas::matrix<T>>;
		std::vector<int> m_sizes;
		BiasesVector biases;
		WeightsVector weights;
```
## The cost functions
In Chapter 3 Michael discusses different cost functions we can use on the network. In C++ these can be implemented as cost policy classes. Here is the code for the QuadraticCost function:
```c++
	template<typename T>
	class QuadraticCost {
	public:
		T cost_fn(const ublas::vector<T>& a,
			const ublas::vector<T>& y) const
		{
			return 0.5 *pow(norm_2(a - y));
		}
		ublas::vector<T> cost_delta(const ublas::vector<T>& z, const ublas::vector<T>& a,
			const ublas::vector<T>& y) const
		{
			auto zp = z;
			sigmoid_prime(zp);
			return element_prod(a - y, zp);
		}
	};
```
And here is the code for the CrossEntropyCost function.
```c++
	template<typename T>
	class CrossEntropyCost {
	public:
		// Return the cost associated with an output ``a`` and desired output
		// ``y``.  Note that np.nan_to_num is used to ensure numerical
		// stability. If both ``a`` and ``y`` have a 1.0
		// in the same slot, then the expression (1 - y)*np.log(1 - a)
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
		// Return the error delta from the output layer. 
		ublas::vector<T> cost_delta(const ublas::vector<T>& z, const ublas::vector<T>& a,
			const ublas::vector<T>& y) const
		{
			z; // not used by design
			return a - y;
		}
	};
```
The imported feature to note with policy classes is they must use the same interface.

We can create decide at compile time which class we would like to use as our cost function by defining the Network class as follows:
```c++
	template<typename T, typename CostPolicy>
	class Network : private CostPolicy {
	private:
		using BiasesVector = std::vector<ublas::vector<T>>;
		using WeightsVector = std::vector<ublas::matrix<T>>;
		std::vector<int> m_sizes;
		BiasesVector biases;
		WeightsVector weights;
```
To create an implementation of the Network using floats and CrossEntropyCost we can create this as follows:
```c++
    NeuralNet::Network<double, NeuralNet::CrossEntropyCost<double>> net({ 10, 748, 30, 10 });
```
Naturally that’s not nice to type out or read C++ so we define it as a type:
```c++
   using NetCrossEntropyCost=NeuralNet::Network<double, NeuralNet::CrossEntropyCost<double>>;
   NetCrossEntropyCost net({ 10, 748, 30, 10 });
```
This helps when we want to obtain the definition of the training data:
```c++
	NetCrossEntropyCost::TrainingData
	// Instead of
	NeuralNet::Network<double, NeuralNet::CrossEntropyCost<double>>::TrainingData
```

## Feedback function
Users of the Network class need to see feedback from each round of fitting using a C++ lambda is a very clean approach to achieving this goal. This can be achieved by adding a std::function parameter to our interface.
```c++
		void SGD(typename std::vector<TrainingData>::iterator td_begin,
			typename std::vector<TrainingData>::iterator td_end,
			int epochs, int mini_batch_size, T eta, T lmbda,
			std::function<void(const Network &,int Epoc)> feedback)
```
By passing a referece to the Network the user of the class can integerate the class as to the current cost and accuracy of the network as follows:
```c++
	NeuralNet1 net({ 784, 30, 10 });
	net.SGD(td.begin(), td.end(), 30, 10, 0.5, Lmbda, [&testData,&td,Lmbda](const NeuralNet1 &network, int Epoch) {
		std::cout << "Epoch " << Epoch << " : " << network.accuracy(testData.begin(), testData.end()) << " / " << testData.size() << std::endl;
		std::cout << "Epoch " << Epoch << " : " << network.accuracy(td.begin(), td.end()) << " / " << td.size() << std::endl;
		std::cout << "Cost : " << network.total_cost(td.begin(), td.end(), Lmbda) << std::endl;
		std::cout << "Cost : " << network.total_cost(testData.begin(), testData.end(), Lmbda) << std::endl;
	});
```
This I hope you will agree results in a nice clean interface for our class.
