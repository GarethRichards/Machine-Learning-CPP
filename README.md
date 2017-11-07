# Machine-Learning-C++

The C++ code in this repository is a hopefully accurate port of the python code in Michael Nielsen's book 
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/). I recommend you read 
Micheal Nielson's book and if you wish to use C++ rather than python you can use the code
in this repository to supplement your understanding. I would also like to thank Grant Sanderson, [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw/featured) 
for his entertaining and educational videos on [Machine learning](https://www.youtube.com/watch?v=aircAruvnKk&t=68s) and
for introducing me to Michael's book.
If you're new to Machine learning I can also recommend as an excellent introduction
[Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome) 
by Stanford University on [Coursera](https://www.coursera.org). Finally thanks to the people who created the 
[MNIST dataset](http://yann.lecun.com/exdb/mnist/) for making their dataset available to us.

I've aimed to produce code as simple and concise as possible using only the features in the standard C++17, but as
the standard library does not contain a linear algebra library I've used the [uBLAS](http://www.boost.org). I've
tied to showcase some of the new features of Modern C++ here, STL algorithms and lambda functions. 

## [Chapter 1](https://github.com/GarethRichards/Machine-Learning-CPP/blob/master/Chapter1.md)
In chapter 1 I convert network.py to C++ in Python style all the code is in one cpp file.

## [Chapter 3](https://github.com/GarethRichards/Machine-Learning-CPP/blob/master/Chapter3.md)
In chapter 3 I have a go at converting network2.py to C++ this time I put the code into a header file
construct some [policy classes](https://en.wikipedia.org/wiki/Policy-based_design) and demonstrate 
how they can be used in Network2.cpp.

## Future
The code above is all single threaded and it is probably thanks to the compilers vectorization skills it is 
able to it's job in a reasonable amount of time. I'm looking forward to using the new 
[execution policies](http://en.cppreference.com/w/cpp/algorithm/execution_policy_tag) to
use all the cores on my computer.

Gareth Richards 
03/11/2017

