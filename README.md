# Machine-Learning-C++

The C++ code in this repository is a hopefully accurate port of the python code in Michael Nielsen's book 
[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/). I recommend you read 
Michael Nielson's book and if you wish to use C++ rather than python you can use the code
in this repository to supplement your understanding. I would also like to thank Grant Sanderson, [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw/featured) 
for his entertaining and educational videos on [Machine learning](https://www.youtube.com/watch?v=aircAruvnKk&t=68s) and
for introducing me to Michael's book.
If you're new to Machine learning I can also recommend as an excellent introduction
[Machine Learning](https://www.coursera.org/learn/machine-learning/home/welcome) 
by Stanford University on [Coursera](https://www.coursera.org). Finally, thanks to the people who created the 
[MNIST dataset](http://yann.lecun.com/exdb/mnist/) for making their dataset available to us.

I've aimed to produce code as simple and concise as possible using only the features in the standard C++17, but as
the standard library does not contain a linear algebra library I've used the [uBLAS](http://www.boost.org). I've
tied to showcase some of the new features of Modern C++ here, STL algorithms and lambda functions. With just a couple of 100 lines of code we can produce a program which can recognize handwritten digits.

## [Chapter 1](https://github.com/GarethRichards/Machine-Learning-CPP/blob/master/Chapter1.md)
In chapter 1 A C++ version of network.py is constructed in Python style as all the code is in one cpp file. The purpose of this version is to create code similar to the python code, rather than the best C++ possible, to allow you to read [Using neural nets to recognize handwritten digits](http://neuralnetworksanddeeplearning.com/chap1.html) and follow the online book with C++.

## Chapter 2
Michael explains how the backpropagation algorithm [works](http://neuralnetworksanddeeplearning.com/chap2.html) but add no 
more Python code to his book. Giving me time to read up on some C++17 features I'd like to try out.

## [Chapter 3](https://github.com/GarethRichards/Machine-Learning-CPP/blob/master/Chapter3.md)
This time I'm aiming to produce C++ worthy of being called a library, but still maintaining code which is recognisable to the python network2.py from [chapter 3](http://neuralnetworksanddeeplearning.com/chap2.html) of Michael's book. Of course the main body of the code goes into a header file  and I show how to construct some [policy classes](https://en.wikipedia.org/wiki/Policy-based_design) and how to use them in Network2.cpp.

## [Future](https://github.com/GarethRichards/Machine-Learning-CPP/blob/master/Future.md)
The code written so far all runs in a single thread. Of course thanks to the C++ compilers vectorization skills it is able to its job in a reasonable amount of time. In C++ 17 the Extensions for parallelism [TR](http://en.cppreference.com/w/cpp/experimental/parallelism)
lets us change the execution plan of algorithms. I introduce Structured bindings into the codebase which increase conciseness and readability. Finally, I use some ideas outlined in Michael's book and add different activation functions to the library. 

## Final thoughts
I hope you enjoyed this excursion into both C++ and Machine learning. I started writing this code as I was unable to find any easily approachable 
code for experimenting with Machine learning in C++. Once again Thanks to Michael Nielsen for writing such an accessible and readable book.

Gareth Richards 
14/11/2017
