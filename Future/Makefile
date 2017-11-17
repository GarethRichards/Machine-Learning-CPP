appname := NeuralNet3

CC=gcc
CXX=g++
RM=rm -f
CXXFLAGS=-O3 -I../../boost_1_65_1 -I../Loader -std=gnu++1z
LDFLAGS=-g 
LDLIBS=

# clang++ -Ipackages\boost.1.65.1.0\lib\native\include -I../loader -std=c++17 -O3 -D_SILENCE_CXX17_ITERATOR_BASE_CLASS_DEPRECATION_WARNING NeuralNet3.cpp -D_SILENCE_CXX17_OLD_ALLOCATOR_MEMBERS_DEPRECATION_WARNING

SRCS := $(shell find . -maxdepth 1 -name "*.cpp")
OBJS  := $(patsubst %.cpp, %.o, $(SRCS))

all: $(appname) 

$(appname): $(OBJS)
	$(CXX) $(LDFLAGS) -o $(appname) $(OBJS) $(LDLIBS)

depend: .depend

.depend: $(SRCS)
	$(RM) ./.depend
	$(CXX) $(CXXFLAGS) -MM $^>>./.depend;

clean:
	$(RM) $(OBJS)

distclean: clean
	$(RM) .depend

include .depend
