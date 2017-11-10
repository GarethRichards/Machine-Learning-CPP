appname := NeuralNet2

CC=gcc
CXX=g++
RM=rm -f
CXXFLAGS=-O3 -I../../boost_1_65_1 -I../Loader
LDFLAGS=-g 
LDLIBS=

SRCS := $(shell find . -name "*.cpp")
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
