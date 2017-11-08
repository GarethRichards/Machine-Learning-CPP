CC      = g++
CFLAGS  = -O3 -IC:\Development\boost_1_65_1 -I"C:\Development\Machine Learning CPP\Loader"
LDFLAGS = 

all: NeuralNet3 clean

NeuralNet3: NeuralNet3.o
	$(CC) -o $@ $^ $(LDFLAGS)

NeuralNet3.o: NeuralNet3.cpp NeuralNet.h
	$(CC) -c $(CFLAGS) $<

.PHONY: clean cleanest

clean:
	del *.o

cleanest: clean
	del NeuralNet2.exe
