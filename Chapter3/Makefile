CC      = g++
CFLAGS  = -O3 -IC:\Development\boost_1_65_1 -I"C:\Development\Machine Learning CPP\Loader"
LDFLAGS = 

all: NeuralNet2 clean

NeuralNet2: NeuralNet2.o
	$(CC) -o $@ $^ $(LDFLAGS)

NeuralNet2.o: NeuralNet2.cpp NeuralNet.h
	$(CC) -c $(CFLAGS) $<

.PHONY: clean cleanest

clean:
	del *.o

cleanest: clean
	del NeuralNet2.exe
