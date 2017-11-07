CC      = g++
CFLAGS  = -O3 -IC:\Development\boost_1_65_1 -I"C:\Development\Machine Learning CPP\Loader"
LDFLAGS = 

all: NeuralNet clean

NeuralNet: NeuralNet1.o
	$(CC) -o $@ $^ $(LDFLAGS)

NeuralNet1.o: NeuralNet1.cpp 
	$(CC) -c $(CFLAGS) $<

.PHONY: clean cleanest

clean:
	del *.o

cleanest: clean
	del NeuralNet.exe
