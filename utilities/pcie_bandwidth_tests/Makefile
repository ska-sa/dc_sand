OBJECTS= pcieRateTest.o cudaPcieRateTest.o memRateTest.o main.o Utils.o

CUDA= /usr/local/cuda

INCLUDES= -I$(CUDA)/include \
		-I../../common

LIBRARIES=-L$(CUDA)/lib64 \
        -L$(CUDA)/lib64/stubs -lcuda -lOpenCL -lboost_program_options

%.o: %.cpp
	nvcc -o $@ -c $< $(INCLUDES) -Xcompiler -fopenmp

test_bandwidth: $(OBJECTS)
	nvcc -o $@ $^ $(LIBRARIES) -Xcompiler -fopenmp

Utils.o: ../../common/Utils.cpp
	nvcc -o $@ -c $< $(INCLUDES) -Xcompiler -fopenmp

clean::
	rm -f test_bandwidth *.o
