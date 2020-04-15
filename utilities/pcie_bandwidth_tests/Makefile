OBJECTS= pcieRateTest.o cudaPcieRateTest.o openclPcieRateTest.o memRateTest.o main.o 

CUDA= /usr/local/cuda-10.1

INCLUDES= -I$(CUDA)/include
#LIBRARIES= -lcuda -lcublas 

LIBRARIES=-L$(CUDA)/lib64 \
        -L$(CUDA)/lib64/stubs -lcuda -lOpenCL -lboost_program_options

%.o: %.cpp
	nvcc -o $@ -c $< $(INCLUDES) -Xcompiler -fopenmp

test_bandwidth: $(OBJECTS)
	nvcc -o $@ $^ $(LIBRARIES) -Xcompiler -fopenmp

clean::
	rm -f test_bandwidth *.o
