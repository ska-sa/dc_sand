# Example makefile for DC_SAND

# Compiler and linker
CC := nvcc
CCFLAGS := -arch=sm_61 --ptxas-options='-v -warn-double-usage -warn-lmem-usage -warn-spills'

# Target executable program
TARGET = BeamformerCoefficientTest

INCDIR := ../common

CUDADIR := /usr/local/cuda
CUDAINC := -I$(CUDADIR)/include -lcuda

OBJDIR := obj

all: $(TARGET)

remake: clean all

$(TARGET):
	$(CC) $(CCFLAGS) -o $(TARGET) -I $(INCDIR) $(CUDAINC) -L $(INCDIR) *.cpp *.cu $(INCDIR)/*.cpp

clean:
	rm -f BeamformerCoefficientTest *.o
