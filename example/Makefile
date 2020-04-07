# Example makefile for DC_SAND

# Compiler and linker
CC := nvcc
CCFLAGS := -arch=sm70 --ptxas-options=-v

# Target executable program
TARGET = VectorAddTest

INCDIR := ../common

CUDADIR := /usr/local/cuda
CUDAINC := -I$(CUDADIR)/include -lcuda

OBJDIR := obj

all: $(TARGET)

remake: clean all

$(TARGET):
	$(CC) -o $(TARGET) -I $(INCDIR) $(CUDAINC) -L $(INCDIR) *.cpp *.cu $(INCDIR)/*.cpp

clean:
	rm -f VectorAddTest *.o
