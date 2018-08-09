TARGET	= cudablockproc
TARGET_OBJS	= main.obj
SRC_DIR = src
INCLUDE_DIR = include

VPATH := $(OBJ_DIR);$(SRC_DIR);$(INCLUDE_DIR);

OPT	= -g -O3
PIC = #-fpic
XPIC  = #-Xcompiler #-fpic
XOPT  = -O3 -lineinfo #-Xptxas=-v # use -lineinfo for profiler, use -G for debugging
XARCH =
DEF   =

CC = "C:/Program Files (x86)\Microsoft Visual Studio 14.0/VC/bin/x86_amd64"

CXX	= nvcc
CXXFLAGS = -ccbin $(CC) $(XARCH) $(XOPT) $(XPIC) $(DEF)

CUDA_PATH ?= "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/9.1"
INCLUDES ?= -I"$(CUDA_PATH)/include" -I"$(CUDA_PATH)/samples/common/inc" -I$(INCLUDE_DIR)

XLIBS	= -lcublas
.PHONY: $(TARGET)

$(TARGET): $(TARGET_OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(INCLUDES) $^ $(XLIBS)

.SUFFIXES: .cu .cuh .obj
.cu.obj:
	$(CXX) -o $@ -c $^ $(CXXFLAGS) $(INCLUDES)

clean:
	rm -f $(TARGET) $(TARGET).lib $(TARGET).exp $(TARGET_OBJS)
