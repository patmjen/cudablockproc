TARGET	= test_cudablockproc
TARGET_OBJS	= main.obj test_blockindexiter.obj test_cudablockproc.obj
SRC_DIR = src

VPATH := $(SRC_DIR)

OPT	= -g -O3 -Wall -Wextra
PIC = #-fpic
XPIC  = #-Xcompiler #-fpic
XOPT  = -O3 -lineinfo #-Xptxas=-v # use -lineinfo for profiler, use -G for debugging
XARCH =
DEF   =

CC = "C:/Program Files (x86)\Microsoft Visual Studio 14.0/VC/bin/x86_amd64"

CXX	= nvcc
CXXFLAGS = -ccbin $(CC) $(XARCH) $(XOPT) $(XPIC) $(DEF)

CUDA_PATH ?= "C:/Program Files/NVIDIA GPU Computing Toolkit/CUDA/9.1"
GTEST_PATH ?= "D:/libs/googletest"
INCLUDES ?= -I"$(CUDA_PATH)/include" -I"$(CUDA_PATH)/samples/common/inc" -I$(SRC_DIR) -I$(GTEST_PATH)/include

XLIBS	= -lcublas -lgtest -L$(GTEST_PATH)/msvc/x64/Release
.PHONY: $(TARGET)

$(TARGET): $(TARGET_OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(INCLUDES) $^ $(XLIBS)

test: $(TARGET)
	./$(TARGET)

.SUFFIXES: .cu .cuh .obj
.cu.obj:
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCLUDES)

clean:
	rm -f $(TARGET) $(TARGET).lib $(TARGET).exp $(TARGET_OBJS)

main.obj:
test_blockindexiter.obj: blockindexiter.cuh helper_math.cuh util.cuh
test_cudablockproc.obj: cudablockproc.cuh blockindexiter.cuh helper_math.cuh util.cuh util_test.cuh