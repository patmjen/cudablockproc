TARGET = test_cudablockproc
TARGET_OBJS = main.obj test_blockindexiter.obj test_cudablockproc.obj test_util.obj
SRC_DIR = src
TEST_DIR = test

VPATH := $(SRC_DIR):$(TEST_DIR)

OPT	= -O2
XOPT  = -O3 -lineinfo

CXX	= nvcc
CXXFLAGS = $(XOPT)

GTEST_DIR ?= "D:/libs/googletest"
INCLUDES ?= -I"$(CUDA_PATH)/include" -I"$(CUDA_PATH)/samples/common/inc" -I$(SRC_DIR) -I$(GTEST_DIR)/include

LINK_GTEST_LIB_DIR ?= "$(GTEST_DIR)\msvc\x64\Release"
XLIBS	= -lgtest -L $(LINK_GTEST_LIB_DIR)
.PHONY: clean

$(TARGET): $(TARGET_OBJS)
	$(CXX) -o $@ $(CXXFLAGS) $(INCLUDES) $^ $(XLIBS) --compiler-options="$(OPT)"

test: $(TARGET)
	./$(TARGET)

.SUFFIXES: .cu .cuh .obj
.cu.obj:
	$(CXX) -o $@ -c $< $(CXXFLAGS) $(INCLUDES) $(XLIBS) --compiler-options="$(OPT)"

clean:
	rm -f $(TARGET) $(TARGET).lib $(TARGET).exp $(TARGET_OBJS)

main.obj: cudablockproc.cuh blockindexiter.cuh helper_math.cuh util.cuh zip.cuh blockindexiter.inl cudablockproc.inl
test_blockindexiter.obj: blockindexiter.cuh helper_math.cuh util.cuh blockindexiter.inl
test_cudablockproc.obj: cudablockproc.cuh blockindexiter.cuh helper_math.cuh util.cuh util_test.cuh zip.cuh cudablockproc.inl
test_util.obj: util.cuh util_test.cuh
