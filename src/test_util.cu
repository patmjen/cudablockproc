#include "gtest/gtest.h"

#include "util.cuh"
#include "util_test.cuh"

CUDA_TEST(UtilTest, GetMemLocation)
{
    int x;
    void *hn_ptr, *hp_ptr, *d_ptr;

    // Test normal host memory
    hn_ptr = &x;
    EXPECT_EQ(HOST_NORMAL, getMemLocation(hn_ptr));
    EXPECT_TRUE(memLocationIs<HOST_NORMAL>(hn_ptr));

    hn_ptr = nullptr;
    hn_ptr = malloc(1);
    ASSERT_NE(nullptr, hn_ptr) << "Could allocate normal host memory";
    EXPECT_EQ(HOST_NORMAL, getMemLocation(hn_ptr));
    EXPECT_TRUE(memLocationIs<HOST_NORMAL>(hn_ptr));
    free(hn_ptr);

    // Test pinned host memory
    ASSERT_EQ(cudaSuccess, cudaMallocHost(&hp_ptr, 1)) << "Could not allocate pinned host memory.";
    EXPECT_EQ(HOST_PINNED, getMemLocation(hp_ptr));
    EXPECT_TRUE(memLocationIs<HOST_PINNED>(hp_ptr));
    ASSERT_EQ(cudaSuccess, cudaFreeHost(hp_ptr)) << "Could not free pinned host memory";

    // Test device memory
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_ptr, 1)) << "Could not allocate device memory";
    EXPECT_EQ(DEVICE, getMemLocation(d_ptr));
    EXPECT_TRUE(memLocationIs<DEVICE>(d_ptr));
    ASSERT_EQ(cudaSuccess, cudaFree(d_ptr)) << "Could not free device memory";
}