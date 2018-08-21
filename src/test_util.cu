#include "gtest/gtest.h"

#include "util.cuh"

TEST(UtilTest, GetMemLocation)
{
    int x;
    void *hn_ptr, *hp_ptr, *d_ptr;
    cudaError_t err;

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
    err = cudaMallocHost(&hp_ptr, 1);
    ASSERT_EQ(cudaSuccess, err) << "Could not allocate pinned host memory.";
    EXPECT_EQ(HOST_PINNED, getMemLocation(hp_ptr));
    EXPECT_TRUE(memLocationIs<HOST_PINNED>(hp_ptr));
    err = cudaFreeHost(hp_ptr);
    ASSERT_EQ(cudaSuccess, err) << "Could not free pinned host memory";

    // Test device memory
    err = cudaMalloc(&d_ptr, 1);
    ASSERT_EQ(cudaSuccess, err) << "Could not allocate device memory";
    EXPECT_EQ(DEVICE, getMemLocation(d_ptr));
    EXPECT_TRUE(memLocationIs<DEVICE>(d_ptr));
    err = cudaFree(d_ptr);
    ASSERT_EQ(cudaSuccess, err) << "Could not free device memory";
}