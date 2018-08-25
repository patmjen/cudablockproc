#include "gtest/gtest.h"

#include "util.cuh"
#include "util_test.cuh"

class GetMemLocationTest : public CudaTest {};

TEST_F(GetMemLocationTest, HostMemoryNormal)
{
    int x;
    void *hn_ptr = &x;

    EXPECT_EQ(HOST_NORMAL, getMemLocation(hn_ptr));
    EXPECT_TRUE(memLocationIs<HOST_NORMAL>(hn_ptr));

    hn_ptr = nullptr;
    hn_ptr = malloc(1);
    ASSERT_NE(nullptr, hn_ptr) << "Could not allocate normal host memory";
    EXPECT_EQ(HOST_NORMAL, getMemLocation(hn_ptr));
    EXPECT_TRUE(memLocationIs<HOST_NORMAL>(hn_ptr));
    free(hn_ptr);
    syncAndAssertCudaSuccess();
}

TEST_F(GetMemLocationTest, HostMemoryPinned)
{
    void *hp_ptr = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMallocHost(&hp_ptr, 1)) << "Could not allocate pinned host memory.";
    EXPECT_EQ(HOST_PINNED, getMemLocation(hp_ptr));
    EXPECT_TRUE(memLocationIs<HOST_PINNED>(hp_ptr));
    ASSERT_EQ(cudaSuccess, cudaFreeHost(hp_ptr)) << "Could not free pinned host memory";
    syncAndAssertCudaSuccess();
}

TEST_F(GetMemLocationTest, DeviceMemory)
{
    void *d_ptr = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_ptr, 1)) << "Could not allocate device memory";
    EXPECT_EQ(DEVICE, getMemLocation(d_ptr));
    EXPECT_TRUE(memLocationIs<DEVICE>(d_ptr));
    ASSERT_EQ(cudaSuccess, cudaFree(d_ptr)) << "Could not free device memory";
    syncAndAssertCudaSuccess();
}
