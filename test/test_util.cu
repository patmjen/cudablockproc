#include "gtest/gtest.h"

#include "cudablockproc.cuh"
#include "util_test.cuh"

TEST(TypeSizeTest, MatchesSizeofPod)
{
    EXPECT_EQ(sizeof(int), cbp::detail::typeSize<int>());
    EXPECT_EQ(sizeof(double), cbp::detail::typeSize<double>());
}

TEST(TypeSizeTest, MatchesSizeofReference)
{
    EXPECT_EQ(sizeof(int), cbp::detail::typeSize<int&>());
    EXPECT_EQ(sizeof(int), cbp::detail::typeSize<const int&>());
}

TEST(TypeSizeTest, MatchesSizeofClass)
{
    EXPECT_EQ(sizeof(std::vector<int>), cbp::detail::typeSize<std::vector<int>>());
}

TEST(TypeSizeTest, HandlesVoid)
{
    EXPECT_EQ(1, cbp::detail::typeSize<void>());
}

class GetMemLocationTest : public CudaTest {};

TEST_F(GetMemLocationTest, HostMemoryNormal)
{
    int x;
    void *hn_ptr = &x;

    EXPECT_EQ(cbp::HOST_NORMAL, cbp::getMemLocation(hn_ptr));
    EXPECT_TRUE(cbp::memLocationIs<cbp::HOST_NORMAL>(hn_ptr));

    hn_ptr = nullptr;
    hn_ptr = malloc(1);
    ASSERT_NE(nullptr, hn_ptr) << "Could not allocate normal host memory";
    EXPECT_EQ(cbp::HOST_NORMAL, cbp::getMemLocation(hn_ptr));
    EXPECT_TRUE(cbp::memLocationIs<cbp::HOST_NORMAL>(hn_ptr));
    free(hn_ptr);
    syncAndAssertCudaSuccess();
}

TEST_F(GetMemLocationTest, HostMemoryPinned)
{
    void *hp_ptr = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMallocHost(&hp_ptr, 1)) << "Could not allocate pinned host memory.";
    EXPECT_EQ(cbp::HOST_PINNED, cbp::getMemLocation(hp_ptr));
    EXPECT_TRUE(cbp::memLocationIs<cbp::HOST_PINNED>(hp_ptr));
    ASSERT_EQ(cudaSuccess, cudaFreeHost(hp_ptr)) << "Could not free pinned host memory";
    syncAndAssertCudaSuccess();
}

TEST_F(GetMemLocationTest, DeviceMemory)
{
    void *d_ptr = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_ptr, 1)) << "Could not allocate device memory";
    EXPECT_EQ(cbp::DEVICE, cbp::getMemLocation(d_ptr));
    EXPECT_TRUE(cbp::memLocationIs<cbp::DEVICE>(d_ptr));
    ASSERT_EQ(cudaSuccess, cudaFree(d_ptr)) << "Could not free device memory";
    syncAndAssertCudaSuccess();
}

TEST_F(GetMemLocationTest, NullptrIsHostNormal)
{
    void *ptr = nullptr;

    EXPECT_EQ(cbp::HOST_NORMAL, cbp::getMemLocation(ptr));
    EXPECT_TRUE(cbp::memLocationIs<cbp::HOST_NORMAL>(ptr));
    syncAndAssertCudaSuccess();
}
