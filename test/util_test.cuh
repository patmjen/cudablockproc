#ifndef UTIL_TEST_CUH__
#define UTIL_TEST_CUH__

#include "gtest/gtest.h"
#include "helper_math.cuh"

inline __host__ __device__ size_t getIdx(const int x, const int y, const int z, const int3 siz)
{
    return x + y * siz.x + z * siz.x * siz.y;
}

inline __host__ __device__ size_t getIdx(const int3 pos, const int3 siz)
{
    return getIdx(pos.x, pos.y, pos.z, siz);
}

#define TYPE_TRAIT_TEST(pred, type) \
    static_assert(pred<type>::value, #type " failed " #pred)

#define EXPECT_ARRAY_EQ(expected, actual, n) \
    EXPECT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return assertArrayEqual(e1, e2, a1, a2, n); }, expected, actual)

#define ASSERT_ARRAY_EQ(expected, actual, n) \
    ASSERT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return assertArrayEqual(e1, e2, a1, a2, n); }, expected, actual)

#define EXPECT_ARRAY_NE(expected, actual, n) \
    EXPECT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return !assertArrayEqual(e1, e2, a1, a2, n); }, expected, actual)

#define ASSERT_ARRAY_NE(expected, actual, n) \
    ASSERT_PRED_FORMAT2([=](auto e1, auto e2, auto a1, auto a2) \
        { return !assertArrayEqual(e1, e2, a1, a2, n); }, expected, actual)

template <typename Ty>
::testing::AssertionResult assertArrayEqual(const char *expr1, const char *expr2, const Ty *a1, const Ty *a2,
    const int n)
{
    if (n <= 0) {
        return ::testing::AssertionSuccess() << "Arrays are empty.";
    }
    int numFail = 0;
    for (int i = 0; i < n; i++) {
        if (a1[i] != a2[i]) {
            numFail++;
        }
    }
    if (numFail == 0) {
        return ::testing::AssertionSuccess() << expr1 << " and " << expr2 << " are equal.";
    }
    auto out = ::testing::AssertionFailure() << expr1 << " and " << expr2 << " differ in " << numFail <<
        " elements.\nExpected: [" << a1[0];
    for (int i = 1; i < n; i++) {
        out << ", " << a1[i];
    }
    out << "]\nActual:   [" << a2[0];
    for (int i = 1; i < n; i++) {
        out << ", " << a2[i];
    }
    out << "]";
    return out;
}

class CudaTest : public ::testing::Test {
public:
    static void TearDownTestCase()
    {
        cudaDeviceReset();
    }

protected:
    void SetUp() override
    {
        cudaDeviceReset();
    }

    void assertCudaSuccess(cudaError_t err) const
    {
        ASSERT_EQ(cudaSuccess, err) << "Got CUDA error: " << cudaGetErrorString(err);
    }

    void assertCudaSuccess() const
    {
        cudaError_t err = cudaPeekAtLastError();
        assertCudaSuccess(err);
    }

    void syncAndAssertCudaSuccess() const
    {
        cudaError_t err = cudaDeviceSynchronize();
        assertCudaSuccess(err);
    }

    cudaError_t expectCudaSuccess(cudaError_t err) const
    {
        EXPECT_EQ(cudaSuccess, err) << "Got CUDA error: " << cudaGetErrorString(err);
        return err;
    }

    cudaError_t expectCudaSuccess() const
    {
        cudaError_t err = cudaPeekAtLastError();
        return expectCudaSuccess(err);
    }

    bool syncAndExpectCudaSuccess() const
    {
        cudaError_t err = cudaDeviceSynchronize();
        return expectCudaSuccess(err);
    }
};

#endif // UTIL_TEST_CUH__
