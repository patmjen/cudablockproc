#ifndef __UTIL_TEST_CUH__
#define __UTIL_TEST_CUH__

#include "gtest/gtest.h"
#include "helper_math.cuh"

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

#endif // __UTIL_TEST_CUH__
