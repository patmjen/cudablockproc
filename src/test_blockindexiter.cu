#include <vector>
#include <type_traits>
#include <iterator>
#include <utility>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "util_test.cuh"
#include "blockindexiter.cuh"

using namespace cbp;

// Compile time tests
TYPE_TRAIT_TEST(std::is_default_constructible, BlockIndexIterator);
TYPE_TRAIT_TEST(std::is_move_constructible, BlockIndexIterator);
TYPE_TRAIT_TEST(std::is_copy_constructible, BlockIndexIterator);
TYPE_TRAIT_TEST(std::is_copy_assignable, BlockIndexIterator);
TYPE_TRAIT_TEST(std::is_destructible, BlockIndexIterator);
static_assert(std::is_same<std::iterator_traits<BlockIndexIterator>::reference, const BlockIndex&>::value,
    "BlockIndexIterator reference type is not const BlockIndex&");

TEST(BlockIndexIteratorTest, Equality)
{
    const int3 volSize1 = make_int3(3), volSize2 = make_int3(5);
    const int3 blockSize1 = make_int3(2), blockSize2 = make_int3(4);
    EXPECT_EQ(BlockIndexIterator(), BlockIndexIterator());
    EXPECT_EQ(BlockIndexIterator(volSize1, blockSize1), BlockIndexIterator(volSize1, blockSize1));
    EXPECT_EQ(BlockIndexIterator(volSize1, blockSize1) + 1, BlockIndexIterator(volSize1, blockSize1) + 1);

    EXPECT_NE(BlockIndexIterator(volSize1, blockSize1), BlockIndexIterator(volSize2, blockSize1));
    EXPECT_NE(BlockIndexIterator(volSize1, blockSize1), BlockIndexIterator(volSize1, blockSize2));
    EXPECT_NE(BlockIndexIterator(volSize1, blockSize1), BlockIndexIterator(volSize1, blockSize1, blockSize1));
    EXPECT_NE(BlockIndexIterator(volSize1, blockSize1) + 1, BlockIndexIterator(volSize1, blockSize1));
}

TEST(BlockIndexIteratorTest, Swap)
{
    BlockIndexIterator a(make_int3(3), make_int3(2)), aOld(a);
    BlockIndexIterator b(make_int3(5), make_int3(4)), bOld(b);
    ASSERT_NE(a, b);
    ASSERT_NE(aOld, bOld);
    ASSERT_EQ(a, aOld);
    ASSERT_EQ(b, bOld);
    std::swap(a, b);
    EXPECT_NE(a, b);
    EXPECT_NE(a, aOld);
    EXPECT_NE(b, bOld);
    EXPECT_EQ(aOld, b);
    EXPECT_EQ(bOld, a);
}

TEST(BlockIndexIteratorTest, PseudoRandomInputIteratorCompliant)
{
    // See https://en.cppreference.com/w/cpp/named_req/InputIterator  and
    // https://en.cppreference.com/w/cpp/named_req/RandomAccessIterator for requirements
    BlockIndexIterator b(make_int3(10), make_int3(2));
    BlockIndexIterator bOrig = b;
    BlockIndexIterator expected;

    // Iterator requirements
    ASSERT_EQ(0, b.linearIndex());
    ASSERT_EQ(BlockIndex(0, 2), *b);
    ++b;
    EXPECT_EQ(1, b.linearIndex());
    EXPECT_EQ(BlockIndex(make_int3(2,0,0), make_int3(4,2,2)), *b);

    // InputIterator requirements
    expected = ++b;
    EXPECT_EQ((*b).startIdx, b->startIdx);
    EXPECT_EQ(expected, b);

    (void)b++; // Passes if this is compilable

    expected = b;
    ASSERT_EQ(expected, b);
    BlockIndex expectedBlockIndex = *b;
    BlockIndex actualBlockIndex = *b++;
    EXPECT_EQ(++expected, b);
    EXPECT_EQ(expectedBlockIndex, actualBlockIndex);

    // NOTE: from this point, only the requirements BlockIndexIterator actually conforms to are tested

    // ForwardIterator requirements
    expected = b;
    ASSERT_EQ(expected, b);
    EXPECT_EQ(++expected, ++b);
    BlockIndexIterator bOld = b++;
    EXPECT_EQ(expected, bOld);
    EXPECT_NE(bOld, b);

    expected = b;
    EXPECT_EQ(*expected, *b++);

    // BidirectionalIterator requirements
    expected = b;
    ASSERT_EQ(expected, b);
    EXPECT_EQ(--expected, --b);
    EXPECT_EQ(*expected, *b);

    expected = b;
    EXPECT_EQ(*expected, *b--);

    // RandomAccessIterator requirements
    expected = b;
    ASSERT_EQ(expected, b);
    ++(++(++expected)); // Move forward 3 times
    ASSERT_NE(expected, b);
    EXPECT_EQ(expected, b + 3);
    EXPECT_EQ(expected, 3 + b);
    EXPECT_EQ(expected, b += 3);
    EXPECT_EQ(expected, b);

    expected = b;
    ASSERT_EQ(expected, b);
    --(--(--expected)); // Move back 3 times
    ASSERT_NE(expected, b);
    EXPECT_EQ(expected, b - 3);
    EXPECT_EQ(expected, b -= 3);
    EXPECT_EQ(expected, b);

    b = bOrig + 3;;
    expected = bOrig + 2;
    EXPECT_EQ(expected, 5 - b);

    expected = b;
    ASSERT_EQ(expected, b);
    EXPECT_TRUE(expected <= b);
    EXPECT_TRUE(expected >= b);

    ++b;
    EXPECT_TRUE(expected <= b);
    EXPECT_TRUE(expected < b);

    b = expected;
    ++expected;
    EXPECT_TRUE(b <= expected);
    EXPECT_TRUE(b < expected);

    b = bOrig;
    EXPECT_EQ(*(b + 2), b[2]);

    b += 5;
    EXPECT_EQ(b - bOrig, 5);
    EXPECT_EQ(bOrig - b, -5);
}

// Helper function to build a BlockIndexIterator and extract its numBlocks field
inline int3 numBlocks(int3 volSize, int3 blockSize, int3 borderSize=make_int3(0))
{
    return BlockIndexIterator(volSize, blockSize, borderSize).numBlocks();
}

TEST(BlockIndexIteratorTest, CalcNumBlocks)
{
    EXPECT_EQ(make_int3(2), numBlocks(make_int3(4), make_int3(2)));
    EXPECT_EQ(make_int3(2), numBlocks(make_int3(3), make_int3(2)));
    EXPECT_EQ(make_int3(1), numBlocks(make_int3(2), make_int3(2)));
    EXPECT_EQ(make_int3(4, 2, 3), numBlocks(make_int3(8, 6, 12), make_int3(2, 3, 4)));
    EXPECT_EQ(make_int3(4, 2, 3), numBlocks(make_int3(7, 5, 11), make_int3(2, 3, 4)));
    EXPECT_EQ(make_int3(2), numBlocks(make_int3(4), make_int3(2), make_int3(2)));
    EXPECT_EQ(make_int3(2), numBlocks(make_int3(3), make_int3(2), make_int3(2)));
}

// Helper function to build a BlockIndexIterator and extract its blkIdx field
inline int maxBlkIdx(int3 volSize, int3 blockSize, int3 borderSize=make_int3(0))
{
    return BlockIndexIterator(volSize, blockSize, borderSize).maxLinearIndex();
}

TEST(BlockIndexIteratorTest, CalcMaxIdx)
{
    EXPECT_EQ(7, maxBlkIdx(make_int3(4), make_int3(2)));
    EXPECT_EQ(7, maxBlkIdx(make_int3(3), make_int3(2)));
    EXPECT_EQ(0, maxBlkIdx(make_int3(2), make_int3(2)));
    EXPECT_EQ(23, maxBlkIdx(make_int3(8, 6, 12), make_int3(2, 3, 4)));
    EXPECT_EQ(23, maxBlkIdx(make_int3(7, 5, 11), make_int3(2, 3, 4)));
    EXPECT_EQ(7, maxBlkIdx(make_int3(4), make_int3(2), make_int3(2)));
    EXPECT_EQ(7, maxBlkIdx(make_int3(3), make_int3(2), make_int3(2)));
}

TEST(BlockIndexIteratorTest, BeginEnd)
{
    const BlockIndexIterator bii1(make_int3(2), make_int3(1));
    EXPECT_EQ(0, bii1.linearIndex());
    EXPECT_EQ(0, bii1.begin().linearIndex());
    EXPECT_EQ(2*2*2, bii1.end().linearIndex());

    const BlockIndexIterator bii2(make_int3(8, 6, 12), make_int3(2, 3, 4));
    EXPECT_EQ(0, bii2.linearIndex());
    EXPECT_EQ(0, bii2.begin().linearIndex());
    EXPECT_EQ(4*2*3, bii2.end().linearIndex());
}

// Helper functions which check if block indices are computed correctly
void checkBlockIndices(const BlockIndexIterator& bii,
    std::vector<int> xIdxStart, std::vector<int> yIdxStart, std::vector<int> zIdxStart,
    std::vector<int> xIdxEnd, std::vector<int> yIdxEnd, std::vector<int> zIdxEnd,
    std::vector<int> xIdxStartBdr, std::vector<int> yIdxStartBdr, std::vector<int> zIdxStartBdr,
    std::vector<int> xIdxEndBdr, std::vector<int> yIdxEndBdr, std::vector<int> zIdxEndBdr)
{
    // X-lists
    ASSERT_EQ(xIdxStart.size(), xIdxEnd.size());
    ASSERT_EQ(xIdxStart.size(), xIdxStartBdr.size());
    ASSERT_EQ(xIdxStart.size(), xIdxEndBdr.size());

    // Y-lists
    ASSERT_EQ(yIdxStart.size(), yIdxEnd.size());
    ASSERT_EQ(yIdxStart.size(), yIdxStartBdr.size());
    ASSERT_EQ(yIdxStart.size(), yIdxEndBdr.size());

    // Z-lists
    ASSERT_EQ(zIdxStart.size(), zIdxEnd.size());
    ASSERT_EQ(zIdxStart.size(), zIdxStartBdr.size());
    ASSERT_EQ(zIdxStart.size(), zIdxEndBdr.size());

    int i = 0;
    for (int zi = 0; zi < zIdxStart.size(); zi++) {
        for (int yi = 0; yi < yIdxStart.size(); yi++) {
            for (int xi = 0; xi < xIdxStart.size(); xi++) {
                const BlockIndex b = bii.blockIndexAt(i);
                const int3 xyzIdx = make_int3(xi, yi, zi);
                const int3 startIdx = make_int3(xIdxStart[xi], yIdxStart[yi], zIdxStart[zi]);
                const int3 endIdx = make_int3(xIdxEnd[xi], yIdxEnd[yi], zIdxEnd[zi]);
                const int3 startIdxBdr = make_int3(xIdxStartBdr[xi], yIdxStartBdr[yi], zIdxStartBdr[zi]);
                const int3 endIdxBdr = make_int3(xIdxEndBdr[xi], yIdxEndBdr[yi], zIdxEndBdr[zi]);
                EXPECT_EQ(startIdx, b.startIdx) <<
                    "Wrong start index at " << xyzIdx << ", i=" << i;
                EXPECT_EQ(endIdx, b.endIdx) <<
                    "Wrong end index at " << xyzIdx << ", i=" << i;
                EXPECT_EQ(startIdxBdr, b.startIdxBorder) <<
                    "Wrong start index w/ border at " << xyzIdx << ", i=" << i;
                EXPECT_EQ(endIdxBdr, b.endIdxBorder) <<
                    "Wrong end index w/ border at " << xyzIdx << ", i=" << i;
                i++;
            }
        }
    }
}

void checkBlockIndices(const BlockIndexIterator& bii,
    std::vector<int> xIdxStart, std::vector<int> yIdxStart, std::vector<int> zIdxStart,
    std::vector<int> xIdxEnd, std::vector<int> yIdxEnd, std::vector<int> zIdxEnd)
{
    checkBlockIndices(bii,
        xIdxStart, yIdxStart, zIdxStart, xIdxEnd, yIdxEnd, zIdxEnd,
        xIdxStart, yIdxStart, zIdxStart, xIdxEnd, yIdxEnd, zIdxEnd);
}

TEST(BlockIndexIteratorTest, AlignedIndices)
{
    const int3 blockSize = make_int3(2, 3, 4);
    const int3 volSize = make_int3(6, 6, 8);
    const BlockIndexIterator bii(volSize, blockSize);

    std::vector<int> xStart = { 0, 2, 4 }, yStart = { 0, 3 }, zStart = { 0, 4 };
    std::vector<int> xEnd = { 2, 4, 6 }, yEnd = { 3, 6 }, zEnd = { 4, 8 };
    checkBlockIndices(bii, xStart, yStart, zStart, xEnd, yEnd, zEnd);
}

TEST(BlockIndexIteratorTest, UnalignedIndices)
{
    const int3 blockSize = make_int3(2, 3, 4);
    const int3 volSize = make_int3(5, 5, 6);
    const BlockIndexIterator bii(volSize, blockSize);

    std::vector<int> xStart = { 0, 2, 4 }, yStart = { 0, 3 }, zStart = { 0, 4 };
    std::vector<int> xEnd = { 2, 4, 5 }, yEnd = { 3, 5 }, zEnd = { 4, 6 };
    checkBlockIndices(bii, xStart, yStart, zStart, xEnd, yEnd, zEnd);
}

TEST(BlockIndexIteratorTest, IndicesWithBorders)
{
    const int3 blockSize = make_int3(2, 3, 4);
    const int3 borderSize = make_int3(2, 0, 3);
    const int3 volSize = make_int3(6, 6, 8);
    const BlockIndexIterator bii(volSize, blockSize, borderSize);

    std::vector<int> xStart = { 0, 2, 4 }, yStart = { 0, 3 }, zStart = { 0, 4 };
    std::vector<int> xStartBdr = { 0, 0, 2 }, yStartBdr = { 0, 3 }, zStartBdr = { 0, 1 };
    std::vector<int> xEnd = { 2, 4, 6 }, yEnd = { 3, 6 }, zEnd = { 4, 8 };
    std::vector<int> xEndBdr = { 4, 6, 6 }, yEndBdr = { 3, 6 }, zEndBdr = { 7, 8 };
    checkBlockIndices(bii, xStart, yStart, zStart, xEnd, yEnd, zEnd,
        xStartBdr, yStartBdr, zStartBdr, xEndBdr, yEndBdr, zEndBdr);
}

TEST(BlockIndexIteratorTest, Iteration)
{
    const int3 blockSize = make_int3(2, 3, 4);
    const int3 volSize = make_int3(6, 6, 8);
    const BlockIndexIterator bii(volSize, blockSize);
    const int NUM_ITERS = 3*2*2;

    BlockIndexIterator b = bii.begin();
    EXPECT_EQ(bii.begin(), b);
    for (int i = 0; i < NUM_ITERS; i++) {
        EXPECT_EQ(i, b.linearIndex());
        ++b;
    }
    EXPECT_EQ(bii.end(), b);
    ++b;
    EXPECT_EQ(bii.end(), b);

    int nIter = 0;
    for (auto b = bii.begin(); b != bii.end(); ++b) {
        EXPECT_EQ(nIter, b.linearIndex());
        nIter++;
        if (nIter > NUM_ITERS) {
            // Safe guard to ensure the test will finish
            nIter = -1;
            break;
        }
    }
    EXPECT_EQ(NUM_ITERS, nIter);

    nIter = 0;
    for (BlockIndex b : bii) {
        nIter++;
        if (nIter > NUM_ITERS) {
            // Safe guard to ensure the test will finish
            nIter = -1;
            break;
        }
    }
    EXPECT_EQ(NUM_ITERS, nIter);
}