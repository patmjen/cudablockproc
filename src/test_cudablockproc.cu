#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "cudablockproc.cuh"
#include "util.cuh"
#include "util_test.cuh"

using namespace cbp;

template <typename Ty>
void fillVolWithIndices(Ty *vol, const int n, const Ty offset=0)
{
    for (int i = 0; i < n; i++) {
        vol[i] = static_cast<Ty>(i) + offset;
    }
}

template <typename Ty>
void fillVolWithValue(Ty *vol, const Ty& val, const int n)
{
    for (int i = 0; i < n; i++) {
        vol[i] = val;
    }
}

TEST(TransferBlockTest, BlockFitsWholeVol)
{
    static const size_t nvol = 4*4*4;
    int vol[nvol], block[nvol];
    const int3 volSize = make_int3(4);
    const BlockIndex bi(make_int3(0), volSize);

    fillVolWithIndices(vol, nvol);
    fillVolWithValue(block, 100, nvol);

    transferBlock(vol, block, bi, volSize, VOL_TO_BLOCK);
    EXPECT_ARRAY_EQ(vol, block, nvol);

    fillVolWithValue(vol, 0, nvol);

    transferBlock(vol, block, bi, volSize, BLOCK_TO_VOL);
    EXPECT_ARRAY_EQ(block, vol, nvol);
}

TEST(TransferBlockTest, BlocksAlignedWithSize)
{
    static const size_t nvol = 4*4*4;
    static const size_t nblk = 2*2*2;
    int vol[nvol], volExpected[nvol], block[nblk];
    fillVolWithIndices(volExpected, nvol);
    const int3 volSize = make_int3(4);
    const int expectedBlocks[][nblk] = {
        { 0, 1, 4, 5, 16, 17, 20, 21 },
        { 2, 3, 6, 7, 18, 19, 22, 23 },
        { 42, 43, 46, 47, 58, 59, 62, 63 }
    };
    const BlockIndex bis[] = {
        BlockIndex(make_int3(0), make_int3(2)),
        BlockIndex(make_int3(2, 0, 0), make_int3(4, 2, 2)),
        BlockIndex(make_int3(2), make_int3(4))
    };

    for (int i = 0; i < 3; i++) {
        memcpy(vol, volExpected, nvol*sizeof(*vol)); // Reset vol
        fillVolWithValue(block, 100, nblk);

        transferBlock(vol, block, bis[i], volSize, VOL_TO_BLOCK);
        EXPECT_ARRAY_EQ(expectedBlocks[i], block, nblk);

        transferBlock(vol, block, bis[i], volSize, BLOCK_TO_VOL);
        EXPECT_ARRAY_EQ(volExpected, vol, nvol);
    }
}

TEST(TransferBlockTest, BlocksWithBordersWithoutClamping)
{
    static const size_t nvol = 4*4*4;
    int vol[nvol], volExpected[nvol], block[nvol];
    const BlockIndex bi(make_int3(1),make_int3(3),make_int3(0),make_int3(4));
    const int3 volSize = make_int3(4);

    fillVolWithIndices(vol, nvol);
    fillVolWithIndices(volExpected, nvol);
    fillVolWithValue(block, 100, nvol);

    transferBlock(vol, block, bi, volSize, VOL_TO_BLOCK);
    EXPECT_ARRAY_EQ(volExpected, block, nvol);

    transferBlock(vol, block, bi, volSize, BLOCK_TO_VOL);
    EXPECT_ARRAY_EQ(volExpected, vol, nvol);
}

TEST(TransferBlockTest, BlocksWithBordersWithClamping)
{
    static const size_t nvol = 4*4*4;
    int vol[nvol], volExpected[nvol], block[nvol];
    const int3 volSize = make_int3(4);
    const BlockIndex bis[] = {
        BlockIndex(make_int3(0), make_int3(2), make_int3(0), make_int3(3)),
        BlockIndex(make_int3(2), make_int3(4), make_int3(1), make_int3(4)),
        BlockIndex(make_int3(1,2,0), make_int3(3, 4, 2), make_int3(0,1,0), make_int3(4, 4, 3))
    };
    // TODO: Maybe find a better way to make these? But then we would likely need testing for the test code...
    const int expectedBlocks[][nvol] = {
        {
            0, 1, 2, 4, 5, 6, 8, 9, 10, 16, 17, 18, 20, 21, 22,
            24, 25, 26, 32, 33, 34, 36, 37, 38, 40, 41, 42, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100
        }, {
            21, 22, 23, 25, 26, 27, 29, 30, 31, 37, 38, 39, 41, 42, 43,
            45, 46, 47, 53, 54, 55, 57, 58, 59, 61, 62, 63, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100
        }, {
            4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
            20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
            36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
            100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100, 100,
        }
    };

    fillVolWithIndices(volExpected, nvol);
    for (int i = 0; i < 3; i++) {
        memcpy(vol, volExpected, nvol*sizeof(*vol)); // Reset vol
        fillVolWithValue(block, 100, nvol);

        transferBlock(vol, block, bis[i], volSize, VOL_TO_BLOCK);
        EXPECT_ARRAY_EQ(expectedBlocks[i], block, nvol);

        transferBlock(vol, block, bis[i], volSize, BLOCK_TO_VOL);
        EXPECT_ARRAY_EQ(volExpected, vol, nvol);
    }
}