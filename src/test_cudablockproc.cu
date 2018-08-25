#include <vector>

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

class AllocBlocksTest : public CudaTest, public ::testing::WithParamInterface<MemLocation> {};
TEST_F(AllocBlocksTest, InvalidLocation)
{
    std::vector<int *> tmp;
    int3 blockSize = make_int3(1);
    MemLocation loc = static_cast<MemLocation>(0xFF); // Known invalid location
    ASSERT_EQ(CBP_INVALID_VALUE, allocBlocks(tmp, 1, loc, blockSize));
}

TEST_P(AllocBlocksTest, BlockLocation)
{
    std::vector<int *> blocks;
    int3 blockSize = make_int3(1);
    const MemLocation loc = GetParam();
    ASSERT_EQ(CBP_SUCCESS, allocBlocks(blocks, 3, loc, blockSize));
    int i = 0;
    for (auto b : blocks) {
        EXPECT_EQ(loc, getMemLocation(b)) << "Block " << i << " has wrong location.";
        i++;
    }
    syncAndAssertCudaSuccess();
}

INSTANTIATE_TEST_CASE_P(ValidLocations, AllocBlocksTest, ::testing::Values(HOST_NORMAL, HOST_PINNED, DEVICE));

TEST(BlockProcTest, SingleInSingleOutNoTmpNoBorder)
{
    static const size_t nvol = 8*8*8, nblk = 3*3*3;
    const int3 volSize = make_int3(8);
    const int3 blockSize = make_int3(3);
    auto identity = [](auto b, auto in, auto out, auto tmp){
        memcpy(out[0], in[0], b.numel()*sizeof(*in[0])); };
    int inVol[nvol], outVol[nvol], inBlk[nblk], outBlk[nblk];
    std::vector<int *> inVols = { inVol }, inBlocks = { inBlk };
    std::vector<int *> outVols = { outVol }, outBlocks = { outBlk }, tmpBlocks = { nullptr };

    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol, 100, nvol);
    blockProc(identity, inVols, outVols, inBlocks, outBlocks, tmpBlocks, volSize, blockSize);
    EXPECT_ARRAY_EQ(inVol, outVol, nvol);
}

TEST(BlockProcTest, MultipleInMultipleOutNoTmpNoBorder)
{
    static const size_t nvol = 8*8*8, nblk = 3*3*3;
    const int3 volSize = make_int3(8);
    const int3 blockSize = make_int3(3);

    auto identityMIMO = [](auto b, auto in, auto out, auto tmp){
        for (int i = 0; i < in.size(); i++) memcpy(out[i], in[i], b.numel()*sizeof(*in[i])); };
    int inVol1[nvol], inVol2[nvol], outVol1[nvol], outVol2[nvol];
    int inBlk1[nblk], inBlk2[nblk], outBlk1[nblk], outBlk2[nblk];

    std::vector<int *> inVols = { inVol1, inVol2 }, inBlocks = { inBlk1, inBlk2 };
    std::vector<int *> outVols = { outVol1, outVol2 }, outBlocks = { outBlk1, outBlk2 };
    std::vector<int *> tmpBlocks = { nullptr };
    fillVolWithIndices(inVol1, nvol);
    fillVolWithIndices(inVol2, nvol, static_cast<int>(nvol));
    fillVolWithValue(outVol1, 100, nvol);
    fillVolWithValue(outVol2, 100, nvol);
    blockProc(identityMIMO, inVols, outVols, inBlocks, outBlocks, tmpBlocks, volSize, blockSize);
    EXPECT_ARRAY_EQ(inVol1, outVol1, nvol);
    EXPECT_ARRAY_EQ(inVol2, outVol2, nvol);
}

TEST(BlockProcTest, SingleInMultipleOutNoTmpNoBorder)
{
    static const size_t nvol = 8*8*8, nblk = 3*3*3;
    const int3 volSize = make_int3(8);
    const int3 blockSize = make_int3(3);

    auto identitySIMO = [](auto b, auto in, auto out, auto tmp){
        for (int i = 0; i < out.size(); i++) memcpy(out[i], in[0], b.numel()*sizeof(*in[i])); };
    int inVol[nvol], outVol1[nvol], outVol2[nvol];
    int inBlk[nblk], outBlk1[nblk], outBlk2[nblk];

    std::vector<int *> inVols = { inVol }, inBlocks = { inBlk };
    std::vector<int *> outVols = { outVol1, outVol2 }, outBlocks = { outBlk1, outBlk2 };
    std::vector<int *> tmpBlocks = { nullptr };
    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol1, 100, nvol);
    fillVolWithValue(outVol2, 100, nvol);
    blockProc(identitySIMO, inVols, outVols, inBlocks, outBlocks, tmpBlocks, volSize, blockSize);
    EXPECT_ARRAY_EQ(inVol, outVol1, nvol);
    EXPECT_ARRAY_EQ(inVol, outVol2, nvol);
}

TEST(BlockProcTest, MultipleInSingleOutNoTmpNoBorder)
{
    static const size_t nvol = 8*8*8, nblk = 3*3*3;
    const int3 volSize = make_int3(8);
    const int3 blockSize = make_int3(3);

    int inVol1[nvol], inVol2[nvol], outVol[nvol];
    int inBlk1[nblk], inBlk2[nblk], outBlk[nblk];
    std::vector<int *> inVols = { inVol1, inVol2 }, inBlocks = { inBlk1, inBlk2 };
    std::vector<int *> outVols = { outVol }, outBlocks = { outBlk }, tmpBlocks = { nullptr };

    auto sumAllIn = [=](auto b, auto in, auto out, auto tmp){
        auto outVol = out[0];
        fillVolWithValue(outVol, 0, b.numel());
        for (int i = 0; i < 2; i++) {
            for (int j = 0; j < b.numel(); j++) {
                outVol[j] += in[i][j];
            }
        }
    };

    int expectedSumVol[nvol];

    inVols = { inVol1, inVol2 };
    inBlocks = { inBlk1, inBlk2 };
    outVols = { outVol };
    outBlocks = { outBlk };
    tmpBlocks = { nullptr };
    fillVolWithIndices(inVol1, nvol);
    fillVolWithIndices(inVol2, nvol, static_cast<int>(nvol));
    fillVolWithValue(outVol, 100, nvol);
    for (int i = 0; i < nvol; i++) {
        expectedSumVol[i] = inVol1[i] + inVol2[i];
    }
    blockProc(sumAllIn, inVols, outVols, inBlocks, outBlocks, tmpBlocks, volSize, blockSize);
    EXPECT_ARRAY_EQ(expectedSumVol, outVol, nvol);
}

TEST(BlockProcTest, SingleInSingleOutNoTmpWithBorder)
{
    static const size_t nvol = 8*8*8, nblk = 5*5*5;
    const int3 volSize = make_int3(8);
    const int3 blockSize = make_int3(3);
    const int3 borderSize = make_int3(1);

    int inVol[nvol], outVol[nvol], inBlk[nblk], outBlk[nblk], expectedOutVol[nvol];
    std::vector<int *> inVols = { inVol }, inBlocks = { inBlk };
    std::vector<int *> outVols = { outVol }, outBlocks = { outBlk }, tmpBlocks = { nullptr };
    std::vector<int *> expectedVols = { expectedOutVol };

    auto sum3x3x3 = [](auto b, auto in, auto out, auto tmp){
        int *vi = in[0], *vo = out[0];
        int3 size = b.blockSizeBorder();
        fillVolWithValue(vo, 0, b.numel());
        for (int x = 1; x < size.x - 1; x++) {
            for (int y = 1; y < size.y - 1; y++) {
                for (int z = 1; z < size.z - 1; z++) {
                    for (int ix = -1; ix <= 1; ix++) {
                        for (int iy = -1; iy <= 1; iy++) {
                            for (int iz = -1; iz <= 1; iz++) {
                                vo[getIdx(x,y,z,size)] += vi[getIdx(x+ix,y+iy,z+iz,size)];
                            }
                        }
                    }
                }
            }
        }
    };

    // Sanity test sum3x3x3 - not meant to be thorough
    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol, 100, 3*3*3);
    fillVolWithValue(expectedOutVol, 0, 3*3*3);
    expectedOutVol[getIdx(1,1,1,make_int3(3))] = 351; // = sum k from 0 to 3*3*3-1
    sum3x3x3(BlockIndex(3), inVols, outVols, tmpBlocks);
    ASSERT_ARRAY_EQ(expectedOutVol, outVol, 3*3*3) << "Test function sum3x3x3 does not work";

    // Test blockProc
    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol, 100, nvol);
    fillVolWithValue(expectedOutVol, 100, nvol);
    sum3x3x3(BlockIndex(volSize), inVols, expectedVols, tmpBlocks);
    blockProc(sum3x3x3, inVols, outVols, inBlocks, outBlocks, tmpBlocks, volSize, blockSize);
    EXPECT_ARRAY_NE(expectedOutVol, outVol, nvol);

    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol, 100, nvol);
    fillVolWithValue(expectedOutVol, 100, nvol);
    sum3x3x3(BlockIndex(volSize), inVols, expectedVols, tmpBlocks);
    blockProc(sum3x3x3, inVols, outVols, inBlocks, outBlocks, tmpBlocks, volSize, blockSize, borderSize);
    EXPECT_ARRAY_EQ(expectedOutVol, outVol, nvol);
}
