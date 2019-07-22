#include <vector>
#include <algorithm>

#include "gtest/gtest.h"
#include "helper_math.cuh"

#include "cudablockproc.cuh"
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

    blockVolumeTransfer<VOL_TO_BLOCK>(vol, block, bi, volSize, 0);
    EXPECT_ARRAY_EQ(vol, block, nvol);

    fillVolWithValue(vol, 0, nvol);

    blockVolumeTransfer<BLOCK_TO_VOL>(vol, block, bi, volSize, 0);
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

        blockVolumeTransfer<VOL_TO_BLOCK>(vol, block, bis[i], volSize, 0);
        EXPECT_ARRAY_EQ(expectedBlocks[i], block, nblk);

        blockVolumeTransfer<BLOCK_TO_VOL>(vol, block, bis[i], volSize, 0);
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

    blockVolumeTransfer<VOL_TO_BLOCK>(vol, block, bi, volSize, 0);
    EXPECT_ARRAY_EQ(volExpected, block, nvol);

    blockVolumeTransfer<BLOCK_TO_VOL>(vol, block, bi, volSize, 0);
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

        blockVolumeTransfer<VOL_TO_BLOCK>(vol, block, bis[i], volSize, 0);
        EXPECT_ARRAY_EQ(expectedBlocks[i], block, nvol);

        blockVolumeTransfer<BLOCK_TO_VOL>(vol, block, bis[i], volSize, 0);
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

// TODO: Make this a typed test
class BlockProcTest : public CudaTest {
protected:
    std::vector<int *> volsIn;
    std::vector<int *> volsOut;
    std::vector<int *> hostBlocksIn;
    std::vector<int *> hostBlocksOut;
    std::vector<int *> deviceBlocksIn;
    std::vector<int *> deviceBlocksOut;
    int *deviceTmpMem;

    template <typename Func>
    CbpResult invokeBlockProcWith(Func func, cbp::BlockIndexIterator blockIter)
    {
        return blockProcMultiple(func, volsIn, volsOut,
            hostBlocksIn, hostBlocksOut,
            deviceBlocksIn, deviceBlocksOut, blockIter, deviceTmpMem);
    }

    CbpResult allocBlocksBasedOnVols(const int3 blockSize, const int3 borderSize=make_int3(0),
        const size_t tmpSize=0)
    {
        CbpResult err;
        err = allocBlocks(hostBlocksIn, volsIn.size(), HOST_PINNED, blockSize, borderSize);
        if (err != CBP_SUCCESS) return err;
        err = allocBlocks(hostBlocksOut, volsOut.size(), HOST_PINNED, blockSize, borderSize);
        if (err != CBP_SUCCESS) return err;
        err = allocBlocks(deviceBlocksIn, volsIn.size(), DEVICE, blockSize, borderSize);
        if (err != CBP_SUCCESS) return err;
        err = allocBlocks(deviceBlocksOut, volsOut.size(), DEVICE, blockSize, borderSize);
        if (err != CBP_SUCCESS) return err;
        if (tmpSize > 0) {
            if (cudaMalloc(&deviceTmpMem, tmpSize) != cudaSuccess) {
                err = CBP_DEVICE_MEM_ALLOC_FAIL;
            } else {
                err = CBP_SUCCESS;
            }
        } else {
            deviceTmpMem = nullptr;
        }
        return err;
    }

    void TearDown() override
    {
        std::for_each(hostBlocksIn.begin(), hostBlocksIn.end(), cudaFreeHost);
        std::for_each(hostBlocksOut.begin(), hostBlocksOut.end(), cudaFreeHost);
        std::for_each(deviceBlocksIn.begin(), deviceBlocksIn.end(), cudaFree);
        std::for_each(deviceBlocksOut.begin(), deviceBlocksOut.end(), cudaFree);
        cudaFree(deviceTmpMem);
    }
};

TEST_F(BlockProcTest, SingleInSingleOutNoTmpNoBorder)
{
    static const size_t nvol = 2*2*2;
    const int3 volSize = make_int3(2,2,2);
    const int3 blockSize = make_int3(3);
    auto identity = [](auto b, auto s, auto in, auto out, auto tmp){
        cudaMemcpy(out[0], in[0], b.numel()*sizeof(*in[0]), cudaMemcpyDeviceToDevice);
    };
    int inVol[nvol], outVol[nvol];
    volsIn = { inVol };
    volsOut = { outVol };
    ASSERT_EQ(CBP_SUCCESS, allocBlocksBasedOnVols(blockSize));

    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol, 100, nvol);
    ASSERT_EQ(CBP_SUCCESS, invokeBlockProcWith(identity, cbp::BlockIndexIterator(volSize, blockSize)));
    EXPECT_ARRAY_EQ(inVol, outVol, nvol);
}

TEST_F(BlockProcTest, MultipleInMultipleOutNoTmpNoBorder)
{
    static const size_t nvol = 8*8*8;
    const int3 volSize = make_int3(8);
    const int3 blockSize = make_int3(3);

    auto identityMIMO = [](auto b, auto s, auto in, auto out, auto tmp){
        for (int i = 0; i < in.size(); i++)
            cudaMemcpy(out[i], in[i], b.numel()*sizeof(*in[i]), cudaMemcpyDeviceToDevice);
    };
    int inVol1[nvol], inVol2[nvol], outVol1[nvol], outVol2[nvol];
    volsIn = { inVol1, inVol2 };
    volsOut = { outVol1, outVol2 };
    ASSERT_EQ(CBP_SUCCESS, allocBlocksBasedOnVols(blockSize));

    fillVolWithIndices(inVol1, nvol);
    fillVolWithIndices(inVol2, nvol, static_cast<int>(nvol));
    fillVolWithValue(outVol1, 100, nvol);
    fillVolWithValue(outVol2, 100, nvol);
    ASSERT_EQ(CBP_SUCCESS, invokeBlockProcWith(identityMIMO, cbp::BlockIndexIterator(volSize, blockSize)));
    EXPECT_ARRAY_EQ(inVol1, outVol1, nvol);
    EXPECT_ARRAY_EQ(inVol2, outVol2, nvol);
}

TEST_F(BlockProcTest, SingleInMultipleOutNoTmpNoBorder)
{
    static const size_t nvol = 8*8*8;
    const int3 volSize = make_int3(8);
    const int3 blockSize = make_int3(3);

    auto identitySIMO = [](auto b, auto s, auto in, auto out, auto tmp){
        for (int i = 0; i < out.size(); i++)
            cudaMemcpy(out[i], in[0], b.numel()*sizeof(*in[i]), cudaMemcpyDeviceToDevice);
        };
    int inVol[nvol], outVol1[nvol], outVol2[nvol];
    volsIn = { inVol };
    volsOut = { outVol1, outVol2 };
    ASSERT_EQ(CBP_SUCCESS, allocBlocksBasedOnVols(blockSize));

    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol1, 100, nvol);
    fillVolWithValue(outVol2, 100, nvol);
    ASSERT_EQ(CBP_SUCCESS, invokeBlockProcWith(identitySIMO, cbp::BlockIndexIterator(volSize, blockSize)));
    EXPECT_ARRAY_EQ(inVol, outVol1, nvol);
    EXPECT_ARRAY_EQ(inVol, outVol2, nvol);
}

__global__ void sumAllInKernel(int *out, const int *in1, const int *in2)
{
    out[threadIdx.x] = in1[threadIdx.x] + in2[threadIdx.x];
}
TEST_F(BlockProcTest, MultipleInSingleOutNoTmpNoBorder)
{
    static const size_t nvol = 8*8*8;
    const int3 volSize = make_int3(8);
    const int3 blockSize = make_int3(3);

    auto sumAllIn = [=](auto b, auto s, auto in, auto out, auto tmp){
        sumAllInKernel<<<1,b.numel(), 0, s>>>(out[0], in[0], in[1]);
    };
    int inVol1[nvol], inVol2[nvol], outVol[nvol], expectedSumVol[nvol];
    volsIn = { inVol1, inVol2 };
    volsOut = { outVol };
    ASSERT_EQ(CBP_SUCCESS, allocBlocksBasedOnVols(blockSize));

    fillVolWithIndices(inVol1, nvol);
    fillVolWithIndices(inVol2, nvol, static_cast<int>(nvol));
    fillVolWithValue(outVol, 100, nvol);
    for (int i = 0; i < nvol; i++) {
        expectedSumVol[i] = inVol1[i] + inVol2[i];
    }
    ASSERT_EQ(CBP_SUCCESS, invokeBlockProcWith(sumAllIn, cbp::BlockIndexIterator(volSize, blockSize)));
    EXPECT_ARRAY_EQ(expectedSumVol, outVol, nvol);
}

__global__ void sum3x3x3Kernel(int *out, const int *in, const int3 size)
{
    const int3 pos = make_int3(threadIdx);
    if (pos >= 0 && pos < size) {
        out[getIdx(pos,size)] = 0;
    }
    if (pos >= 1 && pos < size - 1) {
        for (int ix = -1; ix <= 1; ix++) {
            for (int iy = -1; iy <= 1; iy++) {
                for (int iz = -1; iz <= 1; iz++) {
                    out[getIdx(pos,size)] += in[getIdx(pos + make_int3(ix,iy,iz),size)];
                }
            }
        }
    }
}
TEST_F(BlockProcTest, Sum3x3x3Kernel)
{
    static const size_t nvol = 4*3*3;
    const int3 volSize = make_int3(4,3,3);
    int inVol[nvol], outVol[nvol], expectedOutVol[nvol], *d_inVol, *d_outVol;

    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol, 100, nvol);
    fillVolWithValue(expectedOutVol, 0, nvol);
    expectedOutVol[getIdx(1,1,1,volSize)] = 459;
    expectedOutVol[getIdx(2,1,1,volSize)] = 486;

    assertCudaSuccess(cudaMalloc(&d_inVol, nvol*sizeof(*d_inVol)));
    assertCudaSuccess(cudaMalloc(&d_outVol, nvol*sizeof(*d_outVol)));

    assertCudaSuccess(cudaMemcpy(d_inVol, inVol, nvol*sizeof(*d_inVol), cudaMemcpyHostToDevice));
    sum3x3x3Kernel<<<1,dim3(volSize.x,volSize.y,volSize.z)>>>(d_outVol, d_inVol, volSize);
    assertCudaSuccess(cudaMemcpy(outVol, d_outVol, nvol*sizeof(*d_inVol), cudaMemcpyDeviceToHost));
    EXPECT_ARRAY_EQ(expectedOutVol, outVol, nvol);
    syncAndAssertCudaSuccess();
}

TEST_F(BlockProcTest, SingleInSingleOutNoTmpWithBorder)
{
    static const size_t nvol = 8*8*8;
    const int3 volSize = make_int3(8);
    const int3 blockSize = make_int3(3);
    const int3 borderSize = make_int3(1);

    auto sum3x3x3 = [](auto b, auto s, auto in, auto out, auto tmp){
        const int3 siz = b.blockSizeBorder();
        sum3x3x3Kernel<<<1,dim3(siz.x,siz.y,siz.z),0,s>>>(out[0], in[0], siz);
    };
    int inVol[nvol], outVol[nvol], expectedOutVol[nvol];
    volsIn = { inVol };
    volsOut = { outVol };
    ASSERT_EQ(CBP_SUCCESS, allocBlocksBasedOnVols(blockSize, borderSize));

    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol, 100, nvol);
    fillVolWithValue(expectedOutVol, 0, nvol);
    for (int x = 1; x < volSize.x - 1; x++) {
        for (int y = 1; y < volSize.y - 1; y++) {
            for (int z = 1; z < volSize.z - 1; z++) {
                // Sum 3x3x3 window
                for (int ix = -1; ix <= 1; ix++) {
                    for (int iy = -1; iy <= 1; iy++) {
                        for (int iz = -1; iz <= 1; iz++) {
                            expectedOutVol[getIdx(x,y,z,volSize)] += inVol[getIdx(x+ix,y+iy,z+iz,volSize)];
                        }
                    }
                }
            }
        }
    }

    ASSERT_EQ(CBP_SUCCESS, invokeBlockProcWith(sum3x3x3, cbp::BlockIndexIterator(volSize, blockSize)));
    EXPECT_ARRAY_NE(expectedOutVol, outVol, nvol) << "blockProcMultiple worked without borders";

    fillVolWithIndices(inVol, nvol);
    fillVolWithValue(outVol, 100, nvol);
    ASSERT_EQ(CBP_SUCCESS,
              invokeBlockProcWith(sum3x3x3, cbp::BlockIndexIterator(volSize, blockSize, borderSize)));
    EXPECT_ARRAY_EQ(expectedOutVol, outVol, nvol) << "blockProcMultiple didn't work with borders";
    syncAndAssertCudaSuccess();
}

TEST_F(BlockProcTest, InvalidLocation)
{
    auto nop = [](auto b, auto s, auto in, auto out, auto tmp){};
    const int3 volSize = make_int3(1), blockSize = make_int3(1);

    int x, *pptr, *dptr;
    assertCudaSuccess(cudaMalloc(&dptr, sizeof(*dptr)));
    assertCudaSuccess(cudaMallocHost(&pptr, sizeof(*pptr)));
    ASSERT_EQ(HOST_NORMAL, getMemLocation(&x));
    ASSERT_EQ(HOST_PINNED, getMemLocation(pptr));
    ASSERT_EQ(DEVICE, getMemLocation(dptr));

    volsIn.push_back(&x);
    hostBlocksIn.push_back(pptr);
    deviceBlocksIn.push_back(pptr);
    EXPECT_EQ(CBP_INVALID_MEM_LOC, invokeBlockProcWith(nop, cbp::BlockIndexIterator(volSize, blockSize)));

    hostBlocksIn[0] = dptr;
    deviceBlocksIn[0] = dptr;
    EXPECT_EQ(CBP_INVALID_MEM_LOC, invokeBlockProcWith(nop, cbp::BlockIndexIterator(volSize, blockSize)));

    hostBlocksIn.clear();
    deviceBlocksIn.clear();
    syncAndAssertCudaSuccess();
    assertCudaSuccess(cudaFreeHost(pptr));
    assertCudaSuccess(cudaFree(dptr));
}

TEST_F(BlockProcTest, VoidVolumes)
{
    static const int nvol = 3*3*3;
    auto identityInts = [](auto b, auto s, auto in, auto out, auto tmp){
        cudaMemcpy(out[0], in[0], b.numel(), cudaMemcpyDeviceToDevice);
    };
    const int3 volSize = make_int3(3*sizeof(int), 3, 3), blockSize = make_int3(2*sizeof(int), 2, 2);
    int intInVol[nvol], intOutVol[nvol], expectedIntVol[nvol];
    void *voidInVol = static_cast<void *>(&intInVol[0]), *voidOutVol = static_cast<void *>(&intOutVol[0]);
    for (int i = 0; i < nvol; i++) {
        intInVol[i] = i;
        expectedIntVol[i] = i;
        intOutVol[i] = 0;
    }
    BlockIndexIterator blockIter(volSize, blockSize);
    std::vector<void *> inVols = { voidInVol };
    std::vector<void *> outVols = { voidOutVol };

    EXPECT_EQ(CBP_SUCCESS, blockProcMultiple(identityInts, inVols, outVols, blockIter));
    EXPECT_ARRAY_EQ(expectedIntVol, intInVol, nvol);
    EXPECT_ARRAY_EQ(expectedIntVol, intOutVol, nvol);
    syncAndAssertCudaSuccess();
}
