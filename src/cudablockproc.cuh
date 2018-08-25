#ifndef CUDABLOCKPROC_CUH__
#define CUDABLOCKPROC_CUH__

#include <vector>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include "helper_math.cuh"
#include "blockindexiter.cuh"
#include "util.cuh"

using std::vector;

namespace cbp {

enum BlockTransferKind {
    VOL_TO_BLOCK,
    BLOCK_TO_VOL
};

enum CbpResult {
    CBP_SUCCESS,
    CBP_INVALID_VALUE,
    CBP_MEM_ALLOC_FAIL
};

template <typename Ty>
void transferBlock(Ty *vol, Ty *block, const BlockIndex& bi, int3 volSize, BlockTransferKind kind)
{
    // TODO: Allow vol or block to be a const pointer - maybe use templates?
    // TODO: Allow caller to specify which axis corresponds to consecutive values.
    int3 start, end, bsize;
    if (kind == VOL_TO_BLOCK) {
        start = bi.startIdxBorder;
        end = bi.endIdxBorder;
        bsize = bi.blockSizeBorder();
    } else {
        // If we are transferring back to the volume we need to discard the border
        start = bi.startIdx;
        end = bi.endIdx;
        bsize = bi.blockSize();
    }
    const int3 blockSizeBorder = bi.blockSizeBorder();
    for (int zv = start.z, zb = 0; zv < end.z; zv++, zb++) {
        for (int yv = start.y, yb = 0; yv < end.y; yv++, yb++) {
            Ty *dst, *src;
            if (kind == VOL_TO_BLOCK) {
                src = &(vol[getIdx(start.x, yv, zv, volSize)]);
                dst = &(block[getIdx(0, yb, zb, blockSizeBorder)]);
            } else {
                const int3 bdr = bi.startBorder();
                dst = &(vol[getIdx(start.x, yv, zv, volSize)]);
                src = &(block[getIdx(bdr.x, yb + bdr.y, zb + bdr.z, blockSizeBorder)]);
            }
            memcpy(static_cast<void *>(dst), static_cast<void *>(src), bsize.x*sizeof(Ty));
        }
    }
}

template <typename Ty>
CbpResult allocBlocks(vector<Ty *>& blocks, const size_t n, const MemLocation loc, const int3 blockSize,
    const int3 borderSize=make_int3(0))
{
    const int3 totalSize = blockSize + 2*borderSize;
    const size_t nbytes = sizeof(Ty)*(totalSize.x * totalSize.y * totalSize.z);
    blocks.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Ty *ptr;
        if (loc == HOST_NORMAL) {
            ptr = static_cast<Ty *>(malloc(nbytes));
            if (ptr == nullptr) {
                return CBP_MEM_ALLOC_FAIL;
            }
        } else if (loc == HOST_PINNED) {
            if (cudaMallocHost(&ptr, nbytes) != cudaSuccess) {
                return CBP_MEM_ALLOC_FAIL;
            }
        } else if (loc == DEVICE) {
            if (cudaMalloc(&ptr, nbytes) != cudaSuccess) {
                return CBP_MEM_ALLOC_FAIL;
            }
        } else {
            return CBP_INVALID_VALUE;
        }
        blocks.push_back(ptr);
    }
    return CBP_SUCCESS;
}

template <typename Arr>
void freeAll(const Arr& arr)
{
    for (auto ptr : arr) {
        switch (getMemLocation(ptr)) {
        case HOST_NORMAL:
            free(ptr);
            break;
        case HOST_PINNED:
            cudaFreeHost(ptr);
            break;
        case DEVICE:
            cudaFree(ptr);
            break;
        }
    }
}

template <typename InTy, typename OutTy=InTy, typename TmpTy=InTy, typename Func>
CbpResult blockProcNoValidate(Func func, const vector<InTy *>& inVols, const vector<OutTy *>& outVols,
    const vector<InTy *>& inBlocks, const vector<OutTy *>& outBlocks, const vector<TmpTy *> tmpBlocks,
    const vector<InTy *>& d_inBlocks, const vector<OutTy *>& d_outBlocks, const vector<TmpTy *> d_tmpBlocks,
    const int3 volSize, const int3 blockSize, const int3 borderSize=make_int3(0))
{
    for (BlockIndex b : BlockIndexIterator(volSize, blockSize, borderSize)) {
        for (auto iv = inVols.begin(), ib = inBlocks.begin(), d_ib = d_inBlocks.begin();
            iv != inVols.end(); ++iv, ++ib, ++d_ib) {
            transferBlock(*iv, *ib, b, volSize, VOL_TO_BLOCK);
            cudaMemcpy(*d_ib, *ib, b.numel()*sizeof(InTy), cudaMemcpyHostToDevice);
        }
        func(b, d_inBlocks, d_outBlocks, d_tmpBlocks);
        for (auto ov = outVols.begin(), ob = outBlocks.begin(), d_ob = d_outBlocks.begin();
            ov != outVols.end(); ++ov, ++ob, ++d_ob) {
            cudaMemcpy(*ob, *d_ob, b.numel()*sizeof(OutTy), cudaMemcpyDeviceToHost);
            transferBlock(*ov, *ob, b, volSize, BLOCK_TO_VOL);
        }
    }
    return CBP_SUCCESS;
}

template <typename InTy, typename OutTy=InTy, typename TmpTy=InTy, typename Func>
CbpResult blockProc(Func func, const vector<InTy *>& inVols, const vector<OutTy *>& outVols,
    const vector<InTy *>& inBlocks, const vector<OutTy *>& outBlocks, const vector<TmpTy *> tmpBlocks,
    const vector<InTy *>& d_inBlocks, const vector<OutTy *>& d_outBlocks, const vector<TmpTy *> d_tmpBlocks,
    const int3 volSize, const int3 blockSize, const int3 borderSize=make_int3(0))
{
    if (inVols.size() != inBlocks.size() || outVols.size() != outBlocks.size() ||
        inVols.size() != d_inBlocks.size() || outVols.size() != d_outBlocks.size()) {
        return CBP_INVALID_VALUE;
    }
    // Verify all blocks are pinned memory
    for (auto blockArray : { inBlocks, outBlocks, tmpBlocks }) {
        if (!std::all_of(blockArray.begin(), blockArray.end(), memLocationIs<HOST_PINNED>)) {
            return CBP_INVALID_VALUE;
        }
    }
    // Verify all device blocks are on the device
    for (auto d_blockArray : { d_inBlocks, d_outBlocks, d_tmpBlocks }) {
        if (!std::all_of(d_blockArray.begin(), d_blockArray.end(), memLocationIs<DEVICE>)) {
            return CBP_INVALID_VALUE;
        }
    }

    return blockProcNoValidate(func, inVols, outVols,
        inBlocks, outBlocks, tmpBlocks, d_inBlocks, d_outBlocks, d_tmpBlocks,
        volSize, blockSize, borderSize);
}

}

#endif // CUDABLOCKPROC_CUH__