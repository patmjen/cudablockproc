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
    CBP_INVALID_MEM_LOC,
    CBP_MEM_ALLOC_FAIL
};

template <typename Ty>
void transferBlock(Ty *vol, Ty *block, const BlockIndex& bi, int3 volSize, BlockTransferKind kind)
{
    // TODO: Allow vol or block to be a const pointer - maybe use templates?
    // TODO: Allow caller to specify which axis corresponds to consecutive values.
    int3 start, bsize;
    const int3 blkSizeBdr = bi.blockSizeBorder();
    cudaMemcpy3DParms params = { 0 };
    const auto volPtr = make_cudaPitchedPtr(vol, volSize.x * sizeof(Ty), volSize.x, volSize.y);
    const auto blockPtr = make_cudaPitchedPtr(block, blkSizeBdr.x * sizeof(Ty), blkSizeBdr.x, blkSizeBdr.y);
    if (kind == VOL_TO_BLOCK) {
        start = bi.startIdxBorder;
        bsize = bi.blockSizeBorder();

        params.srcPtr = volPtr;
        params.dstPtr = blockPtr;
        params.srcPos = make_cudaPos(start.x * sizeof(Ty), start.y, start.z);
        params.dstPos = make_cudaPos(0, 0, 0);
    } else {
        // If we are transferring back to the volume we need to discard the border
        start = bi.startIdx;
        bsize = bi.blockSize();
        int3 startBorder = bi.startBorder();

        params.dstPtr = volPtr;
        params.srcPtr = blockPtr;
        params.dstPos = make_cudaPos(start.x * sizeof(Ty), start.y, start.z);
        params.srcPos = make_cudaPos(startBorder.x * sizeof(Ty), startBorder.y, startBorder.z);
    }

    params.kind = cudaMemcpyHostToHost;
    params.extent = make_cudaExtent(bsize.x * sizeof(Ty), bsize.y, bsize.z);

    cudaMemcpy3D(&params);
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
        if (ptr == nullptr) {
            // Don't try to free null pointers
            continue;
        }
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
    // Verify all sizes match
    if (inVols.size() != inBlocks.size() || outVols.size() != outBlocks.size() ||
        inVols.size() != d_inBlocks.size() || outVols.size() != d_outBlocks.size() ||
        tmpBlocks.size() != d_tmpBlocks.size()) {
        return CBP_INVALID_VALUE;
    }
    // Verify all blocks are pinned memory
    for (auto blockArray : { inBlocks, outBlocks, tmpBlocks }) {
        if (!std::all_of(blockArray.begin(), blockArray.end(), memLocationIs<HOST_PINNED>)) {
            return CBP_INVALID_MEM_LOC;
        }
    }
    // Verify all device blocks are on the device
    for (auto d_blockArray : { d_inBlocks, d_outBlocks, d_tmpBlocks }) {
        if (!std::all_of(d_blockArray.begin(), d_blockArray.end(), memLocationIs<DEVICE>)) {
            return CBP_INVALID_MEM_LOC;
        }
    }

    return blockProcNoValidate(func, inVols, outVols,
        inBlocks, outBlocks, tmpBlocks, d_inBlocks, d_outBlocks, d_tmpBlocks,
        volSize, blockSize, borderSize);
}

template <typename InTy, typename OutTy=InTy, typename TmpTy=InTy, typename Func>
CbpResult blockProc(Func func, const vector<InTy *>& inVols, const vector<OutTy *>& outVols,
    const size_t numTmp, const int3 volSize, const int3 blockSize, const int3 borderSize=make_int3(0))
{
    vector<InTy *> inBlocks, d_inBlocks;
    vector<OutTy *> outBlocks, d_outBlocks;
    vector<TmpTy *> tmpBlocks, d_tmpBlocks;
    CbpResult res;
    res = allocBlocks(inBlocks, inVols.size(), HOST_PINNED, blockSize, borderSize);
    if (res == CBP_SUCCESS) {
        res = allocBlocks(d_inBlocks, inVols.size(), DEVICE, blockSize, borderSize);
    }
    if (res == CBP_SUCCESS) {
        res = allocBlocks(outBlocks, outVols.size(), HOST_PINNED, blockSize, borderSize);
    }
    if (res == CBP_SUCCESS) {
        res = allocBlocks(d_outBlocks, outVols.size(), DEVICE, blockSize, borderSize);
    }
    if (res == CBP_SUCCESS) {
        res = allocBlocks(tmpBlocks, numTmp, HOST_PINNED, blockSize, borderSize);
    }
    if (res == CBP_SUCCESS) {
        res = allocBlocks(d_tmpBlocks, numTmp, DEVICE, blockSize, borderSize);
    }

    if (res != CBP_SUCCESS) {
        freeAll(inBlocks);
        freeAll(d_inBlocks);
        freeAll(outBlocks);
        freeAll(d_outBlocks);
        freeAll(tmpBlocks);
        freeAll(d_tmpBlocks);
        return res;
    }

    res = blockProcNoValidate(func, inVols, outVols,
        inBlocks, outBlocks, tmpBlocks, d_inBlocks, d_outBlocks, d_tmpBlocks,
        volSize, blockSize, borderSize);

    freeAll(inBlocks);
    freeAll(d_inBlocks);
    freeAll(outBlocks);
    freeAll(d_outBlocks);
    freeAll(tmpBlocks);
    freeAll(d_tmpBlocks);
    return res;
}

}

#endif // CUDABLOCKPROC_CUH__