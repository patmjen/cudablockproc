#ifndef CUDABLOCKPROC_CUH__
#define CUDABLOCKPROC_CUH__

#include <vector>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <tuple>
#include <type_traits>
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
void transferBlock(Ty *vol, Ty *block, const BlockIndex& bi, int3 volSize, BlockTransferKind kind,
    cudaStream_t stream)
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

    cudaMemcpy3DAsync(&params, stream);
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
        switch (cbp::getMemLocation(ptr)) {
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

template <typename VolArr, typename BlkArr>
void transferAllBlocks(const VolArr& volArray, const BlkArr& blockArray, const BlockIndex& blkIdx,
    int3 volSize, BlockTransferKind kind, cudaStream_t stream)
{
    typename VolArr::value_type volPtr;
    typename BlkArr::value_type blkPtr;
    static_assert(std::is_same<decltype(volPtr), decltype(blkPtr)>::value,
        "Volume and block must have same type");
    for (auto ptrs : zip(volArray, blockArray)) {
        std::tie(volPtr, blkPtr) = ptrs;
        cbp::transferBlock(volPtr, blkPtr, blkIdx, volSize, kind, stream);
    }
}

template <typename DstArr, typename SrcArr>
void copyAllBlocks(const DstArr& dstArray, const SrcArr& srcArray, const BlockIndex& blkIdx,
    cudaMemcpyKind kind, cudaStream_t stream)
{
    typename DstArr::value_type dstPtr;
    typename SrcArr::value_type srcPtr;
    static_assert(std::is_same<decltype(dstPtr), decltype(srcPtr)>::value,
        "Destination and source must have same type");
    for (auto ptrs : zip(dstArray, srcArray)) {
        std::tie(dstPtr, srcPtr) = ptrs;
        cudaMemcpyAsync(dstPtr, srcPtr, blkIdx.numel()*sizeof(*dstPtr), kind, stream);
    }
}

template <typename InTy, typename OutTy=InTy, typename TmpTy=InTy, typename Func>
CbpResult blockProcNoValidate(Func func, const vector<InTy *>& inVols, const vector<OutTy *>& outVols,
    const vector<InTy *>& inBlocks, const vector<OutTy *>& outBlocks, const vector<TmpTy *> tmpBlocks,
    const vector<InTy *>& d_inBlocks, const vector<OutTy *>& d_outBlocks, const vector<TmpTy *> d_tmpBlocks,
    cbp::BlockIndexIterator blockIter)
{
    const int3 volSize = blockIter.volSize();
    size_t blockCount = blockIter.maxLinearIndex() + 1;
    vector<cudaStream_t> streams(blockCount);
    vector<cudaEvent_t> events(blockCount);
    // Create streams and events

    for (auto& s : streams) {
        cudaStreamCreate(&s);
    }
    for (auto& e : events) {
        cudaEventCreate(&e);
    }
    auto crntBlockIdx = blockIter[0];
    auto crntStream = streams[0];

    // Start transfer and copy for first block
    cbp::transferAllBlocks(inVols, inBlocks, crntBlockIdx, volSize, VOL_TO_BLOCK, crntStream);
    cbp::copyAllBlocks(d_inBlocks, inBlocks, crntBlockIdx, cudaMemcpyHostToDevice, crntStream);

    // Process remaining blocks
    auto zipped = zip(events, streams, blockIter);
    auto prevBlockIdx = crntBlockIdx;
    auto prevStream = crntStream;
    cudaEvent_t crntEvent;
    // We skip the first block as we started it above
    for (auto crnt = ++(zipped.begin()); crnt != zipped.end(); ++crnt) {
        std::tie(crntEvent, crntStream, crntBlockIdx) = *crnt;

        cudaEventRecord(crntEvent);
        func(prevBlockIdx, prevStream, d_inBlocks, d_outBlocks, d_tmpBlocks);

        cudaStreamWaitEvent(crntStream, crntEvent, 0);
        cbp::transferAllBlocks(inVols, inBlocks, crntBlockIdx, volSize, VOL_TO_BLOCK, crntStream);
        cbp::copyAllBlocks(d_inBlocks, inBlocks, crntBlockIdx, cudaMemcpyHostToDevice, crntStream);
        cbp::copyAllBlocks(outBlocks, d_outBlocks, prevBlockIdx, cudaMemcpyDeviceToHost, prevStream);
        cbp::transferAllBlocks(outVols, outBlocks, prevBlockIdx, volSize, BLOCK_TO_VOL, prevStream);

        prevBlockIdx = crntBlockIdx;
        prevStream = crntStream;
    }
    func(prevBlockIdx, prevStream, d_inBlocks, d_outBlocks, d_tmpBlocks);
    cbp::copyAllBlocks(outBlocks, d_outBlocks, prevBlockIdx, cudaMemcpyDeviceToHost, prevStream);
    cbp::transferAllBlocks(outVols, outBlocks, prevBlockIdx, volSize, BLOCK_TO_VOL, prevStream);
    cudaStreamSynchronize(prevStream);

    for (auto& s : streams) {
        cudaStreamDestroy(s);
    }
    return CBP_SUCCESS;
}

template <typename InTy, typename OutTy=InTy, typename TmpTy=InTy, typename Func>
CbpResult blockProc(Func func, const vector<InTy *>& inVols, const vector<OutTy *>& outVols,
    const vector<InTy *>& inBlocks, const vector<OutTy *>& outBlocks, const vector<TmpTy *> tmpBlocks,
    const vector<InTy *>& d_inBlocks, const vector<OutTy *>& d_outBlocks, const vector<TmpTy *> d_tmpBlocks,
    cbp::BlockIndexIterator blockIter)
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

    return cbp::blockProcNoValidate(func, inVols, outVols,
        inBlocks, outBlocks, tmpBlocks, d_inBlocks, d_outBlocks, d_tmpBlocks, blockIter);
}

template <typename InTy, typename OutTy=InTy, typename TmpTy=InTy, typename Func>
CbpResult blockProc(Func func, const vector<InTy *>& inVols, const vector<OutTy *>& outVols,
    const size_t numTmp, cbp::BlockIndexIterator blockIter)
{
    const int3 blockSize = blockIter.blockSize();
    const int3 borderSize = blockIter.borderSize();
    vector<InTy *> inBlocks, d_inBlocks;
    vector<OutTy *> outBlocks, d_outBlocks;
    vector<TmpTy *> tmpBlocks, d_tmpBlocks;
    CbpResult res;
    res = cbp::allocBlocks(inBlocks, inVols.size(), HOST_PINNED, blockSize, borderSize);
    if (res == CBP_SUCCESS) {
        res = cbp::allocBlocks(d_inBlocks, inVols.size(), DEVICE, blockSize, borderSize);
    }
    if (res == CBP_SUCCESS) {
        res = cbp::allocBlocks(outBlocks, outVols.size(), HOST_PINNED, blockSize, borderSize);
    }
    if (res == CBP_SUCCESS) {
        res = cbp::allocBlocks(d_outBlocks, outVols.size(), DEVICE, blockSize, borderSize);
    }
    if (res == CBP_SUCCESS) {
        res = cbp::allocBlocks(tmpBlocks, numTmp, HOST_PINNED, blockSize, borderSize);
    }
    if (res == CBP_SUCCESS) {
        res = cbp::allocBlocks(d_tmpBlocks, numTmp, DEVICE, blockSize, borderSize);
    }

    if (res != CBP_SUCCESS) {
        cbp::freeAll(inBlocks);
        cbp::freeAll(d_inBlocks);
        cbp::freeAll(outBlocks);
        cbp::freeAll(d_outBlocks);
        cbp::freeAll(tmpBlocks);
        cbp::freeAll(d_tmpBlocks);
        return res;
    }

    res = cbp::blockProcNoValidate(func, inVols, outVols,
        inBlocks, outBlocks, tmpBlocks, d_inBlocks, d_outBlocks, d_tmpBlocks, blockIter);

    cbp::freeAll(inBlocks);
    cbp::freeAll(d_inBlocks);
    cbp::freeAll(outBlocks);
    cbp::freeAll(d_outBlocks);
    cbp::freeAll(tmpBlocks);
    cbp::freeAll(d_tmpBlocks);
    return res;
}

}

#endif // CUDABLOCKPROC_CUH__