#ifndef CUDABLOCKPROC_CUH__
#define CUDABLOCKPROC_CUH__

#include <vector>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <tuple>
#include <type_traits>
#include <exception>
#include "helper_math.cuh"
#include "blockindexiter.cuh"
#include "util.cuh"

using std::vector;

namespace cbp {

enum BlockTransferKind {
    VOL_TO_BLOCK,
    BLOCK_TO_VOL
};

enum CbpResult : int {
    CBP_SUCCESS = 0x0,
    CBP_INVALID_VALUE = 0x1,
    CBP_INVALID_MEM_LOC = 0x2,
    CBP_HOST_MEM_ALLOC_FAIL = 0x4,
    CBP_DEVICE_MEM_ALLOC_FAIL = 0x8
};

inline CbpResult operator| (CbpResult lhs, CbpResult rhs)
{
    return static_cast<CbpResult>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline CbpResult operator& (CbpResult lhs, CbpResult rhs)
{
    return static_cast<CbpResult>(static_cast<int>(lhs) | static_cast<int>(rhs));
}

inline CbpResult& operator|= (CbpResult& lhs, CbpResult rhs)
{
    lhs = lhs | rhs;
    return lhs;
}

inline CbpResult& operator&= (CbpResult& lhs, CbpResult rhs)
{
    lhs = lhs & rhs;
    return lhs;
}

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
    const int3 borderSize=make_int3(0)) noexcept
{
    const int3 totalSize = blockSize + 2*borderSize;
    const size_t nbytes = sizeof(Ty)*(totalSize.x * totalSize.y * totalSize.z);
    blocks.reserve(n);
    for (size_t i = 0; i < n; i++) {
        Ty *ptr;
        if (loc == HOST_NORMAL) {
            ptr = static_cast<Ty *>(malloc(nbytes));
            if (ptr == nullptr) {
                return CBP_HOST_MEM_ALLOC_FAIL;
            }
        } else if (loc == HOST_PINNED) {
            if (cudaMallocHost(&ptr, nbytes) != cudaSuccess) {
                return CBP_HOST_MEM_ALLOC_FAIL;
            }
        } else if (loc == DEVICE) {
            if (cudaMalloc(&ptr, nbytes) != cudaSuccess) {
                return CBP_DEVICE_MEM_ALLOC_FAIL;
            }
        } else {
            return CBP_INVALID_VALUE;
        }
        blocks.push_back(ptr);
    }
    return CBP_SUCCESS;
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

template <class InArr, class OutArr, class InHBlkArr, class OutHBlkArr, class InDBlkArr, class OutDBlkArr,
    class Func>
CbpResult blockProcNoValidate(Func func, const InArr& inVols, const OutArr& outVols,
    const InHBlkArr& inBlocks, const OutHBlkArr& outBlocks,
    const InDBlkArr& d_inBlocks, const OutDBlkArr& d_outBlocks,
    cbp::BlockIndexIterator blockIter, void *d_tmpMem=nullptr)
{
    static_assert(std::is_same<typename InArr::value_type, typename InHBlkArr::value_type>::value,
        "Value types for input volumes and host input blocks must be equal");
    static_assert(std::is_same<typename InArr::value_type, typename InDBlkArr::value_type>::value,
        "Value types for input volumes and device input blocks must be equal");
    static_assert(std::is_same<typename OutArr::value_type, typename OutHBlkArr::value_type>::value,
        "Value types for output volumes and host output blocks must be equal");
    static_assert(std::is_same<typename OutArr::value_type, typename OutDBlkArr::value_type>::value,
        "Value types for output volumes and device output blocks must be equal");

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
        func(prevBlockIdx, prevStream, d_inBlocks, d_outBlocks, d_tmpMem);

        cudaStreamWaitEvent(crntStream, crntEvent, 0);
        cbp::transferAllBlocks(inVols, inBlocks, crntBlockIdx, volSize, VOL_TO_BLOCK, crntStream);
        cbp::copyAllBlocks(d_inBlocks, inBlocks, crntBlockIdx, cudaMemcpyHostToDevice, crntStream);
        cbp::copyAllBlocks(outBlocks, d_outBlocks, prevBlockIdx, cudaMemcpyDeviceToHost, prevStream);
        cbp::transferAllBlocks(outVols, outBlocks, prevBlockIdx, volSize, BLOCK_TO_VOL, prevStream);

        prevBlockIdx = crntBlockIdx;
        prevStream = crntStream;
    }
    func(prevBlockIdx, prevStream, d_inBlocks, d_outBlocks, d_tmpMem);
    cbp::copyAllBlocks(outBlocks, d_outBlocks, prevBlockIdx, cudaMemcpyDeviceToHost, prevStream);
    cbp::transferAllBlocks(outVols, outBlocks, prevBlockIdx, volSize, BLOCK_TO_VOL, prevStream);
    cudaStreamSynchronize(prevStream);

    for (auto& s : streams) {
        cudaStreamDestroy(s);
    }
    return CBP_SUCCESS;
}

template <class InArr, class OutArr, class InHBlkArr, class OutHBlkArr, class InDBlkArr, class OutDBlkArr,
    class Func>
CbpResult blockProc(Func func, const InArr& inVols, const OutArr& outVols,
    const InHBlkArr& inBlocks, const OutHBlkArr& outBlocks,
    const InDBlkArr& d_inBlocks, const OutDBlkArr& d_outBlocks,
    cbp::BlockIndexIterator blockIter, void *d_tmpMem=nullptr)
{
    // Verify all blocks are pinned memory
    for (auto blockArray : { inBlocks, outBlocks }) {
        if (!std::all_of(blockArray.begin(), blockArray.end(), memLocationIs<HOST_PINNED>)) {
            return CBP_INVALID_MEM_LOC;
        }
    }
    // Verify all device blocks are on the device
    for (auto d_blockArray : { d_inBlocks, d_outBlocks }) {
        if (!std::all_of(d_blockArray.begin(), d_blockArray.end(), memLocationIs<DEVICE>)) {
            return CBP_INVALID_MEM_LOC;
        }
    }
    if (d_tmpMem != nullptr && !memLocationIs<DEVICE>(d_tmpMem)) {
        return CBP_INVALID_MEM_LOC;
    }

    return cbp::blockProcNoValidate(func, inVols, outVols,
        inBlocks, outBlocks, d_inBlocks, d_outBlocks, blockIter, d_tmpMem);
}

template <class InArr, class OutArr, class Func>
CbpResult blockProc(Func func, const InArr& inVols, const OutArr& outVols,
    cbp::BlockIndexIterator blockIter, const size_t tmpSize=0)
{
    const int3 blockSize = blockIter.blockSize();
    const int3 borderSize = blockIter.borderSize();
    vector<typename InArr::value_type> inBlocks, d_inBlocks;
    vector<typename OutArr::value_type> outBlocks, d_outBlocks;
    void *d_tmpMem = nullptr;

    // TODO: Use a scope guard
    auto cleanUp = [&](){
        std::for_each(inBlocks.begin(), inBlocks.end(), cudaFreeHost);
        std::for_each(d_inBlocks.begin(), d_inBlocks.end(), cudaFree);
        std::for_each(outBlocks.begin(), outBlocks.end(), cudaFreeHost);
        std::for_each(d_outBlocks.begin(), d_outBlocks.end(), cudaFree);
        cudaFree(d_tmpMem);
    };

    CbpResult res = cbp::allocBlocks(inBlocks, inVols.size(), HOST_PINNED, blockSize, borderSize);
    res |= cbp::allocBlocks(d_inBlocks, inVols.size(), DEVICE, blockSize, borderSize);
    res |= cbp::allocBlocks(outBlocks, outVols.size(), HOST_PINNED, blockSize, borderSize);
    res |= cbp::allocBlocks(d_outBlocks, outVols.size(), DEVICE, blockSize, borderSize);
    if (tmpSize > 0) {
        if (cudaMalloc(&d_tmpMem, tmpSize) != cudaSuccess) {
            res |= CBP_DEVICE_MEM_ALLOC_FAIL;
        }
    }
    if (!res) {
        cleanUp();
        return res;
    }

    try {
        res = cbp::blockProcNoValidate(func, inVols, outVols, inBlocks, outBlocks,
                                       d_inBlocks, d_outBlocks, blockIter, d_tmpMem);
        cleanUp();
    } catch (...) {
        // In case an exception was thrown, ensure memory is freed and then rethrow
        cleanUp();
        std::rethrow_exception(std::current_exception());
    }
    return res;
}

}

#endif // CUDABLOCKPROC_CUH__