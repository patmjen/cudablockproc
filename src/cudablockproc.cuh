#ifndef CUDABLOCKPROC_CUH__
#define CUDABLOCKPROC_CUH__

#include <array>
#include <vector>
#include <cstdlib>
#include <algorithm>
#include <type_traits>
#include <exception>
#include "blockindexiter.cuh"

using std::vector;

namespace cbp {

enum MemLocation {
    HOST_NORMAL = 0x01,
    HOST_PINNED = 0x02,
    DEVICE      = 0x10
};

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

template <class InTy, class OutTy, class Func>
CbpResult blockProc(Func func, InTy *inVol, OutTy *outVol,
    cbp::BlockIndexIterator blockIter, const size_t tmpSize=0);

template <class InArr, class OutArr, class Func>
CbpResult blockProcMultiple(Func func, const InArr& inVols, const OutArr& outVols,
    cbp::BlockIndexIterator blockIter, const size_t tmpSize=0);

template <class InArr, class OutArr, class InHBlkArr, class OutHBlkArr, class InDBlkArr, class OutDBlkArr,
    class Func>
CbpResult blockProcMultiple(Func func, const InArr& inVols, const OutArr& outVols,
    const InHBlkArr& inBlocks, const OutHBlkArr& outBlocks,
    const InDBlkArr& d_inBlocks, const OutDBlkArr& d_outBlocks,
    cbp::BlockIndexIterator blockIter, void *d_tmpMem=nullptr);

template <class InArr, class OutArr, class InHBlkArr, class OutHBlkArr, class InDBlkArr, class OutDBlkArr,
    class Func>
CbpResult blockProcMultipleNoValidate(Func func, const InArr& inVols, const OutArr& outVols,
    const InHBlkArr& inBlocks, const OutHBlkArr& outBlocks,
    const InDBlkArr& d_inBlocks, const OutDBlkArr& d_outBlocks,
    cbp::BlockIndexIterator blockIter, void *d_tmpMem=nullptr);

inline MemLocation getMemLocation(const void *ptr);

template <MemLocation loc>
inline bool memLocationIs(const void *ptr);

template <typename VolArr, typename BlkArr>
void blockVolumeTransferAll(const VolArr& volArray, const BlkArr& blockArray, const BlockIndex& blkIdx,
    int3 volSize, BlockTransferKind kind, cudaStream_t stream);

template <typename DstArr, typename SrcArr>
void hostDeviceTransferAll(const DstArr& dstArray, const SrcArr& srcArray, const BlockIndex& blkIdx,
    cudaMemcpyKind kind, cudaStream_t stream);

template <typename Ty>
void blockVolumeTransfer(Ty *vol, Ty *block, const BlockIndex& bi, int3 volSize, BlockTransferKind kind,
    cudaStream_t stream);

template <typename Ty>
CbpResult allocBlocks(vector<Ty *>& blocks, const size_t n, const MemLocation loc, const int3 blockSize,
    const int3 borderSize=make_int3(0)) noexcept;

} // namespace cbp

#include "cudablockproc.inl"

#endif // CUDABLOCKPROC_CUH__