#ifndef CUDABLOCKPROC_CUH__
#define CUDABLOCKPROC_CUH__

#include <vector>
#include <cassert>
#include <cstdlib>
#include <algorithm>
#include <mutex>
#include <thread>
#include <utility>
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
void processSingleBlock_(Func func, const vector<InTy *>& inVols, const vector<OutTy *>& outVols,
    const vector<InTy *>& inBlocks, const vector<OutTy *>& outBlocks, const vector<TmpTy *> tmpBlocks,
    const vector<InTy *>& d_inBlocks, const vector<OutTy *>& d_outBlocks, const vector<TmpTy *> d_tmpBlocks,
    const int3 volSize, const BlockIndex& crntBlock, const cudaStream_t crntStream,
    std::unique_lock<std::mutex> ownRdyLock, std::unique_lock<std::mutex> prevRdyLock,
    cudaEvent_t *ownH2d, cudaEvent_t *ownComp, cudaEvent_t *ownD2h,
    cudaEvent_t *prevH2d, cudaEvent_t *prevComp, cudaEvent_t *prevD2h)
{
    if (prevRdyLock.mutex() != nullptr && prevRdyLock.mutex() != ownRdyLock.mutex()) {
        // The previous thread has a different mutex than us, so wait until the lock is released
        prevRdyLock.lock();
    }
    // Previous thread has now setup its events, so we can start to wait on them
    cudaEventSynchronize(*prevH2d);
    for (auto iv = inVols.begin(), ib = inBlocks.begin(); iv != inVols.end(); ++iv, ++ib) {
        transferBlock(*iv, *ib, crntBlock, volSize, VOL_TO_BLOCK);
    }
    cudaStreamWaitEvent(crntStream, *prevComp, 0);
    for (auto ib = inBlocks.begin(), d_ib = d_inBlocks.begin(); ib != inBlocks.end(); ++ib, ++d_ib) {
        cudaMemcpyAsync(*d_ib, *ib, crntBlock.numel()*sizeof(InTy), cudaMemcpyHostToDevice, crntStream);
    }
    cudaEventRecord(*ownH2d, crntStream);
    cudaStreamWaitEvent(crntStream, *prevD2h, 0);
    func(crntBlock, d_inBlocks, d_outBlocks, d_tmpBlocks);
    cudaEventRecord(*ownComp, crntStream);
    for (auto ob = outBlocks.begin(), d_ob = d_outBlocks.begin(); ob != outBlocks.end(); ++ob, ++d_ob) {
        cudaMemcpyAsync(*ob, *d_ob, crntBlock.numel()*sizeof(OutTy), cudaMemcpyDeviceToHost, crntStream);
    }
    cudaEventRecord(*ownD2h, crntStream);
    ownRdyLock.unlock();
    cudaStreamSynchronize(crntStream);
    for (auto ov = outVols.begin(), ob = outBlocks.begin(); ov != outVols.end(); ++ov, ++ob) {
        transferBlock(*ov, *ob, crntBlock, volSize, BLOCK_TO_VOL);
    }
}

template <typename InTy, typename OutTy=InTy, typename TmpTy=InTy, typename Func>
CbpResult blockProcNoValidate(Func func, const vector<InTy *>& inVols, const vector<OutTy *>& outVols,
    const vector<InTy *>& inBlocks, const vector<OutTy *>& outBlocks, const vector<TmpTy *> tmpBlocks,
    const vector<InTy *>& d_inBlocks, const vector<OutTy *>& d_outBlocks, const vector<TmpTy *> d_tmpBlocks,
    const int3 volSize, const int3 blockSize, const int3 borderSize=make_int3(0))
{
    // TODO: Is there a more elegant / faster way to do thread synchronization?
    BlockIndexIterator blockIter = BlockIndexIterator(volSize, blockSize, borderSize);
    const size_t totalBlockCount = blockIter.maxLinearIndex() + 1;
    vector<cudaStream_t> streams(totalBlockCount);
    vector<cudaEvent_t> h2dEvents(totalBlockCount);
    vector<cudaEvent_t> compEvents(totalBlockCount);
    vector<cudaEvent_t> d2hEvents(totalBlockCount);
    vector<std::mutex> mutexes(totalBlockCount);
    vector<std::thread> threads;
    threads.reserve(totalBlockCount);

    // Create all CUDA streams
    for (auto& s : streams) {
        cudaStreamCreate(&s);
    }
    // Create all CUDA events
    for (auto& e : h2dEvents) {
        cudaEventCreate(&e, cudaEventBlockingSync | cudaEventDisableTiming);
    }
    for (auto& e : compEvents) {
        cudaEventCreate(&e, cudaEventBlockingSync | cudaEventDisableTiming);
    }
    for (auto& e : d2hEvents) {
        cudaEventCreate(&e, cudaEventBlockingSync | cudaEventDisableTiming);
    }
    cudaEvent_t prevH2d = h2dEvents[0];
    cudaEvent_t prevComp = compEvents[0];
    cudaEvent_t prevD2h = d2hEvents[0];
    std::unique_lock<std::mutex> prevRdyLock(mutexes[0], std::defer_lock);
    auto streamIter = streams.begin();
    auto mutexIter = mutexes.begin();
    auto h2dIter = h2dEvents.begin();
    auto compIter = compEvents.begin();
    auto d2hIter = d2hEvents.begin();
    for (; blockIter != blockIter.end();
        ++streamIter, ++mutexIter, ++h2dIter, ++compIter, ++d2hIter, ++blockIter) {
        auto crntStream = *streamIter;
        auto crntBlock = *blockIter;
        auto crntH2d = *h2dIter;
        auto crntComp = *compIter;
        auto crntD2h = *d2hIter;

        std::unique_lock<std::mutex> rdyLock(*mutexIter); // Lock mutex for this block
        std::thread t(processSingleBlock_<InTy, OutTy, TmpTy, Func>, func, inVols, outVols,
            inBlocks, outBlocks, tmpBlocks, d_inBlocks, d_outBlocks, d_tmpBlocks,
            volSize, crntBlock, crntStream, std::move(rdyLock), std::move(prevRdyLock),
            &crntH2d, &crntComp, &crntD2h, &prevH2d, &prevComp, &prevD2h);
        threads.push_back(std::move(t));

        prevH2d = crntH2d;
        prevComp = crntComp;
        prevD2h = crntD2h;
        prevRdyLock = std::unique_lock<std::mutex>(*mutexIter, std::defer_lock); // Setup lock for next thread
    }
    // Wait for all threads
    for (auto& t : threads) {
        t.join();
    }
    // Destroy all CUDA events
    for (auto& e : h2dEvents) {
        cudaEventDestroy(e);
    }
    for (auto& e : compEvents) {
        cudaEventDestroy(e);
    }
    for (auto& e : d2hEvents) {
        cudaEventDestroy(e);
    }
    // Destroy all CUDA streams
    for (auto& s : streams) {
        cudaStreamDestroy(s);
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