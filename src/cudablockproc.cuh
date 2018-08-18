#ifndef __CUDABLOCKPROC_CUH__
#define __CUDABLOCKPROC_CUH__

#include <vector>
#include "helper_math.cuh"
#include "blockindexiter.cuh"
#include "util.cuh"

namespace cbp {

enum BlockTransferKind {
    VOL_TO_BLOCK,
    BLOCK_TO_VOL
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

}

#endif // __CUDABLOCKPROC_CUH__