#ifndef __BLOCKINDEXITER_CUH__
#define __BLOCKINDEXITER_CUH__

#include <iterator>
#include "helper_math.cuh"
#include "util.cuh"

namespace cbp {

struct BlockIndex {
    int3 startIdx;
    int3 endIdx;
    int3 startIdxBorder;
    int3 endIdxBorder;

    explicit BlockIndex() = default;
    BlockIndex(int endIdx) :
        startIdx(make_int3(0)),
        endIdx(make_int3(endIdx)),
        startIdxBorder(make_int3(0)),
        endIdxBorder(make_int3(endIdx)) {}
    BlockIndex(int3 endIdx) :
        startIdx(make_int3(0)),
        endIdx(endIdx),
        startIdxBorder(make_int3(0)),
        endIdxBorder(endIdx) {}
    explicit BlockIndex(int3 startIdx, int3 endIdx) :
        startIdx(startIdx),
        endIdx(endIdx),
        startIdxBorder(startIdx),
        endIdxBorder(endIdx) {}
    explicit BlockIndex(int startIdx, int endIdx) :
        startIdx(make_int3(startIdx)),
        endIdx(make_int3(endIdx)),
        startIdxBorder(make_int3(startIdx)),
        endIdxBorder(make_int3(endIdx)) {}
    explicit BlockIndex(int3 startIdx, int3 endIdx, int3 startIdxBorder, int3 endIdxBorder) :
        startIdx(startIdx),
        endIdx(endIdx),
        startIdxBorder(startIdxBorder),
        endIdxBorder(endIdxBorder) {}
    explicit BlockIndex(int startIdx, int endIdx, int startIdxBorder, int endIdxBorder) :
        startIdx(make_int3(startIdx)),
        endIdx(make_int3(endIdx)),
        startIdxBorder(make_int3(startIdxBorder)),
        endIdxBorder(make_int3(endIdxBorder)) {}

    int3 blockSizeBorder() const
    {
        return endIdxBorder - startIdxBorder;
    }

    int3 blockSize() const
    {
        return endIdx - startIdx;
    }

    int3 startBorder() const
    {
        return startIdx - startIdxBorder;
    }

    int3 endBorder() const
    {
        return endIdxBorder - endIdx;
    }

    int numel() const
    {
        const int3 bs = blockSizeBorder();
        return bs.x*bs.y*bs.z;
    }
};

class BlockIndexIterator : public std::iterator<std::forward_iterator_tag, BlockIndex, int> {
    int3 blockSize_;
    int3 borderSize_;
    int3 volSize_;

    int3 numBlocks_;
    int maxBlkIdx_;

    int blkIdx_;

public:
    explicit BlockIndexIterator() = default;
    explicit BlockIndexIterator(const int3 volSize, const int3 blockSize,
        const int3 borderSize=make_int3(0)) :
        blockSize_(blockSize),
        borderSize_(borderSize),
        volSize_(volSize),
        numBlocks_(make_int3(
            gridLineBlocks(blockSize.x, volSize.x),
            gridLineBlocks(blockSize.y, volSize.y),
            gridLineBlocks(blockSize.z, volSize.z))),
        maxBlkIdx_(-1),
        blkIdx_(0)
    {
        maxBlkIdx_ = numBlocks_.x * numBlocks_.y * numBlocks_.z - 1;
    }

    bool operator==(const BlockIndexIterator& rhs) const
    {
        // Since numBlocks and maxBlkIdx are computed from the other values,
        // there is no reason to compare these
        return blockSize_ == rhs.blockSize_ && borderSize_ == rhs.borderSize_ && volSize_ == rhs.volSize_ &&
            blkIdx_ == rhs.blkIdx_;
    }

    bool operator!=(const BlockIndexIterator& rhs) const
    {
        return !(*this == rhs);
    }

    BlockIndexIterator& operator++()
    {
        if (blkIdx_ <= maxBlkIdx_) {
            blkIdx_++;
        }
        return *this;
    }

    BlockIndexIterator operator++(int)
    {
        BlockIndexIterator out = *this;
        if (blkIdx_ <= maxBlkIdx_) {
            blkIdx_++;
        }
        return out;
    }

    BlockIndexIterator& operator--()
    {
        if (blkIdx_ > 0 ) {
            blkIdx_--;
        }
        return *this;
    }

    BlockIndexIterator operator--(int)
    {
        BlockIndexIterator out = *this;
        if (blkIdx_ > 0) {
            blkIdx_--;
        }
        return out;
    }

    BlockIndexIterator operator+=(int n)
    {
        blkIdx_ += n;
        if (blkIdx_ > maxBlkIdx_ + 1) {
            blkIdx_ = maxBlkIdx_ + 1;
        }
        return *this;
    }

    BlockIndexIterator operator-=(int n)
    {
        blkIdx_ -= n;
        if (blkIdx_ < 0) {
            blkIdx_ = 0;
        }
        return *this;
    }

    BlockIndex operator*() const
    {
        return blockIndex();
    }

    BlockIndex operator[](const int n) const
    {
        return blockIndex(n);
    }

    int maxLinearIndex() const { return maxBlkIdx_; }

    int3 numBlocks() const { return numBlocks_; }

    int3 blockSize() const { return blockSize_; }

    int3 volSize() const { return volSize_; }

    int3 borderSize() const { return borderSize_; }

    BlockIndexIterator begin() const
    {
        BlockIndexIterator out = *this;
        out.blkIdx_ = 0;
        return out;
    }

    BlockIndexIterator end() const
    {
        BlockIndexIterator out = *this;
        out.blkIdx_ = maxBlkIdx_ + 1;
        return out;
    }

    int linearBlockIndex() const
    {
        return blkIdx_;
    }

    BlockIndex blockIndex() const
    {
        return blockIndex(blkIdx_);
    }

    BlockIndex blockIndex(const int bi) const
    {
        // TODO: Allow iterating over the axes in different order
        const int xi = bi % numBlocks_.x;
        const int yi = (bi / numBlocks_.x) % numBlocks_.y;
        const int zi = bi / (numBlocks_.x * numBlocks_.y);

        BlockIndex out;
        out.startIdx = make_int3(xi, yi, zi) * blockSize_;
        out.endIdx = min(out.startIdx + blockSize_, volSize_);
        out.startIdxBorder = max(out.startIdx - borderSize_, make_int3(0));
        out.endIdxBorder = min(out.endIdx + borderSize_, volSize_);

        return out;
    }
};

inline BlockIndexIterator operator+(const BlockIndexIterator& it, const int n)
{
    BlockIndexIterator out = it;
    out += n;
    return out;
}

inline BlockIndexIterator operator-(const BlockIndexIterator& it, const int n)
{
    BlockIndexIterator out = it;
    out -= n;
    return out;
}

}

#endif // __BLOCKINDEXITER_CUH__