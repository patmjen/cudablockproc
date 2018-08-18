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

    bool operator==(const BlockIndex& rhs) const
    {
        return startIdx == rhs.startIdx && endIdx == rhs.endIdx &&
            startIdxBorder == rhs.startIdxBorder && endIdxBorder == rhs.endIdxBorder;
    }

    bool operator!=(const BlockIndex& rhs) const
    {
        return !(*this == rhs);
    }
};

class BlockIndexIterator : public std::iterator<
    std::input_iterator_tag, BlockIndex, int, const BlockIndex*, const BlockIndex&> {
    // NOTE: In many ways, this iterator behaves exactly like a random access iterator, as it can jump to any
    // position in constant time. However, it *cannot* return references when dereferenced which remain valid
    // once the iterator has been moved, which disqualifies it as a ForwardIterator.
    int3 blockSize_;
    int3 borderSize_;
    int3 volSize_;

    int3 numBlocks_;
    int maxBlkIdx_;

    int linBlkIdx_;
    BlockIndex blkIdx_;

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
        linBlkIdx_(0),
        blkIdx_()
    {
        maxBlkIdx_ = numBlocks_.x * numBlocks_.y * numBlocks_.z - 1;
        updateBlockIndex_();
    }

    bool operator==(const BlockIndexIterator& rhs) const
    {
        // Since numBlocks and maxBlkIdx are computed from the other values,
        // there is no reason to compare these
        return blockSize_ == rhs.blockSize_ && borderSize_ == rhs.borderSize_ && volSize_ == rhs.volSize_ &&
            linBlkIdx_ == rhs.linBlkIdx_;
    }

    bool operator!=(const BlockIndexIterator& rhs) const
    {
        return !(*this == rhs);
    }

    bool operator<=(const BlockIndexIterator& rhs) const
    {
        return linBlkIdx_ <= rhs.linBlkIdx_;
    }

    bool operator>=(const BlockIndexIterator& rhs) const
    {
        return linBlkIdx_ >= rhs.linBlkIdx_;
    }

    bool operator<(const BlockIndexIterator& rhs) const
    {
        return linBlkIdx_ < rhs.linBlkIdx_;
    }

    bool operator>(const BlockIndexIterator& rhs) const
    {
        return linBlkIdx_ > rhs.linBlkIdx_;
    }

    BlockIndexIterator& operator++()
    {
        if (linBlkIdx_ <= maxBlkIdx_) {
            linBlkIdx_++;
        }
        updateBlockIndex_();
        return *this;
    }

    BlockIndexIterator operator++(int)
    {
        BlockIndexIterator out = *this;
        if (linBlkIdx_ <= maxBlkIdx_) {
            linBlkIdx_++;
        }
        updateBlockIndex_();
        return out;
    }

    BlockIndexIterator& operator--()
    {
        if (linBlkIdx_ > 0 ) {
            linBlkIdx_--;
        }
        updateBlockIndex_();
        return *this;
    }

    BlockIndexIterator operator--(int)
    {
        BlockIndexIterator out = *this;
        if (linBlkIdx_ > 0) {
            linBlkIdx_--;
        }
        updateBlockIndex_();
        return out;
    }

    BlockIndexIterator operator+=(int n)
    {
        linBlkIdx_ += n;
        if (linBlkIdx_ > maxBlkIdx_ + 1) {
            linBlkIdx_ = maxBlkIdx_ + 1;
        }
        updateBlockIndex_();
        return *this;
    }

    BlockIndexIterator operator-=(int n)
    {
        linBlkIdx_ -= n;
        if (linBlkIdx_ < 0) {
            linBlkIdx_ = 0;
        }
        updateBlockIndex_();
        return *this;
    }

    int operator-(const BlockIndexIterator& rhs)
    {
        return linBlkIdx_ - rhs.linBlkIdx_;
    }

    const BlockIndex& operator*() const
    {
        return blockIndex();
    }

    const BlockIndex *operator->()
    {
        return &(this->blkIdx_);
    }

    const BlockIndex operator[](const int i)
    {
        return blockIndexAt(i);
    }

    int maxLinearIndex() const { return maxBlkIdx_; }

    int3 numBlocks() const { return numBlocks_; }

    int3 blockSize() const { return blockSize_; }

    int3 volSize() const { return volSize_; }

    int3 borderSize() const { return borderSize_; }

    int linearIndex() const { return linBlkIdx_; }

    BlockIndexIterator begin() const
    {
        BlockIndexIterator out = *this;
        out.updateBlockIndex_(0);
        return out;
    }

    BlockIndexIterator end() const
    {
        BlockIndexIterator out = *this;
        out.updateBlockIndex_(maxBlkIdx_ + 1);
        return out;
    }

    BlockIndex blockIndexAt(const int i) const
    {
        return calcBlockIndex_(i);
    }

    const BlockIndex& blockIndex() const
    {
        return blkIdx_;
    }

private:
    BlockIndex calcBlockIndex_(const int i) const
    {
        // TODO: Allow iterating over the axes in different order
        const int xi = i % numBlocks_.x;
        const int yi = (i / numBlocks_.x) % numBlocks_.y;
        const int zi = i / (numBlocks_.x * numBlocks_.y);

        BlockIndex out;
        out.startIdx = make_int3(xi, yi, zi) * blockSize_;
        out.endIdx = min(out.startIdx + blockSize_, volSize_);
        out.startIdxBorder = max(out.startIdx - borderSize_, make_int3(0));
        out.endIdxBorder = min(out.endIdx + borderSize_, volSize_);

        return out;
    }

    void updateBlockIndex_()
    {
        blkIdx_ = calcBlockIndex_(linBlkIdx_);
    }

    void updateBlockIndex_(const int bi)
    {
        linBlkIdx_ = bi;
        updateBlockIndex_();
    }
};

inline BlockIndexIterator operator+(const BlockIndexIterator& it, const int n)
{
    BlockIndexIterator out = it;
    out += n;
    return out;
}

inline BlockIndexIterator operator+(const int n, const BlockIndexIterator& it)
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

inline BlockIndexIterator operator-(const int n, const BlockIndexIterator& it)
{
    BlockIndexIterator out = it.begin();
    out += (n - it.linearIndex());
    return out;
}

inline void swap(BlockIndexIterator& a, BlockIndexIterator& b)
{
    BlockIndexIterator aNew(b);
    b = a;
    a = aNew;
}

}

#endif // __BLOCKINDEXITER_CUH__