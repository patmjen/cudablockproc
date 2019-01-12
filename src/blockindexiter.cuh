#ifndef BLOCKINDEXITER_CUH__
#define BLOCKINDEXITER_CUH__

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

    int3 blockSizeBorder() const;

    int3 blockSize() const;

    int3 startBorder() const;

    int3 endBorder() const;

    int numel() const;

    bool operator==(const BlockIndex& rhs) const;

    bool operator!=(const BlockIndex& rhs) const;
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
            cbp::gridLineBlocks(blockSize.x, volSize.x),
            cbp::gridLineBlocks(blockSize.y, volSize.y),
            cbp::gridLineBlocks(blockSize.z, volSize.z))),
        maxBlkIdx_(-1),
        linBlkIdx_(0),
        blkIdx_()
    {
        maxBlkIdx_ = numBlocks_.x * numBlocks_.y * numBlocks_.z - 1;
        updateBlockIndex_();
    }

    bool operator==(const BlockIndexIterator& rhs) const;

    bool operator!=(const BlockIndexIterator& rhs) const;

    bool operator<=(const BlockIndexIterator& rhs) const;

    bool operator>=(const BlockIndexIterator& rhs) const;

    bool operator<(const BlockIndexIterator& rhs) const;

    bool operator>(const BlockIndexIterator& rhs) const;

    BlockIndexIterator& operator++();
    BlockIndexIterator operator++(int);

    BlockIndexIterator& operator--();
    BlockIndexIterator operator--(int);

    BlockIndexIterator operator+=(int n);

    BlockIndexIterator operator-=(int n);

    int operator-(const BlockIndexIterator& rhs);

    const BlockIndex& operator*() const;

    const BlockIndex *operator->();

    const BlockIndex operator[](const int i);

    int maxLinearIndex() const;

    int3 numBlocks() const;

    int3 blockSize() const;

    int3 volSize() const;

    int3 borderSize() const;

    int linearIndex() const;

    BlockIndexIterator begin() const;

    BlockIndexIterator end() const;

    BlockIndex blockIndexAt(const int i) const;

    const BlockIndex& blockIndex() const;

private:
    BlockIndex calcBlockIndex_(const int i) const;

    void updateBlockIndex_();

    void updateBlockIndex_(const int bi);
};

BlockIndexIterator operator+(const BlockIndexIterator& it, const int n);

BlockIndexIterator operator+(const int n, const BlockIndexIterator& it);

BlockIndexIterator operator-(const BlockIndexIterator& it, const int n);

BlockIndexIterator operator-(const int n, const BlockIndexIterator& it);

void swap(BlockIndexIterator& a, BlockIndexIterator& b);

} // namespace cbp

#endif // BLOCKINDEXITER_CUH__