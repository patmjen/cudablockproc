#include <iterator>
#include "helper_math.cuh"

namespace cbp {

int gridLineBlocks(const int bs, const int siz)
{
    return siz/bs + ((siz % bs != 0) ? 1 : 0);
}

struct BlockIndex {
    int3 startIdx;
    int3 endIdx;
    int3 startIdxBorder;
    int3 endIdxBorder;
};

class BlockIndexIterator : public std::iterator<std::forward_iterator_tag, BlockIndex, int> {
    int3 blockSize;
    int3 borderSize;
    int3 volSize;

    int3 numBlocks;
    int maxBlkIdx;

    int blkIdx;

public:
    explicit BlockIndexIterator() = default;
    explicit BlockIndexIterator(const int3 volSize, const int3 blockSize,
        const int3 borderSize=make_int3(0)) :
        blockSize(blockSize),
        borderSize(borderSize),
        volSize(volSize),
        numBlocks(make_int3(
            gridLineBlocks(blockSize.x, volSize.x),
            gridLineBlocks(blockSize.y, volSize.y),
            gridLineBlocks(blockSize.z, volSize.z))),
        maxBlkIdx(-1),
        blkIdx(0)
    {
        maxBlkIdx = numBlocks.x * numBlocks.y * numBlocks.z - 1;
    }

    bool operator==(const BlockIndexIterator& rhs) const
    {
        // Since numBlocks and maxBlkIdx are computed from the other values,
        // there is no reason to compare these
        return blockSize == rhs.blockSize && borderSize == rhs.borderSize && volSize == rhs.volSize &&
            blkIdx == rhs.blkIdx;
    }

    bool operator!=(const BlockIndexIterator& rhs) const
    {
        return !(*this == rhs);
    }

    BlockIndexIterator& operator++()
    {
        if (blkIdx <= maxBlkIdx) {
            blkIdx++;
        }
        return *this;
    }

    BlockIndexIterator operator++(int)
    {
        BlockIndexIterator out = *this;
        if (blkIdx <= maxBlkIdx) {
            blkIdx++;
        }
        return out;
    }

    BlockIndexIterator& operator--()
    {
        if (blkIdx > 0 ) {
            blkIdx--;
        }
        return *this;
    }

    BlockIndexIterator operator--(int)
    {
        BlockIndexIterator out = *this;
        if (blkIdx > 0) {
            blkIdx--;
        }
        return out;
    }

    BlockIndexIterator operator+=(int n)
    {
        blkIdx += n;
        if (blkIdx > maxBlkIdx + 1) {
            blkIdx = maxBlkIdx + 1;
        }
        return *this;
    }

    BlockIndexIterator operator-=(int n)
    {
        blkIdx -= n;
        if (blkIdx < 0) {
            blkIdx = 0;
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

    int getMaxLinearIndex() const
    {
        return maxBlkIdx;
    }

    int3 getNumBlocks() const
    {
        return numBlocks;
    }

    int3 getBlockSize() const
    {
        return blockSize;
    }

    int3 getVolSize() const
    {
        return volSize;
    }

    int3 getBorderSize() const
    {
        return borderSize;
    }

    BlockIndexIterator begin() const
    {
        BlockIndexIterator out = *this;
        out.blkIdx = 0;
        return out;
    }

    BlockIndexIterator end() const
    {
        BlockIndexIterator out = *this;
        out.blkIdx = maxBlkIdx + 1;
        return out;
    }

    int linearBlockIndex() const
    {
        return blkIdx;
    }

    BlockIndex blockIndex() const
    {
        return blockIndex(blkIdx);
    }

    BlockIndex blockIndex(const int bi) const
    {
        // TODO: Allow iterating over the axes in different order
        const int xi = bi % numBlocks.x;
        const int yi = (bi / numBlocks.x) % numBlocks.y;
        const int zi = bi / (numBlocks.x * numBlocks.y);

        BlockIndex out;
        out.startIdx = make_int3(xi, yi, zi) * blockSize;
        out.endIdx = min(out.startIdx + blockSize, volSize);
        out.startIdxBorder = max(out.startIdx - borderSize, make_int3(0));
        out.endIdxBorder = min(out.endIdx + borderSize, volSize);

        return out;
    }
};

BlockIndexIterator operator+(const BlockIndexIterator& it, const int n)
{
    BlockIndexIterator out = it;
    out += n;
    return out;
}

BlockIndexIterator operator-(const BlockIndexIterator& it, const int n)
{
    BlockIndexIterator out = it;
    out -= n;
    return out;
}

}