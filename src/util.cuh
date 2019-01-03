#ifndef UTIL_CUH__
#define UTIL_CUH__

#include <type_traits>
#include "zip.cuh"

namespace cbp {

template <class Ty>
struct typeSize : public std::integral_constant<size_t, sizeof(Ty)> {};
template <>
struct typeSize<void> : public std::integral_constant<size_t, 1> {};

enum MemLocation {
    HOST_NORMAL = 0x01,
    HOST_PINNED = 0x02,
    DEVICE      = 0x10
};

inline MemLocation getMemLocation(const void *ptr)
{
    cudaPointerAttributes attr;
    const cudaError_t err = cudaPointerGetAttributes(&attr, ptr);
    if (err == cudaSuccess) {
        if (attr.memoryType == cudaMemoryTypeHost) {
            return HOST_PINNED;
        } else {
            return DEVICE;
        }
    } else {
        cudaGetLastError(); // Pop error so we don't disturb future calls
        return HOST_NORMAL;
    }
}

template <MemLocation loc>
inline bool memLocationIs(const void *ptr)
{
    return cbp::getMemLocation(ptr) == loc;
}

inline __host__ __device__ int gridLineBlocks(const int bs, const int siz)
{
    return siz/bs + ((siz % bs != 0) ? 1 : 0);
}

inline __host__ __device__ size_t getIdx(const int x, const int y, const int z, const int3 siz)
{
    return x + y * siz.x + z * siz.x * siz.y;
}
inline __host__ __device__ size_t getIdx(const int x, const int y, const int z, const dim3 siz)
{
    return x + y * siz.x + z * siz.x * siz.y;
}

inline __host__ __device__ size_t getIdx(const int3 pos, const int3 siz)
{
    return cbp::getIdx(pos.x, pos.y, pos.z, siz);
}
inline __host__ __device__ size_t getIdx(const int3 pos, const dim3 siz)
{
    return cbp::getIdx(pos.x, pos.y, pos.z, siz);
}
inline __host__ __device__ size_t getIdx(const dim3 pos, const dim3 siz)
{
    return cbp::getIdx((const int)pos.x, (const int)pos.y, (const int)pos.z, siz);
}

}

#endif // UTIL_CUH__