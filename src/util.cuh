#ifndef __UTIL_CUH__
#define __UTIL_CUH__

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
    return getIdx(pos.x, pos.y, pos.z, siz);
}
inline __host__ __device__ size_t getIdx(const int3 pos, const dim3 siz)
{
    return getIdx(pos.x, pos.y, pos.z, siz);
}
inline __host__ __device__ size_t getIdx(const dim3 pos, const dim3 siz)
{
    return getIdx((const int)pos.x, (const int)pos.y, (const int)pos.z, siz);
}

#endif // __UTIL_CUH__