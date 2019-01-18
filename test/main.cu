#include <string>
#include <cstdio>
#include <array>
#include "gtest/gtest.h"
#include "cudablockproc.cuh"
#include "util_test.cuh"

// TODO: Merge this with the version in util.cuh
inline size_t gridLineBlocks(size_t nthr, const unsigned int siz)
{
    return siz/nthr + ((siz % nthr != 0) ? 1 : 0);
}

inline dim3 gridBlocks(const dim3 thrConfig, const int3 siz)
{
    return dim3(
        gridLineBlocks(thrConfig.x, siz.x),
        gridLineBlocks(thrConfig.y, siz.y),
        gridLineBlocks(thrConfig.z, siz.z)
    );
}

__global__ void sum3x3x3Kernel(float *out, const float *in, const int3 size)
{
    const int3 pos = make_int3(threadIdx);
    if (pos >= 0 && pos < size) {
        out[getIdx(pos,size)] = 0;
    }
    if (pos >= 1 && pos < size - 1) {
        for (int ix = -1; ix <= 1; ix++) {
            for (int iy = -1; iy <= 1; iy++) {
                for (int iz = -1; iz <= 1; iz++) {
                    out[getIdx(pos,size)] += in[getIdx(pos + make_int3(ix,iy,iz),size)];
                }
            }
        }
    }
}

int main(int argc, char *argv[])
{
    std::string cmd;
    if (argc > 1) {
        cmd = argv[1];
    } else {
        cmd = "test";
    }

    if (cmd == "test") {
        testing::InitGoogleTest(&argc, argv);
        return RUN_ALL_TESTS();
    } else if (cmd == "bench") {
        if (argc < 5) {
            printf("Must supply dimensions of volume\n");
            return -1;
        }
        const int3 volSize = make_int3(std::stoi(argv[2]), std::stoi(argv[3]), std::stoi(argv[4]));
        const int3 blockSize = make_int3(256,256,256);
        const size_t numElemVol = volSize.x * volSize.y * volSize.z;

        std::vector<float> volIn(numElemVol);
        std::vector<float> volOut(numElemVol);

        auto sum3x3x3 = [](auto b, auto s, auto in, auto out, auto tmp){
            const int3 siz = b.blockSizeBorder();
            const dim3 blockDim = dim3(8,8,8);
            const dim3 gridDim = gridBlocks(blockDim, siz);
            sum3x3x3Kernel<<<gridDim,blockDim,0,s>>>(out[0], in[0], siz);
        };
        cbp::BlockIndexIterator blockIter(volSize, blockSize);
        cbp::CbpResult res = cbp::blockProc(sum3x3x3, volIn.data(), volOut.data(), blockIter);
        if (res != cbp::CBP_SUCCESS) {
            printf("Got error!\n");
        } else {
            printf("Success\n");
        }

        return 0;
    } else {
        printf("Unknown command\n");
        return -1;
    }
}