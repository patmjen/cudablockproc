#include <string>
#include <cstdio>
#include "gtest/gtest.h"
#include "cudablockproc.cuh"

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
        const int3 blockSize = make_int3(500,500,500);
        const size_t numElemVol = volSize.x * volSize.y * volSize.z;

        std::vector<float> volIn(numElemVol);
        std::vector<float> volOut(numElemVol);
        std::vector<float *> volsIn = { volIn.data() };
        std::vector<float *> volsOut = { volOut.data() };

        auto nop = [](auto b, auto i, auto o, auto t){};
        cbp::CbpResult res = cbp::blockProc(nop, volsIn, volsOut, 0, volSize, blockSize);
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