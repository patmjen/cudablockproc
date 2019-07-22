# CUDABLOCKPROC

A small header-only library for CUDA block processing.

The purpose of cudablockproc is to make it simple to work with 3D datasets which are too large to fit in GPU memory. 
You simply provide a function, a collections of in- and output volumes and a block size; cudablockproc handles the rest.
Specifically, the volumes are split into blocks which are then copied to the GPU and passed to your function as pointers.
After processing, the output blocks are placed back in the output volumes.

The library itself is header only. To use it, copy the contents of the `lib` directory to your project.
To compile the tests you can (maybe) use the provided makefile - however you will likely have to modify some of the paths.
The tests are written using the googletest library.

## Examples
These examples show how the library can be used. For all examples, assume that these utility functions are defined:
```cuda
inline __host__ __device__ size_t getIdx(const int x, const int y, const int z, const int3 siz)
{
    return x + y * siz.x + z * siz.x * siz.y;
}

inline __host__ __device__ size_t getIdx(const int3 pos, const int3 siz)
{
    return getIdx(pos.x, pos.y, pos.z, siz);
}

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
```

### Add 1
This example simply adds `1.0f` to every voxel in the input volume and places the results in the output volume.
Let the following kernel be defined:
```cuda
__global__ void addOneKernel(float *out, const float *in, const int3 size)
{
    const int3 pos = make_int3(threadIdx);
    if (pos >= 0 && pos < size) {
        out[getIdx(pos,size)] = in[getIdx(pos,size)] + 1.0f;
    }
}
```
The block processing code is:
```cuda
// Specify sizes
const int3 volSize = make_int3(10,10,10);
const int3 blockSize = make_int3(2,2,2);
const size_t numElemVol = volSize.x * volSize.y * volSize.z;

// Alloc host volumes
std::vector<float> volIn(numElemVol);
std::vector<float> volOut(numElemVol);

// Prepare function
auto addOne = [](auto b, auto s, auto in, auto out, auto tmp){
    // Get size of block
    const int3 siz = b.blockSize();
    // Set up compute grid
    const dim3 blockDim = dim3(8,8,8);
    const dim3 gridDim = gridBlocks(blockDim, siz);
    // Launch kernel
    addOneKernel<<<gridDim,blockDim,0,s>>>(out[0], in[0], siz);
};

// Do block processing
cbp::BlockIndexIterator blockIter(volSize, blockSize);
cbp::CbpResult res = cbp::blockProc(addOne, volIn.data(), volOut.data(), blockIter);
```

### Sum 3x3x3
This example sums the voxels in a 3x3x3 window and places the results in the output volume. Let the following kernel be defined:
```cuda
__global__ void sum3x3x3Kernel(float *out, const float *in, const int3 size)
{
    const int3 pos = make_int3(threadIdx);
    if (pos >= 0 && pos < size) {
        out[getIdx(pos,size)] = 0;
    }
    // Loop over 3x3x3 window
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
```
The block processing code is:
```cuda
// Specify sizes
const int3 volSize = make_int3(10,10,10);
const int3 blockSize = make_int3(2,2,2);
const int3 borderSize = make_int3(1,1,1);
const size_t numElemVol = volSize.x * volSize.y * volSize.z;

// Alloc host volumes
std::vector<float> volIn(numElemVol);
std::vector<float> volOut(numElemVol);

// Prepare function
auto sum3x3x3 = [](auto b, auto s, auto in, auto out, auto tmp){
     // Get size of block including the border
    const int3 siz = b.blockSizeBorder();
    // Setup compute grid
    const dim3 blockDim = dim3(8,8,8);
    const dim3 gridDim = gridBlocks(blockDim, siz);
    // Launch kernel
    sum3x3x3Kernel<<<gridDim,blockDim,0,s>>>(out[0], in[0], siz);
};

// Do block processing
cbp::BlockIndexIterator blockIter(volSize, blockSize, borderSize); // NOTE: We supply a borderSize too
cbp::CbpResult res = cbp::blockProc(sum3x3x3, volIn.data(), volOut.data(), blockIter);
```

## Function signature
You can supply any type of callable as a function as long as it can be called with the following signature:
```cuda
RTy func(BlockIndex bi, cudaStream_t s, std::vector<InTy> inBlocks, std::vector<OutTy> outBlocks, void *tmp) 
```
where:

* `RTy` is the return type of the funcion. **Note**, any return value is ignored.
* `bi` is the current `BlockIndex` which gives the position and size of the current block. More on this in the next section.
* `s` is the cuda stream associated to this block. If your function is asynchronous (i.e. launches kernels) you **must** supply this 
stream, see the section on asynchronous functions.
* `inBlocks` an `std::vector` containing pointers to the input blocks. The order of the blocks is the same as the order of the input
volumes. Their size is given by `bi`.
* `outBlocks` an `std::vector` containing pointers to the output blocks. The order of the blocks is the same as the order of the output
volumes. Their size is given by `bi`.
* `tmp` is a pointer to temporary device memory which won't be copied back to the host.

The function must run on the host, so any kernels should be launced by you - cudablockproc **does not** launch kernels.

### Asynchronous functions
In order to reduce the overhead of block processing, cudablockproc will attempt to move data between the volumes to / from a set of 
page-locked (pinned) host buffers while your function runs.
In order to do this, the transfer to and from the GPU is done asynchronously using CUDA streams to ensure the correct order of operations. Specifically, the order is:

1. Copy block from input volume to host input buffer
2. Copy input host block to device input block (async)
3. Your function runs (maybe async)
4. Copy device block to host output buffer (async)
5. Copy host output buffer block to output volume

Thus, when launching kernels (or other async work) you must supply the provided CUDA stream
to ensure no memory transfers occur while your functions runs. If you do no async work, then call `cudaStreamSynchronize`
at the beginning of your function to ensure all data has been transfered to the device before processing.

## The `BlockIndex` and `BlockIndexIterator`
You specify the size of the blocks by making a `BlockIndexIterator`, where you supply:

* The size of the volumes. Currently, the library assumes all volumes have the same size.
* The size of the blocks. 
* (Optional) The size of the block border. This specifies the amount of overlap between the blocks which is useful if your functions
processes a small window of voxels - e.g. for smoothing. By default, a border of 0 voxels is assumed.

You can iterate over a `BlockIndexIterator` like a normal iterator - e.g. using `begin` and `end`.
When it is dereferenced it returns a `BlockIndex` which contains the start- and end xyz-indices for the current block - 
both with and without borders.
You can also get the size of the current block by calling `BlockIndex::blockSize` or `BlockIndex::blockSizeBorder` to include the
size of the border.
