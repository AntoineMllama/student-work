#include "scan_gpu.cuh"

#define BLOCK_DIM 1024

__device__ constexpr int garbage_val = -27;

__global__ void build_predicate_vector(raft::device_span<int> d_predicate, raft::device_span<int> d_to_fix_buffer) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_to_fix_buffer.size()) {
        if (d_to_fix_buffer[idx] != garbage_val)
            d_predicate[idx] = 1;
    }
}

__global__ void scatter_the_coorresp_addresses(raft::device_span<int> d_predicate, raft::device_span<int> to_fix) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < d_predicate.size() && to_fix[idx] != garbage_val)
    {
        to_fix[d_predicate[idx]] = to_fix[idx];
    }
}

__global__ void scan_kernel_exclusive(raft::device_span<int> buffer, int* partSums) {

    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    __shared__ int buffer_s[BLOCK_DIM];

    if (i < buffer.size()) {
        buffer_s[threadIdx.x] = buffer[i];
    } else {
        buffer_s[threadIdx.x] = 0;
    }
    __syncthreads();


    for (unsigned int stride = 1; stride < blockDim.x; stride *= 2) {
        int temp = 0;
        if (threadIdx.x >= stride) {
            temp = buffer_s[threadIdx.x - stride];
        }
        __syncthreads();
        if (threadIdx.x >= stride) {
            buffer_s[threadIdx.x] += temp;
        }
        __syncthreads();
    }

    if (threadIdx.x == blockDim.x - 1) {
        partSums[blockIdx.x] = buffer_s[threadIdx.x];
    }

    int res = (threadIdx.x > 0) ? buffer_s[threadIdx.x - 1] : 0;
    __syncthreads();

    if (i < buffer.size()) {
        buffer[i] = res;
    }
}

__global__ void add_kernel(raft::device_span<int> buffer, const int* partSums)
{
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx < buffer.size() && blockIdx.x > 0) {
        int sum = 0;
        for (int j = 0; j < blockIdx.x; j++) {
            sum += partSums[j];
        }
        buffer[idx] += sum;
    }
}

__global__ void scan_inclusive(raft::device_span<int> buffer)
{
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int buffer_s1[BLOCK_DIM];
    __shared__ int buffer_s2[BLOCK_DIM];
    int* inBuffer = buffer_s1;
    int* outBuffer = buffer_s2;

    if (i < buffer.size()) {
        inBuffer[threadIdx.x] = buffer[i];
    } else {
        inBuffer[threadIdx.x] = 0;
    }
    __syncthreads();

    for (unsigned int stride = 1; stride <= BLOCK_DIM / 2; stride *= 2) {
        if (threadIdx.x >= stride) {
            outBuffer[threadIdx.x] = inBuffer[threadIdx.x] + inBuffer[threadIdx.x - stride];
        } else {
            outBuffer[threadIdx.x] = inBuffer[threadIdx.x];
        }
        __syncthreads();
        int* tmp = inBuffer;
        inBuffer = outBuffer;
        outBuffer = tmp;
    }

    if (i < buffer.size()) {
        buffer[i] = inBuffer[threadIdx.x];
    }
}