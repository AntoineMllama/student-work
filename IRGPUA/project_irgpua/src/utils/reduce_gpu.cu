#include "reduce_gpu.cuh"

template <typename T, int BLOCK_SIZE>
__device__
void warp_reduce(T* sdata, int tid)
{
    if (BLOCK_SIZE >= 64) { sdata[tid] += sdata[tid + 32]; __syncwarp(); }
    if (BLOCK_SIZE >= 32) { sdata[tid] += sdata[tid + 16]; __syncwarp(); }
    if (BLOCK_SIZE >= 16) { sdata[tid] += sdata[tid + 8]; __syncwarp(); }
    if (BLOCK_SIZE >= 8) { sdata[tid] += sdata[tid + 4]; __syncwarp(); }
    if (BLOCK_SIZE >= 4) { sdata[tid] += sdata[tid + 2]; __syncwarp(); }
    if (BLOCK_SIZE >= 2) { sdata[tid] += sdata[tid + 1]; __syncwarp(); }
}



template <typename T, int BLOCK_SIZE>
__global__
void kernel_your_reduce(raft::device_span<const T> buffer, raft::device_span<T> total)
{

    int n = buffer.size();

    extern __shared__ int sdata[];

    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    unsigned int gridSize = BLOCK_SIZE * 2 * gridDim.x;

    sdata[tid] = 0;
    while (i < n) {
        sdata[tid] += buffer[i];
        if (i + BLOCK_SIZE < n)
            sdata[tid] += buffer[i + BLOCK_SIZE];
        i += gridSize;
    }
    __syncthreads();

    if constexpr (BLOCK_SIZE >= 512) {
        if (tid < 256) { sdata[tid] += sdata[tid + 256]; } __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 256) {
        if (tid < 128) { sdata[tid] += sdata[tid + 128]; } __syncthreads();
    }
    if constexpr (BLOCK_SIZE >= 128) {
        if (tid < 64) { sdata[tid] += sdata[tid + 64]; } __syncthreads();
    }
    if (tid < 32)
      warp_reduce<T, BLOCK_SIZE>(sdata, tid);

    if (tid == 0)
        atomicAdd(total.data(), sdata[0]);
}

void your_reduce(raft::device_span<int> buffer,
                 raft::device_span<int> total)
{
    int thread = 512 ;
    int block = std::ceil(static_cast<float>(buffer.size()) / thread);

	kernel_your_reduce<int, 512><<<block, thread, sizeof(int) * thread>>>(
        raft::device_span<const int>(buffer.data(), buffer.size()),
        raft::device_span<int>(total.data(), 1));

    cudaDeviceSynchronize();
}
/*
void your_reduce(raft::device_span<int> buffer,
                 raft::device_span<int> total, const raft::handle_t handle)
{
    int thread = 1024 ;
    int block = std::ceil(static_cast<float>(buffer.size()) / 1024);

	kernel_your_reduce<int, 512><<<block, 512, sizeof(int) * 1024, handle.get_stream()>>>(
        raft::device_span<const int>(buffer.data(), buffer.size()),
        raft::device_span<int>(total.data(), 1));

    cudaStreamSynchronize(handle.get_stream());
}*/