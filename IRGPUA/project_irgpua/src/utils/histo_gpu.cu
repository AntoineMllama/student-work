#include "histo_gpu.cuh"


/*__global__ void histogram(raft::device_span<int> d_histo, raft::device_span<int> d_to_fix_buffer,  int image_size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < image_size)
        atomicAdd(&d_histo[d_to_fix_buffer[idx]], 1);
}*/

__global__
void histogram(raft::device_span<int> histo, raft::device_span<int> g_idata, int N) {
    constexpr int hist_size = 256;
    int tid = threadIdx.x + blockIdx.x * blockDim.x;

    __shared__ int s_histo[hist_size];

    for (int i = threadIdx.x; i < hist_size; i += blockDim.x) {
        s_histo[i] = 0;
    }
    __syncthreads();

    if (tid < N) {
        atomicAdd(&s_histo[g_idata[tid]], 1);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < hist_size; i += blockDim.x) {
        atomicAdd(&histo[i], s_histo[i]);
    }
}


__global__ void find_min_histo(raft::device_span<int> d_histo, raft::device_span<int> min)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < 256)
    {
        if (d_histo[idx] != 0 && d_histo[idx] < min[0])
        {
            atomicMin(min.data(), d_histo[idx]);
        }
    }
}


__global__ void transform(raft::device_span<int> d_histo, raft::device_span<int> d_to_fix_buffer, int image_size, raft::device_span<int> min)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int first_none_zero_val = min[0];
    if (idx < image_size)
    {
        int pixel = d_to_fix_buffer[idx];
            d_to_fix_buffer[idx] = roundf(((d_histo[pixel] - first_none_zero_val) / static_cast<float>(image_size - first_none_zero_val)) * 255.0f);
    }
}
