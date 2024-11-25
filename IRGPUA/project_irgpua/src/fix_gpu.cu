#include "fix_gpu.cuh"
#include "utils/scan_gpu.cuh"
#include "utils/histo_gpu.cuh"

#include <raft/core/device_span.hpp>
//#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include "cuda_error_checking.cuh"

#define THREAD 1024

__global__ void fix_buffer(raft::device_span<int> d_to_fix_buffer, const int image_size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < image_size) {
        if (idx % 4 == 0)
            d_to_fix_buffer[idx] += 1;
        else if (idx % 4 == 1)
            d_to_fix_buffer[idx] -= 5;
        else if (idx % 4 == 2)
            d_to_fix_buffer[idx] += 3;
        else if (idx % 4 == 3)
            d_to_fix_buffer[idx] -= 8;

    }
}


void fix_image_gpu(const int image_size, raft::device_span<int> d_to_fix_buffer, raft::device_span<int> d_predicate, raft::device_span<int> d_histo, raft::device_span<int> min) {

    // #1 Compact

    int BLOCK = std::ceil(static_cast<float>(d_to_fix_buffer.size()) / THREAD);

    // Build predicate vector
    build_predicate_vector<<<BLOCK, THREAD, 0>>>(
        raft::device_span<int>(d_predicate.data(), d_predicate.size()),
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size())
    );
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // Exclusive scan
    const unsigned int nbBlocks = (d_predicate.size() + THREAD - 1) / THREAD;
    rmm::device_uvector<int> partSumsEx(nbBlocks, rmm::cuda_stream_default);

    scan_kernel_exclusive<<<nbBlocks, THREAD, 0, rmm::cuda_stream_default>>>(
        raft::device_span<int>(d_predicate.data(), d_predicate.size()),
        partSumsEx.data()
    );
    CUDA_CHECK_ERROR(cudaStreamSynchronize(rmm::cuda_stream_default));

    add_kernel<<<nbBlocks, THREAD, 0, rmm::cuda_stream_default>>>(
        raft::device_span<int>(d_predicate.data(), d_predicate.size()),
        partSumsEx.data()
    );
    CUDA_CHECK_ERROR(cudaStreamSynchronize(rmm::cuda_stream_default)); 

    // Scatter to the corresponding addresses
    scatter_the_coorresp_addresses<<<BLOCK, THREAD, 0>>>(
        raft::device_span<int>(d_predicate.data(), d_predicate.size()),
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size())
    );
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // #2 Apply map to fix pixels

    BLOCK = std::ceil(static_cast<float>(image_size) / THREAD);
    fix_buffer<<<BLOCK, THREAD, 0>>>(
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
        image_size);

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    // #3 Histogram equalization

    // Histogram
    histogram<<<BLOCK, THREAD, 0>>>(
        raft::device_span<int>(d_histo.data(), d_histo.size()),
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
        image_size);

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    const unsigned int nbBlocksSI = (d_histo.size() + THREAD - 1) / THREAD;
    scan_inclusive<<<nbBlocksSI, THREAD, 0>>>(
        raft::device_span<int>(d_histo.data(), d_histo.size())
    );

    find_min_histo<<<1, 256, 0>>>(
        raft::device_span<int>(d_histo.data(), d_histo.size()),
        raft::device_span<int>(min.data(), min.size())
    );

    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    transform<<<BLOCK, THREAD, 0>>>(
        raft::device_span<int>(d_histo.data(), d_histo.size()),
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
        image_size,
        raft::device_span<int>(min.data(), min.size()));
        
    CUDA_CHECK_ERROR(cudaDeviceSynchronize());

    return;
}

/*void fix_image_gpu(const int image_size, raft::device_span<int> d_to_fix_buffer, raft::device_span<int> d_predicate, raft::device_span<int> d_histo, raft::device_span<int> min, const raft::handle_t handle) {

    // #1 Compact

    int BLOCK = std::ceil(static_cast<float>(d_to_fix_buffer.size()) / THREAD);

    // Build predicate vector
    build_predicate_vector<<<BLOCK, THREAD, 0, handle.get_stream()>>>(
        raft::device_span<int>(d_predicate.data(), d_predicate.size()),
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
        d_to_fix_buffer.size());

    CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));

    // Scan
    scan_gpu<<<1, 1, 0, handle.get_stream()>>>(
        raft::device_span<int>(d_predicate.data(), d_predicate.size()),
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()));

    CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));

    // #2 Apply map to fix pixels

    BLOCK = std::ceil(static_cast<float>(image_size) / THREAD);
    fix_buffer<<<BLOCK, THREAD, 0, handle.get_stream()>>>(
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
        image_size);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));

    // #3 Histogram equalization

    // Histogram
    histogram<<<BLOCK, THREAD, 0, handle.get_stream()>>>(
        raft::device_span<int>(d_histo.data(), d_histo.size()),
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
        image_size);

    CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));

    scan_inclusive<<<1, 1, 0, handle.get_stream()>>>(
        raft::device_span<int>(d_histo.data(), d_histo.size()),
        d_histo.size());

    CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));

    find_min_histo<<<1, 256, 0, handle.get_stream()>>>(
        raft::device_span<int>(d_histo.data(), d_histo.size()),
        raft::device_span<int>(min.data(), min.size())
    );

    CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));

    transform<<<BLOCK, THREAD, 0, handle.get_stream()>>>(
        raft::device_span<int>(d_histo.data(), d_histo.size()),
        raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
        image_size,
        raft::device_span<int>(min.data(), min.size()));
        
    CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));

    return;
}*/
