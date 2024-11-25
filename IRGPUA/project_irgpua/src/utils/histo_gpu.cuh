#pragma once

#include <raft/core/device_span.hpp>

__global__ void histogram(raft::device_span<int> d_histo, raft::device_span<int> d_to_fix_buffer,  int image_size);

__global__ void find_min_histo(raft::device_span<int> d_histo, raft::device_span<int> min);

__global__ void transform(raft::device_span<int> d_histo, raft::device_span<int> d_to_fix_buffer, int image_size, raft::device_span<int> min);