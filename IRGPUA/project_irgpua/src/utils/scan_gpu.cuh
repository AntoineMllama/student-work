#pragma once

#include <raft/core/device_span.hpp>

__global__ void build_predicate_vector(raft::device_span<int> d_predicate, raft::device_span<int> d_to_fix_buffer);
__global__ void scatter_the_coorresp_addresses(raft::device_span<int> d_predicate, raft::device_span<int> to_fix);

__global__ void scan_kernel_exclusive(raft::device_span<int> buffer, int* partSums);
__global__ void add_kernel(raft::device_span<int> buffer, const int* partSums);

__global__ void scan_inclusive(raft::device_span<int> buffer);