#pragma once

#include "image.hh"
#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

void fix_image_gpu(const int image_size, raft::device_span<int> d_to_fix_buffer, raft::device_span<int> d_predicate, raft::device_span<int> d_histo, raft::device_span<int> min);
//void fix_image_gpu(const int image_size, raft::device_span<int> d_to_fix_buffer, raft::device_span<int> d_predicate, raft::device_span<int> d_histo, raft::device_span<int> min, const raft::handle_t handle);