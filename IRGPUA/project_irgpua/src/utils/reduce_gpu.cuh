#pragma once

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

void your_reduce(raft::device_span<int> buffer, raft::device_span<int> total);
//void your_reduce(raft::device_span<int> buffer, raft::device_span<int> total, const raft::handle_t handle);