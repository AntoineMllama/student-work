#pragma once

#include <thrust/device_vector.h>

void fix_image_gpu_inc(const int image_size, thrust::device_vector<int>& d_to_fix_buffer, thrust::device_vector<int>& d_predicate, thrust::device_vector<int>& d_histo, thrust::device_vector<int>& min);