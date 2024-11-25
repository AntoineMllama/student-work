#include "fix_gpu_inc.cuh"

#include <thrust/transform.h>
#include <thrust/scan.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/count.h>
#include <thrust/fill.h>
#include <thrust/for_each.h>
#include <thrust/copy.h>

__device__ constexpr int garbage_val = -27;

void fix_image_gpu_inc(const int image_size, thrust::device_vector<int>& d_to_fix_buffer, thrust::device_vector<int>& d_predicate, thrust::device_vector<int>& d_histo, thrust::device_vector<int>& min) {

    thrust::transform(d_to_fix_buffer.begin(),
                      d_to_fix_buffer.end(),
                      d_predicate.begin(),
                      [=] __device__ (int val) {
                          return val != garbage_val ? 1 : 0;
                      });

    thrust::device_vector<int> d_exclu_scan(d_predicate.size());
    thrust::exclusive_scan(d_predicate.begin(),
                           d_predicate.end(),
                           d_exclu_scan.begin());

    thrust::device_vector<int> d_scatter(d_predicate.size());
    thrust::scatter_if(d_to_fix_buffer.begin(),
                       d_to_fix_buffer.end(),
                       d_exclu_scan.begin(),
                       d_predicate.begin(),
                       d_scatter.begin()
                       );

    thrust::copy(d_scatter.begin(), d_scatter.end(), d_to_fix_buffer.begin());

    thrust::for_each(thrust::make_counting_iterator(0),
                     thrust::make_counting_iterator(static_cast<int>(d_to_fix_buffer.size())),
                     [d_to_fix_buffer = thrust::raw_pointer_cast(d_to_fix_buffer.data())]
                     __device__ (int x) {
                        switch (x % 4) {
                            case 0: d_to_fix_buffer[x] += 1; break;
                            case 1: d_to_fix_buffer[x] -= 5; break;
                            case 2: d_to_fix_buffer[x] += 3; break;
                            case 3: d_to_fix_buffer[x] -= 8; break;
                        }
                     });

    thrust::for_each(thrust::make_counting_iterator(0),
                    thrust::make_counting_iterator(image_size),
                    [d_to_fix_buffer = thrust::raw_pointer_cast(d_to_fix_buffer.data()), 
                    d_histo = thrust::raw_pointer_cast(d_histo.data())]
                    __device__ (int i) {
                        int pixel_value = d_to_fix_buffer[i];
                        atomicAdd(&d_histo[pixel_value], 1);
                 });

    thrust::inclusive_scan(d_histo.begin(),
                           d_histo.end(),
                           d_histo.begin());

    auto it = thrust::find_if(d_histo.begin(),
                              d_histo.end(),
                              [] __device__ (int val) { return val != 0; });

    const int first_non_zero = *it;

    thrust::for_each(thrust::make_counting_iterator(0),
                      thrust::make_counting_iterator(image_size),
                      [d_to_fix_buffer = thrust::raw_pointer_cast(d_to_fix_buffer.data()),
                        d_histo = thrust::raw_pointer_cast(d_histo.data()),
                        first_non_zero, image_size]
                        __device__ (int i) {
                          int pixel = d_to_fix_buffer[i];
                          d_to_fix_buffer[i] = static_cast<int>(roundf(((d_histo[pixel] - first_non_zero) / static_cast<float>(image_size - first_non_zero)) * 255.0f));
                      });


    return;
}