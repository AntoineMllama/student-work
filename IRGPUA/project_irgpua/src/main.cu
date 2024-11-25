#include "image.hh"
#include "pipeline.hh"
#include "fix_cpu.cuh"
#include "fix_gpu.cuh"
#include "fix_gpu_inc.cuh"

#include <vector>
#include <iostream>
#include <algorithm>
#include <sstream>
#include <filesystem>
#include <numeric>

#include <rmm/mr/device/cuda_async_memory_resource.hpp>
#include <rmm/mr/device/owning_wrapper.hpp>
#include <rmm/mr/device/pool_memory_resource.hpp>

#include <raft/core/device_span.hpp>
#include <raft/core/handle.hpp>

#include <rmm/device_uvector.hpp>
#include <rmm/device_scalar.hpp>

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/reduce.h>

#include "cuda_error_checking.cuh"
#include "utils/reduce_gpu.cuh"

bool GPU = false;
bool INC = false;

//FROM TP
auto make_async() {
    return std::make_shared<rmm::mr::cuda_async_memory_resource>();
}

auto make_pool() {
    // Allocate 1.0 Go. To much ? Propably ...
    size_t initial_pool_size = std::pow(2, 30);
    return rmm::mr::make_owning_wrapper<rmm::mr::pool_memory_resource>(
        make_async(),
        initial_pool_size
    );
}


int main([[maybe_unused]] int argc, [[maybe_unused]] char* argv[])
{
    if(argc >= 2 && argv[1] == std::string("-gpu")) {
        GPU = true;
        std::cout << "Mode GPU" << std::endl;
    }
    if(argc >= 2 && argv[1] == std::string("-inc")) {
        GPU = true;
        INC = true;
        std::cout << "Mode GPU INDUSTRIE" << std::endl;
    }
    if (!(GPU || INC)) {
        std::cout << "Mode CPU" << std::endl;
    }


    auto memory_resource = make_pool();
    rmm::mr::set_current_device_resource(memory_resource.get());

    // -- Pipeline initialization

    std::cout << "File loading..." << std::endl;

    // - Get file paths

    using recursive_directory_iterator = std::filesystem::recursive_directory_iterator;
    std::vector<std::string> filepaths;
    for (const auto& dir_entry : recursive_directory_iterator("/afs/cri.epita.fr/resources/teach/IRGPUA/images"))
        filepaths.emplace_back(dir_entry.path());

    // - Init pipeline object

    Pipeline pipeline(filepaths);

    // -- Main loop containing image retring from pipeline and fixing

    const int nb_images = pipeline.images.size();
    std::vector<Image> images(nb_images);



    // - One CPU thread is launched for each image

    std::cout << "Done, starting compute" << std::endl;

    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        // TODO : make it GPU compatible (aka faster)
        // You will need to copy images one by one on the GPU
        // You can store the images the way you want on the GPU
        // But you should treat the pipeline as a pipeline :
        // You *must not* copy all the images and only then do the computations
        // You must get the image from the pipeline as they arrive and launch computations right away
        // There are still ways to speeds this process of course
        images[i] = pipeline.get_image(i);
        if(!GPU) {
            fix_image_cpu(images[i]);
        }
        else {
            
            Image img_actu = images[i];
            const int image_size = img_actu.width * img_actu.height;

            if (INC) {
                thrust::device_vector<int> d_to_fix_buffer(img_actu.size());
                thrust::device_vector<int> d_predicate(img_actu.size());
                thrust::device_vector<int> d_histo(256);
                thrust::device_vector<int> d_first_none_zero_val(1, 255);

                thrust::fill(d_predicate.begin(), d_predicate.end(), 0);
                thrust::fill(d_histo.begin(), d_histo.end(), 0);

                thrust::copy(
                    img_actu.buffer,
                    img_actu.buffer + img_actu.size(),
                    d_to_fix_buffer.begin()
                    );

                fix_image_gpu_inc(
                    image_size,
                    d_to_fix_buffer,
                    d_predicate,
                    d_histo,
                    d_first_none_zero_val
                    );

                thrust::copy(
                    d_to_fix_buffer.begin(),
                    d_to_fix_buffer.end(),
                    img_actu.buffer
                    );
            }
            else {
            
                //cudaStream_t stream;
                //cudaStreamCreate(&stream);
                const raft::handle_t handle{};
                //const raft::handle_t handle{stream};

                rmm::device_uvector<int> d_to_fix_buffer(img_actu.size(), handle.get_stream());
                rmm::device_uvector<int> d_predicate(img_actu.size(), handle.get_stream());
                rmm::device_uvector<int> d_histo(256, handle.get_stream());
                rmm::device_scalar<int> d_first_none_zero_val(255, handle.get_stream());

                CUDA_CHECK_ERROR(cudaMemset(d_predicate.data(), 0, sizeof(int) * d_to_fix_buffer.size()));
                CUDA_CHECK_ERROR(cudaMemset(d_histo.data(), 0, sizeof(int) * 256));
                CUDA_CHECK_ERROR(cudaMemcpy(d_to_fix_buffer.data(), img_actu.buffer, img_actu.size() * sizeof(int), cudaMemcpyHostToDevice));

                //CUDA_CHECK_ERROR(cudaMemsetAsync(d_predicate.data(), 0, sizeof(int) * d_to_fix_buffer.size(), handle.get_stream()));
                //CUDA_CHECK_ERROR(cudaMemsetAsync(d_histo.data(), 0, sizeof(int) * 256, handle.get_stream()));
                //CUDA_CHECK_ERROR(cudaMemcpyAsync(d_to_fix_buffer.data(), img_actu.buffer, img_actu.size() * sizeof(int), cudaMemcpyHostToDevice, handle.get_stream()));

                //CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));

                fix_image_gpu(
                    image_size,
                    raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
                    raft::device_span<int>(d_predicate.data(), d_predicate.size()),
                    raft::device_span<int>(d_histo.data(), d_histo.size()),
                    raft::device_span<int>(d_first_none_zero_val.data(), 1)
                    );

                /*fix_image_gpu(
                    image_size,
                    raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
                    raft::device_span<int>(d_predicate.data(), d_predicate.size()),
                    raft::device_span<int>(d_histo.data(), d_histo.size()),
                    raft::device_span<int>(d_first_none_zero_val.data(), 1),
                    handle
                    );*/

                CUDA_CHECK_ERROR(cudaMemcpy(img_actu.buffer, d_to_fix_buffer.data(), img_actu.size() * sizeof(int), cudaMemcpyDeviceToHost));
                //CUDA_CHECK_ERROR(cudaMemcpyAsync(img_actu.buffer, d_to_fix_buffer.data(), img_actu.size() * sizeof(int), cudaMemcpyDeviceToHost, handle.get_stream()));

                //CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));
                //CUDA_CHECK_ERROR(cudaStreamDestroy(handle.get_stream()));
            }
        }
    }

    std::cout << "Done with compute, starting stats" << std::endl;

    // -- All images are now fixed : compute stats (total then sort)

    // - First compute the total of each image

    // TODO : make it GPU compatible (aka faster)
    // You can use multiple CPU threads for your GPU version using openmp or not
    // Up to you :)
    #pragma omp parallel for
    for (int i = 0; i < nb_images; ++i)
    {
        auto& image = images[i];
        const int image_size = image.width * image.height;
        if (GPU)
        {
            if (INC) {
                thrust::device_vector<int> d_to_fix_buffer(image.size());
                thrust::copy(
                    image.buffer,
                    image.buffer + image.size(),
                    d_to_fix_buffer.begin()
                    );

                image.to_sort.total = thrust::reduce(d_to_fix_buffer.begin(), d_to_fix_buffer.end(), 0, thrust::plus<int>());
            }
            else {
                //cudaStream_t stream;
                //cudaStreamCreate(&stream);
                const raft::handle_t handle{};
                //const raft::handle_t handle{stream};

                rmm::device_uvector<int> d_to_fix_buffer(image_size, handle.get_stream());
                rmm::device_scalar<int> total(0, handle.get_stream());

                CUDA_CHECK_ERROR(cudaMemcpy(d_to_fix_buffer.data(), image.buffer, image_size * sizeof(int), cudaMemcpyHostToDevice));
            
                //CUDA_CHECK_ERROR(cudaMemcpyAsync(d_to_fix_buffer.data(), image.buffer, image_size * sizeof(int), cudaMemcpyHostToDevice, handle.get_stream()));
                //CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));

                your_reduce(
                    raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
                    raft::device_span<int>(total.data(), 1)
                );

                /*your_reduce(
                    raft::device_span<int>(d_to_fix_buffer.data(), d_to_fix_buffer.size()),
                    raft::device_span<int>(total.data(), 1),
                    handle
                    );*/

                CUDA_CHECK_ERROR(cudaMemcpy(&image.to_sort.total, total.data(), sizeof(int), cudaMemcpyDeviceToHost));
                //CUDA_CHECK_ERROR(cudaMemcpyAsync(&image.to_sort.total, total.data(), sizeof(int), cudaMemcpyDeviceToHost, handle.get_stream()));

                //CUDA_CHECK_ERROR(cudaStreamSynchronize(handle.get_stream()));
                //CUDA_CHECK_ERROR(cudaStreamDestroy(handle.get_stream()));
                }
        }
        else
            image.to_sort.total = std::reduce(image.buffer, image.buffer + image_size, 0);
    }

    // - All totals are known, sort images accordingly (OPTIONAL)
    // Moving the actual images is too expensive, sort image indices instead
    // Copying to an id array and sort it instead

    // TODO OPTIONAL : for you GPU version you can store it the way you want
    // But just like the CPU version, moving the actual images while sorting will be too slow
    using ToSort = Image::ToSort;
    std::vector<ToSort> to_sort(nb_images);
    std::generate(to_sort.begin(), to_sort.end(), [n = 0, images] () mutable
    {
        return images[n++].to_sort;
    });

    // TODO OPTIONAL : make it GPU compatible (aka faster)
    std::sort(to_sort.begin(), to_sort.end(), [](ToSort a, ToSort b) {
        return a.total < b.total;
    });

    // TODO : Test here that you have the same results
    // You can compare visually and should compare image vectors values and "total" values
    // If you did the sorting, check that the ids are in the same order
    for (int i = 0; i < nb_images; ++i)
    {
        std::cout << "Image #" << images[i].to_sort.id << " total : " << images[i].to_sort.total << std::endl;
        std::ostringstream oss;
        oss << "Image#" << images[i].to_sort.id << ".pgm";
        std::string str = oss.str();
        images[i].write(str);
    }

    std::cout << "Done, the internet is safe now :)" << std::endl;

    // Cleaning
    // TODO : Don't forget to update this if you change allocation style
    for (int i = 0; i < nb_images; ++i)
        free(images[i].buffer);

    return 0;
}
