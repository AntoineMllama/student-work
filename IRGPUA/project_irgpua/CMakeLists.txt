cmake_minimum_required(VERSION 3.26.4 FATAL_ERROR)
project(project_irgpua LANGUAGES CXX CUDA)

set(RAFT_NVTX ON)
include(fetch_rapids.cmake)
include(rapids-cmake)
include(rapids-cpm)
include(rapids-cuda)
include(rapids-find)

rapids_cuda_init_architectures(tp_reduce)

rapids_cpm_init()
include(cmake/thirdparty/get_cccl.cmake)
include(cmake/thirdparty/get_rmm.cmake)
include(cmake/thirdparty/get_raft.cmake)

find_package(OpenMP REQUIRED)

# Set the CUDA standard
set(CMAKE_CUDA_STANDARD 17)
set(CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
set(CUDA_ARCHITECTURES "75")

include_directories(src)

add_executable(main src/main.cu src/fix_cpu.cu src/fix_gpu.cu src/fix_gpu_inc.cu src/utils/scan_gpu.cu src/utils/histo_gpu.cu src/utils/reduce_gpu.cu)
target_compile_options(main PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=${OpenMP_CXX_FLAGS}>)
target_link_libraries(main PUBLIC OpenMP::OpenMP_CXX rmm::rmm raft::raft)
