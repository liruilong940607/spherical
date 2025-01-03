#include <c10/cuda/CUDAGuard.h>
#include <torch/extension.h>
#include <tuple>

#define N_THREADS 256

#define CHECK_CUDA(x)                                                   \
    TORCH_CHECK(x.is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x)                                             \
    TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x)                                                  \
    CHECK_CUDA(x);                                                      \
    CHECK_CONTIGUOUS(x)
#define DEVICE_GUARD(_ten)                                              \
    const at::cuda::OptionalCUDAGuard device_guard(device_of(_ten));

#define PRAGMA_UNROLL _Pragma("unroll")


std::tuple<torch::Tensor, torch::Tensor> wignerD_fwd(
    const torch::Tensor& eulers, // [N, 3]
    const int j
);