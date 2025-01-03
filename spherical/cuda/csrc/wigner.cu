/*
Reference: https://en.wikipedia.org/wiki/Wigner_D-matrix
*/
#include "bindings.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

inline __device__ int factorial(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}

inline __device__ float log_factorial(int n) {
    if (n == 0 || n == 1) return 0.0f;
    float log_fact = 0.0f;
    for (int i = 2; i <= n; i++) {
        log_fact += logf(i);
    }
    return log_fact;
}

__device__ float wigner_small_d(
    const int j,
    const int mp,
    const int m,
    const float beta
) {
    // Check ranges
    if (abs(m) > j || abs(mp) > j) {
        return 0.0f;
    }

    // Precompute factorial terms
    // float prefactor = sqrtf(
    //     factorial(j + mp) * factorial(j - mp) * factorial(j + m) * factorial(j - m)
    // );
    float prefactor = expf(
        0.5f * (
            log_factorial(j + mp) + log_factorial(j - mp) + log_factorial(j + m) + log_factorial(j - m)
        )
    );
    // printf("prefactor: %f\n", prefactor);

    // The summation index k must be chosen so that factorial arguments are non-negative:
    // Conditions:
    //  (j - m' - k)! >= 0  => k <= j - m'
    //  (j + m - k)! >= 0   => k <= j + m
    //  (k - m + m')! >= 0  => k >= m - m'
    //  (k)! >= 0           => k >= 0
    //
    // Combine:
    // k >= max(0, m - m')
    // k <= min(j - m', j + m)
    int s_min = max(0, m - mp);
    int s_max = min(j - mp, j + m);

    float half_beta = beta / 2.0f;
    float cos = cosf(half_beta);
    float sin = sinf(half_beta);
    float val = 0.0f;

    for (int s = s_min; s <= s_max; s++) {
        float numerator = (
            pow(-1, s + mp - m) *
            (cos > 1e-8f ? powf(cos, 2 * j + m - mp - 2 * s) : 0.f) *
            (sin > 1e-8f ? powf(sin, mp - m + 2 * s): 0.f)
        );
        float denom = expf(
            log_factorial(j - mp - s) + log_factorial(j + m - s) + log_factorial(s - m + mp) + log_factorial(s)
        );
        val += numerator / denom;
        // printf("s: %d, numerator: %f, denom: %f, val: %f\n", s, numerator, denom, val);
    }

    return val * prefactor;
}



__device__ float wignerD(
    const int j,
    const float alpha,
    const float beta,
    const float gamma,
    float* Dreal, // [2*j+1, 2*j+1]
    float* Dimag // [2*j+1, 2*j+1]
) {
    int dim = 2 * j + 1;
    for (int idx_mp = 0; idx_mp < dim; idx_mp++) {
        int mp = idx_mp - j;
        for (int idx_m = 0; idx_m < dim; idx_m++) {
            int m = idx_m - j;

            float d = wigner_small_d(j, mp, m, beta);
            // element = cmath.exp(-1j * mp * alpha) * d * cmath.exp(-1j * m * gamma)
            float angle = mp * alpha + m * gamma;
            Dreal[idx_mp * dim + idx_m] = d * cosf(angle);
            Dimag[idx_mp * dim + idx_m] = -d * sinf(angle);
        }
    }
}


__global__ void wignerD_fwd_kernel(
    const uint32_t N,
    const float* __restrict__ eulers, // [N, 3]
    const int j,
    float* __restrict__ Dreal, // [N, 2*j+1, 2*j+1]
    float* __restrict__ Dimag  // [N, 2*j+1, 2*j+1]
) {
    uint32_t idx = cg::this_grid().thread_rank();
    if (idx >= N) {
        return;
    }

    float alpha = eulers[idx * 3 + 0];
    float beta = eulers[idx * 3 + 1];
    float gamma = eulers[idx * 3 + 2];

    int dim = 2 * j + 1;
    Dreal += idx * dim * dim;
    Dimag += idx * dim * dim;

    wignerD(j, alpha, beta, gamma, Dreal, Dimag);
}

std::tuple<torch::Tensor, torch::Tensor> wignerD_fwd(
    const torch::Tensor& eulers, // [N, 3]
    const int j
) {
    DEVICE_GUARD(eulers);
    CHECK_INPUT(eulers);
    TORCH_CHECK(eulers.size(-1) == 3, "eulers must have last dimension 3");
    TORCH_CHECK(eulers.dim() == 2, "eulers must have 2 dimensions");
    uint32_t N = eulers.size(0);

    uint32_t dim = 2 * j + 1;
    torch::Tensor Dreal = torch::empty({N, dim, dim}, eulers.options());
    torch::Tensor Dimag = torch::empty({N, dim, dim}, eulers.options());
    
    if (N) {
        at::cuda::CUDAStream stream = at::cuda::getCurrentCUDAStream();
        wignerD_fwd_kernel
            <<<(N + N_THREADS - 1) / N_THREADS,
                N_THREADS,
                0,
                stream>>>(
                N,
                eulers.data_ptr<float>(),
                j,
                Dreal.data_ptr<float>(),
                Dimag.data_ptr<float>()
            );
    }
    return std::make_tuple(Dreal, Dimag);
}
