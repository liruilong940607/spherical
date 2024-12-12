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

__device__ float wigner_small_d(
    const int j,
    const int m_prime,
    const int m,
    const float beta
) {
    // Check ranges
    if (abs(m) > j || abs(m_prime) > j) {
        return 0.0f;
    }

    // Precompute factorial terms
    float prefactor = sqrtf(
        factorial(j + m_prime) * factorial(j - m_prime) * factorial(j + m) * factorial(j - m)
    );

    float d_val = 0.0f;
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
    int k_min = max(0, m - m_prime);
    int k_max = min(j - m_prime, j + m);

    float half_beta = beta / 2.0f;
    float c = cosf(half_beta);
    float s = sinf(half_beta);
    // printf("k_min: %d, k_max: %d\n", k_min, k_max);
    // printf("c: %f, s: %f\n", c, s);

    for (int k = k_min; k <= k_max; k++) {
        float numerator = (
            pow(-1, k - m_prime + m) *
            prefactor *
            powf(c, 2 * j + m - m_prime - 2 * k) *
            powf(s, m_prime - m + 2 * k)
        );
        float denom = factorial(j - m_prime - k) * factorial(j + m - k) * factorial(k - m + m_prime) * factorial(k);
        // printf("numerator: %f, denom: %f\n", numerator, denom);
        // printf("prefactor: %f\n", prefactor);
        // printf("%f, %f, %f\n", pow(-1, k - m_prime + m), powf(c, 2 * j + m - m_prime - 2 * k), powf(s, m_prime - m + 2 * k));
        d_val += numerator / denom;
    }

    return d_val;
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
    for (int idx_m_prime = 0; idx_m_prime < dim; idx_m_prime++) {
        int m_prime = idx_m_prime - j;
        for (int idx_m = 0; idx_m < dim; idx_m++) {
            int m = idx_m - j;

            float d = wigner_small_d(j, m_prime, m, beta);
            // element = cmath.exp(+1j * m_prime * alpha) * d * cmath.exp(+1j * m * gamma)
            float real = d * cosf(m_prime * alpha + m * gamma);
            float imag = d * sinf(m_prime * alpha + m * gamma);
            // printf("d: %f, real: %f, imag: %f\n", d, real, imag);
            Dreal[idx_m_prime * dim + idx_m] = real;
            Dimag[idx_m_prime * dim + idx_m] = imag;
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
    // printf("idx: %d, alpha: %f, beta: %f, gamma: %f, dim: %d\n", idx, alpha, beta, gamma, dim);

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
