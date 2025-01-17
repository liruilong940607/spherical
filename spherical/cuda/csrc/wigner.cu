/*
Reference: https://en.wikipedia.org/wiki/Wigner_D-matrix
*/
#include "bindings.h"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

__constant__ float LOG_FACTORIAL_TABLE[30] = {
    0.0f, 0.0f, 0.69314718055994531f, 
    1.79175946922805500f, 3.17805383034794562f, 4.78749174278204599f, 
    6.57925121201010100f, 8.52516136106541430f, 10.60460290274525023f, 
    12.80182748008146961f, 15.10441257307551530f, 17.50230784587388584f, 
    19.98721449566188615f, 22.55216385312342289f, 25.19122118273868150f,
    27.89927138384089157f, 30.67186010608067280f, 33.50507345013688888f,
    36.39544520803305358f, 39.33988418719949404f, 42.33561646075348503f,
    45.38013889847690803f, 48.47118135183522388f, 51.60667556776437357f,
    54.78472939811231919f, 58.00360522298051994f, 61.26170176100200198f,
    64.55753862700633106f, 67.88974313718153498f, 71.25703896716800901f
};

// source: https://github.com/jakemannix/Mahout/blob/5d4a8391b9da4d21de6e48e9f49cd2be2d1b1ba3/math/src/main/java/org/apache/mahout/math/jet/math/Arithmetic.java
inline __device__ float log_factorial(int n) {
    if (n < 30) return LOG_FACTORIAL_TABLE[n];

    // Use Stirling's approximation with corrections for n >= 30
    float r = 1.0f / n;
    float rr = r * r;
    float c7 = -5.95238095238095238e-04f;
    float c5 = 7.93650793650793651e-04f;
    float c3 = -2.77777777777777778e-03f;
    float c1 = 8.33333333333333333e-02f;
    float c0 = 9.18938533204672742e-01f;

    return (n + 0.5f) * logf(n) - n + c0 + r * (c1 + rr * (c3 + rr * (c5 + rr * c7)));
}

__device__ float wigner_small_d(
    const int j,
    const int mp,
    const int m,
    const float beta
) {
    // Check ranges
    if (abs(m) > j || abs(mp) > j) return 0.0f;

    // Precompute factorial terms
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

    PRAGMA_UNROLL
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

    PRAGMA_UNROLL
    for (int idx_mp = 0; idx_mp < dim; idx_mp++) {
        int mp = idx_mp - j;
        PRAGMA_UNROLL
        for (int idx_m = 0; idx_m < dim; idx_m++) {
            int m = idx_m - j;

            // element = cmath.exp(-1j * mp * alpha) * d * cmath.exp(-1j * m * gamma)
            float d = wigner_small_d(j, mp, m, beta);
            float angle = mp * alpha + m * gamma;
            Dreal[idx_mp * dim + idx_m] = d * cosf(angle);
            Dimag[idx_mp * dim + idx_m] = -d * sinf(angle);
        }
    }
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
