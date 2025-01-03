import cmath
import math
import numpy as np


def wigner_small_d(j, m_prime, m, beta):
    """
    Compute the Wigner small-d matrix element d^{(j)}_{m',m}(beta).

    Parameters
    ----------
    j : int or float
        The total angular momentum quantum number. Typically an integer or half-integer.
    m_prime : int or float
        The m' quantum number (between -j and j).
    m : int or float
        The m quantum number (between -j and j).
    beta : float
        The Euler angle beta (in radians).

    Returns
    -------
    complex
        The d^{(j)}_{m',m}(beta) element.
    """
    # Convert j, m_prime, m to integers if they are half-integers (e.g. j=1/2)
    # Typically j, m, m' are integers or half-integers, but the formula requires factorials.
    # For half-integers, we can rescale by 2*j and use double factorials or gamma functions.
    # Here, we assume j, m, m' are integers for simplicity.
    # For half-integers, you would use gamma functions instead of factorials.

    # Ensure that j, m_prime, m are integers
    j_int = int(j)
    # The code below assumes integer j;
    # If your problem involves half-integers, you must adapt the factorial calls using gamma functions.

    # Check ranges
    if abs(m) > j_int or abs(m_prime) > j_int:
        assert False
        return 0.0

    # Precompute factorial terms
    # Note: For large j, consider using logarithms or scipy.special.comb for numerical stability.
    f = math.factorial
    prefactor = math.sqrt(
        f(j_int + m_prime) * f(j_int - m_prime) * f(j_int + m) * f(j_int - m)
    )

    d_val = 0.0
    # The summation index k must be chosen so that factorial arguments are non-negative:
    # Conditions:
    #  (j - m' - k)! >= 0  => k <= j - m'
    #  (j + m - k)! >= 0   => k <= j + m
    #  (k - m + m')! >= 0  => k >= m - m'
    #  (k)! >= 0           => k >= 0
    #
    # Combine:
    # k >= max(0, m - m')
    # k <= min(j - m', j + m)
    k_min = max(0, m - m_prime)
    k_max = min(j_int - m_prime, j_int + m)

    half_beta = beta / 2.0
    c = math.cos(half_beta)
    s = math.sin(half_beta)

    for k in range(k_min, k_max + 1):
        numerator = (
            ((-1) ** (k + m_prime - m))
            * (c ** (2 * j_int + m - m_prime - 2 * k))
            * (s ** (m_prime - m + 2 * k))
        )
        denom = f(j_int - m_prime - k) * f(j_int + m - k) * f(k - m + m_prime) * f(k)
        d_val += numerator / denom

    return d_val * prefactor


def wigner_D_matrix(j, alpha, beta, gamma):
    """
    Compute the Wigner D-matrix D^{(j)}(alpha, beta, gamma).

    Parameters
    ----------
    j : int
        The total angular momentum quantum number (integer for this example).
    alpha : float
        The Euler angle alpha in radians.
    beta : float
        The Euler angle beta in radians.
    gamma : float
        The Euler angle gamma in radians.

    Returns
    -------
    list of list of complex
        The (2j+1) x (2j+1) Wigner D-matrix.
    """
    dim = 2 * j + 1
    D = [[0.0j for _ in range(dim)] for __ in range(dim)]

    for idx_m_prime in range(dim):
        m_prime = idx_m_prime - j
        for idx_m in range(dim):
            m = idx_m - j

            d = wigner_small_d(j, m_prime, m, beta)
            element = cmath.exp(-1j * m_prime * alpha) * d * cmath.exp(-1j * m * gamma)
            D[idx_m_prime][idx_m] = element

    return np.array(D)


def wignerD(euler: np.ndarray, j: int):
    real, imag = [], []
    for i in range(euler.shape[0]):
        D = wigner_D_matrix(j, euler[i, 0], euler[i, 1], euler[i, 2])
        real.append(D.real)
        imag.append(D.imag)
    real, imag = np.stack(real), np.stack(imag)
    return real, imag
