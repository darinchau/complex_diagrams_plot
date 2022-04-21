import numpy as np
import scipy.special as sp
import time

PI = np.pi


# Helper function for a loading bar-esque progress function
lpl = 0
def fprint(fr, total_fr, func_name):
    global lpl
    len_bar = 24
    ratio = round((fr + 1)/total_fr * len_bar)
    st = func_name + ": [" + ratio * "=" + (len_bar - ratio) * " " + "]  " + str(fr + 1) + "/" + str(total_fr)
    print("\b" * lpl + st, end = "", flush = True)
    lpl = len(st)


# Make sure these functions are numpy array friendly if you decide to implement something

# The identity function also fixes the bullshit values
def id(z):
    z[z.real == np.nan] = np.infty
    z[z.imag == np.nan] = np.infty
    return z


def three(z):
    return 3 * z


# By default we treat all nan as infinity
def inv(z):
    res = np.zeros_like(z)
    res[z != 0] = 1 / z[z != 0]
    res[z == 0] = np.infty
    return id(res)

def exp(z):
    return np.exp(z)


def sin(z):
    return np.sin(z)


def cos(z):
    return np.cos(z)

# e ^ 1/z. Nice one to plot because of it's essential singularity


def e1z(z):
    return exp(inv(z))


def gamma(z):
    return sp.gamma(z)

# True zeta is the one without analytic continuation
def true_zeta(z, k=100):
    res = np.zeros_like(z)
    for i in range(1, k):
        res += inv(exp(z * np.log(i)))
    return res

# The usual Riemann zeta function that we talk about
def zeta(z):
    res = np.zeros_like(z)
    # Main part
    res[z.real > 1] = true_zeta(z[z.real > 1])
    # Analytic continuation
    res[z.real <= 1] = exp(z[z.real <= 1] * np.log(2)) * exp((z[z.real <= 1] - 1) * np.log(PI)) * sin(PI * z[z.real <= 1] / 2) * gamma(1 - z[z.real <= 1]) * true_zeta(1 - z[z.real <= 1])
    return res

# w1 and w2 are the two parameters
# If you want to plot WP, you need a helper function like:
# def Fancy_P(z):
#     return Weierstrass_P(z, 1-3j, 2 + 1j)
def Weierstrass_P(z, w1=2 - 1j, w2=1 + 1j):
    res = inv(z ** 2)
    for i in range(-20, 20):
        for k in range(-20, 20):
            if i == 0 and k == 0:
                continue
            l = i * w1 + k * w2
            res += inv((z - l) ** 2) - 1 / l ** 2
    return res

# Partial sums of zeta
def zeta_partial(z, i):
    res = np.zeros_like(z)
    res[z.real > 1] = inv(exp(z[z.real > 1] * np.log(i)))
    res[z.real <= 1] = exp(z[z.real <= 1] * np.log(2)) * exp((z[z.real <= 1] - 1) * np.log(PI)) * sin(PI * z[z.real <= 1] / 2) * gamma(1 - z[z.real <= 1]) * inv(exp((1-z[z.real <= 1]) * np.log(i)))
    return res


# f is the function and i is the number of iterations
def Newtons_Attractor(z, f, df, iter):
    for _ in range(iter):
        z = z - f(z) * inv(df(z))
    return z

# Eisenstein Series E2k
def ES(z, k, precision = 20, show_progress = True):
    q = exp(2 * PI * 1j * z)
    summ = np.ones_like(q)

    # Calculate E4 using the summation definition
    q_powers = q
    c = round(2 / sp.zeta(1 - 2 * k), 5)
    for n in range(1, precision):
        if show_progress: fprint(n, precision, "Calculating E{}".format(2 * k))
        assert np.count_nonzero(np.abs(q_powers - q ** n) > 1e-10) == 0
        summ += c * (n ** (2 * k - 1)) * q_powers / (1 - q_powers)
        q_powers = np.multiply(q_powers, q)
    if show_progress: print()
    return summ

def j(z, precision = 10, show_progress = True):
    res = np.zeros_like(z)
    E4t3 = ES(z, 2, precision, show_progress) ** 3
    E6t2 = ES(z, 3, precision, show_progress) ** 2
    res = 1728 * E4t3 * inv(E4t3 - E6t2)
    return res

# # A faster implementation of j for repeated summations
# E4_vals, E6_vals, initialized = [1], [1], False
# def j_fast_init(z, max_precision = 300):
#     q = exp(2 * PI * 1j * z)
#     for n in range(1, max_precision):
#         E4_vals.append(240 * (n ** 3 * q ** n / (1 - q ** n)))
#         E6_vals.append(-504 * (n ** 5 * q ** n / (1 - q ** n)))


# def j_fast(z, precision):
#     q = exp(2 * PI * 1j * z)
#     E4t3, E6t2 = np.ones_like(q), np.ones_like(q)
#     for n in range(1, precision):
#         E4t3 += E4_vals[n]
#         E6t2 += E6_vals[n]
#     E4t3 = E4t3 ** 3
#     E6t2 = E6t2 ** 2
#     return 1728 * E4t3 * inv(E4t3 - E6t2)


# Alternative implementation using power series, much faster but cannot compute to arbittrary precision since we do not have a way to multiply stuff by such large coefficients
# def j2(z):
#     q = exp(2 * PI * 1j * z[z.imag > 0])
#     res = np.zeros_like(z)
#     coeffs = np.array([744, 196884, 21493760, 864299970, 20245856256, 333202640600, 4252023300096, 44656994071935, 401490886656000, 3176440229784420, 22567393309593600, 146211911499519294, 874313719685775360])
#     #, 4872010111798142520, 25497827389410525184, 126142916465781843075
#     res[z.imag > 0] = inv(q) + 744
#     for i in range(1, len(coeffs)):
#         res[z.imag > 0] += coeffs[i] * q ** i
#     return res