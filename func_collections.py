from re import L
import numpy as np
import scipy.special as sp

PI = np.pi

# Make sure these functions are numpy array friendly if you decide to implement something
def id(z):
    return z

def three(z):
    return 3 * z

# By default we treat all nan as infinity
def inv(z):
    res = np.zeros_like(z)
    res[z != 0] = 1 / z[z != 0]
    res[z == 0] = np.infty
    return res

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
def true_zeta(z, k = 100):
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
def Weierstrass_P(z, w1 = 2 - 1j, w2 = 1 + 1j):
    res = inv(z ** 2)
    for i in range(-20, 20):
        for k in range(-20, 20):
            if i == 0 and k == 0: continue
            l = i * w1 + k * w2
            res += inv((z - l) ** 2) - 1 / l ** 2
    return res

# You can specify how many iterations to plot
def iter_zeta(z, i):
    res = np.zeros_like(z)
    res[z.real > 1] = true_zeta(z[z.real > 1], i)
    res[z.real <= 1] = exp(z[z.real <= 1] * np.log(2)) * exp((z[z.real <= 1] - 1) * np.log(PI)) * sin(PI * z[z.real <= 1] / 2) * gamma(1 - z[z.real <= 1]) * true_zeta(1 - z[z.real <= 1], i)
    return res


# f is the function and i is the number of iterations
def Newtons_Attractor(z, f, f_prime, iter):
    for _ in range(iter):
        z = z - f(z) * inv(f_prime(z))
    return z