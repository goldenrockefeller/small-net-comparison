import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import scipy
from scipy.interpolate import BSpline

from pathlib import Path

rng = np.random.default_rng()

n = 60

epsilon = 0.000000001

relu_underspread_model = (
    rng.uniform(-1, 1., size = (n)) ,
    np.ones(n) ,
    np.array([0.]),
    np.array([0.]),
)

mu = rng.uniform(0, 1., size = (n))
std = np.ones(n)


# def get_fstd(mu, x_s):
#     def fstd(std):
#         error = 0.
#
#         for x in x_s:
#             diff = (x-mu) / std
#             diff_sqr = diff * diff
#             activation = np.exp(-diff_sqr)
#             error += (np.sum(activation) - 1.5) * (np.sum(activation) - 1.5)
#
#         return error
#     return fstd
#
#
#
# def restat(std, mu, x_s):
#     result = scipy.optimize.minimize(get_fstd(mu, x_s), std, method = "CG")
#     return result.x
vn = 2
def get_std_mu(v):
    nv = 0. *v
    nv[0] = v[0]
    nv[1] = v[0] + v[1] ** 2
    for i in range(len(v)-2):
        nv[i+2] = nv[i+1] + 2**np.tanh(v[i+2]) * (nv[i+1] - nv[i]) # 2 is a good number for random
    v = nv

    std = [0.for i in range(len(v)-vn)]
    mu = [0.for i in range(len(v)-vn)]

    for i in range(len(v)-vn):
        std[i] =  (v[i+vn] - v[i]) # multiply by 1-4, best is 1
        mu[i] =  0.5 * (v[i+vn] + v[i])

    return np.array(std), np.array(mu)

def restat(std, mu, x_s):
    def cross_entropy(v):
        new_std, new_mu = get_std_mu(v)
        entropy = 0

        for x in x_s:
            activation = np.exp(-0.5 *(new_mu-x)**2/new_std**2)/new_std
            w = np.sum(activation)
            entropy -= np.log(w)
        return entropy

    v = np.zeros(len(std) + vn)
    v[0] = -20
    v[1] = 1
    print(cross_entropy(v))
    result = scipy.optimize.minimize(cross_entropy, v, method = "CG")
    print(result.x, "v")
    print(cross_entropy(result.x), np.sum(result.x), flush = True)
    std, mu = get_std_mu(result.x)
    return std, mu



# constant_bias = rng.uniform(-1, 1., size = (n))
@jax.jit
def eval_model(model, x):
    m = model[0]
    b = model[1]
    m1 = model[2]
    b1 = model[3]

    diff = (x-mu) /std
    diff_sqr = diff * diff
    activation = jax.numpy.exp(-diff_sqr)

    return np.sum(activation * (m * x + b)) + m1 * x + b1

def model_from_vector(vector):
    return (
        vector[:n],
        vector[n:2*n],
        vector[2*n],
        vector[2*n+1]
    )

@jax.jit
def error(model, x, y):
    return 0.5 * np.sum((eval_model(model, x) - y) * (eval_model(model, x) - y))

v_error = jax.vmap(error, (None, 0, 0))

@jax.jit
def jf(vector, x_s, y_s):
    print("eval")
    error_total = 0
    model = model_from_vector(vector)
    errors = v_error(model, x_s, y_s)
    return np.sum(errors)

jdf = jax.jit(jax.grad(jf))

def f(vector):
    x_s = np.arange(0, 1, 0.01)
    y_s = np.sin(6 * np.pi * x_s)
    return jf(vector, x_s, y_s)

def df(vector):
    x_s = np.arange(0, 1, 0.01)
    y_s = np.sin(6 * np.pi * x_s)
    return jdf(vector, x_s, y_s)

def plot_model(model, x_s):
    y_s = x_s * 0.
    for i in range(len(x_s)):
        y_s[i] = eval_model(model, x_s[i])
    plt.plot(x_s, y_s)

def plot_sin(x_s):
    y_s = np.sin(6 * np.pi * x_s)
    plt.plot(x_s, y_s)
#
x =  np.hstack(relu_underspread_model)
x_s = np.arange(0, 1, 0.01)
x_s = rng.uniform(0, 1, len(x_s))
std, mu = restat(std, mu, x_s)

n_epochs_elapsed = 0
def callback(x):
    global n_epochs_elapsed
    print(f"{f(x)}, {n_epochs_elapsed=}")
    n_epochs_elapsed += 1
    return x


y_s = np.sin(6 * np.pi * x_s)
print(jf(x, x_s, y_s))
print(jdf(x, x_s, y_s))
print("JIT Complete")
# #
old_x = x
result = scipy.optimize.minimize(f, x, method = "L-BFGS-B", jac = df)
x = result.x
# print("Result Complete")

x_s = np.arange(0, 1, 0.001)
fig = plt.figure()
fig.suptitle(Path(__file__).name)
print(Path(__file__).name)
plot_sin(x_s)
plot_model(model_from_vector(x), x_s)
plot_model(model_from_vector(old_x), x_s)
plt.show()
