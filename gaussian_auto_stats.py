import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import scipy
from scipy.interpolate import BSpline

from pathlib import Path

rng = np.random.default_rng()

n = 30

epsilon = 0.000000001

relu_underspread_model = (
    rng.uniform(-1, 1., size = (n)) ,
    rng.uniform(-1, 1., size = (n)) ,
    np.array([0.]),
    np.array([0.]),
)

mu = rng.uniform(0, 1., size = (n)) + 10
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



def restat(std, mu, x_s):
    for i in range(100):
        moment = std * 0.
        total_activation = std * 0.
        for x in x_s:
            diff = (x-mu) / std
            diff_sqr = diff * diff
            activation = np.exp(-diff_sqr) / std
            weighted_activation = activation/np.sum(activation)
            x_sqr = (x-mu) *(x-mu)
            x_quart = x_sqr * x_sqr
            total_activation += weighted_activation * 3 * x_sqr
            moment += 2*x_quart* weighted_activation
        var = moment/total_activation
        std = np.sqrt(var)
    return std



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

    return np.sum(activation * (0*m * x + b)) + m1 * x + b1

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
std = restat(std, mu, x_s)

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
#
# result = scipy.optimize.minimize(f, x, method = "CG", jac = df, callback = callback)
# x = result.x
print("Result Complete")

x_s = np.arange(0, 1, 0.001)
fig = plt.figure()
fig.suptitle(Path(__file__).name)
print(Path(__file__).name)
plot_sin(x_s)
plot_model(model_from_vector(x), x_s)
plt.show()
