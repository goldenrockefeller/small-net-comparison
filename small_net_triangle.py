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

#Ideas
"""
Double Gated
Partial Nestorov Restart

Constraints:
Same frequency
Same phase
Same amplitude
Gate Phase

Gradient:
Batch Norm Variance 1 (results in large-scaling)
Change of variables/ Gradient scaling/ADAM




"""

relu_underspread_model = (
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(1, 2., size = (n)),
    -rng.uniform(-10, -9., size = (n))
)

@jax.jit
def eval_model(model, x):
    a = model[0]
    b = model[1]
    c = model[2]
    dd = np.arange(n)
    # d = model[3]
    d = model[3][0] * dd
    a1 = model[4]

    w = jax.numpy.exp(-a*x)*b* (jax.numpy.sin(c) * jax.numpy.sin(d * x) + jax.numpy.cos(c)  * jax.numpy.cos(d*x))
    # w = 1/(1 + jax.numpy.exp(-a*(x+a1)))*b

    return np.sum(w)

def model_from_vector(vector):
    return (
        vector[:n],
        vector[n:2*n],
        vector[2*n:3*n],
        vector[3*n:4*n],
        vector[4*n:5*n]
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
    y_s = np.abs( 20*(x_s-0.5))
    return jf(vector, x_s, y_s)

def df(vector):
    x_s = np.arange(0, 1, 0.01)
    y_s = np.abs( 20*(x_s-0.5))
    return jdf(vector, x_s, y_s)

def plot_model(model, x_s):
    y_s = x_s * 0.
    for i in range(len(x_s)):
        y_s[i] = eval_model(model, x_s[i])
    plt.plot(x_s, y_s)

def plot_sin(x_s):
    y_s = np.abs( 20*(x_s-0.5))
    plt.plot(x_s, y_s)
#
x =  np.hstack(relu_underspread_model)
x_s = np.arange(0, 1, 0.01)

n_epochs_elapsed = 0
def callback(x):
    global n_epochs_elapsed
    print(f"{f(x)}, {n_epochs_elapsed=}")
    n_epochs_elapsed += 1
    return x


y_s = np.abs( 20*(x_s-0.5))
print(jf(x, x_s, y_s))
print(jdf(x, x_s, y_s))
print("JIT Complete")
# #
result = scipy.optimize.minimize(f, x, method = "CG", jac = df)
x = result.x
# print("Result Complete")

x_s = np.arange(-1, 2, 0.001)
fig = plt.figure()
fig.suptitle(Path(__file__).name)
print(Path(__file__).name)
plot_sin(x_s)
plot_model(model_from_vector(x), x_s)
plt.show()
