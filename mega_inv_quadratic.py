import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import scipy
from scipy.interpolate import BSpline

from pathlib import Path

rng = np.random.default_rng()

n = 6

epsilon = 0.000000001

relu_underspread_model = (
    rng.uniform(-1, 1., size = (n)) ,
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    np.array([0.]),
    np.array([0.]),
)


# constant_bias = rng.uniform(-1, 1., size = (n))
@jax.jit
def eval_model(model, x):
    bias = model[0]
    c =bias
    mc = np.sum(c)/50
    vc = np.sum((c-mc) * (c-mc)) /50
    c = c - mc
    # c = c/jax.numpy.sqrt(vc * 12/2)
    # bias = c + 0.5

    m = model[1]
    b = model[2]
    p = model[3]
    q = model[4]
    m1 = model[5]
    b1 = model[6]

    diff = (x - bias)
    sqr_diff = diff * diff

    ep = jax.numpy.exp(p)
    eq = jax.numpy.exp(q)
    v = (1 + ep * sqr_diff) / (1+(ep + eq) * sqr_diff)
    # w = v/np.sum(v)
    w = v

    pos_relu = jax.nn.relu(diff) * model[1]
    neg_relu = -jax.nn.relu(-diff) * model[2]

    return np.sum(w * (m * x + b)) + m1 * x + b1

def model_from_vector(vector):
    return (
        vector[:n],
        vector[n:2*n],
        vector[2*n:3*n],
        vector[3*n:4*n],
        vector[4*n:5*n],
        vector[5*n],
        vector[5*n+1]
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

x =  np.hstack(relu_underspread_model)
x_s = np.arange(0, 1, 0.01)
y_s = np.sin(6 * np.pi * x_s)
print(jf(x, x_s, y_s))
print(jdf(x, x_s, y_s))
print("JIT Complete")

result = scipy.optimize.minimize(f, x, method = "CG", jac = df)
x = result.x
print("Result Complete")

x_s = np.arange(0, 1, 0.001)
fig = plt.figure()
fig.suptitle(Path(__file__).name)
print(Path(__file__).name)
plot_sin(x_s)
plot_model(model_from_vector(x), x_s)
plt.show()
