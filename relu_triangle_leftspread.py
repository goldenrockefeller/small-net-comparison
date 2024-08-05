import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import scipy
from scipy.interpolate import BSpline

from pathlib import Path

rng = np.random.default_rng()

n = 100

epsilon = 0.000000001


relu_underspread_model = (
    np.hstack((rng.uniform(-1., 1., size = n)))+10,
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    np.array([0.]),
)

@jax.jit
def eval_model(model, x):
    bias = model[0]
    diff = (x - bias)
    md = model[3]*diff
    # pos_relu = jax.nn.softplus(model[4]*md) * model[1]
    # neg_relu = -jax.nn.softplus(model[4]*-md) * model[2]
    # pos_relu = jax.nn.relu(model[4]*md) * model[1]
    # neg_relu = -jax.nn.relu(model[4]*-md) * model[2]
    pos_relu = md * jax.nn.sigmoid(model[4]*md) * model[1]
    neg_relu = -md * jax.nn.sigmoid(-model[4]*md) * model[2]

    return np.sum(pos_relu + neg_relu) + model[5]

def model_from_vector(vector):
    return (
        vector[:n],
        vector[n:2*n],
        vector[2*n:3*n],
        vector[3*n:4*n],
        vector[4*n:5*n],
        vector[5*n],
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
    y_s = np.sin(12*np.pi * x_s)
    return jf(vector, x_s, y_s)

def df(vector):
    x_s = np.arange(0, 1, 0.01)
    y_s = np.sin(12*np.pi * x_s)
    return jdf(vector, x_s, y_s)

def plot_model(model, x_s):
    y_s = x_s * 0.
    for i in range(len(x_s)):
        y_s[i] = eval_model(model, x_s[i])
    plt.plot(x_s, y_s)

def plot_sin(x_s):
    y_s = np.sin(12*np.pi * x_s)
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


y_s = np.sin(12*np.pi * x_s)
print(jf(x, x_s, y_s))
print(jdf(x, x_s, y_s))
print("JIT Complete")
# #
result = scipy.optimize.minimize(f, x, method = "BFGS", jac = df)
x = result.x
# print("Result Complete")

x_s = np.arange(0, 1, 0.001)
fig = plt.figure()
fig.suptitle(Path(__file__).name)
print(Path(__file__).name)
plot_sin(x_s)
plot_model(model_from_vector(x), x_s)
plt.show()
