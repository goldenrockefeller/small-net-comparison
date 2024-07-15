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
    rng.uniform(-1, 1., size = (n)) +3,
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    rng.uniform(-1, 1., size = (n)),
    np.array([0.]),
)



def spline(x, c0, c1, c2, c3, t0, t1, t2, t3):
    v = (x-t1) / (t2- t1 + epsilon)
    h00 = (1 + 2*v) * (1-v) * (1-v)
    h10 = v*(1-v) *(1-v)
    h01 = v * v * (3-2 *v)
    h11 = v * v * (v-1)

    m0 =0 * (c2 - c0) / (t2 - t0 )
    m1 =0 * (c3 - c1) / (t3 - t1 )
    #
    # m0 =  0.5 * ((c2-c1)/(t2-t1) + (c1-c0)/(t1-t0))
    # m1 =  0.5 * ((c3-c2)/(t3-t2) + (c2-c1)/(t2-t1))

    return h00 * c1 + h10 * (t2 - t1) * m0 + h01 * c2 + h11 * (t2 - t1) * m1

def spline2(x, c1, c2, m0, m1, t1, t2):
    """
    Cubic Hermite Spline, Might try B-splines again later but it didn't work
    """

    v = (x-t1) / (t2+epsilon - t1)
    h00 = (1 + 2*v) * (1-v) * (1-v)
    h10 = v*(1-v) *(1-v)
    h01 = v * v * (3-2 *v)
    h11 = v * v * (v-1)

    return h00 * c1 + h10 * (t2 - t1) * m0 + h01 * c2 + h11 * (t2 - t1) * m1


@jax.jit
def smaller_than(val, candidate):
    val = jax.numpy.where(candidate <= val, candidate, val)
    return val, val

@jax.jit
def bigger_less_than_x(data, candidate):
    val, skip_count, x = data

    # Candidate is equal to x, skip it.
    old_skip_count = skip_count
    old_candidate = candidate

    cond = jax.numpy.where(candidate == x, skip_count > 0, False)
    skip_count = jax.numpy.where(cond, skip_count-1, skip_count)
    # If candidate is greater than val but less than x, get candidate
    old_val = val
    old_candidate = candidate
    candidate = jax.numpy.where(cond, val, candidate)

    val = jax.numpy.where(candidate > val, candidate, old_val)
    val = jax.numpy.where(candidate <= x, val, old_val)

    data = (val, skip_count, x)
    return data, val

@jax.jit
def get_one_smaller(x, c, skip_count):
    val = c[0]

    # Get smallest value
    val, ignored = jax.lax.scan(smaller_than, val, c)
    data = (val, skip_count, x)
    data, ignored = jax.lax.scan(bigger_less_than_x, data, c)
    val, skip_count, x = data

    return val

@jax.jit
def bigger_than(val, candidate):
    val = jax.numpy.where(candidate >= val, candidate, val)
    return val, val

@jax.jit
def smaller_greater_than_x(data, candidate):
    val, skip_count, x = data

    # Candidate is equal to x, skip it.
    old_skip_count = skip_count
    old_candidate = candidate

    cond = jax.numpy.where(candidate == x, skip_count > 0, False)
    skip_count = jax.numpy.where(cond, skip_count-1, skip_count)
    candidate = jax.numpy.where(cond, val, candidate)

    # If candidate is greater than val but less than x, get candidate
    old_val = val
    old_candidate = candidate

    val = jax.numpy.where(candidate < val, candidate, old_val)
    val = jax.numpy.where(candidate >= x, val, old_val)

    data = (val, skip_count, x)
    return data, val


@jax.jit
def get_one_bigger(x, c, skip_count):
    val = c[0]

    # Get smallest value
    val, ignored = jax.lax.scan(bigger_than, val, c)
    data = (val, skip_count, x)
    data, ignored = jax.lax.scan(smaller_greater_than_x, data, c)
    val, skip_count, x = data

    return val

@jax.jit
def eval_relu(model, x):
    bias = model[0]
    c =bias
    mc = np.sum(c)/50
    vc = np.sum((c-mc) * (c-mc)) /50
    c = c - mc
    c = c/jax.numpy.sqrt(vc * 12/2)
    # bias = c + 0.5
#
    diff = (x - bias)
    pos_relu = jax.nn.relu(diff) * model[1]
    neg_relu = -jax.nn.relu(-diff) * model[2]

    return np.sum(pos_relu + neg_relu) + model[4]

@jax.jit
def eval_bspline(model, x):
    bias = model[0]
    c =bias
    mc = np.sum(c)/50
    vc = np.sum((c-mc) * (c-mc)) /50
    c = c - mc
    std = jax.numpy.sqrt(vc * 12)
    c = c/std
    # bias = c + 0.5

    m = model[3]

    t1 = get_one_smaller(x, bias, 0)
    t2 = get_one_bigger(t1, bias, n)

    t1_ind = 1. * (t1 == bias)
    t2_ind = 1. * (t2 == bias)

    m0 = np.sum(t1_ind * model[3]) / np.sum(t1_ind)
    m1 = np.sum(t2_ind * model[3]) / np.sum(t2_ind)


    c1 = eval_relu(model, t1)
    c2 = eval_relu(model, t2)

    ans = spline2(x, c1, c2, m0, m1, t1, t2)

    return ans


#
# def eval_relu(model, x):
#     bias = model[0]
#     c =bias
#     mc = np.sum(c)/50
#     vc = np.sum((c-mc) * (c-mc)) /50
#     c = c - mc
#     c = c/jax.numpy.sqrt(vc * 12)
#     bias = c + 0.5
#
#     diff = (x - bias)
#     pos_relu = jax.nn.relu(diff) * model[1]
#     neg_relu = -jax.nn.relu(-diff) * model[2]
#
#     return np.sum(pos_relu + neg_relu) + model[3]

def model_from_vector(vector):
    return (
        vector[:n],
        vector[n:2*n],
        vector[2*n:3*n],
        vector[3*n:4*n],
        vector[4*n],
    )

@jax.jit
def error(model, x, y):
    return 0.5 * np.sum((eval_relu(model, x) - y) * (eval_relu(model, x) - y))

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

def plot_relu(model, x_s):
    y_s = x_s * 0.
    for i in range(len(x_s)):
        y_s[i] = eval_relu(model, x_s[i])
    plt.plot(x_s, y_s)

def plot_spline(model, x_s):
    y_s = x_s * 0.
    for i in range(len(x_s)):
        y_s[i] = eval_bspline(model, x_s[i])
    plt.plot(x_s, y_s)

def plot_sin(x_s):
    y_s = np.sin(6 * np.pi * x_s)
    plt.plot(x_s, y_s)

def backtracking_rate(old_lr, f, df, x_mom, descent_dir):
    lr = 3 * old_lr
    m = -np.dot(descent_dir, descent_dir)

    c = 0.5
    d = 0.5

    n_lr_evals = 2
    t = -c * m
    fx = f(x_mom)
    fx_new = f(x_mom + lr * descent_dir)

    if fx - fx_new >= lr * t: # initial step is too small
        while fx - fx_new >= lr * t:
            lr *= 2
            n_lr_evals += 1
            fx_new = f(x_mom + lr * descent_dir)
        lr *= 0.5
    else: # initial step is too big
        while fx - fx_new < lr * t:
            lr *= 0.5
            n_lr_evals += 1
            fx_new = f(x_mom + lr * descent_dir)

    return (d * lr, n_lr_evals)


x =  np.hstack(relu_underspread_model)
# x_prev = x
# epsilon = 1e-16
# lr = epsilon
# n_evals_elapsed = []
# evals = []
# n_evals = 0
# n_steps = 0
# restarts = True
# cum_mom = 0*x
# cum_grad = 0*x
# cum_inv_rate = epsilon
# horizon = 1 # all the same horizon value!!!
# beta = 1 - (1/horizon)
#
# print(f(x))
# restarted = False
#
# avg_log_lr = 1.
# avg_log_lr_count = 0.
# backtrack_counter = horizon
# backtrack_on_restart = False
#
# for i in range(400):
#     small_mom = (x - x_prev)
#     cum_mom = small_mom + beta * cum_mom
#     x_mom = x + horizon * small_mom
#     descent_dir = -df(x_mom)
#     cum_grad = descent_dir + beta * cum_grad
#     #cum_grad = (descent_dir/cum_inv_rate/horizon + beta * cum_grad) / (1 + beta)
#
#     if restarts:
#         restart_condition = np.dot(cum_grad, cum_mom) < 0
#         #restart_condition = np.dot(cum_grad, small_mom + cum_grad) < 0
#
#         if restart_condition:
#             print("Restart")
#             # restart
#             x_mom = x
#             descent_dir = -df(x_mom)
#             cum_mom = 0*x
#             cum_grad = descent_dir
#             small_mom = 0*x
#             n_evals += 1
#             avg_log_lr_count = 1.
#             backtrack_counter = horizon
#             # cum_inv_rate is not zeroed out!
#
#     # skippable? YES!
#     if backtrack_counter > 0 or not backtrack_on_restart:
#         lr, n_lr_evals = backtracking_rate(2 ** avg_log_lr, f, df, x_mom, descent_dir)
#         avg_log_lr = (
#             (np.log2(lr) + avg_log_lr_count * avg_log_lr)
#             / (1 + avg_log_lr_count)
#         )
#         avg_log_lr_count += 1.
#         backtrack_counter -= 1
#         cum_inv_rate = 1 / (lr+ epsilon) + beta * cum_inv_rate
#     else:
#         n_lr_evals = 0
#
#     grad_step = descent_dir / cum_inv_rate / horizon
#     x_prev = x
#     x = x + small_mom + grad_step
#
#     n_evals += 1 + n_lr_evals
#     n_steps += 1
#
#     print(f"loss = {f(x)}")
#     n_evals_elapsed.append(n_evals)
#     evals.append(f(x))

x_s = np.arange(0, 1, 0.01)
y_s = np.sin(6 * np.pi * x_s)
print(jf(x, x_s, y_s))
print(jdf(x, x_s, y_s))
print("JIT Complete")

result = scipy.optimize.minimize(f, x, method = "L-BFGS-B", jac = df)
x = result.x
print("Result Complete")

x_s = np.arange(0, 1, 0.001)
fig = plt.figure()
fig.suptitle(Path(__file__).name)
print(Path(__file__).name)
plot_sin(x_s)
plot_relu(model_from_vector(x), x_s)
# plot_spline(model_from_vector(x), x_s)
plt.show()

def hermite_interpolation(x, t, c):
    val = 0

    for i in range(len(t)):
        if i -1 >= len(t):
            continue
        if i + 2 >= len(t):
            continue
        if x > t[i] and x <= t[i+1]:
            v = (x-t[i]) / (t[i+1] - t[i])
            h00 = (1 + 2*v) * (1-v) * (1-v)
            h10 = v*(1-v) *(1-v)
            h01 = v * v * (3-2 *v)
            h11 = v * v * (v-1)

            m0 = (c[i+1] - c[i-1]) / (t[i+1] - t[i-1])
            m1 = (c[i+2] - c[i]) / (t[i+2] - t[i])

            return h00 * c[i] + h10 * (t[i+1] - t[i]) * m0 + h01 * c[i+1] + h11 * (t[i+1] - t[i]) * m1

    return val

k = 2
c = np.sort(model_from_vector(x)[0])

t = c
c = [eval_relu(model_from_vector(x), ti) for ti in t]
# t = np.arange(0, 1, 0.1)
# c = t * 0; c[5] = 1
y_spl = [hermite_interpolation(xi, t, c) for xi in x_s]
plt.plot(x_s,y_spl)
# plot_relu(model_from_vector(x), x_s)
plt.show()
