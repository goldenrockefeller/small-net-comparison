import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import scipy

from pathlib import Path

rng = np.random.default_rng()

model = (
    rng.uniform(-1, 1., size = (30)),
    rng.uniform(-1, 1., size = (30)),
    np.hstack((rng.uniform(-20, -10., size = (15)),rng.uniform(10, 20., size = (15))))
)

def eval_relu(model, x):
    b = model[0]
    m = model[1]
    c = model[2]
    mc = np.sum(c)/30
    vc = np.sum((c-mc) * (c-mc)) /30
    c = c - (np.sum(c)/30)
    c = c/jax.numpy.sqrt(vc * 12)
    c = c + 0.5

    sqr_diff = (x-c) * (x-c) + 1/30
    sum_inv_sqr_diff = np.sum(1/sqr_diff)
    v = 1 / (sqr_diff * sum_inv_sqr_diff)
    w = v/np.sum(v)
    return np.sum((b +  m * (x-c)) * w)

def model_from_vector(vector):
    return (
        vector[:30],
        vector[30:60],
        vector[60:90]
    )

def jf(vector, x_s, y_s):
    error_total = 0
    model = model_from_vector(vector)
    for x, y in zip(x_s, y_s):
        error_total += 0.5 * np.sum((eval_relu(model, x) - y) * (eval_relu(model, x) - y))
    return error_total

jf = jax.jit(jf)
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


x =  np.hstack(model)

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
# for i in range(4000):
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

result = scipy.optimize.minimize(f, x, method = "L-BFGS-B", jac = df)
x = result.x

x_s = np.arange(0, 1, 0.01)
fig = plt.figure()
fig.suptitle(Path(__file__).name)
print(Path(__file__).name)
plot_sin(x_s)
plot_relu(model_from_vector(x), x_s)
plt.show()