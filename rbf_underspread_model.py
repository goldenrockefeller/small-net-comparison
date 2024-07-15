import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)

from pathlib import Path

rng = np.random.default_rng()

model = (
    rng.uniform(0.45, 0.55, size = (30)) +3,
    rng.uniform(-1, 1., size = (30)),
    rng.uniform(-1, 1., size = (30)),
)

def deBoor(x, c0, c1, c2, c3, t1, t2, t3, t4, t5, t6):
    """
    Unrolled algorithm from https://en.wikipedia.org/wiki/De_Boor%27s_algorithm
    for cubic B-splines
    """

    d0 = c0
    d1 = c1
    d2 = c2
    d3 = c3

    epsilon = 0.000000001

    alpha = (x - t3 + epsilon)  / (t6 - t3 + epsilon)
    d3 = (1.0 - alpha) * d2 + alpha * d3

    alpha = (x - t2 + epsilon) / (t5 - t2 + epsilon)
    d2 = (1.0 - alpha) * d1 + alpha * d2

    alpha = (x - t1 + epsilon) / (t4 - t1 + epsilon)
    d1 = (1.0 - alpha) * d0 + alpha * d1

    alpha = (x - t3 + epsilon) / (t5 - t3 + epsilon)
    d3 = (1.0 - alpha) * d2 + alpha * d3

    alpha = (x - t2 + epsilon) / (t4 - t2 + epsilon)
    d2 = (1.0 - alpha) * d1 + alpha * d2

    alpha = (x - t3 + epsilon) / (t4 - t3 + epsilon)
    d3 = (1.0 - alpha) * d2 + alpha * d3

    return d3

def get_one_smaller(x, c):
    val = c[0]

    for candidate in c:

        # Get the smaller number of val and candidate
        cond = candidate < val
        smaller = (
            cond * candidate
            + (1- cond) * val
        )

        # If candidate is greater than val, get the larger of the two.
        cond = candidate > val
        val = (
            cond * candidate
            + (1- cond) * val
        )

        # If val is greater than x, get the smaller of val and candidate
        cond = val > x
        val = (
            cond * smaller
            + (1-cond) * val
        )

    return val

def get_one_bigger(x, c):
    val = c[0]

    for candidate in c:

        # Get the bigger number of val and candidate
        cond = candidate >= val
        bigger = (
            cond * candidate
            + (1- cond) * val
        )

        # If candidate is less than val, get the smaller of the two.
        cond = candidate < val
        val = (
            cond * candidate
            + (1- cond) * val
        )

        # If val is less/eq than x, get the bigger of val and candidate
        cond = val <= x
        val = (
            cond * bigger
            + (1-cond) * val
        )

    return val

def eval_relu(model, x):
    c = model[0]
    b = model[1]
    m = model[2]
    mc = jax.numpy.mean(c)
    vc = jax.numpy.var(c)

    c = c - mc
    c = c/jax.numpy.sqrt(vc * 12)
    c = c + 0.5

    e = x-c
    e2 = e * e
    v = e/ (1+jax.numpy.abs(e))
    w = v/np.sum(v)
    return np.sum(w * (b+m * x))

def model_from_vector(vector):
    return (
        vector[:30],
        vector[30:60],
        vector[60:90]
    )

def get_ent(x):
    x = x.reshape(2,4)
    x = abs(x)
    vsum = np.sum(x, axis = 1)
    asum = np.sum(vsum)
    hsum = np.sum(x, axis = 0)
    ent = 0
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            p =  x[i,j] *x[i,j]  /vsum[i]/ hsum[j]
            if p > 0:
                ent -= p*np.log(p)
    return ent

def f(vector):
    x_s = np.arange(0, 1, 0.01)
    y_s = np.sin(6 * np.pi * x_s)
    error_total = 0
    model = model_from_vector(vector)
    for x, y in zip(x_s, y_s):
        error_total += 0.5 * np.sum((eval_relu(model, x) - y) * (eval_relu(model, x) - y))
    return error_total

f = jax.jit(f)
df = jax.jit(jax.grad(f))


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
    if not np.isfinite(fx):
        raise ValueError
    if not np.isfinite(m):
        print(descent_dir)
        raise ValueError
    fx_new = f(x_mom + lr * descent_dir)

    if fx - fx_new >= lr * t and np.isfinite(fx - fx_new): # initial step is too small
        while fx - fx_new >= lr * t:
            lr *= 2
            n_lr_evals += 1
            fx_new = f(x_mom + lr * descent_dir)
        lr *= 0.5
    else: # initial step is too big
        while fx - fx_new < lr * t or not np.isfinite(fx - fx_new):
            lr *= 0.5
            n_lr_evals += 1
            fx_new = f(x_mom + lr * descent_dir)

    return (d * lr, n_lr_evals)


x =  np.hstack(model)
x_prev = x
epsilon = 1e-16
lr = epsilon
n_evals_elapsed = []
evals = []
n_evals = 0
n_steps = 0
restarts = True
cum_mom = 0*x
cum_grad = 0*x
cum_inv_rate = epsilon
horizon = 1 # all the same horizon value!!!
beta = 1 - (1/horizon)

print(f(x))
restarted = False

avg_log_lr = 1.
avg_log_lr_count = 0.
backtrack_counter = horizon
backtrack_on_restart = False

for i in range(4000):
    small_mom = (x - x_prev)
    cum_mom = small_mom + beta * cum_mom
    x_mom = x + horizon * small_mom
    descent_dir = -df(x_mom)
    cum_grad = descent_dir + beta * cum_grad
    #cum_grad = (descent_dir/cum_inv_rate/horizon + beta * cum_grad) / (1 + beta)

    if restarts:
        restart_condition = np.dot(cum_grad, cum_mom) < 0
        #restart_condition = np.dot(cum_grad, small_mom + cum_grad) < 0

        if restart_condition:
            print("Restart")
            # restart
            x_mom = x
            descent_dir = -df(x_mom)
            cum_mom = 0*x
            cum_grad = descent_dir
            small_mom = 0*x
            n_evals += 1
            avg_log_lr_count = 1.
            backtrack_counter = horizon
            # cum_inv_rate is not zeroed out!

    # skippable? YES!
    if backtrack_counter > 0 or not backtrack_on_restart:
        lr, n_lr_evals = backtracking_rate(2 ** avg_log_lr, f, df, x_mom, descent_dir)
        avg_log_lr = (
            (np.log2(lr) + avg_log_lr_count * avg_log_lr)
            / (1 + avg_log_lr_count)
        )
        avg_log_lr_count += 1.
        backtrack_counter -= 1
        cum_inv_rate = 1 / (lr+ epsilon) + beta * cum_inv_rate
    else:
        n_lr_evals = 0

    grad_step = descent_dir / cum_inv_rate / horizon
    x_prev = x
    x = x + small_mom + grad_step

    n_evals += 1 + n_lr_evals
    n_steps += 1

    print(f"loss = {f(x)}")
    n_evals_elapsed.append(n_evals)
    evals.append(f(x))

x_s = np.arange(0, 1, 0.01)
fig = plt.figure()
fig.suptitle(Path(__file__).name)
print(Path(__file__).name)
plot_sin(x_s)
plot_relu(model_from_vector(x), x_s)
plt.show()
