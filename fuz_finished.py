import numpy as np
import matplotlib.pyplot as plt
import jax
jax.config.update("jax_enable_x64", True)
import scipy

from pathlib import Path

rng = np.random.default_rng()

fuz = (
    rng.uniform(-1, 1., size = (30)),
    rng.uniform(-1, 1., size = (30)),
    rng.uniform(-200, -100., size = (30)),
    np.ones(30) * 10000,
)

batch_norm = (np.array([0.]), np.array([1.]))
fuz_std = (np.ones(30), np.array([1.]))
model_stats = (batch_norm, fuz_std)
model = (None, fuz)
q = 1.

def eval_fuz(model, model_stats, x):
    b = model[0]
    m = model[1]
    c = model[2]
    smoothing = model[3]
    mc = np.sum(c)/30
    vc = np.sum((c-mc) * (c-mc)) /30
    # c = c - (np.sum(c)/30)
    # c = c/jax.numpy.sqrt(vc*q)

    # sqr_diff = (x-c) * (x-c) + jax.nn.sigmoid(smoothing[0])
    # sum_inv_sqr_diff = np.sum(1/sqr_diff)
    # v = 1 / (sqr_diff * sum_inv_sqr_diff)
    #
    sqr_diff = (x-c) * (x-c)
    v = jax.numpy.exp(-sqr_diff*4)

    w = v/np.sum(v)

    return np.sum((b +  m * x) * w)

def get_densities(c, x_s, d):
    means = c

    for i in range(100):

        # total = 0. * c
        # sum_weights = 0. * c
        # for x in x_s:
        #     # Calculate difference
        #     sqr_diff = (x-means) * (x-means) /d
        #
        #     v = np.exp(-sqr_diff)
        #
        #     w = v/np.sum(v)
        #
        #     # Add weighted square diff
        #     total += w * x
        #     sum_weights +=  w
        # means = total / sum_weights
        #

        # total = 0. * c
        # sum_weights = 0. * c
        # for x in x_s:
        #     # Calculate difference
        #     sqr_diff = (x-means) * (x-means) /d
        #
        #     v = np.exp(-sqr_diff)
        #
        #     w = v/np.sum(v)
        #
        #     # Add weighted square diff
        #     total += w * x
        #     sum_weights +=  w
        # means = total / sum_weights
        #

        total = 0. * c
        sum_weights = 0. * c
        for x in x_s:
            # Calculate difference
            sqr_diff = (x-c) * (x-c) /d

            v = np.exp(-sqr_diff)

            w = v/np.sum(v)

            # Add weighted square diff
            total += w * (x-c) * (x-c)
            sum_weights +=  w
        variance = total / sum_weights
        d = variance

        print(f"{means=}")
        print(f"{variance=}")
        print()

def eval_rbf(c, x, d):
    sqr_diff = (x-c) * (x-c) /d
    v = np.exp(-sqr_diff)

    return np.sum(v)


def plot_rbf(c, d, x_s):
    y_s = x_s * 0.
    for i in range(len(x_s)):
        y_s[i] = eval_rbf(c,x_s[i], d)
    plt.plot(x_s, y_s)


def eval_batch_norm(model, model_stats, x):
    mean = model_stats[0]
    std = model_stats[1]

    return (x - mean) / std

def eval_model(model, model_stats, x):
    x = eval_batch_norm(model[0], model_stats[0], x)
    x = eval_fuz(model[1], model_stats[1], x)
    return x

def project_vector(vector, model_stats):
    return vector_from_model(project_model(model_from_vector(vector), model_stats))

def project_model(model, model_stats):
    batch_norm = model[0]
    fuz = project_fuz(model[1], model_stats[1])
    return (batch_norm, fuz)

def project_fuz(model, model_stats):
    b = model[0]
    m = model[1]
    c = model[2]
    smoothing = model[3]
    mc = np.sum(c)/30
    vc = np.sum((c-mc) * (c-mc)) /30
    c = c - (np.sum(c)/30)
    c = c/jax.numpy.sqrt(vc*q)

    # smoothing = np.clip(smoothing, -0.1, 0.1)


    return (b, m, c, smoothing)

def restat_model(model, model_mom, model_stats, x_s, rate):
    batch_norm_model, batch_norm_mom, batch_norm_stats = restat_batch_norm(model[0], model_mom[0], model_stats[0], x_s, rate)

    y_s = 0 * x_s
    for x_i, x in enumerate(x_s):
        y_s[x_i] = eval_batch_norm(model[0], batch_norm, x)

    fuz_model,fuz_mom,fuz_stats  = restat_fuz(model[1], model_mom[1], model_stats[1], y_s, rate)

    return (
        (batch_norm_model, fuz_model),
        (batch_norm_mom, fuz_mom),
        (batch_norm_stats, fuz_stats)
    )

def restat_batch_norm(model,model_mom, model_stats, x_s, rate):
    batch_norm = model_stats
    var = batch_norm[1] * batch_norm[1]
    return (
        None,
        None,
        (
            (1 - rate) * batch_norm[0] + rate * np.mean(x_s),
            np.sqrt((1 - rate) * var + rate * np.var(x_s))
        )
    )

def get_entropy(c, x_s):
    def entropy(smoothing):
        val = 0

        for x in x_s:
            # Calculate difference
            sqr_diff = (x-c) * (x-c) + smoothing * smoothing

            # Calculate weights
            sqr_diff_eps = sqr_diff
            sum_inv_sqr_diff = np.sum(1/sqr_diff_eps)
            v = 1. / (sqr_diff_eps * sum_inv_sqr_diff)

            w = v/np.sum(v)

            val += -np.sum(w * np.log2(w))

        return val
    return entropy

def restat_fuz(model,model_mom, model_stats, x_s, rate):
    b = model[0]
    m = model[1]
    c = model[2]
    smoothing = model[3]

    b_mom = model_mom[0]
    m_mom = model_mom[1]
    c_mom = model_mom[2]
    smoothing_mom = model_mom[3]


    mc = np.sum(c)/30
    vc = np.sum((c-mc) * (c-mc)) /30
    c = c - (np.sum(c)/30)
    c = c/np.sqrt(vc*q)

    x_std = model_stats[1]

    fuz_std = model_stats[0]
    fuz_variance = fuz_std * fuz_std
    old_fuz_variance = fuz_variance

    M = 0. * c
    sum_weights = 0. * c
    for x in x_s:
        # Calculate difference
        sqr_diff = (x-c) * (x-c) + smoothing

        # Calculate weights
        sqr_diff_eps = sqr_diff
        sum_inv_sqr_diff = np.sum(1/sqr_diff_eps)
        v = 1. / (sqr_diff_eps * sum_inv_sqr_diff)

        # sqr_diff = (x-c) * (x-c)
        # v = np.exp(sqr_diff / smoothing)

        w = v/np.sum(v)

        # Add weighted square diff
        M += w * (x-c) * (x-c)
        sum_weights +=  w

    fuz_variance = M / sum_weights
    fuz_variance = (1 - rate) *  old_fuz_variance + rate * fuz_variance

    rescale = (old_fuz_variance / fuz_variance)
    # m = m * rescale
    # m_mom = m_mom * rescale

    model = (b, m, c, smoothing)
    model_mom = (b_mom, m_mom, c_mom, smoothing_mom)
    model_stats = (np.sqrt(fuz_variance), x_std)

    # print(smoothing[0])

    return model, model_mom, model_stats

def fuz_from_vector(vector):
    return (
        vector[:30],
        vector[30:60],
        vector[60:90],
        vector[90:120],
    )

def model_from_vector(vector):
    return (None, fuz_from_vector(vector))

def vector_from_fuz(fuz):
    return np.hstack(fuz)

def vector_from_model(model):
    return vector_from_fuz(model[1])

def jf(vector, model_stats, x_s, y_s):
    error_total = 0
    model = model_from_vector(vector)
    for x, y in zip(x_s, y_s):
        e = eval_model(model, model_stats, x) - y
        error_total += 0.5 * np.sum(e * e)
    return error_total

jf = jax.jit(jf)
jdf = jax.jit(jax.grad(jf))

def f(vector, data):
    model_stats = data[0]
    x_s = data[1]
    y_s = data[2]
    return jf(vector, model_stats, x_s, y_s)

def df(vector, data):
    model_stats = data[0]
    x_s = data[1]
    y_s = data[2]
    return jdf(vector, model_stats, x_s, y_s)

def plot_data(model, model_stats, x_s):
    y_s = x_s * 0.
    for i in range(len(x_s)):
        y_s[i] = eval_model(model, model_stats, x_s[i])
    plt.plot(x_s, y_s)

def plot_sin(x_s):
    y_s = np.sin(6 * np.pi * x_s)
    plt.plot(x_s, y_s)

def backtracking_rate(old_lr, f, df, x_mom, descent_dir, data):
    lr = 3 * old_lr
    m = -np.dot(descent_dir, descent_dir)

    c = 0.5
    d = 0.5

    n_lr_evals = 2
    t = -c * m
    fx = f(x_mom, data)
    fx_new = f(x_mom + lr * descent_dir, data)

    if fx - fx_new >= lr * t: # initial step is too small
        while fx - fx_new >= lr * t:
            lr *= 2
            n_lr_evals += 1
            fx_new = f(x_mom + lr * descent_dir, data)
        lr *= 0.5
    else: # initial step is too big
        while fx - fx_new < lr * t:
            lr *= 0.5
            n_lr_evals += 1
            fx_new = f(x_mom + lr * descent_dir, data)

    return (d * lr, n_lr_evals)

x_s = np.arange(0, 1, 0.01)
y_s = np.sin(6 * np.pi * x_s)

x = vector_from_model(model)


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

restat_rate = 0.001

print(f(x, (model_stats, x_s, y_s)))
restarted = False

avg_log_lr = 1.
avg_log_lr_count = 0.
backtrack_counter = horizon
backtrack_on_restart = True

for i in range(8000):
    small_mom = (x - x_prev)
    cum_mom = small_mom + beta * cum_mom
    model, model_mom, model_stats = (
        restat_model(
            model_from_vector(x),
            model_from_vector(small_mom),
            model_stats,
            x_s,
            restat_rate
        )
    )
    x = vector_from_model(model)
    small_mom = vector_from_model(model_mom)
    x_mom = x + horizon * small_mom
    data = (model_stats, x_s, y_s)
    descent_dir = -df(x_mom, data)
    cum_grad = descent_dir + beta * cum_grad
    #cum_grad = (descent_dir/cum_inv_rate/horizon + beta * cum_grad) / (1 + beta)

    if restarts:
        restart_condition = np.dot(cum_grad, cum_mom) < 0
        #restart_condition = np.dot(cum_grad, small_mom + cum_grad) < 0

        if restart_condition:
            print("Restart")
            # restart
            x_mom = x
            model, model_mom, model_stats = (
                restat_model(
                    model_from_vector(x_mom),
                    model_from_vector(0*x),
                    model_stats,
                    x_s,
                    restat_rate
                )
            )
            x_mom = vector_from_model(model)
            data = (model_stats, x_s, y_s)
            descent_dir = -df(x_mom, data)
            cum_mom = 0*x
            cum_grad = descent_dir
            small_mom = 0*x
            n_evals += 1
            avg_log_lr_count = 1.
            backtrack_counter = horizon
            # cum_inv_rate is not zeroed out!

    # skippable? YES!
    if backtrack_counter > 0 or not backtrack_on_restart:
        lr, n_lr_evals = backtracking_rate(2 ** avg_log_lr, f, df, x_mom, descent_dir,  (model_stats, x_s, y_s))
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
    #x = project_vector(x, model_stats)

    n_evals += 1 + n_lr_evals
    n_steps += 1

    print(f"loss = {f(x,  (model_stats, x_s, y_s))}, epoch = {i}")
    n_evals_elapsed.append(n_evals)
    evals.append(f(x,  (model_stats, x_s, y_s)))

# result = scipy.optimize.minimize(f, x, method = "L-BFGS-B", jac = df)
# x = result.x

x_s = np.arange(-6, 6, 0.0001)
fig = plt.figure()
fig.suptitle(Path(__file__).name)
print(Path(__file__).name)
plot_sin(x_s)
plot_data(model_from_vector(x), model_stats, x_s)
plt.show()