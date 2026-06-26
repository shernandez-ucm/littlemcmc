import numpy as np
import matplotlib.pyplot as plt

import jax
# NUTS (littlemcmc) corre en numpy/float64: igualamos la precisión de JAX para
# que el logp y su gradiente sean estables en el muestreador.
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
from jax.scipy.special import gammaln

from littlemcmc.nuts_jax import sample_vmapped_nuts_chains

RANDOM_STATE = 42   # semilla única para todo el laboratorio (reproducibilidad)


def f(x):
    '''Función verdadera (media condicional) sin ruido.'''
    return x * np.sin(x)


def train_test_split(X, y, test_size=0.3, random_state=0):
    '''División entrenamiento/prueba sin dependencias externas (sklearn).'''
    rng = np.random.RandomState(random_state)
    idx = rng.permutation(len(X))
    n_test = int(round(test_size * len(X)))
    test_idx, train_idx = idx[:n_test], idx[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def standardize_X(a):
    return (a - x_mean) / x_sd


# ---------------------------------------------------------------------------
# MLP funcional (lista de pares (W, b)); compatible con el bucle SGD de abajo.
# ---------------------------------------------------------------------------
def init_mlp(key, sizes):
    '''Inicializa los parámetros de un MLP con inicialización tipo He.'''
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for k, din, dout in zip(keys, sizes[:-1], sizes[1:]):
        W = jax.random.normal(k, (din, dout)) * jnp.sqrt(2.0 / din)
        b = jnp.zeros((dout,))
        params.append((W, b))
    return params


def mlp_forward(params, x):
    '''Pasada hacia adelante: capas ReLU ocultas y salida lineal.'''
    for W, b in params[:-1]:
        x = jax.nn.relu(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def init_sgd(params):
    '''Vector de velocidad (momentum) inicializado en cero.'''
    return tree_map(jnp.zeros_like, params)


def sgd_step(params, grads, velocity, lr=1e-2, momentum=0.9):
    '''Una actualización de SGD con momentum sobre el árbol de parámetros.'''
    velocity = tree_map(lambda v_, g: momentum * v_ + g, velocity, grads)
    params = tree_map(lambda p, v_: p - lr * v_, params, velocity)
    return params, velocity


def train(loss_fn, params, X, y, n_epochs=4000, lr=1e-2, log_every=1000):
    '''Entrena `params` minimizando `loss_fn(params, X, y)` con SGD (momentum).'''
    velocity = init_sgd(params)

    @jit
    def step(params, velocity):
        loss, grads = value_and_grad(loss_fn)(params, X, y)
        params, velocity = sgd_step(params, grads, velocity, lr=lr)
        return params, velocity, loss

    history = []
    for epoch in range(1, n_epochs + 1):
        params, velocity, loss = step(params, velocity)
        history.append(float(loss))
        if epoch % log_every == 0 or epoch == 1:
            print(f"  época {epoch:5d}  |  pérdida = {loss:.4f}")
    return params, history


def tstudent_loss(params, x, y):
    '''NLL de una t de Student (location-scale) con cabeza heterocedástica.'''
    out = mlp_forward(params, x)               # (N, 3): mu, log(sigma^2), nu_raw
    mu = out[:, 0]
    log_var = jnp.clip(out[:, 1], -7.0, 7.0)   # estabilidad numérica
    nu = 2.0 + jax.nn.softplus(out[:, 2])      # grados de libertad nu > 2 (varianza finita)
    z = (y - mu) / jnp.exp(0.5 * log_var)
    # NLL de la t de Student: colas más pesadas -> robusta a outliers
    nll = (gammaln(nu / 2.0) - gammaln((nu + 1.0) / 2.0)
           + 0.5 * jnp.log(nu * jnp.pi) + 0.5 * log_var
           + 0.5 * (nu + 1.0) * jnp.log1p(z ** 2 / nu))
    return jnp.mean(nll)


def predict_params(params, x):
    '''Devuelve (mu, sigma, nu) de la cabeza t-Student en espacio estandarizado.'''
    out = mlp_forward(params, x)
    mu = out[:, 0]
    sigma = jnp.exp(0.5 * jnp.clip(out[:, 1], -7.0, 7.0))
    nu = 2.0 + jax.nn.softplus(out[:, 2])
    return mu, sigma, nu


def tstudent_logpdf(y, mu, sigma, nu):
    '''log-densidad t-Student (location-scale) por punto.'''
    log_var = 2.0 * jnp.log(sigma)
    z = (y - mu) / sigma
    return -(gammaln(nu / 2.0) - gammaln((nu + 1.0) / 2.0)
             + 0.5 * jnp.log(nu * jnp.pi) + 0.5 * log_var
             + 0.5 * (nu + 1.0) * jnp.log1p(z ** 2 / nu))


# ---------------------------------------------------------------------------
# Datos sintéticos
# ---------------------------------------------------------------------------
rng = np.random.RandomState(RANDOM_STATE)
n_samples = 1000
X = rng.uniform(0.0, 10.0, size=n_samples).reshape(-1, 1)
expected_y = f(X).ravel()
# Ruido log-normal cuya dispersión aumenta con x (heterocedástico y asimétrico)
sigma = 0.5 + X.ravel() / 10.0
noise = rng.lognormal(mean=0.0, sigma=sigma) - np.exp(sigma**2 / 2.0)
y = expected_y + noise
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=RANDOM_STATE
)
print(f"Entrenamiento: {X_train.shape[0]} muestras")
print(f"Prueba:        {X_test.shape[0]} muestras")


x_mean, x_sd = X_train.mean(), X_train.std()
y_mean, y_sd = y_train.mean(), y_train.std()
# Tensores JAX estandarizados para el entrenamiento
Xtr = jnp.asarray(standardize_X(X_train))
ytr = jnp.asarray((y_train - y_mean) / y_sd)
Xte = jnp.asarray(standardize_X(X_test))
yte = jnp.asarray((y_test - y_mean) / y_sd)

# Malla ordenada para graficar las curvas suaves
xx = np.linspace(0.0, 10.0, 400).reshape(-1, 1)
Xgrid = jnp.asarray(standardize_X(xx))


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
random_key = jax.random.PRNGKey(RANDOM_STATE)
params_tstudent_key, random_key = jax.random.split(random_key)
params_tstudent = init_mlp(params_tstudent_key, [1, 64, 64, 3])
print("Entrenando MLP con pérdida t-student...")
params_tstudent, hist_tstudent = train(
    tstudent_loss, params_tstudent, Xtr, ytr, n_epochs=20000, lr=1e-2
)


# ---------------------------------------------------------------------------
# Predicción en la malla (des-estandarizada) y gráfica
# ---------------------------------------------------------------------------
out_grid = mlp_forward(params_tstudent, Xgrid)
mu_grid = np.asarray(out_grid[:, 0]) * y_sd + y_mean
sigma_grid = np.asarray(jnp.exp(0.5 * jnp.clip(out_grid[:, 1], -7.0, 7.0))) * y_sd

plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, s=8, alpha=0.3, label="datos de entrenamiento")
plt.plot(xx, f(xx), "k--", label="media verdadera")
plt.plot(xx, mu_grid, "C1", label="media predicha (MLP)")
plt.fill_between(
    xx.ravel(), mu_grid - 2 * sigma_grid, mu_grid + 2 * sigma_grid,
    color="C1", alpha=0.2, label="±2σ predicho"
)
plt.legend()
plt.title("MLP con verosimilitud t-Student entrenada con SGD")
plt.tight_layout()
plt.savefig("bayesian_neural_net.png", dpi=120)
print("Gráfica guardada en bayesian_neural_net.png")


# ---------------------------------------------------------------------------
# Modelo bayesiano: NUTS (littlemcmc) sobre los pesos de un MLP pequeño.
#
# Usamos una red más pequeña [1, 8, 3] (43 parámetros) para que el muestreo del
# posterior sea tratable. La verosimilitud es la misma t-Student y añadimos una
# previa N(0, 1) débilmente informativa sobre todos los pesos:
#     log p(w | datos)  =  sum_i  log t(y_i | mu(x_i;w), sigma(x_i;w), nu(x_i;w))
#                          - 0.5 * ||w||^2
# ---------------------------------------------------------------------------
PRIOR_SD = 1.0

key_b, random_key = jax.random.split(random_key)
params_small = init_mlp(key_b, [1, 64,64 , 3])
print("\nEntrenando MLP pequeño (SGD) para inicializar NUTS...")
params_small, _ = train(tstudent_loss, params_small, Xtr, ytr, n_epochs=8000, lr=1e-2)

# Aplanamos el árbol de parámetros a un vector (lo que NUTS muestrea).
flat0, unflatten = ravel_pytree(params_small)
model_ndim = int(flat0.size)


def log_posterior(flat):
    '''log-posterior no normalizado del MLP bayesiano.'''
    params = unflatten(flat)
    mu, sigma, nu = predict_params(params, Xtr)
    loglik = jnp.sum(tstudent_logpdf(ytr, mu, sigma, nu))
    logprior = -0.5 * jnp.sum(flat ** 2) / (PRIOR_SD ** 2)
    return loglik + logprior


# Contrato del backend JAX (nuts_jax): q (JAX array) -> (logp escalar, dlogp),
# todo en JAX (sin np.asarray) para que sea trazable por vmap+jit on-device.
logp_dlogp_func = value_and_grad(log_posterior)


print(f"\nMuestreando el posterior con NUTS-JAX vmapped ({model_ndim} parámetros)...")
trace, stats = sample_vmapped_nuts_chains(
    logp_dlogp_func,
    model_ndim=model_ndim,
    draws=400,
    tune=400,
    chains=2,
    start=flat0,   # (model_ndim,) se difunde a todas las cadenas
    random_seed=RANDOM_STATE,
)
samples = jnp.asarray(trace.reshape(-1, model_ndim))   # (n_muestras, ndim)
n_div = int(np.sum(stats["diverging"]))
print(f"  muestras posteriores: {samples.shape[0]}  |  divergencias: {n_div}")


# --- Predicción posterior (posterior predictive) sobre la malla -------------
def grid_mu(flat):
    mu, _, _ = predict_params(unflatten(flat), Xgrid)
    return mu


post_mu_grid = vmap(grid_mu)(samples) * y_sd + y_mean        # (S, G) en unidades reales
bayes_mean_grid = np.asarray(jnp.mean(post_mu_grid, axis=0))
bayes_lo_grid = np.asarray(jnp.percentile(post_mu_grid, 2.5, axis=0))
bayes_hi_grid = np.asarray(jnp.percentile(post_mu_grid, 97.5, axis=0))


# ---------------------------------------------------------------------------
# Comparación cuantitativa en el conjunto de prueba
# ---------------------------------------------------------------------------
def rmse(pred_real):
    return float(np.sqrt(np.mean((np.asarray(pred_real) - y_test) ** 2)))


# (1) MLP determinista grande (SGD, [1,64,64,3]).
mu_det, sigma_det, nu_det = predict_params(params_tstudent, Xte)
det_rmse = rmse(np.asarray(mu_det) * y_sd + y_mean)
det_ll = float(jnp.mean(tstudent_logpdf(yte, mu_det, sigma_det, nu_det)))

# (2) MLP determinista pequeño (SGD, [1,8,3]) = la moda del modelo bayesiano.
mu_map, sigma_map, nu_map = predict_params(params_small, Xte)
map_rmse = rmse(np.asarray(mu_map) * y_sd + y_mean)
map_ll = float(jnp.mean(tstudent_logpdf(yte, mu_map, sigma_map, nu_map)))


# (3) Modelo bayesiano (NUTS): media y verosimilitud predictiva posterior.
def test_pred(flat):
    mu, sigma, nu = predict_params(unflatten(flat), Xte)
    return mu, tstudent_logpdf(yte, mu, sigma, nu)


post_mu_test, post_lp_test = vmap(test_pred)(samples)        # (S, N_test) cada uno
bayes_mu_test = jnp.mean(post_mu_test, axis=0) * y_sd + y_mean
bayes_rmse = rmse(bayes_mu_test)
# log-verosimilitud predictiva: log( (1/S) sum_s p_s(y) ) promediada sobre puntos
S = samples.shape[0]
bayes_ll = float(jnp.mean(jax.scipy.special.logsumexp(post_lp_test, axis=0) - jnp.log(S)))

print("\n=== Comparación en el conjunto de prueba ===")
print(f"{'Modelo':<34}{'RMSE':>10}{'log-lik/pto':>14}")
print(f"{'MLP determinista [1,64,64,3] (SGD)':<34}{det_rmse:>10.4f}{det_ll:>14.4f}")
print(f"{'MLP determinista [1,8,3]   (SGD/MAP)':<34}{map_rmse:>10.4f}{map_ll:>14.4f}")
print(f"{'Bayesiano [1,8,3]          (NUTS)':<34}{bayes_rmse:>10.4f}{bayes_ll:>14.4f}")
print("(log-lik mayor es mejor; el bayesiano integra la incertidumbre de los pesos)")


# ---------------------------------------------------------------------------
# Gráfica comparativa: determinista vs bayesiano (NUTS)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, s=8, alpha=0.2, color="gray", label="datos")
plt.plot(xx, f(xx), "k--", label="media verdadera")
plt.plot(xx, mu_grid, "C1", label="MLP determinista (SGD)")
plt.plot(xx, bayes_mean_grid, "C0", label="Bayesiano NUTS (media posterior)")
plt.fill_between(
    xx.ravel(), bayes_lo_grid, bayes_hi_grid,
    color="C0", alpha=0.25, label="IC 95% (incertidumbre epistémica)"
)
plt.legend()
plt.title("Determinista (SGD) vs Bayesiano (NUTS) — verosimilitud t-Student")
plt.tight_layout()
plt.savefig("bayesian_neural_net_nuts.png", dpi=120)
print("\nGráfica comparativa guardada en bayesian_neural_net_nuts.png")
