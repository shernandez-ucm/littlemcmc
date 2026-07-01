import numpy as np
import matplotlib.pyplot as plt
import arviz as az

import jax
# NUTS (littlemcmc) corre en numpy/float64: igualamos la precisión de JAX para
# que el logp y su gradiente sean estables en el muestreador.
#jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from jax import jit, vmap, value_and_grad
from jax.tree_util import tree_map
from jax.flatten_util import ravel_pytree
import flax.linen as nn

import distrax
# distrax no trae una t-Student nativa, así que envolvemos la de TFP sobre el
# sustrato JAX (trazable por vmap+jit y compatible con float64).
from tensorflow_probability.substrates import jax as tfp
tfd = tfp.distributions

from littlemcmc.nuts_jax import sample_vmapped_nuts_chains
from littlemcmc.hmc_jax import sample_vmapped_chains

# ---------------------------------------------------------------------------
# Variante con previa horseshoe (Carvalho, Polson & Scott) sobre los pesos.
#
# `bayesian_neural_net.py` usa una previa N(0, 1) débilmente informativa e
# idéntica para todos los pesos. Aquí la reemplazamos por la previa horseshoe
# global-local jerárquica tal como la formula Kohns & Szendrei, "Horseshoe
# prior Bayesian quantile regression" (J. R. Stat. Soc. C, 2024,
# https://academic.oup.com/jrsssc/article/73/1/193/7336940), ec. de la previa:
#     beta_j | lambda_j^2, nu^2 ~ N(0, lambda_j^2 * nu^2)
#     lambda_j ~ C+(0, 1)   (escala LOCAL, una por peso -> permite pesos != 0)
#     nu       ~ C+(0, 1)   (escala GLOBAL, una sola -> encoge la red entera)
# Las colas pesadas de la Half-Cauchy en lambda_j dejan "escapar" del encogimiento
# global a los pesos realmente relevantes, mientras que nu empuja al resto hacia
# cero: es la previa canónica para regularización/selección de variables
# bayesiana, aplicada aquí a los pesos de una NN pequeña en vez de a
# coeficientes de una regresión.
# ---------------------------------------------------------------------------

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
# MLP (flax.linen): cuerpo (capas ocultas ReLU) y cabeza (capa lineal final)
# como submódulos con nombre ('body'/'head'), para poder congelar el cuerpo y
# muestrear solo la cabeza en el esquema bayesiano de última capa más abajo.
# ---------------------------------------------------------------------------
HIDDEN_SIZES = (64, 64)
OUT_SIZE = 3


class MLPBody(nn.Module):
    '''Capas ocultas ReLU; produce las características phi(x).'''
    hidden_sizes: tuple

    @nn.compact
    def __call__(self, x):
        for h in self.hidden_sizes:
            x = nn.relu(nn.Dense(h, kernel_init=jax.nn.initializers.he_normal())(x))
        return x


class MLPHead(nn.Module):
    '''Capa lineal final: phi(x) -> (mu, log(sigma^2), nu_raw).'''
    out_size: int

    @nn.compact
    def __call__(self, phi):
        return nn.Dense(self.out_size, kernel_init=jax.nn.initializers.he_normal())(phi)


class MLP(nn.Module):
    '''MLP completo = MLPBody + MLPHead, con submódulos nombrados 'body'/'head'.'''
    hidden_sizes: tuple
    out_size: int

    def setup(self):
        self.body = MLPBody(self.hidden_sizes)
        self.head = MLPHead(self.out_size)

    def __call__(self, x):
        return self.head(self.body(x))


mlp = MLP(hidden_sizes=HIDDEN_SIZES, out_size=OUT_SIZE)
mlp_body = MLPBody(hidden_sizes=HIDDEN_SIZES)
mlp_head = MLPHead(out_size=OUT_SIZE)


def init_mlp(key, in_dim=1):
    '''Inicializa los parámetros de `mlp` (dict anidado {'body': ..., 'head': ...}).'''
    return mlp.init(key, jnp.zeros((1, in_dim)))["params"]


def mlp_forward(params, x):
    '''Pasada hacia adelante del MLP completo (flax `nn.Module.apply`).'''
    return mlp.apply({"params": params}, x)


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


def tstudent_dist(mu, sigma, nu):
    '''Distribución t-Student (location-scale) de distrax (envuelve TFP).'''
    return distrax.as_distribution(tfd.StudentT(df=nu, loc=mu, scale=sigma))


def tstudent_loss(params, x, y):
    '''NLL de una t de Student (location-scale) con cabeza heterocedástica.'''
    out = mlp_forward(params, x)               # (N, 3): mu, log(sigma^2), nu_raw
    mu = out[:, 0]
    log_var = jnp.clip(out[:, 1], -7.0, 7.0)   # estabilidad numérica
    nu = 2.0 + jax.nn.softplus(out[:, 2])      # grados de libertad nu > 2 (varianza finita)
    sigma = jnp.exp(0.5 * log_var)
    # NLL de la t de Student: colas más pesadas -> robusta a outliers
    return -jnp.mean(tstudent_dist(mu, sigma, nu).log_prob(y))


def predict_params(params, x):
    '''Devuelve (mu, sigma, nu) de la cabeza t-Student en espacio estandarizado.'''
    out = mlp_forward(params, x)
    mu = out[:, 0]
    sigma = jnp.exp(0.5 * jnp.clip(out[:, 1], -7.0, 7.0))
    nu = 2.0 + jax.nn.softplus(out[:, 2])
    return mu, sigma, nu


def mlp_features(body_params, x):
    '''Extractor de características: `MLPBody` (todas las capas menos la última).

    Devuelve phi(x), las activaciones que alimentan la capa lineal final. En el
    esquema bayesiano de última capa, este cuerpo queda congelado en su valor MAP.
    '''
    return mlp_body.apply({"params": body_params}, x)


def predict_params_last(head_params, phi):
    '''(mu, sigma, nu) a partir de `MLPHead` (capa lineal final) sobre características phi.'''
    out = mlp_head.apply({"params": head_params}, phi)
    mu = out[:, 0]
    sigma = jnp.exp(0.5 * jnp.clip(out[:, 1], -7.0, 7.0))
    nu = 2.0 + jax.nn.softplus(out[:, 2])
    return mu, sigma, nu


def tstudent_logpdf(y, mu, sigma, nu):
    '''log-densidad t-Student (location-scale) por punto (vía distrax).'''
    return tstudent_dist(mu, sigma, nu).log_prob(y)


def tstudent_rvs(key, mu, sigma, nu, sample_shape=()):
    '''Muestras t-Student (location-scale) en float32, vía ``jax.random.t``.

    El muestreador de la t de TFP/distrax cae internamente a un gamma que castea
    a float64 (avisos de truncado con x64 desactivado); ``jax.random.t`` respeta
    el dtype de entrada, así que muestreamos a mano en float32.
    '''
    t = jax.random.t(key, nu, shape=tuple(sample_shape) + mu.shape)
    return mu + sigma * t


# ---------------------------------------------------------------------------
# Previa horseshoe (no centrada): theta = [z, eta_lambda, eta_tau] ->
# w = z * exp(eta_lambda) * exp(eta_tau). El muestreador nunca ve lambda/tau
# directamente (siempre positivos): sample en el logaritmo evita la geometría
# de "embudo" que rompería a HMC/NUTS si se intentara centrar la previa.
# ---------------------------------------------------------------------------
def log_half_cauchy(eta):
    '''log p(eta) cuando lambda = exp(eta) ~ Half-Cauchy(0, 1).

    p(lambda) = 2 / (pi * (1 + lambda^2)); con el jacobiano de eta = log(lambda)
    (d lambda/d eta = lambda) los términos en lambda se cancelan y queda
    log p(eta) = log(2/pi) + eta - softplus(2 eta), estable en ambas colas.
    '''
    return jnp.log(2.0 / jnp.pi) + eta - jax.nn.softplus(2.0 * eta)


def unpack_horseshoe(theta, n_weights):
    '''Separa el vector plano muestreado en (z, eta_lambda, eta_tau).'''
    z = theta[:n_weights]
    eta_lambda = theta[n_weights:2 * n_weights]
    eta_tau = theta[2 * n_weights]
    return z, eta_lambda, eta_tau


def horseshoe_weights(theta, n_weights):
    '''Pesos w = z * lambda * tau (parametrización no centrada del horseshoe).'''
    z, eta_lambda, eta_tau = unpack_horseshoe(theta, n_weights)
    return z * jnp.exp(eta_lambda) * jnp.exp(eta_tau)


def horseshoe_logprior(theta, n_weights):
    '''log p(z, lambda, tau): N(0,1) en z y Half-Cauchy(0,1) en cada lambda_j y en tau.'''
    z, eta_lambda, eta_tau = unpack_horseshoe(theta, n_weights)
    logp_z = -0.5 * jnp.sum(z ** 2)
    logp_lambda = jnp.sum(log_half_cauchy(eta_lambda))
    logp_tau = log_half_cauchy(eta_tau)
    return logp_z + logp_lambda + logp_tau


def init_horseshoe_theta(flat_weights):
    '''theta0 tal que w(theta0) == flat_weights (lambda_j = tau = 1 al inicio).'''
    n_weights = flat_weights.size
    return jnp.concatenate([flat_weights, jnp.zeros(n_weights), jnp.zeros(1)])


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
# Tensores JAX estandarizados para el entrenamiento (float32: x64 está desactivado)
Xtr = jnp.asarray(standardize_X(X_train), dtype=jnp.float32)
ytr = jnp.asarray((y_train - y_mean) / y_sd, dtype=jnp.float32)
Xte = jnp.asarray(standardize_X(X_test), dtype=jnp.float32)
yte = jnp.asarray((y_test - y_mean) / y_sd, dtype=jnp.float32)

# Malla ordenada para graficar las curvas suaves
xx = np.linspace(0.0, 10.0, 400).reshape(-1, 1)
Xgrid = jnp.asarray(standardize_X(xx), dtype=jnp.float32)


# ---------------------------------------------------------------------------
# Entrenamiento
# ---------------------------------------------------------------------------
random_key = jax.random.PRNGKey(RANDOM_STATE)
params_tstudent_key, random_key = jax.random.split(random_key)
params_tstudent = init_mlp(params_tstudent_key)
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
plt.savefig("bayesian_neural_net_horseshoe.png", dpi=120)
print("Gráfica guardada en bayesian_neural_net_horseshoe.png")


# ---------------------------------------------------------------------------
# Modelo bayesiano: NUTS (littlemcmc) sobre los pesos de un MLP pequeño, con
# previa horseshoe global-local en vez de la N(0, 1) del script original:
#     log p(w, lambda, tau | datos) =
#         sum_i log t(y_i | mu(x_i; w), sigma(x_i; w), nu(x_i; w))
#         - 0.5 ||z||^2 + sum_j log C+(lambda_j; 0, 1) + log C+(tau; 0, 1)
#     con w = z * lambda * tau (no centrada, ver comentarios arriba).
# El vector que efectivamente muestrea NUTS/HMC es theta = [z, eta_lambda, eta_tau],
# de dimensión 2 * model_ndim + 1 (el doble de parámetros que la previa gaussiana).
# ---------------------------------------------------------------------------
key_b, random_key = jax.random.split(random_key)
params_small = init_mlp(key_b)
print("\nEntrenando MLP pequeño (SGD) para inicializar NUTS...")
params_small, _ = train(tstudent_loss, params_small, Xtr, ytr, n_epochs=8000, lr=1e-2)

# Aplanamos el árbol de parámetros a un vector (los pesos w, no lo que NUTS muestrea).
flat0, unflatten = ravel_pytree(params_small)
model_ndim = int(flat0.size)                # nº de pesos w
theta_ndim = 2 * model_ndim + 1             # nº de parámetros que muestrea NUTS/HMC
theta0 = init_horseshoe_theta(flat0)


def log_posterior(theta):
    '''log-posterior no normalizado del MLP bayesiano con previa horseshoe.'''
    flat = horseshoe_weights(theta, model_ndim)
    params = unflatten(flat)
    mu, sigma, nu = predict_params(params, Xtr)
    loglik = jnp.sum(tstudent_logpdf(ytr, mu, sigma, nu))
    return loglik + horseshoe_logprior(theta, model_ndim)


# Contrato del backend JAX (nuts_jax): q (JAX array) -> (logp escalar, dlogp),
# todo en JAX (sin np.asarray) para que sea trazable por vmap+jit on-device.
logp_dlogp_func = value_and_grad(log_posterior)


print(f"\nMuestreando el posterior con NUTS-JAX vmapped "
      f"({model_ndim} pesos, {theta_ndim} parámetros con horseshoe)...")
trace, stats = sample_vmapped_nuts_chains(
    logp_dlogp_func,
    model_ndim=theta_ndim,
    draws=400,
    tune=400,
    chains=2,
    start=theta0,   # (theta_ndim,) se difunde a todas las cadenas
    random_seed=RANDOM_STATE,
)
theta_samples = jnp.asarray(trace.reshape(-1, theta_ndim))                     # (n_muestras, theta_ndim)
samples = vmap(lambda th: horseshoe_weights(th, model_ndim))(theta_samples)    # (n_muestras, model_ndim)
n_div = int(np.sum(stats["diverging"]))
print(f"  muestras posteriores: {samples.shape[0]}  |  divergencias: {n_div}")


print(f"\nMuestreando el posterior con HMC-JAX vmapped "
      f"({model_ndim} pesos, {theta_ndim} parámetros con horseshoe)...")
trace_hmc, stats_hmc = sample_vmapped_chains(
    logp_dlogp_func,
    model_ndim=theta_ndim,
    draws=400,
    tune=400,
    chains=2,
    n_leapfrog=16,
    start=theta0,   # (theta_ndim,) se difunde a todas las cadenas
    random_seed=RANDOM_STATE,
)
theta_samples_hmc = jnp.asarray(trace_hmc.reshape(-1, theta_ndim))
samples_hmc = vmap(lambda th: horseshoe_weights(th, model_ndim))(theta_samples_hmc)
n_div_hmc = int(np.sum(stats_hmc["diverging"]))
print(f"  muestras posteriores: {samples_hmc.shape[0]}  |  divergencias: {n_div_hmc}")


def print_convergence(name, trace_chains):
    '''Resume R-hat y ESS (bulk, vía arviz) sobre todos los parámetros de `trace_chains`.

    `trace_chains` tiene forma (chains, draws, ndim); con cientos/miles de parámetros
    reportamos min/media/máx de cada diagnóstico en vez de una tabla por parámetro.
    '''
    idata = az.from_dict({"posterior": {"w": np.asarray(trace_chains)}})
    rhat = az.rhat(idata, var_names=["w"])["w"].values
    ess = az.ess(idata, var_names=["w"])["w"].values
    print(f"  {name}: R-hat  min={rhat.min():.4f}  media={rhat.mean():.4f}  max={rhat.max():.4f}")
    print(f"  {name}: ESS    min={ess.min():.1f}  media={ess.mean():.1f}  max={ess.max():.1f}")


print(f"\nDiagnósticos de convergencia (R-hat, ESS) sobre los {theta_ndim} "
      f"parámetros muestreados (theta = [z, eta_lambda, eta_tau]):")
print_convergence("NUTS", trace)
print_convergence("HMC ", trace_hmc)


def print_shrinkage(name, theta_samples_, n_weights):
    '''Resume la escala global (tau) y local (lambda_j) estimadas por el horseshoe.

    tau chico => la red completa se encoge hacia 0; lambda_j >> 1 para un peso
    puntual => ese peso "escapa" del encogimiento global (señal real detectada).
    '''
    tau = np.asarray(jnp.exp(theta_samples_[:, 2 * n_weights]))
    lam = np.asarray(jnp.exp(theta_samples_[:, n_weights:2 * n_weights]))
    frac_shrunk = float(np.mean((lam * tau[:, None]) < 0.1))
    print(f"  {name}: tau (global)     mediana={np.median(tau):.4f}  "
          f"[{np.percentile(tau, 5):.4f}, {np.percentile(tau, 95):.4f}] (IC90%)")
    print(f"  {name}: lambda_j*tau     mediana={np.median(lam * tau[:, None]):.4f}  "
          f"|  fracción de pesos con lambda_j*tau < 0.1: {frac_shrunk:.2%}")


print("\nEncogimiento horseshoe (posterior de tau y lambda_j * tau):")
print_shrinkage("NUTS", theta_samples, model_ndim)
print_shrinkage("HMC ", theta_samples_hmc, model_ndim)


# ---------------------------------------------------------------------------
# Bayesiano de última capa (HMC-JAX): congelamos el cuerpo MAP (todas las capas
# menos la última) como extractor de características fijo y muestreamos SOLO la
# última capa lineal, también con previa horseshoe. Es mucho más barato
# (2*195+1 vs 2*4483+1 parámetros) y captura la mayor parte de la incertidumbre
# predictiva (last-layer Laplace/Bayes) mientras muestra el mismo mecanismo de
# encogimiento a una escala tratable.
#     log p(W_L, b_L, lambda, tau | datos) =
#         sum_i log t(y_i | head(phi(x_i); W_L, b_L)) + previa horseshoe
#     con phi(x) = cuerpo MAP congelado, w_L = z_L * lambda_L * tau_L.
# ---------------------------------------------------------------------------
body_params = params_small["body"]       # capas congeladas (valor MAP)
last_params = params_small["head"]       # cabeza (MLPHead): lo único que se muestrea
flat0_last, unflatten_last = ravel_pytree(last_params)
model_ndim_last = int(flat0_last.size)
theta_ndim_last = 2 * model_ndim_last + 1
theta0_last = init_horseshoe_theta(flat0_last)

# Características de entrenamiento, congeladas (sin gradiente hacia el cuerpo).
phi_tr = jax.lax.stop_gradient(mlp_features(body_params, Xtr))


def log_posterior_last(theta):
    '''log-posterior no normalizado sobre la última capa (cuerpo congelado), previa horseshoe.'''
    flat = horseshoe_weights(theta, model_ndim_last)
    head_params = unflatten_last(flat)
    mu, sigma, nu = predict_params_last(head_params, phi_tr)
    loglik = jnp.sum(tstudent_logpdf(ytr, mu, sigma, nu))
    return loglik + horseshoe_logprior(theta, model_ndim_last)


logp_dlogp_last = value_and_grad(log_posterior_last)

print(f"\nMuestreando SOLO la última capa con HMC-JAX vmapped "
      f"({model_ndim_last} pesos de {model_ndim}, {theta_ndim_last} parámetros con horseshoe)...")
trace_ll, stats_ll = sample_vmapped_chains(
    logp_dlogp_last,
    model_ndim=theta_ndim_last,
    draws=400,
    tune=400,
    chains=2,
    n_leapfrog=16,
    start=theta0_last,   # (theta_ndim_last,) se difunde a todas las cadenas
    random_seed=RANDOM_STATE,
)
theta_samples_ll = jnp.asarray(trace_ll.reshape(-1, theta_ndim_last))
samples_ll = vmap(lambda th: horseshoe_weights(th, model_ndim_last))(theta_samples_ll)
n_div_ll = int(np.sum(stats_ll["diverging"]))
print(f"  muestras posteriores: {samples_ll.shape[0]}  |  divergencias: {n_div_ll}")
print_shrinkage("HMC última capa", theta_samples_ll, model_ndim_last)

# Características congeladas en malla y prueba (reutilizadas por las métricas/plot).
phi_grid = jax.lax.stop_gradient(mlp_features(body_params, Xgrid))
phi_te = jax.lax.stop_gradient(mlp_features(body_params, Xte))


# --- Predicción posterior (posterior predictive) sobre la malla -------------
def grid_mu(flat):
    mu, _, _ = predict_params(unflatten(flat), Xgrid)
    return mu


post_mu_grid = vmap(grid_mu)(samples) * y_sd + y_mean        # (S, G) en unidades reales
bayes_mean_grid = np.asarray(jnp.mean(post_mu_grid, axis=0))
bayes_lo_grid = np.asarray(jnp.percentile(post_mu_grid, 2.5, axis=0))
bayes_hi_grid = np.asarray(jnp.percentile(post_mu_grid, 97.5, axis=0))

post_mu_grid_hmc = vmap(grid_mu)(samples_hmc) * y_sd + y_mean
hmc_mean_grid = np.asarray(jnp.mean(post_mu_grid_hmc, axis=0))


def grid_mu_last(flat):
    head_params = unflatten_last(flat)
    mu, _, _ = predict_params_last(head_params, phi_grid)
    return mu


post_mu_grid_ll = vmap(grid_mu_last)(samples_ll) * y_sd + y_mean
ll_mean_grid = np.asarray(jnp.mean(post_mu_grid_ll, axis=0))


# ---------------------------------------------------------------------------
# Comparación cuantitativa en el conjunto de prueba
# ---------------------------------------------------------------------------
def rmse(pred_real):
    return float(np.sqrt(np.mean((np.asarray(pred_real) - y_test) ** 2)))


def coverage95(pred_real):
    '''Cobertura empírica del IC predictivo central del 95% (ideal: 0.95).'''
    qlo = jnp.percentile(pred_real, 2.5, axis=0)
    qhi = jnp.percentile(pred_real, 97.5, axis=0)
    return float(jnp.mean((yte_real >= qlo) & (yte_real <= qhi)))


def crps(pred_real):
    '''CRPS medio estimado de muestras predictivas (estimador ordenado, O(M log M)).

    CRPS(F, y) = E|X - y| - 0.5 E|X - X'|, menor es mejor; combina calibración
    y nitidez de la predictiva en una sola métrica (unidades de y).
    '''
    M = pred_real.shape[0]
    term1 = jnp.mean(jnp.abs(pred_real - yte_real[None, :]), axis=0)
    xs = jnp.sort(pred_real, axis=0)
    i = jnp.arange(1, M + 1)[:, None]
    term2 = jnp.sum((2.0 * i - M - 1.0) * xs, axis=0) / (M ** 2)
    return float(jnp.mean(term1 - term2))


N_PRED = 800   # muestras predictivas por punto para cobertura y CRPS
yte_real = jnp.asarray(y_test, dtype=jnp.float32)
key_pred, random_key = jax.random.split(random_key)


def det_pred_samples(params, key, n=N_PRED):
    '''Muestras de la predictiva (t-Student única) de un modelo determinista, unidades reales.'''
    mu, sigma, nu = predict_params(params, Xte)
    ys = tstudent_rvs(key, mu, sigma, nu, sample_shape=(n,))                 # (n, N_test)
    return ys * y_sd + y_mean


def bayes_pred_samples(post_samples, key):
    '''Muestras de la predictiva posterior (una y por muestra posterior y punto), unidades reales.'''
    keys = jax.random.split(key, post_samples.shape[0])

    def one(flat, k):
        mu, sigma, nu = predict_params(unflatten(flat), Xte)
        return tstudent_rvs(k, mu, sigma, nu)

    ys = vmap(one)(post_samples, keys)                                       # (S, N_test)
    return ys * y_sd + y_mean


# (1) MLP determinista grande (SGD, [1,64,64,3]).
mu_det, sigma_det, nu_det = predict_params(params_tstudent, Xte)
det_rmse = rmse(np.asarray(mu_det) * y_sd + y_mean)
det_ll = float(jnp.mean(tstudent_logpdf(yte, mu_det, sigma_det, nu_det)))
k1, key_pred = jax.random.split(key_pred)
det_pred = det_pred_samples(params_tstudent, k1)
det_cov, det_crps = coverage95(det_pred), crps(det_pred)

# (2) MLP determinista pequeño (SGD/MAP) = la moda de la parte de verosimilitud
# del modelo bayesiano (no coincide con la moda de la previa horseshoe, que
# encoge hacia 0; se reporta igualmente como referencia determinista).
mu_map, sigma_map, nu_map = predict_params(params_small, Xte)
map_rmse = rmse(np.asarray(mu_map) * y_sd + y_mean)
map_ll = float(jnp.mean(tstudent_logpdf(yte, mu_map, sigma_map, nu_map)))
k2, key_pred = jax.random.split(key_pred)
map_pred = det_pred_samples(params_small, k2)
map_cov, map_crps = coverage95(map_pred), crps(map_pred)


# (3)/(4)/(5) Modelos bayesianos (NUTS, HMC, última capa): media y verosimilitud
# predictiva posterior. `test_pred_fn` y `pred_samples_fn` desacoplan la red
# completa del esquema de última capa (cuerpo congelado).
def test_pred(flat):
    mu, sigma, nu = predict_params(unflatten(flat), Xte)
    return mu, tstudent_logpdf(yte, mu, sigma, nu)


def test_pred_last(flat):
    head_params = unflatten_last(flat)
    mu, sigma, nu = predict_params_last(head_params, phi_te)
    return mu, tstudent_logpdf(yte, mu, sigma, nu)


def lastlayer_pred_samples(post_samples, key):
    '''Predictiva posterior de última capa (una y por muestra y punto), unidades reales.'''
    keys = jax.random.split(key, post_samples.shape[0])

    def one(flat, k):
        head_params = unflatten_last(flat)
        mu, sigma, nu = predict_params_last(head_params, phi_te)
        return tstudent_rvs(k, mu, sigma, nu)

    ys = vmap(one)(post_samples, keys)                                       # (S, N_test)
    return ys * y_sd + y_mean


def bayes_metrics(post_samples, key, test_pred_fn=test_pred,
                  pred_samples_fn=bayes_pred_samples):
    post_mu_test, post_lp_test = vmap(test_pred_fn)(post_samples)   # (S, N_test) cada uno
    mu_test = jnp.mean(post_mu_test, axis=0) * y_sd + y_mean
    S = post_samples.shape[0]
    # log-verosimilitud predictiva: log( (1/S) sum_s p_s(y) ) promediada sobre puntos
    ll = float(jnp.mean(jax.scipy.special.logsumexp(post_lp_test, axis=0) - jnp.log(S)))
    pred = pred_samples_fn(post_samples, key)
    return rmse(mu_test), ll, coverage95(pred), crps(pred)


k3, key_pred = jax.random.split(key_pred)
nuts_rmse, nuts_ll, nuts_cov, nuts_crps = bayes_metrics(samples, k3)
k4, key_pred = jax.random.split(key_pred)
hmc_rmse, hmc_ll, hmc_cov, hmc_crps = bayes_metrics(samples_hmc, k4)
k5, key_pred = jax.random.split(key_pred)
ll_rmse, ll_ll, ll_cov, ll_crps = bayes_metrics(
    samples_ll, k5, test_pred_fn=test_pred_last, pred_samples_fn=lastlayer_pred_samples
)

print("\n=== Comparación en el conjunto de prueba (previa horseshoe) ===")
hdr = f"{'Modelo':<38}{'RMSE':>9}{'log-lik/pto':>13}{'cob.95%':>10}{'CRPS':>9}"
print(hdr)
print(f"{'MLP determinista [1,64,64,3] (SGD)':<38}{det_rmse:>9.4f}{det_ll:>13.4f}{det_cov:>10.3f}{det_crps:>9.4f}")
print(f"{'MLP determinista [1,64,64,3] (MAP)':<38}{map_rmse:>9.4f}{map_ll:>13.4f}{map_cov:>10.3f}{map_crps:>9.4f}")
print(f"{'Bayesiano horseshoe [1,64,64,3] (NUTS)':<38}{nuts_rmse:>9.4f}{nuts_ll:>13.4f}{nuts_cov:>10.3f}{nuts_crps:>9.4f}")
print(f"{'Bayesiano horseshoe [1,64,64,3] (HMC)':<38}{hmc_rmse:>9.4f}{hmc_ll:>13.4f}{hmc_cov:>10.3f}{hmc_crps:>9.4f}")
print(f"{'Bayesiano horseshoe última capa (HMC, {n} par.)'.format(n=model_ndim_last):<38}{ll_rmse:>9.4f}{ll_ll:>13.4f}{ll_cov:>10.3f}{ll_crps:>9.4f}")
print("(log-lik mayor es mejor; CRPS menor es mejor; cobertura 95% ideal ≈ 0.95)")


# ---------------------------------------------------------------------------
# Gráfica comparativa: determinista vs bayesiano (NUTS, previa horseshoe)
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 5))
plt.scatter(X_train, y_train, s=8, alpha=0.2, color="gray", label="datos")
plt.plot(xx, f(xx), "k--", label="media verdadera")
plt.plot(xx, mu_grid, "C1", label="MLP determinista (SGD)")
plt.plot(xx, bayes_mean_grid, "C0", label="Bayesiano NUTS horseshoe (media posterior)")
plt.plot(xx, hmc_mean_grid, "C2", label="Bayesiano HMC horseshoe (media posterior)")
plt.plot(xx, ll_mean_grid, "C3", label="Bayesiano última capa horseshoe (media posterior)")
plt.fill_between(
    xx.ravel(), bayes_lo_grid, bayes_hi_grid,
    color="C0", alpha=0.25, label="IC 95% NUTS (incertidumbre epistémica)"
)
plt.legend()
plt.title("Determinista (SGD) vs Bayesiano horseshoe (NUTS/HMC/última capa) — t-Student")
plt.tight_layout()
plt.savefig("bayesian_neural_net_horseshoe_nuts.png", dpi=120)
print("\nGráfica comparativa guardada en bayesian_neural_net_horseshoe_nuts.png")
