"""
Simulacion y estimacion de parametros para un proceso de Ornstein-Uhlenbeck (OU).

dX_t = -mu * X_t dt + sigma dB_t

Incluye:
- Simulacion via Euler-Maruyama
- Estimador de maxima verosimilitud (MLE)
- Estimador basado en la distancia de Wasserstein (comparado como alternativa al MLE)

Desarrollado durante la practica en CR2 / FONDECYT 11230184
("Atmospheric water vapor and precipitation processes in central and southern Chile").
"""

from functools import partial

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.stats import norm


def euler_maruyama_ou(n: int, dt: float, mu: float, sigma: float) -> np.ndarray:
    """Simula una trayectoria de un proceso Ornstein-Uhlenbeck via Euler-Maruyama.

    Args:
        n: numero de pasos.
        dt: espaciado temporal entre observaciones.
        mu: parametro de reversion a la media del proceso OU.
        sigma: parametro de volatilidad del proceso OU.

    Returns:
        Array de numpy con la trayectoria simulada.
    """
    X = np.zeros(n)
    X[0] = np.random.normal(0, sigma / np.sqrt(2 * mu))
    for i in range(n - 1):
        dBt = np.random.normal(0, np.sqrt(dt))
        X[i + 1] = X[i] - mu * X[i] * dt + sigma * dBt
    return X


# ---------------------------------------------------------------------------
# Estimador de maxima verosimilitud (MLE)
# ---------------------------------------------------------------------------

def _objective_function_mle(mu: float, X: np.ndarray, dt: float) -> float:
    """Funcion objetivo (log de suma de residuos) usada para el MLE de mu."""
    return np.log(np.sum((X[1:] - X[:-1] * np.exp(-mu * dt)) ** 2))


def mu_sigma_mle(X: np.ndarray, dt: float) -> list:
    """Estima (mu, sigma) de un proceso OU via maxima verosimilitud.

    Args:
        X: array de observaciones.
        dt: espaciado temporal entre observaciones.

    Returns:
        [mu_hat, sigma_hat]
    """
    objective_function = partial(_objective_function_mle, X=X, dt=dt)
    sol = minimize_scalar(method="bounded", bounds=(0, 10), fun=objective_function)
    mu_hat = sol.x
    n = X.shape[0]
    sigma_hat = np.sqrt(
        (2 * mu_hat / (n * (1 - np.exp(-2 * mu_hat * dt))))
        * np.sum((X[1:] - X[:-1] * np.exp(-mu_hat * dt)) ** 2)
    )
    return [mu_hat, sigma_hat]


# ---------------------------------------------------------------------------
# Estimador de Wasserstein (comparado como alternativa al MLE)
# ---------------------------------------------------------------------------

def empiric_cov(X: np.ndarray, k: int) -> float:
    """Covarianza empirica de X contra si mismo desplazado k pasos."""
    n = X.shape[0]
    X_bar = np.mean(X)
    return np.mean((X[: n - k] - X_bar) * (X[-(n - k):] - X_bar))


def c_x(X: np.ndarray) -> float:
    """Estadistico C_X usado en el estimador de Wasserstein (ver informe de practica)."""
    n = X.shape[0]
    X_tilde = np.sort(X)
    q = np.linspace(0, 1, n + 1)
    return np.sum(X_tilde * (norm.pdf(norm.ppf(q))[:-1] - norm.pdf(norm.ppf(q))[1:]))


def mu_sigma_wasserstein(X: np.ndarray, dt: float, k: int) -> tuple:
    """Estima (mu, sigma) minimizando una distancia de Wasserstein.

    Solucion cerrada obtenida resolviendo el problema de optimizacion
    restringida (ver informe de practica) via condiciones de Lagrange/KKT:

        mu_hat    = (2 / (k*dt)) * log( C_X / sqrt(Cov_k(X)) )
        sigma_hat = sqrt(2 * mu_hat) * C_X

    Nota: en la practica este estimador resulto menos estable que el MLE
    (ver conclusiones del informe): cuando C_X <= 0 o Cov_k(X) <= 0 el
    logaritmo/raiz no estan bien definidos, y mu_hat no es admisible.

    Args:
        X: array de observaciones.
        dt: espaciado temporal entre observaciones.
        k: numero de pasos de desplazamiento usado en la covarianza empirica.

    Returns:
        (mu_hat, sigma_hat)
    """
    cov_k = empiric_cov(X, k)
    cx = c_x(X)
    if cx <= 0 or cov_k <= 0:
        raise ValueError(
            "Estimador de Wasserstein no admisible: se requiere C_X > 0 y "
            f"Cov_k(X) > 0 (se obtuvo C_X={cx:.4f}, Cov_k(X)={cov_k:.4f})."
        )
    mu_hat = (2 / (k * dt)) * np.log(cx / np.sqrt(cov_k))
    sigma_hat = np.sqrt(2 * mu_hat) * cx
    return mu_hat, sigma_hat