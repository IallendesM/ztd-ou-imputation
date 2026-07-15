"""
Imputacion de datos faltantes en series de tiempo, asumiendo que el proceso
subyacente es un Ornstein-Uhlenbeck (OU), via un algoritmo de Maximizacion de
Esperanza (EM):

- Paso E: los segmentos faltantes se rellenan simulando "puentes" del proceso OU.
- Paso M: se re-estiman (mu, sigma) con los datos ya rellenados (ver ou_estimation.py).

Contexto: la aplicacion original es rellenar series de Zenith Total Delay (ZTD)
de estaciones GNSS en Chile central y sur (CR2 / FONDECYT 11230184).

Desarrollado durante la practica en CR2 / FONDECYT 11230184.
"""

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
from scipy.stats import chi2

from ou_estimation import mu_sigma_mle


def separate_by_continuity(arr: np.ndarray) -> list:
    """Separa un array de indices en sub-arrays de indices contiguos."""
    idx_discontinuous = np.where(np.diff(arr) != 1)[0] + 1
    return np.split(arr, idx_discontinuous)


def grad_x_log_p(t: float, x: float, y: float, mu: float, sigma: float) -> float:
    """Gradiente respecto a x del log del kernel de transicion del proceso OU.

    Se usa para simular puentes del proceso condicionados a un punto final.
    """
    return (
        2 * mu * (y - x * np.exp(-mu * t)) * np.exp(-mu * t)
        / ((sigma ** 2) * (1 - np.exp(-2 * mu * t)))
    )


def euler_maruyama_bridge_ou(
    X_0: float, X_n: float, n: int, dt: float, mu: float, sigma: float
) -> np.ndarray:
    """Simula un puente de un proceso OU entre dos valores conocidos (X_0, X_n).

    Args:
        X_0: valor inicial (conocido).
        X_n: valor final (conocido).
        n: numero de pasos, incluyendo extremos.
        dt: espaciado temporal.
        mu: parametro OU.
        sigma: parametro OU.

    Returns:
        Array de largo n con el puente simulado (X[0] == X_0, X[-1] == X_n).
    """
    X = np.zeros(n)
    X[0] = X_0
    X[n - 1] = X_n
    for i in range(n - 2):
        dBt = np.random.normal(0, np.sqrt(dt))
        X[i + 1] = (
            X[i]
            - mu * X[i] * dt
            + (sigma ** 2) * grad_x_log_p((n - i - 1) * dt, X[i], X[n - 1], mu, sigma) * dt
            + sigma * dBt
        )
    return X


def p_val_ou(X: np.ndarray, dt: float, mu: float, sigma: float) -> tuple:
    """Test de bondad de ajuste para H0: X ~ OU(mu, sigma).

    Bajo H0, el estadistico T definido abajo se distribuye chi-cuadrado con
    n-1 grados de libertad.

    Returns:
        (p_val, T)
    """
    T = np.sum(((np.diff(X) + mu * X[:-1] * dt) / sigma) ** 2 / dt)
    n = X.shape[0] - 1
    p_val = 1 - chi2.cdf(T, df=n)
    return p_val, T


def expectation_maximization(
    df1: pd.DataFrame,
    mu_0: float,
    sigma_0: float,
    dt: float,
    n_steps: int,
    mu_sigma_estimator=mu_sigma_mle,
    tol: float = 1e-4,
    window_length: int = 24 * 30,
    polyorder: int = 4,
    extra_sim: int = 3,
    error_estimation: bool = True,
    error_iter: int = 10,
):
    """Rellena datos faltantes en una serie de tiempo asumiendo un proceso OU subyacente.

    Args:
        df1: DataFrame con columnas 'time' y 'Z' (Z puede tener NaNs).
        mu_0, sigma_0: valores iniciales de los parametros OU.
        dt: espaciado temporal entre observaciones (mismas unidades que 'time').
        n_steps: numero maximo de iteraciones del EM.
        mu_sigma_estimator: funcion callable (X, dt) -> (mu, sigma), por defecto MLE.
        tol: tolerancia para detener el EM.
        window_length, polyorder: hiperparametros del filtro Savitzky-Golay usado
            para remover la estacionalidad (Z -> X, ver ecuacion (2.1) del informe).
        extra_sim: numero de trayectorias adicionales simuladas al terminar el EM.
        error_estimation: si se estima el error de (mu, sigma) via normalidad asintotica.
        error_iter: numero de repeticiones usadas para estimar dicho error.

    Returns:
        mu, sigma: estimacion final de los parametros.
        mu_error, sigma_error: error estimado (solo si error_estimation=True).
        df_list: lista de DataFrames con las trayectorias adicionales rellenadas.
        p_val_list: p-valores de bondad de ajuste para cada trayectoria en df_list.
    """
    df = df1.copy()
    df["time"] = pd.to_datetime(df["time"])
    df["logZ"] = np.log(df["Z"])

    hour_mean = df.groupby(df["time"].dt.strftime("%m-%d %H"))["logZ"].mean()
    arr_filtered = savgol_filter(hour_mean.values, window_length, polyorder)
    df["c"] = df["time"].dt.strftime("%m-%d %H").map(dict(zip(hour_mean.index, arr_filtered)))
    df["X"] = df["logZ"] - df["c"]

    index_nans = np.where(pd.isnull(df["X"]))[0]
    sub_index_nans = separate_by_continuity(index_nans)

    mu_list, sigma_list = [mu_0], [sigma_0]
    j = 0
    tol_condition = True

    while tol_condition and j < n_steps:
        for index_na in sub_index_nans:
            X_0 = df["X"][index_na[0] - 1]
            X_f = df["X"][index_na[-1] + 1]
            X_fill = euler_maruyama_bridge_ou(
                X_0, X_f, index_na.shape[0] + 2, dt, mu_list[j], sigma_list[j]
            )
            df.loc[index_na, "X"] = X_fill[1:-1]

        df["logZ"] = df["X"] + df["c"]
        df["Z"] = np.exp(df["logZ"])

        X = df["X"].values
        mu, sigma = mu_sigma_estimator(X, dt)
        distance = np.sqrt((mu - mu_list[j]) ** 2 + (sigma - sigma_list[j]) ** 2)
        mu_list.append(mu)
        sigma_list.append(sigma)
        j += 1
        if distance <= tol:
            tol_condition = False

    mu, sigma = mu_list[-1], sigma_list[-1]

    if error_estimation:
        list_mu, list_sigma = [], []
        for _ in range(error_iter):
            for index_na in sub_index_nans:
                X_0 = df["X"][index_na[0] - 1]
                X_f = df["X"][index_na[-1] + 1]
                X_fill = euler_maruyama_bridge_ou(
                    X_0, X_f, index_na.shape[0] + 2, dt, mu, sigma
                )
                df.loc[index_na, "X"] = X_fill[1:-1]
            X = df["X"].values
            mu_hat, sigma_hat = mu_sigma_estimator(X, dt)
            list_mu.append(mu_hat)
            list_sigma.append(sigma_hat)
        mu, sigma = np.mean(list_mu), np.mean(list_sigma)
        mu_error, sigma_error = 2 * np.std(list_mu), 2 * np.std(list_sigma)

    df_list, p_val_list = [], []
    for _ in range(extra_sim):
        for index_na in sub_index_nans:
            X_0 = df["X"][index_na[0] - 1]
            X_f = df["X"][index_na[-1] + 1]
            X_fill = euler_maruyama_bridge_ou(X_0, X_f, index_na.shape[0] + 2, dt, mu, sigma)
            df.loc[index_na, "X"] = X_fill[1:-1]
        p_val, _ = p_val_ou(df["X"].values, dt, mu, sigma)
        df["logZ"] = df["X"] + df["c"]
        df["Z"] = np.exp(df["logZ"])
        df_list.append(df.copy())
        p_val_list.append(p_val)

    if error_estimation:
        return mu, mu_error, sigma, sigma_error, df_list, p_val_list
    return mu, sigma, df_list, p_val_list
