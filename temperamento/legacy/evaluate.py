"""Fase 3 — Métricas de evaluación.

Son las mismas tres que reportan rostro y voz, con la misma definición,
porque la tabla comparativa final los pone lado a lado.
"""

from __future__ import annotations

import numpy as np


def r2(y: np.ndarray, pred: np.ndarray) -> float:
    """R² clásico: 1 - SSE/SST."""
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - sse / sst


def delta_r2(
    y: np.ndarray,
    pred_base: np.ndarray,
    pred_full: np.ndarray,
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, tuple[float, float]]:
    """ΔR² del modelo completo sobre la línea base, con IC bootstrap.

    El IC es lo que decide el resultado del módulo: si incluye el cero,
    el genotipo no aporta información por encima de la línea base. Para
    que esa conclusión valga, el ΔR² mínimo detectable del diseño tiene
    que estar por debajo del h² de la literatura (ver scripts/power_analysis.py).

    Returns:
        (delta_r2 puntual, (límite inferior, límite superior)).
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    punto = r2(y, pred_full) - r2(y, pred_base)

    muestras = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        muestras[b] = r2(y[idx], pred_full[idx]) - r2(y[idx], pred_base[idx])

    lo, hi = np.quantile(muestras, [alpha / 2, 1 - alpha / 2])
    return punto, (float(lo), float(hi))


def coverage(y: np.ndarray, lo: np.ndarray, hi: np.ndarray) -> float:
    """Fracción de observaciones dentro del intervalo.

    Se compara contra el nivel nominal con la tolerancia de ±5 pp que fija
    la idea directriz (ver contract.TraitResult).
    """
    return float(np.mean((y >= lo) & (y <= hi)))


def crps_gaussian(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """CRPS medio bajo predictiva normal (forma cerrada).

    Penaliza a la vez el sesgo y la sobreconfianza, por eso la hipótesis
    exige mejora de CRPS además de cobertura: un modelo puede tener
    cobertura correcta ensanchando los intervalos sin informar nada.
    """
    from scipy.stats import norm

    z = (y - mu) / sigma
    return float(np.mean(sigma * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))))


def crps_gain(
    y: np.ndarray,
    mu_base: np.ndarray,
    sigma_base: np.ndarray,
    mu_full: np.ndarray,
    sigma_full: np.ndarray,
) -> float:
    """Mejora relativa de CRPS sobre la línea base. Positivo = mejora."""
    base = crps_gaussian(y, mu_base, sigma_base)
    full = crps_gaussian(y, mu_full, sigma_full)
    return (base - full) / base
