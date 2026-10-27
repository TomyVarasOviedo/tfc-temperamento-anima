"""Fase 3 — Modelo heterocedástico.

NO desarrollar de cero: portar desde el repo del módulo de voz de Diego
manteniendo esta interfaz. Lo único que cambia es qué entra en X.

La línea base de temperamento es solo intercepto, no "sexo y edad": en
personalidad esas variables explican casi nada y la comparación honesta
es puntaje poligénico contra media constante. Está parametrizado para
que la decisión quede explícita y documentada, no escondida en el código.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np


class ProbabilisticModel(Protocol):
    """Interfaz común a los tres módulos. No cambiar sin avisar al grupo."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ProbabilisticModel": ...

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Devuelve (mu, sigma) por observación."""
        ...


class InterceptOnly:
    """Línea base de temperamento: media y desvío constantes.

    Es el modelo contra el que se mide ΔR². Si el puntaje poligénico no
    le gana a esto, el resultado del módulo es el nulo esperado.
    """

    def __init__(self) -> None:
        self.mu_: float | None = None
        self.sigma_: float | None = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "InterceptOnly":
        self.mu_ = float(np.mean(y))
        self.sigma_ = float(np.std(y, ddof=1))
        return self

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        n = len(X)
        return np.full(n, self.mu_), np.full(n, self.sigma_)


class HeteroscedasticModel:
    """Modelo con media y varianza condicionadas al predictor.

    TODO (Tomás): portar la implementación de tfc-anima-voz. Mantener la
    firma. Conservar el criterio de parada y el de regularización que ya
    estén validados allá, y anotar en el informe cualquier cambio.
    """

    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HeteroscedasticModel":
        raise NotImplementedError("Portar desde tfc-anima-voz (módulo de Diego)")

    def predict_mean_std(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError("Portar desde tfc-anima-voz (módulo de Diego)")
