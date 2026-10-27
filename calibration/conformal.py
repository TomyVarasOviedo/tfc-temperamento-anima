"""Fase 3 — Calibración conformal (split conformal).

NO desarrollar de cero: portar desde tfc-anima-voz manteniendo la firma.
El mismo calibrador tiene que servir a los tres módulos, porque la
comparabilidad de la cobertura entre dominios es el eje del argumento.
"""

from __future__ import annotations

import numpy as np

from ..models.heteroscedastic import ProbabilisticModel


class SplitConformal:
    """Envuelve un modelo y produce intervalos con cobertura garantizada.

    El conjunto de calibración tiene que ser disjunto del de entrenamiento:
    si se solapan, la cobertura sale optimista y el criterio de ±5 pp deja
    de significar algo.
    """

    def __init__(self, model: ProbabilisticModel, alpha: float = 0.10) -> None:
        self.model = model
        self.alpha = alpha
        self.q_: float | None = None

    @property
    def nominal(self) -> float:
        """Nivel nominal declarado, el que va al TraitResult."""
        return 1.0 - self.alpha

    def calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray) -> "SplitConformal":
        """Calcula el cuantil de los scores de no conformidad.

        TODO (Tomás): portar desde tfc-anima-voz. El score normalizado es
        |y - mu| / sigma, y q_ el cuantil (1-alpha) corregido por tamaño
        finito: ceil((n+1)(1-alpha)) / n.
        """
        raise NotImplementedError("Portar desde tfc-anima-voz (módulo de Diego)")

    def predict_interval(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Devuelve (mu, lo, hi) por observación."""
        raise NotImplementedError("Portar desde tfc-anima-voz (módulo de Diego)")
