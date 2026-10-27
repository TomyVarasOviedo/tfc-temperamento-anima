"""Fase 4 — Reconstrucción individual.

La salida nunca es un perfil puntual: son N perfiles muestreados de la
distribución predictiva más los intervalos calibrados por rasgo. Es el
equivalente a los N rostros muestreados y a los N WAV de los otros módulos.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .calibration.conformal import SplitConformal
from .contract import TraitResult
from .traits.space import TraitSpace

BASELINE_TEMPERAMENTO = "solo intercepto"


@dataclass(frozen=True)
class Reconstruction:
    """Resultado de reconstruir un individuo.

    Attributes:
        samples: perfiles muestreados, forma (n_samples, k).
        results: un TraitResult por rasgo, listo para exportar.
    """

    samples: np.ndarray
    results: list[TraitResult]


def reconstruct(
    genotype: np.ndarray,
    space: TraitSpace,
    calibrators: dict[str, SplitConformal],
    metrics: dict[str, dict],
    n_samples: int = 500,
    seed: int = 0,
) -> Reconstruction:
    """Reconstruye el perfil temperamental de un individuo.

    Args:
        genotype: puntajes poligénicos del individuo, forma (k,).
        calibrators: un SplitConformal ya calibrado por rasgo.
        metrics: por rasgo, el dict con delta_r2, delta_r2_ci, coverage y
            crps_gain calculados sobre el conjunto de validación. No se
            recalculan acá: son propiedades del modelo, no del individuo.

    TODO (Tomás): muestrear preservando la correlación entre rasgos, igual
    que en cohort.generate — un perfil muestreado rasgo por rasgo de forma
    independiente no es un perfil posible.
    """
    raise NotImplementedError

    # Esqueleto de lo que tiene que armar al final:
    # results = [
    #     TraitResult(
    #         domain="temperamento",
    #         trait=name,
    #         mu=..., lo=..., hi=...,
    #         nominal=calibrators[name].nominal,
    #         delta_r2=metrics[name]["delta_r2"],
    #         delta_r2_ci=metrics[name]["delta_r2_ci"],
    #         coverage=metrics[name]["coverage"],
    #         crps_gain=metrics[name]["crps_gain"],
    #         baseline=BASELINE_TEMPERAMENTO,
    #     )
    #     for name in space.names
    # ]


def explain(results: list[TraitResult]) -> str:
    """Capa opcional de explicación en lenguaje natural (LLM).

    Vive FUERA del pipeline estadístico, igual que en el módulo de rostro:
    recibe la salida ya calculada y calibrada en formato estructurado y
    la verbaliza. Nunca participa del cálculo, del muestreo ni de la
    calibración, y nunca puede cambiar un número.
    """
    raise NotImplementedError("Opcional — solo si sobra tiempo")
