"""Fase 2 — Cohorte sintética.

Único lugar donde la matemática de temperamento difiere de voz: los cinco
rasgos están correlacionados entre sí, así que el muestreo es multivariado
(Cholesky sobre la matriz de correlación). Si se muestrea rasgo por rasgo
de forma independiente, la cohorte queda irreal y el ΔR² no significa nada.

Mismo criterio que rostro y voz: el puntaje poligénico se representa como
variable aleatoria por dimensión con varianza anclada en h²-SNP publicada,
sin simular SNPs individuales.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..traits.space import TraitSpace


@dataclass(frozen=True)
class Cohort:
    """Cohorte sintética.

    Attributes:
        G: componente genética estandarizada, forma (n, k).
        Y: fenotipo observado en la escala del instrumento, forma (n, k).
        sex: 0/1, forma (n,).
        age: años, forma (n,).
        h2: heredabilidad usada al generar, forma (k,). Es el valor de
            verdad contra el que se contrasta lo que recupera el modelo.
    """

    G: np.ndarray
    Y: np.ndarray
    sex: np.ndarray
    age: np.ndarray
    h2: np.ndarray


def generate(
    space: TraitSpace,
    n: int,
    h2: np.ndarray | float | None = None,
    seed: int = 0,
    age_range: tuple[int, int] = (18, 80),
    link: bool = True,
) -> Cohort:
    """Genera una cohorte sintética con la estructura de correlación de space.

    Args:
        space: espacio de rasgos cargado en la Fase 1.
        h2: heredabilidad por rasgo. None usa space.h2_snp. Un escalar se
            aplica a todos los rasgos.
        seed: semilla. Fijarla siempre en el informe de validación.
        link: si es False, el genotipo se genera con su varianza normal
            pero se permuta respecto del fenotipo. Esa es la cohorte nula
            correcta para el análisis de potencia: un genotipo que existe
            y no informa nada. Generar la nula con h2=0 es un error —
            produce G idénticamente cero, y cualquier modelo recupera la
            media exacta, con lo que el ΔR² da cero por construcción y el
            mínimo detectable sale artificialmente perfecto.

    Returns:
        Cohort con G e Y correlacionados según space.corr.
    """
    rng = np.random.default_rng(seed)
    k = len(space.names)

    if h2 is None:
        h2 = space.h2_snp
    h2 = np.full(k, float(h2)) if np.isscalar(h2) else np.asarray(h2, dtype=float)
    if np.any(h2 < 0) or np.any(h2 > 1):
        raise ValueError("h2 debe estar entre 0 y 1")
    if np.any(h2 > space.reliability):
        raise ValueError("h2 no puede superar la confiabilidad del instrumento")

    L = space.cholesky
    G = (rng.standard_normal((n, k)) @ L.T) * np.sqrt(h2)
    E = (rng.standard_normal((n, k)) @ L.T) * np.sqrt(1.0 - h2)

    z = G + E
    Y = space.mu + space.sigma * z

    if not link:
        G = G[rng.permutation(n)]

    return Cohort(
        G=G,
        Y=Y,
        sex=rng.integers(0, 2, size=n),
        age=rng.integers(age_range[0], age_range[1] + 1, size=n),
        h2=h2,
    )


def check(cohort: Cohort, space: TraitSpace, tol_corr: float = 0.02, tol_h2: float = 0.01) -> None:
    """Valida la cohorte antes de seguir. Si esto falla, no avanzar de fase.

    Chequea las dos cosas que tienen que ser ciertas por construcción:
    que la correlación entre rasgos se haya preservado, y que la fracción
    de varianza genética sea la que se pidió.
    """
    z = (cohort.Y - space.mu) / space.sigma

    err_corr = np.abs(np.corrcoef(z.T) - space.corr).max()
    if err_corr > tol_corr:
        raise AssertionError(
            f"la correlación empírica se desvía {err_corr:.3f} de space.corr "
            f"(tolerancia {tol_corr}). Subir n o revisar el muestreo."
        )

    if not np.allclose(cohort.G.mean(axis=0), 0.0, atol=0.1):
        raise AssertionError("G debería estar centrada en cero")

    h2_emp = cohort.G.var(axis=0) / z.var(axis=0)
    err_h2 = np.abs(h2_emp - cohort.h2).max()
    if err_h2 > tol_h2:
        raise AssertionError(
            f"la h2 empírica se desvía {err_h2:.3f} de la pedida "
            f"(tolerancia {tol_h2}). Subir n."
        )
