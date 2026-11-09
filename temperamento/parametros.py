"""Parámetros del modelo de Temperamento.

Todas las cifras del módulo salen de este archivo y llevan su fuente al lado.
El orden de los rasgos es fijo en todo el código: ``E, A, C, N, O``.

Fuentes principales:
    - Schwaba, T. et al. (2026). *Robust inference and correlates from
      genetic associations with personality*. Nature. Tabla 1, Figura 1f y
      tablas suplementarias S4, S21 y S36.
    - Johnson, J. A. (2014). *Measuring thirty facets of the Five Factor
      Model with a 120-item public domain inventory*. J. Research in
      Personality, 51, 78-89. Tabla 2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

RASGOS: tuple[str, ...] = ("E", "A", "C", "N", "O")

NOMBRES: dict[str, str] = {
    "E": "Extraversión",
    "A": "Amabilidad",
    "C": "Responsabilidad",
    "N": "Neuroticismo",
    "O": "Apertura a la experiencia",
}

# Johnson (2014), Tabla 2, muestra de internet (N = 619.150), IPIP-120.
ALPHA: dict[str, float] = {"E": 0.89, "A": 0.86, "C": 0.90, "N": 0.90, "O": 0.81}

# Schwaba et al. (2026), Tabla 1, columna "Disattenuated for measurement
# instrument error", valor "all" (alpha = 1). Coincide con la tabla S21.
H2_LATENTE_BASE: dict[str, float] = {
    "E": 0.158,
    "A": 0.108,
    "C": 0.112,
    "N": 0.120,
    "O": 0.152,
}

# Schwaba et al. (2026), Tabla S4, columna "LDSC h2", EUR-like. Son h²
# observados, es decir, medidos con el error del IPIP-NEO-120.
H2_OBSERVADA_LDSC: dict[str, float] = {
    "E": 0.0926,
    "A": 0.0480,
    "C": 0.0713,
    "N": 0.0832,
    "O": 0.0745,
}

# Schwaba et al. (2026), Tabla S36, "Population", EUR-like, cohortes HRS,
# GSOEP, TEDS, NTR y Add Health. Es el R² que logra un puntaje poligénico real.
H2_OBSERVADA_PGI: dict[str, float] = {
    "E": 0.0355,
    "A": 0.0094,
    "C": 0.0269,
    "N": 0.0302,
    "O": 0.0302,
}

# Schwaba et al. (2026), Figura 1f, debajo de la diagonal: correlaciones
# genéticas (LDSC, EUR-like). Orden E, A, C, N, O.
RG: np.ndarray = np.array(
    [
        [1.00, 0.16, 0.19, -0.27, 0.22],
        [0.16, 1.00, 0.20, -0.36, 0.05],
        [0.19, 0.20, 1.00, -0.20, -0.20],
        [-0.27, -0.36, -0.20, 1.00, -0.05],
        [0.22, 0.05, -0.20, -0.05, 1.00],
    ],
    dtype=float,
)

# Schwaba et al. (2026), Figura 1f, arriba de la diagonal: correlaciones
# fenotípicas meta-analíticas. No son específicas del IPIP-NEO-120.
RP: np.ndarray = np.array(
    [
        [1.00, 0.15, 0.16, -0.24, 0.28],
        [0.15, 1.00, 0.29, -0.24, 0.14],
        [0.16, 0.29, 1.00, -0.29, 0.08],
        [-0.24, -0.24, -0.29, 1.00, -0.08],
        [0.28, 0.14, 0.08, -0.08, 1.00],
    ],
    dtype=float,
)

NIVELES: tuple[str, ...] = ("base", "ldsc", "pgi", "nulo")

# Índice estable de cada nivel. Entra en la semilla: no cambiar sin subir
# la versión de los resultados.
INDICE_NIVEL: dict[str, int] = {"base": 0, "ldsc": 1, "pgi": 2, "nulo": 3}

# Semilla raíz del módulo. La cohorte conserva sus propias semillas.
SEMILLA_TEMPERAMENTO: int = 20261109

# Partición usada para ajustar. Las otras dos no participan de ningún ajuste.
PARTICION_ENTRENAMIENTO: str = "entrenamiento"

# Parámetros de los ajustes por mínimos cuadrados.
DDOF_GENOTIPO: int = 2  # intercepto + pendiente
DDOF_MEDIA: int = 1

COHORTE: Path = Path(__file__).resolve().parents[1] / "data" / "cohorte_anima.csv"


def vector(valores: Mapping[str, float]) -> np.ndarray:
    """Convierte un diccionario por rasgo en un vector en el orden canónico.

    Args:
        valores: mapa con una entrada por cada código de ``RASGOS``.

    Returns:
        Array de forma ``(5,)`` en el orden ``E, A, C, N, O``.
    """
    faltan = set(RASGOS) - set(valores)
    if faltan:
        raise ValueError(f"faltan rasgos en el diccionario: {sorted(faltan)}")
    return np.array([valores[rasgo] for rasgo in RASGOS], dtype=float)


def h2_observada(nivel: str) -> np.ndarray:
    """Devuelve la h² observada del nivel, en el orden canónico.

    En ``base`` se calcula como ``ALPHA * H2_LATENTE_BASE``; en ``ldsc`` y
    ``pgi`` se usan los valores publicados; ``nulo`` usa la misma h² que
    ``base`` porque el control se construye permutando el genotipo.

    Args:
        nivel: uno de ``NIVELES``.

    Returns:
        Array de forma ``(5,)`` con la h² observada de cada rasgo.

    Raises:
        ValueError: si el nivel no existe.
    """
    if nivel not in NIVELES:
        raise ValueError(f"nivel desconocido: {nivel!r}; se esperaba uno de {NIVELES}")
    alpha = vector(ALPHA)
    if nivel in ("base", "nulo"):
        return alpha * vector(H2_LATENTE_BASE)
    if nivel == "ldsc":
        return vector(H2_OBSERVADA_LDSC)
    return vector(H2_OBSERVADA_PGI)


def h2_latente(nivel: str) -> np.ndarray:
    """Convierte la h² observada del nivel a h² latente.

    La corrección es ``h2_latente = h2_observada / alpha`` y deshace la
    atenuación por el error de medición del IPIP-NEO-120.

    Args:
        nivel: uno de ``NIVELES``.

    Returns:
        Array de forma ``(5,)`` con la h² latente de cada rasgo.
    """
    return h2_observada(nivel) / vector(ALPHA)
