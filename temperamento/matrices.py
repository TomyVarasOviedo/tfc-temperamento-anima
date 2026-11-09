"""Matrices de covarianza del modelo generativo de Temperamento.

Implementa la sección 4.2 de ``MODEL.md``: corrige la atenuación de las
correlaciones fenotípicas, construye la covarianza genética y la ambiental,
y verifica que ambas sean simétricas y definidas positivas.

No se proyectan matrices al cono positivo: si una matriz no es definida
positiva, se lanza un error para que el problema se vea.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import parametros


@dataclass(frozen=True)
class MatricesNivel:
    """Matrices de covarianza de un nivel de señal.

    Attributes:
        nivel: nombre del nivel (``base``, ``ldsc``, ``pgi`` o ``nulo``).
        h2_observada: h² observada por rasgo, forma ``(5,)``.
        h2_latente: h² latente por rasgo, forma ``(5,)``.
        correlacion_latente: correlación latente de los rasgos, forma ``(5, 5)``.
        G: covarianza genética, forma ``(5, 5)``.
        Sigma_E: covarianza ambiental, forma ``(5, 5)``.
    """

    nivel: str
    h2_observada: np.ndarray
    h2_latente: np.ndarray
    correlacion_latente: np.ndarray
    G: np.ndarray
    Sigma_E: np.ndarray


def construir(
    nivel: str,
    *,
    alpha: np.ndarray | None = None,
    h2_latente: np.ndarray | None = None,
    rg: np.ndarray | None = None,
    rp: np.ndarray | None = None,
) -> MatricesNivel:
    """Construye y valida las matrices de covarianza de un nivel.

    Los argumentos opcionales existen para los tests de robustez; en el uso
    normal se dejan en ``None`` y todo sale de ``parametros.py``.

    Args:
        nivel: uno de ``parametros.NIVELES``.
        alpha: confiabilidades por rasgo, forma ``(5,)``. Por defecto, las de
            Johnson (2014).
        h2_latente: h² latente por rasgo, forma ``(5,)``. Por defecto, la del
            nivel pedido.
        rg: correlación genética, forma ``(5, 5)``. Por defecto, ``RG``.
        rp: correlación fenotípica, forma ``(5, 5)``. Por defecto, ``RP``.

    Returns:
        ``MatricesNivel`` con las matrices construidas y validadas.

    Raises:
        ValueError: si las formas no coinciden, si una matriz no es simétrica
            o si ``G`` o ``Sigma_E`` no son definidas positivas.
    """
    if nivel not in parametros.NIVELES:
        raise ValueError(f"nivel desconocido: {nivel!r}; se esperaba uno de {parametros.NIVELES}")

    k = len(parametros.RASGOS)
    a = parametros.vector(parametros.ALPHA) if alpha is None else _vector_5(alpha, "alpha")
    h = parametros.h2_latente(nivel) if h2_latente is None else _vector_5(h2_latente, "h2_latente")
    rg = parametros.RG.copy() if rg is None else _matriz_5(rg, "rg")
    rp = parametros.RP.copy() if rp is None else _matriz_5(rp, "rp")

    if np.any(a <= 0.0) or np.any(a > 1.0):
        raise ValueError("alpha debe estar en (0, 1]")
    if np.any(h < 0.0):
        raise ValueError("h2_latente no puede ser negativa")

    correlacion_latente = rp / np.sqrt(np.outer(a, a))
    np.fill_diagonal(correlacion_latente, 1.0)

    G = rg * np.sqrt(np.outer(h, h))
    np.fill_diagonal(G, h)

    Sigma_E = correlacion_latente - G

    _validar_simetrica(correlacion_latente, "correlacion_latente")
    _validar_simetrica(G, "G")
    _validar_simetrica(Sigma_E, "Sigma_E")
    _validar_definida_positiva(G, "G", nivel)
    _validar_definida_positiva(Sigma_E, "Sigma_E", nivel)

    return MatricesNivel(
        nivel=nivel,
        h2_observada=a * h,
        h2_latente=h,
        correlacion_latente=correlacion_latente,
        G=G,
        Sigma_E=Sigma_E,
    )


def _vector_5(valor: np.ndarray, nombre: str) -> np.ndarray:
    """Valida y devuelve un vector de cinco rasgos."""
    arr = np.asarray(valor, dtype=float)
    if arr.shape != (len(parametros.RASGOS),):
        raise ValueError(f"{nombre} debe tener forma ({len(parametros.RASGOS)},), tiene {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{nombre} contiene valores no finitos")
    return arr


def _matriz_5(valor: np.ndarray, nombre: str) -> np.ndarray:
    """Valida y devuelve una matriz de 5 x 5."""
    k = len(parametros.RASGOS)
    arr = np.asarray(valor, dtype=float)
    if arr.shape != (k, k):
        raise ValueError(f"{nombre} debe tener forma ({k}, {k}), tiene {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{nombre} contiene valores no finitos")
    return arr


def _validar_simetrica(matriz: np.ndarray, nombre: str, atol: float = 1e-10) -> None:
    """Comprueba que una matriz sea simétrica."""
    if not np.allclose(matriz, matriz.T, atol=atol):
        raise ValueError(f"{nombre} no es simétrica")


def _validar_definida_positiva(matriz: np.ndarray, nombre: str, nivel: str) -> None:
    """Comprueba que todos los autovalores de una matriz sean positivos."""
    autovalores = np.linalg.eigvalsh(matriz)
    if np.any(autovalores <= 0.0):
        raise ValueError(
            f"{nombre} no es definida positiva en el nivel {nivel!r}: "
            f"autovalores {np.round(autovalores, 6).tolist()}"
        )
