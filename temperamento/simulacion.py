"""Simulación del genotipo y del fenotipo de Temperamento.

El esquema es el de la sección 4 de ``MODEL.md``::

    g ~ N(0, G)
    e ~ N(0, Sigma_E), independiente de g
    t = g + e
    y = sqrt(alpha) * t + sqrt(1 - alpha) * epsilon

``g`` es el puntaje genético que ve el predictor, ``t`` el rasgo latente e
``y`` la respuesta observada en el IPIP-NEO-120 (escala z).

Cada paso aleatorio usa un flujo de semilla separado. Así, agregar o quitar
un paso no cambia los números de los demás.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import matrices, parametros


@dataclass(frozen=True)
class Simulacion:
    """Resultado de simular una muestra.

    Attributes:
        nivel: nivel de señal usado.
        g: puntaje genético que ve el predictor, forma ``(n, 5)``. En el
            nivel ``nulo`` está permutado entre personas.
        t: rasgo latente, forma ``(n, 5)``.
        y: puntaje observado del IPIP-NEO-120, forma ``(n, 5)``.
        epsilon: error de medición independiente, forma ``(n, 5)``.
        permutacion: permutación aplicada a ``g`` en el nivel ``nulo``, o
            ``None`` en los demás niveles.
    """

    nivel: str
    g: np.ndarray
    t: np.ndarray
    y: np.ndarray
    epsilon: np.ndarray
    permutacion: np.ndarray | None = None


def simular(n: int, nivel: str, replica: int = 0) -> Simulacion:
    """Simula ``n`` personas para un nivel de señal y una réplica.

    La semilla se deriva de forma determinística de
    ``(SEMILLA_TEMPERAMENTO, replica, INDICE_NIVEL[nivel])``. Los flujos de
    ``g``, ``e``, ``epsilon`` y la permutación se generan con
    ``SeedSequence.spawn`` y son independientes entre sí.

    Args:
        n: cantidad de personas. Debe ser un entero positivo.
        nivel: uno de ``parametros.NIVELES``.
        replica: índice de réplica del barrido. La réplica 0 es la principal.

    Returns:
        ``Simulacion`` con ``g``, ``t``, ``y`` y ``epsilon``.

    Raises:
        ValueError: si ``n`` no es positivo o el nivel no existe.
    """
    if not isinstance(n, (int, np.integer)) or int(n) <= 0:
        raise ValueError("n debe ser un entero positivo")
    if nivel not in parametros.NIVELES:
        raise ValueError(f"nivel desconocido: {nivel!r}; se esperaba uno de {parametros.NIVELES}")

    n = int(n)
    m = matrices.construir(nivel)
    rng_g, rng_e, rng_epsilon, rng_permutacion = _flujos(int(replica), nivel)

    k = len(parametros.RASGOS)
    g = rng_g.standard_normal((n, k)) @ np.linalg.cholesky(m.G).T
    e = rng_e.standard_normal((n, k)) @ np.linalg.cholesky(m.Sigma_E).T
    t = g + e
    epsilon = rng_epsilon.standard_normal((n, k))

    alpha = parametros.vector(parametros.ALPHA)
    y = np.sqrt(alpha) * t + np.sqrt(1.0 - alpha) * epsilon

    permutacion = None
    if nivel == "nulo":
        permutacion = rng_permutacion.permutation(n)
        g = g[permutacion]

    return Simulacion(nivel=nivel, g=g, t=t, y=y, epsilon=epsilon, permutacion=permutacion)


def aplicar_test(t: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Aplica el IPIP-NEO-120 otra vez al mismo rasgo latente.

    Sirve para verificar la confiabilidad: dos aplicaciones comparten ``t``
    pero tienen errores de medición independientes.

    Args:
        t: rasgo latente, forma ``(n, 5)``.
        rng: generador de números aleatorios para el nuevo error.

    Returns:
        Nueva respuesta observada, forma ``(n, 5)``.
    """
    t = np.asarray(t, dtype=float)
    alpha = parametros.vector(parametros.ALPHA)
    epsilon = rng.standard_normal(t.shape)
    return np.sqrt(alpha) * t + np.sqrt(1.0 - alpha) * epsilon


def simular_cohorte(
    nivel: str,
    replica: int = 0,
    ruta=parametros.COHORTE,
) -> tuple[object, Simulacion]:
    """Carga la cohorte compartida y simula sus rasgos.

    La cohorte solo aporta los individuos y sus particiones; no se modifica.

    Args:
        nivel: uno de ``parametros.NIVELES``.
        replica: índice de réplica del barrido.
        ruta: ruta del CSV de la cohorte.

    Returns:
        Tupla ``(spine, simulacion)``.
    """
    from anima_spine import spine as spine_mod

    sp = spine_mod.cargar(ruta)
    return sp, simular(len(sp.filas), nivel, replica)


def mascara_particion(spine: object, particion: str) -> np.ndarray:
    """Devuelve una máscara booleana de las filas de una partición.

    Args:
        spine: objeto ``anima_spine.spine.Spine`` cargado.
        particion: ``entrenamiento``, ``calibracion`` o ``prueba``.

    Returns:
        Array booleano de forma ``(n,)``.
    """
    return np.array([fila["particion"] == particion for fila in spine.filas], dtype=bool)


def _flujos(replica: int, nivel: str) -> tuple[np.random.Generator, ...]:
    """Crea los cuatro flujos independientes de una simulación."""
    secuencia = np.random.SeedSequence(
        [parametros.SEMILLA_TEMPERAMENTO, replica, parametros.INDICE_NIVEL[nivel]]
    )
    return tuple(np.random.default_rng(hijo) for hijo in secuencia.spawn(4))
