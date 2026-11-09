"""Modelos de predicción de Temperamento, sin calibración.

Incluye el modelo con genotipo y las dos líneas base de la sección 5 de
``MODEL.md``. Todos los ajustes reciben únicamente datos de entrenamiento:
la partición de calibración y la de prueba no participan.

El modelo principal es ``y ~ 1 + g`` por rasgo. La línea base de ancestría,
sexo y edad usa variables indicadoras para ``subpoblacion`` y toma como
categoría de referencia la primera en orden alfabético.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import parametros


@dataclass(frozen=True)
class AjusteGenotipo:
    """Ajuste lineal de un rasgo sobre su propio puntaje genético.

    Attributes:
        intercepto: ordenada al origen.
        pendiente: coeficiente del puntaje genético.
        sigma: desvío de los residuos de entrenamiento, con
            ``ddof = DDOF_GENOTIPO``.
    """

    intercepto: float
    pendiente: float
    sigma: float

    def predecir(self, g: np.ndarray) -> np.ndarray:
        """Predice la media para uno o varios puntajes genéticos."""
        return self.intercepto + self.pendiente * np.asarray(g, dtype=float)


@dataclass(frozen=True)
class AjusteMedia:
    """Línea base de solo intercepto.

    Attributes:
        mu: media del rasgo en entrenamiento.
        sigma: desvío del rasgo en entrenamiento, con ``ddof = DDOF_MEDIA``.
    """

    mu: float
    sigma: float

    def predecir(self, n: int) -> np.ndarray:
        """Devuelve la media constante para ``n`` personas."""
        return np.full(int(n), self.mu, dtype=float)


@dataclass(frozen=True)
class AjusteLineal:
    """Línea base de ancestría, sexo y edad.

    Attributes:
        coeficientes: coeficientes del diseño, incluido el intercepto.
        categorias: subpoblaciones en el orden usado para las indicadoras.
            La primera es la categoría de referencia.
        sigma: desvío de los residuos de entrenamiento, con ``ddof`` igual
            a la cantidad de coeficientes ajustados.
    """

    coeficientes: np.ndarray
    categorias: tuple[str, ...]
    sigma: float

    def predecir(
        self,
        sexo: np.ndarray,
        edad: np.ndarray,
        subpoblacion: np.ndarray | list[str],
    ) -> np.ndarray:
        """Predice la media para las personas indicadas."""
        diseno = diseno_ancestria(sexo, edad, subpoblacion, self.categorias)
        return diseno @ self.coeficientes


def ajustar_genotipo(y: np.ndarray, g: np.ndarray) -> AjusteGenotipo:
    """Ajusta ``y ~ 1 + g`` por mínimos cuadrados ordinarios.

    Args:
        y: respuestas de entrenamiento, forma ``(n,)``.
        g: puntajes genéticos de entrenamiento, forma ``(n,)``.

    Returns:
        ``AjusteGenotipo`` con pendiente, intercepto y ``sigma``.

    Raises:
        ValueError: si los vectores no tienen el mismo largo, tienen menos de
            tres observaciones o contienen valores no finitos.
    """
    y = _vector(y, "y")
    g = _vector(g, "g")
    if y.shape != g.shape:
        raise ValueError(f"y y g deben tener el mismo largo: {y.shape} != {g.shape}")
    n = len(y)
    if n <= parametros.DDOF_GENOTIPO:
        raise ValueError(f"hacen falta más de {parametros.DDOF_GENOTIPO} observaciones")

    diseno = np.column_stack([np.ones(n), g])
    coeficientes, *_ = np.linalg.lstsq(diseno, y, rcond=None)
    residuos = y - diseno @ coeficientes
    sigma = float(np.sqrt(np.sum(residuos**2) / (n - parametros.DDOF_GENOTIPO)))

    return AjusteGenotipo(
        intercepto=float(coeficientes[0]),
        pendiente=float(coeficientes[1]),
        sigma=sigma,
    )


def ajustar_solo_intercepto(y: np.ndarray) -> AjusteMedia:
    """Ajusta la línea base de solo intercepto.

    Args:
        y: respuestas de entrenamiento, forma ``(n,)``.

    Returns:
        ``AjusteMedia`` con la media y el desvío de entrenamiento.

    Raises:
        ValueError: si hay menos de dos observaciones o valores no finitos.
    """
    y = _vector(y, "y")
    if len(y) <= parametros.DDOF_MEDIA:
        raise ValueError(f"hacen falta más de {parametros.DDOF_MEDIA} observaciones")
    return AjusteMedia(
        mu=float(np.mean(y)),
        sigma=float(np.std(y, ddof=parametros.DDOF_MEDIA)),
    )


def ajustar_ancestria_sexo_edad(
    y: np.ndarray,
    sexo: np.ndarray,
    edad: np.ndarray,
    subpoblacion: np.ndarray | list[str],
    categorias: tuple[str, ...] | None = None,
) -> AjusteLineal:
    """Ajusta ``y ~ 1 + sexo + edad + subpoblacion``.

    La subpoblación entra con variables indicadoras y una categoría de
    referencia (la primera en orden alfabético si ``categorias`` es ``None``).

    Args:
        y: respuestas de entrenamiento, forma ``(n,)``.
        sexo: código 0/1 de sexo, forma ``(n,)``.
        edad: edades, forma ``(n,)``.
        subpoblacion: subpoblaciones, forma ``(n,)``.
        categorias: orden opcional de las subpoblaciones. Si se omite, se
            calcula ordenando alfabéticamente las observadas.

    Returns:
        ``AjusteLineal`` con los coeficientes y ``sigma``.

    Raises:
        ValueError: si los largos no coinciden, faltan observaciones o hay
            subpoblaciones no declaradas.
    """
    y = _vector(y, "y")
    if categorias is None:
        categorias = tuple(sorted(set(str(v) for v in subpoblacion)))
    diseno = diseno_ancestria(sexo, edad, subpoblacion, categorias)
    if len(y) != diseno.shape[0]:
        raise ValueError(f"y tiene {len(y)} filas y el diseño {diseno.shape[0]}")
    if len(y) <= diseno.shape[1]:
        raise ValueError("hacen falta más observaciones que coeficientes")

    coeficientes, *_ = np.linalg.lstsq(diseno, y, rcond=None)
    residuos = y - diseno @ coeficientes
    sigma = float(np.sqrt(np.sum(residuos**2) / (len(y) - diseno.shape[1])))

    return AjusteLineal(
        coeficientes=coeficientes,
        categorias=tuple(categorias),
        sigma=sigma,
    )


def diseno_ancestria(
    sexo: np.ndarray,
    edad: np.ndarray,
    subpoblacion: np.ndarray | list[str],
    categorias: tuple[str, ...],
) -> np.ndarray:
    """Construye la matriz de diseño de la línea base de ancestría.

    Columnas: intercepto, sexo, edad y una indicadora por cada subpoblación
    distinta de la categoría de referencia ``categorias[0]``.

    Args:
        sexo: códigos 0/1, forma ``(n,)``.
        edad: edades, forma ``(n,)``.
        subpoblacion: subpoblaciones, forma ``(n,)``.
        categorias: subpoblaciones en orden; la primera es la referencia.

    Returns:
        Matriz de diseño, forma ``(n, 3 + len(categorias) - 1)``.

    Raises:
        ValueError: si los largos no coinciden o aparece una subpoblación
            desconocida.
    """
    sexo = _vector(sexo, "sexo")
    edad = _vector(edad, "edad")
    subpoblacion = np.asarray(subpoblacion, dtype=object)
    n = len(sexo)
    if len(edad) != n or len(subpoblacion) != n:
        raise ValueError("sexo, edad y subpoblacion deben tener el mismo largo")
    if not np.all(np.isfinite(edad)):
        raise ValueError("edad contiene valores no finitos")

    categorias = tuple(categorias)
    if not categorias:
        raise ValueError("categorias no puede estar vacío")
    desconocidas = set(str(v) for v in subpoblacion) - set(categorias)
    if desconocidas:
        raise ValueError(f"subpoblaciones no declaradas: {sorted(desconocidas)}")

    columnas = [np.ones(n), sexo, edad]
    for categoria in categorias[1:]:
        columnas.append((subpoblacion == categoria).astype(float))
    return np.column_stack(columnas)


def _vector(valor: np.ndarray, nombre: str) -> np.ndarray:
    """Valida y devuelve un vector numérico de una dimensión."""
    arr = np.asarray(valor, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{nombre} debe ser un vector de una dimensión")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{nombre} contiene valores no finitos")
    return arr
