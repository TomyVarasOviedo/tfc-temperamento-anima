"""El resultado central del TFC: el gradiente entre los tres dominios.

Se corre UNA vez, cuando los tres módulos ya exportaron sus predicciones.
Remuestrea individuos (no filas) de la partición de prueba, y en cada
remuestreo recalcula el ΔR² de los tres dominios sobre LOS MISMOS
individuos. Así la diferencia entre dominios tiene intervalo de confianza.

Eso es lo que no se puede hacer si cada módulo usa su propia cohorte.

La hipótesis se formula en dos niveles (decisión del grupo del 22/10/2026):
el nivel superior (Rostro y Voz) por encima del inferior (Temperamento).
Los contrastes que se AFIRMAN son Rostro - Temperamento y Voz -
Temperamento, y la hipótesis se sostiene si los dos son significativos.
Rostro - Voz se calcula e INFORMA con su intervalo, pero no se afirma: con la
brecha publicada (0,008) su potencia es de 0,12 a 0,19.

Exigir que los dos contrastes afirmados sean significativos, cada uno al
nivel alpha, es una prueba de intersección-unión: el error de tipo I de la
hipótesis conjunta queda acotado por alpha sin corrección por
comparaciones múltiples.
"""

from __future__ import annotations

import numpy as np

from .predictions import DOMINIOS, LINEAS_BASE, como_arrays

CONTRASTES_AFIRMADOS = ("rostro - temperamento", "voz - temperamento")
CONTRASTES_INFORMADOS = ("rostro - voz",)


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - sse / sst if sst > 0 else 0.0


def _delta_r2_por_dominio(
    datos: dict[str, dict[str, np.ndarray]], idx: np.ndarray
) -> dict[str, float]:
    """ΔR² promedio sobre los rasgos de cada dominio, en un remuestreo.

    Promedia por rasgo dentro del dominio: así un dominio con 28 componentes
    no pesa más que uno con 5 rasgos solo por tener más dimensiones.
    """
    out = {}
    for dominio, por_rasgo in datos.items():
        valores = []
        for arrays in por_rasgo.values():
            y = arrays["y_real"][idx]
            valores.append(_r2(y, arrays["mu"][idx]) - _r2(y, arrays["mu_base"][idx]))
        out[dominio] = float(np.mean(valores))
    return out


def comparar(
    predicciones: dict[str, list[dict]],
    rasgos: dict[str, list[str]],
    linea_base: str = "solo intercepto",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    colas: int = 1,
) -> dict:
    """Compara el ΔR² de los tres dominios con bootstrap pareado.

    Args:
        predicciones: por dominio, las filas cargadas de su CSV.
        rasgos: por dominio, qué rasgos entran en el promedio.
        linea_base: la misma para los tres. Es el punto del acuerdo.
        colas: 1 (por defecto) o 2. La hipótesis es direccional
            (rostro y voz por encima de temperamento), así que la sección
            4.7 del documento integrado fija un contraste de una sola cola: el
            escalón queda demostrado si el percentil alpha de la diferencia
            es > 0.
            Con colas=2 se exige que el IC al (1 - alpha) excluya el cero.
            El IC a dos colas se informa siempre.

    Returns:
        Por dominio su ΔR² y su IC, y por par de dominios la diferencia
        con su IC, su rol ("afirmado" o "informado") y si es
        significativa. `hipotesis_dos_niveles` es True si los dos
        contrastes afirmados son significativos; Rostro - Voz no entra.
    """
    datos = {
        dominio: {
            rasgo: como_arrays(filas, rasgo, linea_base) for rasgo in rasgos[dominio]
        }
        for dominio, filas in predicciones.items()
    }

    # Mismos individuos en los tres dominios: el pareo se apoya en esto.
    conjuntos = [
        set(a["individuo_id"].tolist())
        for por_rasgo in datos.values()
        for a in por_rasgo.values()
    ]
    comunes = set.intersection(*conjuntos)
    if len(comunes) != len(conjuntos[0]):
        raise AssertionError(
            "los dominios no cubren los mismos individuos: el bootstrap "
            "pareado no es válido. Revisar la columna vertebral."
        )

    n = len(comunes)
    rng = np.random.default_rng(seed)
    punto = _delta_r2_por_dominio(datos, np.arange(n))

    if colas not in (1, 2):
        raise ValueError("colas debe ser 1 o 2")
    # Orden de la hipótesis: cada par se resta en el sentido esperado.
    nombres = [d for d in DOMINIOS if d in datos]
    pares = [(a, b) for i, a in enumerate(nombres) for b in nombres[i + 1 :]]
    muestras = {d: np.empty(n_boot) for d in nombres}
    dif = {par: np.empty(n_boot) for par in pares}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        valores = _delta_r2_por_dominio(datos, idx)
        for d in nombres:
            muestras[d][b] = valores[d]
        for a, c in pares:
            dif[(a, c)][b] = valores[a] - valores[c]

    q = [alpha / 2, 1 - alpha / 2]

    def _significativa(muestras_dif: np.ndarray) -> bool:
        if colas == 1:
            return bool(np.quantile(muestras_dif, alpha) > 0)
        lo, hi = np.quantile(muestras_dif, q)
        return bool(lo > 0 or hi < 0)

    diferencias = {
        f"{a} - {c}": {
            "diferencia": punto[a] - punto[c],
            "ic": tuple(float(v) for v in np.quantile(dif[(a, c)], q)),
            "cota_inferior_una_cola": float(np.quantile(dif[(a, c)], alpha)),
            "significativa": _significativa(dif[(a, c)]),
            "rol": "afirmado" if f"{a} - {c}" in CONTRASTES_AFIRMADOS else "informado",
        }
        for a, c in pares
    }
    afirmados = [diferencias[k] for k in CONTRASTES_AFIRMADOS if k in diferencias]

    return {
        "linea_base": linea_base,
        "n_individuos": n,
        "n_boot": n_boot,
        "colas": colas,
        "por_dominio": {
            d: {
                "delta_r2": punto[d],
                "ic": tuple(float(v) for v in np.quantile(muestras[d], q)),
            }
            for d in nombres
        },
        "diferencias": diferencias,
        "hipotesis_dos_niveles": (
            len(afirmados) == len(CONTRASTES_AFIRMADOS)
            and all(v["significativa"] for v in afirmados)
        ),
    }


def imprimir(resultado: dict) -> None:
    """Imprime el resultado en formato legible para el informe."""
    print(f"Línea base: {resultado['linea_base']}")
    print(
        f"n = {resultado['n_individuos']}, {resultado['n_boot']} remuestreos, "
        f"contraste a {resultado['colas']} cola(s)\n"
    )

    print(f"{'dominio':>14}  {'ΔR²':>8}  {'IC 95%':>22}")
    print("-" * 48)
    for d, v in resultado["por_dominio"].items():
        ic = f"[{v['ic'][0]:+.4f}, {v['ic'][1]:+.4f}]"
        print(f"{d:>14}  {v['delta_r2']:>8.4f}  {ic:>22}")

    print(f"\n{'comparación':>28}  {'dif':>8}  {'IC 95%':>22}  {'sig':>5}  {'rol':>10}")
    print("-" * 82)
    for par, v in resultado["diferencias"].items():
        ic = f"[{v['ic'][0]:+.4f}, {v['ic'][1]:+.4f}]"
        sig = "sí" if v["significativa"] else "no"
        print(f"{par:>28}  {v['diferencia']:>+8.4f}  {ic:>22}  {sig:>5}  {v['rol']:>10}")

    sostenida = "SE SOSTIENE" if resultado["hipotesis_dos_niveles"] else "NO se sostiene"
    print(f"\nHipótesis en dos niveles (Rostro y Voz > Temperamento): {sostenida}")
    print("Rostro - Voz se informa, no se afirma.")


def comparar_lineas_base(
    predicciones: dict[str, list[dict]],
    rasgos: dict[str, list[str]],
    **kwargs,
) -> dict[str, dict]:
    """Corre `comparar` contra las dos líneas base. Es lo que va al informe."""
    return {lb: comparar(predicciones, rasgos, linea_base=lb, **kwargs) for lb in LINEAS_BASE}
