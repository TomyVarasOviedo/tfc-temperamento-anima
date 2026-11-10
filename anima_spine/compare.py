"""El resultado central del TFC: el gradiente entre los tres dominios.

Se corre UNA vez, cuando los tres módulos ya exportaron sus predicciones.
Remuestrea individuos (no filas) de la partición de prueba, y en cada
remuestreo recalcula el ΔR² de los tres dominios sobre LOS MISMOS
individuos. Así la diferencia entre dominios tiene intervalo de confianza.

Eso es lo que no se puede hacer si cada módulo usa su propia cohorte.

REFORMULACIÓN DEL 3/11/2026, aprobada por el tutor
---------------------------------------------------
Hasta esa fecha la hipótesis se formulaba en dos niveles y AFIRMABA dos
contrastes: Rostro - Temperamento y Voz - Temperamento. Se encontró que las
tres señales de anclaje no eran de la misma clase de cantidad (sección 6.8
del documento integrado) y que, con cualquier lectura coherente de las
fuentes, la probabilidad de que los dos contrastes afirmados resultaran
simultáneamente significativos era de 0,02.

Desde la reformulación:

* el ORDENAMIENTO entre dominios se calcula e informa con sus intervalos,
  y NINGÚN contraste se afirma;
* lo que la hipótesis afirma por dominio es la RECUPERACIÓN del nivel de
  señal declarado, con el criterio de `criterio_recuperacion` de
  predictions.py, más el aporte por encima de la línea base;
* el ΔR² se informa en sus dos formas, parcial y total, y el criterio se
  evalúa sobre el PARCIAL respecto de "ancestria sexo edad", porque las
  tres fuentes estiman sus heredabilidades netas de covariables.

Por eso `comparar` ya no devuelve `hipotesis_dos_niveles`. El código que lo
leía tiene que mirar `ordenamiento` y el criterio de recuperación.
"""

from __future__ import annotations

import numpy as np

from .predictions import DOMINIOS, LINEAS_BASE, como_arrays

# Desde el 3/11/2026 ningún contraste entre dominios se afirma: los tres
# pares se informan. Se conservan los nombres por trazabilidad.
CONTRASTES_AFIRMADOS: tuple[str, ...] = ()
CONTRASTES_INFORMADOS = ("rostro - voz", "rostro - temperamento", "voz - temperamento")


def _r2(y: np.ndarray, pred: np.ndarray) -> float:
    sse = float(np.sum((y - pred) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    return 1.0 - sse / sst if sst > 0 else 0.0


def delta_r2(y: np.ndarray, mu: np.ndarray, mu_base: np.ndarray) -> tuple[float, float]:
    """Devuelve (ΔR² total, ΔR² parcial) de un rasgo.

    El total es la fracción de la varianza TOTAL que el genotipo explica por
    encima de la línea base. El parcial es la fracción de lo que la línea
    base NO explica:

        parcial = (R² completo - R² base) / (1 - R² base)

    Los dos coinciden cuando la línea base no explica nada. Difieren mucho
    cuando sí: en el dominio vocal el sexo explica hasta el 74 % de algunos
    parámetros, y ahí el total queda comprimido por el factor (1 - R² base)
    aunque la heredabilidad sea la misma. Como las tres fuentes estiman sus
    heredabilidades netas de covariables, el parcial es la medida comparable
    con el valor declarado.
    """
    r2_completo = _r2(y, mu)
    r2_base = _r2(y, mu_base)
    total = r2_completo - r2_base
    denominador = 1.0 - r2_base
    # Con R² base >= 1 el parcial no está definido; no puede pasar fuera de
    # muestra, pero si pasa es un error del módulo y no se tapa con un 0.
    if denominador <= 0:
        raise AssertionError(
            f"R² de la línea base = {r2_base:.4f}: el ΔR² parcial no está "
            "definido. Revisar mu_base del módulo."
        )
    return float(total), float(total / denominador)


def _delta_r2_por_dominio(
    datos: dict[str, dict[str, np.ndarray]], idx: np.ndarray
) -> dict[str, dict[str, float]]:
    """ΔR² total y parcial, promediados sobre los rasgos de cada dominio.

    Promedia por rasgo dentro del dominio: así un dominio con 28 componentes
    no pesa más que uno con 5 rasgos solo por tener más dimensiones.
    """
    out = {}
    for dominio, por_rasgo in datos.items():
        totales, parciales = [], []
        for arrays in por_rasgo.values():
            y = arrays["y_real"][idx]
            t, p = delta_r2(y, arrays["mu"][idx], arrays["mu_base"][idx])
            totales.append(t)
            parciales.append(p)
        out[dominio] = {
            "total": float(np.mean(totales)),
            "parcial": float(np.mean(parciales)),
        }
    return out


def comparar(
    predicciones: dict[str, list[dict]],
    rasgos: dict[str, list[str]],
    linea_base: str = "ancestria sexo edad",
    n_boot: int = 2000,
    alpha: float = 0.05,
    seed: int = 0,
    colas: int = 1,
    medida: str = "parcial",
) -> dict:
    """Compara el ΔR² de los tres dominios con bootstrap pareado.

    Args:
        predicciones: por dominio, las filas cargadas de su CSV.
        rasgos: por dominio, qué rasgos entran en el promedio.
        linea_base: la misma para los tres. Es el punto del acuerdo. Por
            defecto "ancestria sexo edad", que es la línea base sobre la que
            se evalúan los criterios desde el 3/11/2026.
        medida: "parcial" (por defecto) o "total". Decide cuál de las dos se
            usa para los intervalos y la significación. Las dos se informan
            siempre, en `por_dominio` y en `diferencias`.
        colas: 1 (por defecto) o 2. La hipótesis es direccional
            (rostro y voz por encima de temperamento), así que la sección
            4.7 del documento integrado fija un contraste de una sola cola: el
            escalón queda demostrado si el percentil alpha de la diferencia
            es > 0.
            Con colas=2 se exige que el IC al (1 - alpha) excluya el cero.
            El IC a dos colas se informa siempre.

    Returns:
        Por dominio el ΔR² en sus dos formas con su IC, y por par de
        dominios la diferencia con su IC y si es significativa. Desde la
        reformulación del 3/11/2026 todos los pares tienen rol
        "informado": el ordenamiento entre dominios no se afirma, de modo
        que la clave `hipotesis_dos_niveles` ya no existe.
    """
    if medida not in ("parcial", "total"):
        raise ValueError('medida debe ser "parcial" o "total"')
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
    formas = ("total", "parcial")
    muestras = {f: {d: np.empty(n_boot) for d in nombres} for f in formas}
    dif = {f: {par: np.empty(n_boot) for par in pares} for f in formas}

    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        valores = _delta_r2_por_dominio(datos, idx)
        for f in formas:
            for d in nombres:
                muestras[f][d][b] = valores[d][f]
            for a, c in pares:
                dif[f][(a, c)][b] = valores[a][f] - valores[c][f]

    q = [alpha / 2, 1 - alpha / 2]

    def _significativa(muestras_dif: np.ndarray) -> bool:
        if colas == 1:
            return bool(np.quantile(muestras_dif, alpha) > 0)
        lo, hi = np.quantile(muestras_dif, q)
        return bool(lo > 0 or hi < 0)

    diferencias = {
        f"{a} - {c}": {
            "diferencia": punto[a][medida] - punto[c][medida],
            "ic": tuple(float(v) for v in np.quantile(dif[medida][(a, c)], q)),
            "cota_inferior_una_cola": float(np.quantile(dif[medida][(a, c)], alpha)),
            "significativa": _significativa(dif[medida][(a, c)]),
            "rol": "afirmado" if f"{a} - {c}" in CONTRASTES_AFIRMADOS else "informado",
            "total": {
                "diferencia": punto[a]["total"] - punto[c]["total"],
                "ic": tuple(float(v) for v in np.quantile(dif["total"][(a, c)], q)),
            },
            "parcial": {
                "diferencia": punto[a]["parcial"] - punto[c]["parcial"],
                "ic": tuple(float(v) for v in np.quantile(dif["parcial"][(a, c)], q)),
            },
        }
        for a, c in pares
    }

    return {
        "linea_base": linea_base,
        "medida": medida,
        "n_individuos": n,
        "n_boot": n_boot,
        "colas": colas,
        "por_dominio": {
            d: {
                "delta_r2": punto[d][medida],
                "ic": tuple(float(v) for v in np.quantile(muestras[medida][d], q)),
                "cota_inferior_una_cola": float(np.quantile(muestras[medida][d], alpha)),
                # El aporte del genotipo aprueba con la cota inferior > 0
                # (sección 4.7). Que el IC excluya el cero no alcanza: con un
                # genotipo sin información el ΔR² fuera de muestra sale
                # levemente negativo.
                "aporte_por_encima_de_la_linea_base": bool(
                    np.quantile(muestras[medida][d], alpha) > 0
                ),
                "total": {
                    "delta_r2": punto[d]["total"],
                    "ic": tuple(float(v) for v in np.quantile(muestras["total"][d], q)),
                },
                "parcial": {
                    "delta_r2": punto[d]["parcial"],
                    "ic": tuple(float(v) for v in np.quantile(muestras["parcial"][d], q)),
                },
            }
            for d in nombres
        },
        "ordenamiento": [
            d for d in sorted(nombres, key=lambda x: -punto[x][medida])
        ],
        "diferencias": diferencias,
    }


def imprimir(resultado: dict) -> None:
    """Imprime el resultado en formato legible para el informe."""
    print(f"Línea base: {resultado['linea_base']}")
    print(
        f"n = {resultado['n_individuos']}, {resultado['n_boot']} remuestreos, "
        f"contraste a {resultado['colas']} cola(s), medida: {resultado['medida']}\n"
    )

    print(f"{'dominio':>14}  {'parcial':>8}  {'IC 95%':>22}  {'total':>8}  {'aporte>0':>9}")
    print("-" * 70)
    for d, v in resultado["por_dominio"].items():
        ic = f"[{v['parcial']['ic'][0]:+.4f}, {v['parcial']['ic'][1]:+.4f}]"
        ap = "sí" if v["aporte_por_encima_de_la_linea_base"] else "no"
        print(
            f"{d:>14}  {v['parcial']['delta_r2']:>8.4f}  {ic:>22}  "
            f"{v['total']['delta_r2']:>8.4f}  {ap:>9}"
        )

    print(f"\n{'comparación':>28}  {'dif':>8}  {'IC 95%':>22}  {'sig':>5}  {'rol':>10}")
    print("-" * 82)
    for par, v in resultado["diferencias"].items():
        ic = f"[{v['ic'][0]:+.4f}, {v['ic'][1]:+.4f}]"
        sig = "sí" if v["significativa"] else "no"
        print(f"{par:>28}  {v['diferencia']:>+8.4f}  {ic:>22}  {sig:>5}  {v['rol']:>10}")

    print(
        "\nOrdenamiento observado: "
        + " > ".join(resultado["ordenamiento"])
    )
    print(
        "Se informa, no se afirma (reformulación del 3/11/2026). El veredicto "
        "de la hipótesis sale de predictions.criterio_recuperacion y del "
        "criterio de calibración, no de este contraste."
    )


def comparar_lineas_base(
    predicciones: dict[str, list[dict]],
    rasgos: dict[str, list[str]],
    **kwargs,
) -> dict[str, dict]:
    """Corre `comparar` contra las dos líneas base. Es lo que va al informe.

    Los criterios se evalúan contra "ancestria sexo edad"; la otra se
    informa al lado.
    """
    return {lb: comparar(predicciones, rasgos, linea_base=lb, **kwargs) for lb in LINEAS_BASE}
