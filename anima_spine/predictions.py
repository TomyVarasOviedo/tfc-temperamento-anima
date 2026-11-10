"""Qué devuelve cada módulo a la columna vertebral.

Cada módulo exporta dos archivos sobre la partición de prueba:

1. `predicciones_<dominio>.csv` — una fila por individuo, por rasgo y por
   línea base, con la media y el desvío predictivos y el intervalo
   calibrado al 90 %. Con estos tres archivos el bootstrap pareado entre
   dominios se calcula una sola vez, de forma central, en compare.py.

2. `intervalos_<dominio>.csv` — una fila por individuo, por rasgo y por
   nivel nominal, con el intervalo calibrado en cada uno de los seis
   niveles de NIVELES. La hipótesis exige cobertura dentro de ±5 puntos en
   todo nivel entre 50 % y 95 %, y eso no se puede verificar con un solo
   intervalo. La cobertura se calcula de forma central, contra el y_real de
   predicciones_<dominio>.csv: no es un número que cada módulo informe.

3. `barrido_<dominio>.csv` — el resumen de REPLICAS corridas completas del
   módulo, cada una con su semilla: por réplica, rasgo y nivel, cuántas
   personas de prueba quedaron dentro de su intervalo. El criterio de
   calibración se evalúa sobre la cobertura PROMEDIO de las réplicas
   (opción A, adoptada el 22/10/2026). La réplica 0 es la corrida principal
   y se recalcula de forma central contra los intervalos.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

DOMINIOS = ("rostro", "voz", "temperamento")

CAMPOS = (
    "individuo_id",   # debe existir en cohorte_anima.csv, partición prueba
    "dominio",        # uno de DOMINIOS
    "rasgo",          # "pc01", "f0_medio", "neuroticismo", ... (cada módulo
                      # usa sus propios nombres: el contrato no fija el formato)
    "y_real",         # fenotipo observado
    "mu",             # media predictiva del modelo completo
    "sigma",          # desvío predictivo del modelo completo
    "lo",             # límite inferior calibrado al 90 %
    "hi",             # límite superior calibrado al 90 %
    "mu_base",        # media predictiva de la línea base
    "sigma_base",     # desvío predictivo de la línea base
    "linea_base",     # "solo intercepto" | "ancestria sexo edad"
)

# Los tres módulos reportan contra AMBAS líneas base. Es un número más por
# rasgo y evita que parte de la diferencia de ΔR² entre dominios venga de
# que cada uno comparó contra otra cosa. Estos textos son el contrato: el
# Anexo C del documento integrado debe usar exactamente los mismos.
LINEAS_BASE = ("solo intercepto", "ancestria sexo edad")

# Nivel del intervalo que va en predicciones_<dominio>.csv.
NIVEL_PRINCIPAL = 0.90

# Niveles nominales en los que se evalúa la calibración (hipótesis, 3.2).
NIVELES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)

CAMPOS_INTERVALOS = ("individuo_id", "dominio", "rasgo", "nivel", "lo", "hi")

_NUMERICOS = ("y_real", "mu", "sigma", "lo", "hi", "mu_base", "sigma_base")
_COLUMNAS_DEL_MODELO = ("y_real", "mu", "sigma", "lo", "hi")


def exportar(filas: list[dict], dominio: str, directorio: str | Path = ".") -> Path:
    """Escribe predicciones_<dominio>.csv. Lo llama cada módulo al final."""
    if dominio not in DOMINIOS:
        raise ValueError(f"dominio debe ser uno de {DOMINIOS}, no {dominio!r}")

    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"predicciones_{dominio}.csv"

    with ruta.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS)
        escritor.writeheader()
        escritor.writerows(filas)
    return ruta


def cargar(ruta: str | Path) -> list[dict]:
    """Lee un archivo de predicciones y castea los campos numéricos."""
    with Path(ruta).open(encoding="utf-8") as fh:
        lector = csv.DictReader(fh)
        if tuple(lector.fieldnames or ()) != CAMPOS:
            raise AssertionError(
                f"{Path(ruta).name}: columnas {lector.fieldnames}, se esperaba {list(CAMPOS)}"
            )
        return [
            {
                **r,
                "individuo_id": int(r["individuo_id"]),
                **{k: float(r[k]) for k in _NUMERICOS},
            }
            for r in lector
        ]


def verificar(filas: list[dict], ids_prueba: set[int], dominio: str) -> None:
    """Valida un archivo de predicciones contra la columna vertebral.

    Esto es lo que corre el que integra, antes de calcular nada. Si falla,
    el módulo tiene que corregir su export: no se parchea acá.
    """
    if not filas:
        raise AssertionError(f"{dominio}: el archivo está vacío")

    malos = {f["dominio"] for f in filas} - {dominio}
    if malos:
        raise AssertionError(f"{dominio}: hay filas con dominio {malos}")

    lb = {f["linea_base"] for f in filas}
    if lb - set(LINEAS_BASE):
        raise AssertionError(f"{dominio}: líneas base no declaradas: {lb - set(LINEAS_BASE)}")
    # Las dos son obligatorias: si un módulo informa contra una sola, parte de
    # la diferencia de ΔR² entre dominios viene de que cada uno comparó contra
    # otra cosa. Antes de la versión 2.1 esto se aceptaba y no debía.
    faltan = set(LINEAS_BASE) - lb
    if faltan:
        raise AssertionError(
            f"{dominio}: falta exportar contra la línea base {sorted(faltan)}. "
            f"Los tres módulos reportan contra las dos de LINEAS_BASE."
        )

    for f in filas:
        for k in _NUMERICOS:
            if not math.isfinite(f[k]):
                raise AssertionError(
                    f"{dominio}: individuo {f['individuo_id']}, rasgo {f['rasgo']}: {k} no es finito"
                )

    ids = {f["individuo_id"] for f in filas}
    fuera = ids - ids_prueba
    if fuera:
        raise AssertionError(
            f"{dominio}: {len(fuera)} individuos no están en la partición de "
            f"prueba de la cohorte compartida. Si aparecen individuos de "
            f"entrenamiento, el ΔR² sale inflado."
        )

    # Cada (rasgo, línea base) cubre a cada individuo de prueba exactamente una vez.
    vistos: dict[tuple, set[int]] = {}
    for f in filas:
        clave = (f["rasgo"], f["linea_base"])
        grupo = vistos.setdefault(clave, set())
        if f["individuo_id"] in grupo:
            raise AssertionError(
                f"{dominio}: individuo {f['individuo_id']} repetido en rasgo "
                f"{f['rasgo']}, línea base {f['linea_base']!r}"
            )
        grupo.add(f["individuo_id"])
    for (rasgo, base), grupo in vistos.items():
        faltan = ids_prueba - grupo
        if faltan:
            raise AssertionError(
                f"{dominio}: rasgo {rasgo}, línea base {base!r}: faltan {len(faltan)} "
                f"individuos de prueba. El bootstrap pareado necesita los mismos "
                f"individuos en los tres dominios."
            )

    # Las dos líneas base de un mismo rasgo describen el mismo modelo completo.
    por_individuo: dict[tuple, tuple] = {}
    for f in filas:
        clave = (f["rasgo"], f["individuo_id"])
        valores = tuple(f[k] for k in _COLUMNAS_DEL_MODELO)
        previo = por_individuo.setdefault(clave, valores)
        if not np.allclose(previo, valores):
            raise AssertionError(
                f"{dominio}: individuo {f['individuo_id']}, rasgo {f['rasgo']}: "
                f"y_real, mu, sigma, lo o hi cambian entre líneas base"
            )

    for f in filas:
        if not f["lo"] <= f["mu"] <= f["hi"]:
            raise AssertionError(
                f"{dominio}: individuo {f['individuo_id']}, rasgo {f['rasgo']}: "
                f"se esperaba lo <= mu <= hi"
            )
        if f["sigma"] <= 0 or f["sigma_base"] <= 0:
            raise AssertionError(f"{dominio}: hay sigma no positivo")


def como_arrays(filas: list[dict], rasgo: str, linea_base: str) -> dict[str, np.ndarray]:
    """Reorganiza las filas de un rasgo en arrays alineados por individuo."""
    sel = [f for f in filas if f["rasgo"] == rasgo and f["linea_base"] == linea_base]
    sel.sort(key=lambda f: f["individuo_id"])
    if not sel:
        raise ValueError(f"no hay filas para rasgo={rasgo!r}, línea base={linea_base!r}")
    return {
        "individuo_id": np.array([f["individuo_id"] for f in sel]),
        **{k: np.array([f[k] for f in sel], dtype=float) for k in _NUMERICOS},
    }


# --- Intervalos en varios niveles --------------------------------------------

def exportar_intervalos(filas: list[dict], dominio: str, directorio: str | Path = ".") -> Path:
    """Escribe intervalos_<dominio>.csv (campos de CAMPOS_INTERVALOS)."""
    if dominio not in DOMINIOS:
        raise ValueError(f"dominio debe ser uno de {DOMINIOS}, no {dominio!r}")
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"intervalos_{dominio}.csv"
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS_INTERVALOS)
        escritor.writeheader()
        escritor.writerows(filas)
    return ruta


def cargar_intervalos(ruta: str | Path) -> list[dict]:
    """Lee un archivo de intervalos y castea los campos numéricos."""
    with Path(ruta).open(encoding="utf-8") as fh:
        lector = csv.DictReader(fh)
        if tuple(lector.fieldnames or ()) != CAMPOS_INTERVALOS:
            raise AssertionError(
                f"{Path(ruta).name}: columnas {lector.fieldnames}, "
                f"se esperaba {list(CAMPOS_INTERVALOS)}"
            )
        return [
            {
                **r,
                "individuo_id": int(r["individuo_id"]),
                "nivel": float(r["nivel"]),
                "lo": float(r["lo"]),
                "hi": float(r["hi"]),
            }
            for r in lector
        ]


def verificar_intervalos(
    intervalos: list[dict], predicciones: list[dict], ids_prueba: set[int], dominio: str
) -> None:
    """Valida intervalos_<dominio>.csv contra la cohorte y contra las predicciones."""
    if not intervalos:
        raise AssertionError(f"{dominio}: el archivo de intervalos está vacío")
    if {f["dominio"] for f in intervalos} - {dominio}:
        raise AssertionError(f"{dominio}: hay intervalos de otro dominio")

    niveles = {round(f["nivel"], 6) for f in intervalos}
    if niveles != {round(v, 6) for v in NIVELES}:
        raise AssertionError(f"{dominio}: niveles {sorted(niveles)}, se esperaba {list(NIVELES)}")

    rasgos_pred = {f["rasgo"] for f in predicciones}
    rasgos_int = {f["rasgo"] for f in intervalos}
    if rasgos_pred != rasgos_int:
        raise AssertionError(f"{dominio}: los rasgos de intervalos y predicciones no coinciden")

    tabla: dict[tuple, dict[float, tuple[float, float]]] = {}
    for f in intervalos:
        if not (math.isfinite(f["lo"]) and math.isfinite(f["hi"]) and f["lo"] <= f["hi"]):
            raise AssertionError(
                f"{dominio}: individuo {f['individuo_id']}, rasgo {f['rasgo']}, "
                f"nivel {f['nivel']}: intervalo inválido"
            )
        celda = tabla.setdefault((f["rasgo"], f["individuo_id"]), {})
        nivel = round(f["nivel"], 6)
        if nivel in celda:
            raise AssertionError(
                f"{dominio}: intervalo repetido para individuo {f['individuo_id']}, "
                f"rasgo {f['rasgo']}, nivel {nivel}"
            )
        celda[nivel] = (f["lo"], f["hi"])

    for rasgo in rasgos_int:
        ids = {i for (r, i) in tabla if r == rasgo}
        if ids != ids_prueba:
            raise AssertionError(f"{dominio}: rasgo {rasgo}: no cubre exactamente la partición de prueba")

    principal = round(NIVEL_PRINCIPAL, 6)
    for f in predicciones:
        lo, hi = tabla[(f["rasgo"], f["individuo_id"])][principal]
        if not (np.isclose(lo, f["lo"]) and np.isclose(hi, f["hi"])):
            raise AssertionError(
                f"{dominio}: el intervalo al 90 % de intervalos_ y de predicciones_ "
                f"no coincide (individuo {f['individuo_id']}, rasgo {f['rasgo']})"
            )

    # Intervalos anidados: un nivel mayor nunca es más angosto que uno menor.
    orden = sorted(round(v, 6) for v in NIVELES)
    for (rasgo, i), celda in tabla.items():
        for a, b in zip(orden, orden[1:]):
            if celda[b][0] > celda[a][0] + 1e-12 or celda[b][1] < celda[a][1] - 1e-12:
                raise AssertionError(
                    f"{dominio}: individuo {i}, rasgo {rasgo}: el intervalo al {b:.0%} "
                    f"no contiene al del {a:.0%}"
                )


def cobertura_por_nivel(
    intervalos: list[dict], predicciones: list[dict]
) -> dict[str, dict[float, float]]:
    """Cobertura empírica por rasgo y nivel, con el y_real de las predicciones."""
    y = {(f["rasgo"], f["individuo_id"]): f["y_real"] for f in predicciones}
    acum: dict[str, dict[float, list[bool]]] = {}
    for f in intervalos:
        yi = y[(f["rasgo"], f["individuo_id"])]
        acum.setdefault(f["rasgo"], {}).setdefault(round(f["nivel"], 6), []).append(
            f["lo"] <= yi <= f["hi"]
        )
    return {
        rasgo: {nivel: float(np.mean(v)) for nivel, v in sorted(niveles.items())}
        for rasgo, niveles in acum.items()
    }


# --- Barrido de réplicas (criterio de calibración, opción A) ------------------
#
# Con 500 personas de calibración y 500 de prueba, la cobertura de una sola
# corrida varía unos 3 puntos al 50 % solo por azar, y evaluar ±5 puntos en
# cada rasgo y cada nivel falla la mayoría de las veces aunque el modelo sea
# perfecto. Por eso el criterio se evalúa sobre la cobertura PROMEDIO de
# REPLICAS corridas completas del módulo (generar fenotipos, ajustar, calibrar
# y evaluar), cada una con su propia semilla: el desvío del promedio baja a
# menos de 1 punto.
#
# Cada módulo exporta barrido_<dominio>.csv con una fila por réplica, rasgo y
# nivel: cuántas personas de prueba quedaron dentro de su intervalo. La réplica
# 0 es la corrida principal, la misma de predicciones_ e intervalos_, y se
# recalcula de forma central para verificar que los conteos son genuinos.

REPLICAS = 20

CAMPOS_BARRIDO = ("replica", "semilla", "dominio", "rasgo", "nivel", "n", "dentro")


def fila_barrido(
    replica: int, semilla: int, dominio: str, rasgo: str, nivel: float,
    y: np.ndarray, lo: np.ndarray, hi: np.ndarray,
) -> dict:
    """Arma una fila de barrido a partir de los valores reales y los intervalos de una réplica."""
    y, lo, hi = (np.asarray(v, dtype=float) for v in (y, lo, hi))
    return {
        "replica": int(replica), "semilla": int(semilla), "dominio": dominio,
        "rasgo": rasgo, "nivel": float(nivel), "n": int(len(y)),
        "dentro": int(np.sum((lo <= y) & (y <= hi))),
    }


def exportar_barrido(filas: list[dict], dominio: str, directorio: str | Path = ".") -> Path:
    """Escribe barrido_<dominio>.csv (campos de CAMPOS_BARRIDO)."""
    if dominio not in DOMINIOS:
        raise ValueError(f"dominio debe ser uno de {DOMINIOS}, no {dominio!r}")
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"barrido_{dominio}.csv"
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS_BARRIDO)
        escritor.writeheader()
        escritor.writerows(filas)
    return ruta


def cargar_barrido(ruta: str | Path) -> list[dict]:
    """Lee un archivo de barrido y castea los campos numéricos."""
    with Path(ruta).open(encoding="utf-8") as fh:
        lector = csv.DictReader(fh)
        if tuple(lector.fieldnames or ()) != CAMPOS_BARRIDO:
            raise AssertionError(
                f"{Path(ruta).name}: columnas {lector.fieldnames}, se esperaba {list(CAMPOS_BARRIDO)}"
            )
        return [
            {
                **r,
                "replica": int(r["replica"]), "semilla": int(r["semilla"]),
                "nivel": float(r["nivel"]), "n": int(r["n"]), "dentro": int(r["dentro"]),
            }
            for r in lector
        ]


def verificar_barrido(
    barrido: list[dict], intervalos: list[dict], predicciones: list[dict],
    ids_prueba: set[int], dominio: str,
) -> None:
    """Valida barrido_<dominio>.csv. La réplica 0 tiene que coincidir con el recálculo central."""
    if not barrido:
        raise AssertionError(f"{dominio}: el archivo de barrido está vacío")
    if {f["dominio"] for f in barrido} - {dominio}:
        raise AssertionError(f"{dominio}: hay filas de barrido de otro dominio")

    replicas = sorted({f["replica"] for f in barrido})
    if replicas != list(range(REPLICAS)):
        raise AssertionError(f"{dominio}: réplicas {replicas}, se esperaban 0..{REPLICAS - 1}")

    semillas = {}
    for f in barrido:
        semillas.setdefault(f["replica"], set()).add(f["semilla"])
    if any(len(s) != 1 for s in semillas.values()):
        raise AssertionError(f"{dominio}: una réplica usa más de una semilla")
    if len({next(iter(s)) for s in semillas.values()}) != REPLICAS:
        raise AssertionError(f"{dominio}: hay réplicas con la misma semilla")

    rasgos = {f["rasgo"] for f in predicciones}
    niveles = {round(v, 6) for v in NIVELES}
    vistos = set()
    for f in barrido:
        clave = (f["replica"], f["rasgo"], round(f["nivel"], 6))
        if clave in vistos:
            raise AssertionError(f"{dominio}: fila de barrido repetida {clave}")
        vistos.add(clave)
        if f["n"] != len(ids_prueba):
            raise AssertionError(f"{dominio}: réplica {f['replica']} evalúa {f['n']} personas, no {len(ids_prueba)}")
        if not 0 <= f["dentro"] <= f["n"]:
            raise AssertionError(f"{dominio}: conteo imposible en {clave}")
    esperadas = {(r, ra, n) for r in range(REPLICAS) for ra in rasgos for n in niveles}
    if vistos != esperadas:
        raise AssertionError(f"{dominio}: el barrido no cubre todas las réplicas, rasgos y niveles")

    # La réplica 0 es la corrida principal: se recalcula contra los intervalos.
    y = {(f["rasgo"], f["individuo_id"]): f["y_real"] for f in predicciones}
    central: dict[tuple, int] = {}
    for f in intervalos:
        clave = (f["rasgo"], round(f["nivel"], 6))
        dentro = f["lo"] <= y[(f["rasgo"], f["individuo_id"])] <= f["hi"]
        central[clave] = central.get(clave, 0) + int(dentro)
    for f in barrido:
        if f["replica"] == 0 and central[(f["rasgo"], round(f["nivel"], 6))] != f["dentro"]:
            raise AssertionError(
                f"{dominio}: la réplica 0 del barrido no coincide con los intervalos exportados "
                f"(rasgo {f['rasgo']}, nivel {f['nivel']})"
            )


def cobertura_promedio(barrido: list[dict]) -> dict[str, dict[float, float]]:
    """Cobertura promedio sobre las réplicas, por rasgo y nivel. Es la que se evalúa."""
    acum: dict[str, dict[float, list[float]]] = {}
    for f in barrido:
        acum.setdefault(f["rasgo"], {}).setdefault(round(f["nivel"], 6), []).append(f["dentro"] / f["n"])
    return {
        rasgo: {nivel: float(np.mean(v)) for nivel, v in sorted(niveles.items())}
        for rasgo, niveles in acum.items()
    }


# ---------------------------------------------------------------------------
# Recuperación del nivel declarado (criterio de la hipótesis reformulada)
#
# El 3/11/2026 el tutor aprobó la reformulación de la hipótesis: lo que el
# trabajo afirma por dominio es que el modelo recupera el nivel de señal
# genética DECLARADO, no un ordenamiento entre dominios. La condición que
# puede fallar es que el ΔR² parcial recuperado quede dentro de TOLERANCIA
# del nivel declarado, evaluado sobre el promedio del barrido de REPLICAS.
#
# La tolerancia se fijó por simulación en dos pasos.
#
# Primero, con la línea base tomada como conocida: el desvío absoluto respecto
# del nivel declarado se mantiene por debajo de 0,0043 en el 95 % de los
# experimentos, y sobre una réplica individual la dispersión es cuatro veces
# mayor (hasta 0,016). De ahí que el criterio SOLO tenga sentido sobre el
# promedio del barrido.
#
# Segundo, y es la corrección del 9/11/2026 que trajo la nota de Rostro: la
# línea base no se conoce, se ESTIMA sobre `entrenamiento` y se evalúa fuera de
# muestra. Estimar sus k coeficientes sobre n personas cuesta alrededor de k/n
# en R² fuera de muestra (con k = 6 y n = 1.000, unos -0,009 cuando el efecto
# real es cero), y ese costo queda solo en el denominador de la línea base, de
# modo que el parcial sale SISTEMÁTICAMENTE POR ENCIMA del nivel declarado.
# Medido por simulación sobre el promedio de 20 réplicas, el corrimiento es de
# +0,0045 a +0,0065 con línea base de efecto nulo (dominios facial y
# temperamental) y de +0,0064 a +0,0087 con línea base de efecto real (dominio
# vocal), con un percentil 95 del desvío absoluto de 0,0091.
#
# La tolerancia de 0,015 absorbe ese corrimiento y su ruido, y sigue pudiendo
# fallar: un error en la línea base, en el bootstrap o en el generador se
# aparta mucho más que 1,5 puntos. El corrimiento se declara en lugar de
# corregirse, porque es una propiedad de estimar la línea base y no del
# circuito del genotipo.
#
# El criterio se evalúa sobre el CASO BASE. El segundo nivel declarado se
# informa con su intervalo y sin criterio de recuperación: con 500 personas de
# prueba, un nivel del orden del 0,15 % no se distingue de cero, y una
# tolerancia absoluta de 0,015 sería diez veces el nivel que pretende verificar.
#
# El ΔR² por réplica no viaja en barrido_<dominio>.csv, que es por nivel
# nominal y no por línea base. Va en su propio archivo.

TOLERANCIA_RECUPERACION = 0.015


class ContratoError(ValueError):
    """Incumplimiento del contrato de archivos entre módulos.

    Definida acá para que la capa de calibración y los validadores del paquete
    levanten el mismo tipo y un módulo pueda capturar uno solo. Subclase de
    ValueError, así el código que ya capturaba ValueError sigue funcionando.
    """

CAMPOS_RECUPERACION = (
    "replica",            # 0 a REPLICAS - 1
    "semilla",            # la de la réplica
    "dominio",            # uno de DOMINIOS
    "rasgo",              # el mismo nombre que en CAMPOS
    "linea_base",         # uno de LINEAS_BASE
    "nivel_declarado",    # la señal declarada del rasgo, de la configuración
    "delta_r2_total",     # sobre la varianza total
    "delta_r2_parcial",   # sobre lo que la línea base no explica
)


def fila_recuperacion(
    replica: int, semilla: int, dominio: str, rasgo: str, linea_base: str,
    nivel_declarado: float, delta_r2_total: float, delta_r2_parcial: float,
) -> dict:
    """Arma una fila de recuperación. El nivel declarado sale de la configuración
    del módulo, nunca de un resultado observado."""
    if dominio not in DOMINIOS:
        raise ValueError(f"dominio debe ser uno de {DOMINIOS}, no {dominio!r}")
    if linea_base not in LINEAS_BASE:
        raise ValueError(f"linea_base debe ser una de {LINEAS_BASE}, no {linea_base!r}")
    return {
        "replica": int(replica), "semilla": int(semilla), "dominio": dominio,
        "rasgo": rasgo, "linea_base": linea_base,
        "nivel_declarado": float(nivel_declarado),
        "delta_r2_total": float(delta_r2_total),
        "delta_r2_parcial": float(delta_r2_parcial),
    }


def exportar_recuperacion(filas: list[dict], dominio: str, directorio: str | Path = ".") -> Path:
    """Escribe recuperacion_<dominio>.csv (campos de CAMPOS_RECUPERACION)."""
    if dominio not in DOMINIOS:
        raise ValueError(f"dominio debe ser uno de {DOMINIOS}, no {dominio!r}")
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)
    ruta = directorio / f"recuperacion_{dominio}.csv"
    with ruta.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS_RECUPERACION)
        escritor.writeheader()
        escritor.writerows(filas)
    return ruta


def cargar_recuperacion(ruta: str | Path) -> list[dict]:
    """Lee recuperacion_<dominio>.csv con los tipos correctos."""
    enteros = ("replica", "semilla")
    flotantes = ("nivel_declarado", "delta_r2_total", "delta_r2_parcial")
    filas = []
    with Path(ruta).open(newline="", encoding="utf-8") as fh:
        for f in csv.DictReader(fh):
            if set(f) != set(CAMPOS_RECUPERACION):
                raise AssertionError(
                    f"{ruta}: columnas {sorted(f)}, se esperaban "
                    f"{sorted(CAMPOS_RECUPERACION)}"
                )
            filas.append({
                **{k: int(f[k]) for k in enteros},
                **{k: float(f[k]) for k in flotantes},
                "dominio": f["dominio"], "rasgo": f["rasgo"],
                "linea_base": f["linea_base"],
            })
    return filas


def verificar_recuperacion(filas: list[dict], dominio: str) -> None:
    """Valida recuperacion_<dominio>.csv antes de evaluar el criterio."""
    if not filas:
        raise AssertionError(f"{dominio}: el archivo de recuperación está vacío")
    malos = {f["dominio"] for f in filas} - {dominio}
    if malos:
        raise AssertionError(f"{dominio}: hay filas con dominio {malos}")
    lb = {f["linea_base"] for f in filas}
    faltan = set(LINEAS_BASE) - lb
    if faltan:
        raise AssertionError(f"{dominio}: falta la línea base {sorted(faltan)}")
    replicas = {f["replica"] for f in filas}
    if replicas != set(range(REPLICAS)):
        raise AssertionError(
            f"{dominio}: se esperaban las réplicas 0 a {REPLICAS - 1}, hay "
            f"{len(replicas)}"
        )
    for f in filas:
        for k in ("nivel_declarado", "delta_r2_total", "delta_r2_parcial"):
            if not math.isfinite(f[k]):
                raise AssertionError(f"{dominio}: {k} no es finito en la réplica {f['replica']}")
        if f["nivel_declarado"] < 0:
            raise AssertionError(f"{dominio}: nivel_declarado negativo en {f['rasgo']}")
    # Un rasgo tiene que declarar el mismo nivel en las 20 réplicas: si cambia,
    # el nivel se eligió mirando resultados.
    por_rasgo: dict[tuple[str, str], set[float]] = {}
    for f in filas:
        por_rasgo.setdefault((f["rasgo"], f["linea_base"]), set()).add(
            round(f["nivel_declarado"], 10)
        )
    movidos = {k: v for k, v in por_rasgo.items() if len(v) > 1}
    if movidos:
        raise AssertionError(
            f"{dominio}: el nivel declarado cambia entre réplicas en "
            f"{sorted(k[0] for k in movidos)}. Tiene que salir de la "
            "configuración y ser el mismo en las 20."
        )
    vistos = {(f["replica"], f["rasgo"], f["linea_base"]) for f in filas}
    if len(vistos) != len(filas):
        raise AssertionError(f"{dominio}: hay filas duplicadas por (réplica, rasgo, línea base)")


def criterio_recuperacion(
    filas: list[dict],
    linea_base: str = "ancestria sexo edad",
    tolerancia: float = TOLERANCIA_RECUPERACION,
) -> dict:
    """Evalúa el criterio de recuperación sobre el promedio del barrido.

    Promedia el ΔR² parcial de cada rasgo sobre las REPLICAS réplicas, lo
    compara con el nivel declarado de ese rasgo, y agrega por dominio
    promediando los rasgos. El criterio se cumple si el desvío absoluto del
    agregado queda dentro de `tolerancia`.

    Devuelve también el desvío por rasgo: si el agregado pasa pero un rasgo
    se aparta mucho, conviene verlo antes de informar.
    """
    sel = [f for f in filas if f["linea_base"] == linea_base]
    if not sel:
        raise ValueError(f"no hay filas para la línea base {linea_base!r}")
    acum: dict[str, dict[str, list[float]]] = {}
    declarado: dict[str, float] = {}
    for f in sel:
        acum.setdefault(f["rasgo"], {}).setdefault("parcial", []).append(f["delta_r2_parcial"])
        acum[f["rasgo"]].setdefault("total", []).append(f["delta_r2_total"])
        declarado[f["rasgo"]] = f["nivel_declarado"]
    por_rasgo = {
        rasgo: {
            "nivel_declarado": declarado[rasgo],
            "parcial_promedio": float(np.mean(v["parcial"])),
            "total_promedio": float(np.mean(v["total"])),
            "desvio": float(np.mean(v["parcial"]) - declarado[rasgo]),
            "replicas": len(v["parcial"]),
        }
        for rasgo, v in sorted(acum.items())
    }
    recuperado = float(np.mean([v["parcial_promedio"] for v in por_rasgo.values()]))
    nivel = float(np.mean([v["nivel_declarado"] for v in por_rasgo.values()]))
    desvio = recuperado - nivel
    return {
        "linea_base": linea_base,
        "tolerancia": float(tolerancia),
        "nivel_declarado": nivel,
        "parcial_recuperado": recuperado,
        "total_recuperado": float(np.mean([v["total_promedio"] for v in por_rasgo.values()])),
        "desvio": desvio,
        "cumple": bool(abs(desvio) <= tolerancia),
        "peor_rasgo": max(por_rasgo, key=lambda r: abs(por_rasgo[r]["desvio"])),
        "por_rasgo": por_rasgo,
    }
