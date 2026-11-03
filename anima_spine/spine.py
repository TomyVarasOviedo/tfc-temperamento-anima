"""Columna vertebral de la cohorte — proyecto Anima.

Genera UNA sola cohorte de individuos que los tres módulos comparten.
No contiene nada de rostro, voz ni temperamento: solo quiénes son los
individuos y en qué partición cae cada uno.

Cada módulo lee este archivo, se queda con la columna `individuo_id` y
genera sus propios valores genéticos latentes para su propio espacio de
rasgos. El PCA de Audrey, los parámetros acústicos de Diego y los cinco
rasgos de Tomás siguen siendo de cada uno.

Se genera UNA vez, con semilla fija, y no se vuelve a tocar. Si hay que
regenerarla, se sube la versión y se avisa a los tres, porque todos los
resultados calculados sobre la versión anterior dejan de ser comparables.

La fuente de verdad es el CSV publicado, no este código: el generador de
números aleatorios de numpy no garantiza la misma secuencia en todas sus
versiones. Por eso los metadatos guardan la huella SHA-256 del CSV y la
versión de numpy con la que se generó, y `cargar` verifica la huella.

Cambios de la versión 2.0 respecto de la 1.1 (ver README):
  - la cohorte es mixta: incluye a las 503 personas europeas reales de
    1000 Genomes (fase 3), con su subpoblación y su sexo reales, y se
    completa con personas simuladas hasta los mismos totales por
    subpoblación que la 1.1;
  - columnas nuevas: `origen` ("1000g" o "simulada") y `muestra_1000g`
    (el identificador de la muestra, vacío en las simuladas);
  - las particiones se estratifican además por origen, así las personas
    reales quedan repartidas en los tres grupos;
  - el listado de las 503 personas viaja con el paquete
    (`datos/muestras_1000g_eur.csv`) y su huella va en los metadatos.
  Los rasgos (voz, cara, personalidad) siguen siendo simulados: 1000
  Genomes no informa ninguno.

Cambios de la versión 1.1 respecto de la 1.0:
  - conteos exactos por subpoblación (antes se sorteaban);
  - sexo balanceado y codificado: 0 = mujer, 1 = hombre;
  - particiones estratificadas por subpoblación y sexo;
  - huella SHA-256 y versión de numpy en los metadatos;
  - `verificar` controla tamaños, valores y huella.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from dataclasses import dataclass
from pathlib import Path

import numpy as np

VERSION = "2.0"
SEMILLA = 20261019

# Tamaños del panel europeo de 1000 Genomes (fase 3), usados como
# proporciones. Verificados contra Ensembl: CEU 99, GBR 91, FIN 99,
# IBS 107, TSI 107 (total 503).
SUBPOBLACIONES = {"CEU": 99, "GBR": 91, "FIN": 99, "IBS": 107, "TSI": 107}

# Codificación de sexo, común a los tres módulos.
SEXO = {0: "mujer", 1: "hombre"}

# Proporciones de la partición. La de calibración no puede solaparse con
# la de entrenamiento: si se solapan, la cobertura sale optimista y el
# criterio de ±5 pp deja de significar algo.
PARTICIONES = {"entrenamiento": 0.50, "calibracion": 0.25, "prueba": 0.25}

# Rango de edad común a los tres módulos. Ver README, "Rango de edad".
RANGO_EDAD = (18, 80)

CAMPOS = (
    "individuo_id", "subpoblacion", "sexo", "edad", "particion", "origen", "muestra_1000g",
)

# Origen de cada persona de la cohorte.
ORIGENES = ("1000g", "simulada")

# Listado de las personas reales que viaja con el paquete. Se prepara una
# sola vez desde el panel público de 1000 Genomes con `preparar_panel`.
PANEL_PAQUETE = Path(__file__).resolve().parent / "datos" / "muestras_1000g_eur.csv"
CAMPOS_PANEL = ("muestra_1000g", "subpoblacion", "sexo")


@dataclass(frozen=True)
class Spine:
    """La cohorte compartida.

    Attributes:
        filas: una por individuo, con las claves de CAMPOS.
        meta: semilla, versión y parámetros de generación. Va al informe.
    """

    filas: list[dict]
    meta: dict

    def ids(self, particion: str | None = None) -> list[int]:
        """Los individuo_id, opcionalmente filtrados por partición."""
        if particion is not None and particion not in PARTICIONES:
            raise ValueError(f"partición desconocida: {particion!r}")
        return [
            f["individuo_id"]
            for f in self.filas
            if particion is None or f["particion"] == particion
        ]


def _mayor_resto(total: int, pesos: dict[str, float]) -> dict[str, int]:
    """Reparte `total` en enteros proporcionales a `pesos` (método del mayor resto)."""
    claves = list(pesos)
    w = np.array([pesos[k] for k in claves], dtype=float)
    cuotas = w / w.sum() * total
    base = np.floor(cuotas).astype(int)
    faltan = total - int(base.sum())
    # desempate estable: mayor resto primero, después el orden declarado
    orden = sorted(range(len(claves)), key=lambda i: (-(cuotas[i] - base[i]), i))
    for i in orden[:faltan]:
        base[i] += 1
    return {k: int(v) for k, v in zip(claves, base)}


def _asignacion_sistematica(n: int, proporciones: dict[str, float]) -> list[str]:
    """Secuencia de n etiquetas con totales exactos y repartidas de forma pareja.

    En cada posición elige la etiqueta más atrasada respecto de su cuota.
    Aplicada sobre las filas ordenadas por estrato, deja cada estrato a
    ±1 persona de las proporciones y los totales exactos.
    """
    totales = _mayor_resto(n, proporciones)
    etiquetas = list(proporciones)
    asignadas = {e: 0 for e in etiquetas}
    salida = []
    for i in range(n):
        deficit = {
            e: totales[e] * (i + 1) / n - asignadas[e]
            for e in etiquetas
            if asignadas[e] < totales[e]
        }
        elegida = max(deficit, key=lambda e: (deficit[e], -etiquetas.index(e)))
        asignadas[elegida] += 1
        salida.append(elegida)
    return salida


def preparar_panel(ruta_panel_1000g: str | Path, destino: str | Path = PANEL_PAQUETE) -> Path:
    """Extrae del panel público de 1000 Genomes las 503 personas europeas.

    El panel es `integrated_call_samples_v3.20130502.ALL.panel` (fase 3):
    texto separado por tabuladores, con las columnas sample, pop,
    super_pop y gender. Se queda con las cinco subpoblaciones de
    SUBPOBLACIONES, codifica el sexo como 0 = mujer y 1 = hombre, y
    verifica que los conteos coincidan con el panel europeo publicado.
    """
    filas = []
    with Path(ruta_panel_1000g).open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh, delimiter="\t"):
            pop = (r.get("pop") or "").strip()
            if pop not in SUBPOBLACIONES:
                continue
            genero = (r.get("gender") or "").strip().lower()
            if genero not in ("female", "male"):
                raise AssertionError(f"{r.get('sample')}: sexo desconocido {genero!r}")
            filas.append({
                "muestra_1000g": r["sample"].strip(),
                "subpoblacion": pop,
                "sexo": 0 if genero == "female" else 1,
            })
    filas.sort(key=lambda f: f["muestra_1000g"])
    _validar_panel(filas)
    destino = Path(destino)
    destino.parent.mkdir(parents=True, exist_ok=True)
    with destino.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS_PANEL, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(filas)
    return destino


def cargar_panel(ruta: str | Path = PANEL_PAQUETE) -> list[dict]:
    """Lee el listado de personas reales que usa `generar`."""
    if not Path(ruta).exists():
        raise FileNotFoundError(
            f"falta {ruta}: se prepara una sola vez con preparar_panel(<panel de 1000 Genomes>). "
            "El panel es integrated_call_samples_v3.20130502.ALL.panel, de la fase 3."
        )
    with Path(ruta).open(encoding="utf-8") as fh:
        filas = [
            {"muestra_1000g": r["muestra_1000g"], "subpoblacion": r["subpoblacion"], "sexo": int(r["sexo"])}
            for r in csv.DictReader(fh)
        ]
    _validar_panel(filas)
    return filas


def _validar_panel(filas: list[dict]) -> None:
    conteo = {k: 0 for k in SUBPOBLACIONES}
    for f in filas:
        conteo[f["subpoblacion"]] += 1
    if conteo != SUBPOBLACIONES:
        raise AssertionError(
            f"el panel no coincide con el panel europeo de 1000 Genomes: {conteo}, "
            f"se esperaba {SUBPOBLACIONES}"
        )
    muestras = [f["muestra_1000g"] for f in filas]
    if len(set(muestras)) != len(muestras) or any(not m for m in muestras):
        raise AssertionError("el panel tiene muestras repetidas o vacías")
    if {f["sexo"] for f in filas} - set(SEXO):
        raise AssertionError("el panel tiene valores de sexo distintos de 0 y 1")


def generar(n: int = 2000, semilla: int = SEMILLA, panel: list[dict] | None = None) -> Spine:
    """Genera la cohorte compartida.

    Args:
        n: cantidad de individuos. Con 2000 (500 de prueba) el contraste
            promediado por dominio alcanza potencia 0,80 a dos colas para una
            brecha de 0,025 en ΔR². Evaluado rasgo por rasgo haría falta una
            cohorte del orden de 16.000.
        semilla: fija y documentada. No cambiarla sin subir la versión.
    """
    rng = np.random.default_rng(semilla)

    # 1. Subpoblación: conteos exactos proporcionales al panel.
    conteos = _mayor_resto(n, SUBPOBLACIONES)

    # 2. Personas reales de 1000 Genomes, con su subpoblación y su sexo.
    if panel is None:
        panel = cargar_panel()
    else:
        _validar_panel(panel)
    reales = sorted(panel, key=lambda f: f["muestra_1000g"])
    for subpob, n_k in conteos.items():
        n_real = sum(1 for f in reales if f["subpoblacion"] == subpob)
        if n_real > n_k:
            raise ValueError(
                f"con n = {n}, {subpob} tiene lugar para {n_k} personas y hay {n_real} reales"
            )

    # 3. Sexo: mitad y mitad dentro de cada subpoblación, contando a las
    #    reales. En las impares, el individuo sobrante alterna entre mujer y
    #    hombre. Las simuladas completan lo que falta de cada sexo.
    filas = []
    impar = 0
    for subpob, n_k in conteos.items():
        mujeres = n_k // 2
        if n_k % 2:
            mujeres += 1 if impar % 2 == 0 else 0
            impar += 1
        propias = [f for f in reales if f["subpoblacion"] == subpob]
        for f in propias:
            filas.append({"subpoblacion": subpob, "sexo": f["sexo"],
                          "origen": "1000g", "muestra_1000g": f["muestra_1000g"]})
        mujeres_sim = max(0, mujeres - sum(1 for f in propias if f["sexo"] == 0))
        n_sim = n_k - len(propias)
        mujeres_sim = min(mujeres_sim, n_sim)
        for s_ in [0] * mujeres_sim + [1] * (n_sim - mujeres_sim):
            filas.append({"subpoblacion": subpob, "sexo": s_,
                          "origen": "simulada", "muestra_1000g": ""})

    # 4. Edad: entera y uniforme en el rango declarado, independiente del resto.
    edades = rng.integers(RANGO_EDAD[0], RANGO_EDAD[1] + 1, size=n)
    for f, e in zip(filas, edades):
        f["edad"] = int(e)

    # 5. Partición estratificada: se ordena por estrato (subpoblación, sexo,
    #    origen) y al azar dentro del estrato, y se asigna en forma sistemática.
    azar = rng.random(n)
    orden = sorted(
        range(n),
        key=lambda i: (filas[i]["subpoblacion"], filas[i]["sexo"], filas[i]["origen"], azar[i]),
    )
    etiquetas = _asignacion_sistematica(n, PARTICIONES)
    for pos, i in enumerate(orden):
        filas[i]["particion"] = etiquetas[pos]

    # 6. Orden final al azar e identificadores del 1 al n.
    filas = [filas[i] for i in rng.permutation(n)]
    filas = [
        {
            "individuo_id": k + 1,
            "subpoblacion": f["subpoblacion"],
            "sexo": f["sexo"],
            "edad": f["edad"],
            "particion": f["particion"],
            "origen": f["origen"],
            "muestra_1000g": f["muestra_1000g"],
        }
        for k, f in enumerate(filas)
    ]

    meta = {
        "version": VERSION,
        "semilla": semilla,
        "n": n,
        "subpoblaciones_panel": SUBPOBLACIONES,
        "conteos_subpoblacion": conteos,
        "sexo_codificacion": {str(k): v for k, v in SEXO.items()},
        "conteos_sexo": {
            str(s): sum(1 for f in filas if f["sexo"] == s) for s in SEXO
        },
        "particiones": PARTICIONES,
        "conteos_particion": _mayor_resto(n, PARTICIONES),
        "estratificacion": "subpoblacion x sexo x origen",
        "origenes": list(ORIGENES),
        "conteos_origen": {o: sum(1 for f in filas if f["origen"] == o) for o in ORIGENES},
        "conteos_origen_particion": {
            o: {p: sum(1 for f in filas if f["origen"] == o and f["particion"] == p) for p in PARTICIONES}
            for o in ORIGENES
        },
        "panel_1000g": {
            "fuente": "1000 Genomes fase 3, integrated_call_samples_v3.20130502.ALL.panel",
            "personas": len(reales),
            "sha256_listado": hashlib.sha256(
                "\n".join(f"{f['muestra_1000g']},{f['subpoblacion']},{f['sexo']}" for f in reales).encode("utf-8")
            ).hexdigest(),
        },
        "rango_edad": list(RANGO_EDAD),
        "distribucion_edad": "uniforme entera",
        "numpy_version": np.__version__,
        "python_version": platform.python_version(),
        "supuestos_de_diseno": [
            "La edad no existe en 1000 Genomes: se genera uniforme entre 18 y 80. "
            "Es un supuesto de diseño común a los tres módulos, no un dato.",
            "El efecto de edad, sexo y ancestría sobre cada fenotipo lo define "
            "cada módulo y debe documentarse como supuesto.",
            "La cohorte es mixta: las 503 personas europeas de 1000 Genomes "
            "son reales en subpoblación, sexo y genoma; el resto son simuladas, "
            "con los mismos totales por subpoblación que la versión 1.1.",
            "Los rasgos de todas las personas, reales o simuladas, son "
            "simulados: 1000 Genomes no informa voz, cara ni personalidad.",
            "Sexo balanceado dentro de cada subpoblación contando a las "
            "personas reales; las simuladas completan lo que falta.",
        ],
    }
    return Spine(filas=filas, meta=meta)


def huella(ruta: str | Path) -> str:
    """SHA-256 del archivo."""
    return hashlib.sha256(Path(ruta).read_bytes()).hexdigest()


def exportar(spine: Spine, directorio: str | Path = ".") -> tuple[Path, Path]:
    """Escribe la cohorte y su metadata (con la huella del CSV). Devuelve (csv, json)."""
    directorio = Path(directorio)
    directorio.mkdir(parents=True, exist_ok=True)

    ruta_csv = directorio / "cohorte_anima.csv"
    with ruta_csv.open("w", newline="", encoding="utf-8") as fh:
        escritor = csv.DictWriter(fh, fieldnames=CAMPOS, lineterminator="\n")
        escritor.writeheader()
        escritor.writerows(spine.filas)

    meta = dict(spine.meta)
    meta["sha256_csv"] = huella(ruta_csv)
    ruta_json = directorio / "cohorte_anima.meta.json"
    ruta_json.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    return ruta_csv, ruta_json


def cargar(ruta: str | Path = "cohorte_anima.csv") -> Spine:
    """Lee la cohorte compartida. Es lo que llama cada módulo.

    Si los metadatos traen la huella del CSV, la verifica: un archivo
    modificado a mano o de otra versión no se carga en silencio.
    """
    ruta = Path(ruta)
    ruta_meta = ruta.with_suffix(".meta.json")
    meta = json.loads(ruta_meta.read_text(encoding="utf-8")) if ruta_meta.exists() else {}

    if "sha256_csv" in meta and huella(ruta) != meta["sha256_csv"]:
        raise AssertionError(
            f"{ruta.name} no coincide con la huella de sus metadatos: el archivo "
            "fue modificado o no corresponde a esa versión de la cohorte."
        )

    with ruta.open(encoding="utf-8") as fh:
        filas = [
            {
                "individuo_id": int(r["individuo_id"]),
                "subpoblacion": r["subpoblacion"],
                "sexo": int(r["sexo"]),
                "edad": int(r["edad"]),
                "particion": r["particion"],
                "origen": r.get("origen") or "simulada",
                "muestra_1000g": r.get("muestra_1000g") or "",
            }
            for r in csv.DictReader(fh)
        ]
    return Spine(filas=filas, meta=meta)


def verificar(spine: Spine) -> None:
    """Chequea lo que tiene que ser cierto. Correr después de generar o cargar."""
    n = len(spine.filas)
    ids = [f["individuo_id"] for f in spine.filas]
    if sorted(ids) != list(range(1, n + 1)):
        raise AssertionError("los individuo_id no son 1..n sin repetidos")

    if "n" in spine.meta and spine.meta["n"] != n:
        raise AssertionError(f"la cohorte tiene {n} filas y los metadatos dicen {spine.meta['n']}")

    malos = {f["subpoblacion"] for f in spine.filas} - set(SUBPOBLACIONES)
    if malos:
        raise AssertionError(f"subpoblaciones desconocidas: {sorted(malos)}")

    if {f["sexo"] for f in spine.filas} - set(SEXO):
        raise AssertionError("hay valores de sexo distintos de 0 (mujer) y 1 (hombre)")

    if any(not RANGO_EDAD[0] <= f["edad"] <= RANGO_EDAD[1] for f in spine.filas):
        raise AssertionError("hay edades fuera del rango declarado")

    if {f.get("origen", "simulada") for f in spine.filas} - set(ORIGENES):
        raise AssertionError(f"hay valores de origen distintos de {ORIGENES}")
    muestras = [f.get("muestra_1000g", "") for f in spine.filas if f.get("origen") == "1000g"]
    if any(not m for m in muestras) or len(set(muestras)) != len(muestras):
        raise AssertionError("hay personas reales sin muestra de 1000 Genomes o con muestras repetidas")
    if any(f.get("muestra_1000g") for f in spine.filas if f.get("origen", "simulada") == "simulada"):
        raise AssertionError("hay personas simuladas con muestra de 1000 Genomes")
    if "conteos_origen" in spine.meta:
        conteo_origen = {o: sum(1 for f in spine.filas if f.get("origen") == o) for o in ORIGENES}
        if conteo_origen != spine.meta["conteos_origen"]:
            raise AssertionError(
                f"conteos por origen {conteo_origen}, los metadatos dicen {spine.meta['conteos_origen']}"
            )

    conteo: dict[str, int] = {}
    for f in spine.filas:
        conteo[f["particion"]] = conteo.get(f["particion"], 0) + 1
    if set(conteo) != set(PARTICIONES):
        raise AssertionError(f"particiones inesperadas: {sorted(conteo)}")
    esperado = _mayor_resto(n, PARTICIONES)
    if conteo != esperado:
        raise AssertionError(f"tamaños de partición {conteo}, se esperaba {esperado}")


if __name__ == "__main__":
    spine = generar()
    verificar(spine)
    csv_path, json_path = exportar(spine)
    print(f"Cohorte {VERSION} generada: {len(spine.filas)} individuos "
          f"({spine.meta['conteos_origen']['1000g']} de 1000 Genomes)")
    for etiqueta in PARTICIONES:
        print(f"  {etiqueta:>14}: {len(spine.ids(etiqueta))}")
    print(f"\n{csv_path}\n{json_path}\nSHA-256: {huella(csv_path)}")
