"""Propiedades del generador de la cohorte (versión 2.0, mixta).

Estos tests no dependen del CSV publicado ni del panel real de 1000
Genomes: arman un panel de prueba con los mismos conteos por
subpoblación y regeneran la cohorte en memoria.
"""

from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from anima_spine import spine

REALES = {"CEU": 99, "FIN": 99, "GBR": 91, "IBS": 107, "TSI": 107}


def panel_de_prueba(semilla: int = 1) -> list[dict]:
    """503 personas con los conteos del panel europeo y sexo al azar."""
    rng = np.random.default_rng(semilla)
    filas, k = [], 0
    for subpob, m in REALES.items():
        for _ in range(m):
            k += 1
            filas.append({"muestra_1000g": f"PRUEBA{k:04d}", "subpoblacion": subpob,
                          "sexo": int(rng.integers(0, 2))})
    return filas


@pytest.fixture(scope="module")
def panel():
    return panel_de_prueba()


@pytest.fixture(scope="module")
def cohorte(panel):
    return spine.generar(panel=panel)


def test_conteos_exactos_por_subpoblacion(cohorte):
    conteo = Counter(f["subpoblacion"] for f in cohorte.filas)
    assert conteo == {"CEU": 394, "FIN": 394, "GBR": 362, "IBS": 425, "TSI": 425}


def test_incluye_a_todas_las_personas_reales(cohorte, panel):
    reales = [f for f in cohorte.filas if f["origen"] == "1000g"]
    assert len(reales) == 503
    assert Counter(f["subpoblacion"] for f in reales) == REALES
    assert {f["muestra_1000g"] for f in reales} == {f["muestra_1000g"] for f in panel}


def test_las_reales_conservan_su_subpoblacion_y_su_sexo(cohorte, panel):
    por_muestra = {f["muestra_1000g"]: f for f in panel}
    for f in cohorte.filas:
        if f["origen"] == "1000g":
            original = por_muestra[f["muestra_1000g"]]
            assert (f["subpoblacion"], f["sexo"]) == (original["subpoblacion"], original["sexo"])


def test_las_simuladas_no_tienen_muestra(cohorte):
    simuladas = [f for f in cohorte.filas if f["origen"] == "simulada"]
    assert len(simuladas) == 1497
    assert all(f["muestra_1000g"] == "" for f in simuladas)


def test_sexo_balanceado_y_codificado(cohorte):
    conteo = Counter(f["sexo"] for f in cohorte.filas)
    assert conteo == {0: 1000, 1: 1000}
    assert spine.SEXO == {0: "mujer", 1: "hombre"}


def test_tamanos_de_particion(cohorte):
    conteo = Counter(f["particion"] for f in cohorte.filas)
    assert conteo == {"entrenamiento": 1000, "calibracion": 500, "prueba": 500}


def test_particiones_estratificadas_por_origen(cohorte):
    """Cada estrato (subpoblación x sexo x origen) queda a ±1 persona de 50/25/25."""
    clave = lambda f: (f["subpoblacion"], f["sexo"], f["origen"])
    estratos = Counter(clave(f) for f in cohorte.filas)
    por_estrato = Counter(clave(f) + (f["particion"],) for f in cohorte.filas)
    for estrato, m in estratos.items():
        for particion, prop in spine.PARTICIONES.items():
            assert abs(por_estrato[estrato + (particion,)] - prop * m) <= 1, (estrato, particion)


def test_las_reales_quedan_repartidas_en_las_tres_particiones(cohorte):
    conteo = Counter(f["particion"] for f in cohorte.filas if f["origen"] == "1000g")
    assert abs(conteo["entrenamiento"] - 251.5) <= 10
    assert abs(conteo["calibracion"] - 125.75) <= 10
    assert abs(conteo["prueba"] - 125.75) <= 10


def test_edades_en_rango_y_todas_presentes(cohorte):
    edades = {f["edad"] for f in cohorte.filas}
    assert edades == set(range(spine.RANGO_EDAD[0], spine.RANGO_EDAD[1] + 1))


def test_misma_semilla_mismo_resultado(panel):
    assert spine.generar(panel=panel).filas == spine.generar(panel=panel).filas


def test_cohorte_grande_para_los_tests_de_los_modulos(panel):
    grande = spine.generar(n=200_000, panel=panel)
    assert len(grande.filas) == 200_000
    assert sum(1 for f in grande.filas if f["origen"] == "1000g") == 503


def test_verificar_acepta_la_cohorte(cohorte):
    spine.verificar(cohorte)


@pytest.mark.parametrize("cambio", [
    {"sexo": 2},
    {"origen": "otro"},
    {"origen": "1000g", "muestra_1000g": ""},
    {"origen": "simulada", "muestra_1000g": "HG00096"},
])
def test_verificar_rechaza_filas_invalidas(cohorte, cambio):
    filas = [dict(f) for f in cohorte.filas]
    i = next(k for k, f in enumerate(filas) if f["origen"] == "simulada")
    filas[i].update(cambio)
    with pytest.raises(AssertionError):
        spine.verificar(spine.Spine(filas=filas, meta={}))


def test_verificar_rechaza_muestras_repetidas(cohorte):
    filas = [dict(f) for f in cohorte.filas]
    reales = [k for k, f in enumerate(filas) if f["origen"] == "1000g"]
    filas[reales[1]]["muestra_1000g"] = filas[reales[0]]["muestra_1000g"]
    with pytest.raises(AssertionError):
        spine.verificar(spine.Spine(filas=filas, meta={}))


def test_exportar_y_cargar_ida_y_vuelta(cohorte, tmp_path):
    ruta_csv, _ = spine.exportar(cohorte, tmp_path)
    leida = spine.cargar(ruta_csv)
    assert leida.filas == cohorte.filas
    spine.verificar(leida)


def test_cargar_rechaza_un_archivo_modificado(cohorte, tmp_path):
    ruta_csv, _ = spine.exportar(cohorte, tmp_path)
    texto = ruta_csv.read_text(encoding="utf-8").replace(",prueba,", ",entrenamiento,", 1)
    ruta_csv.write_text(texto, encoding="utf-8")
    with pytest.raises(AssertionError):
        spine.cargar(ruta_csv)


def test_cargar_lee_la_version_1_1(tmp_path):
    """Un archivo sin las columnas nuevas se lee como todo simulado."""
    ruta = tmp_path / "cohorte_anima.csv"
    ruta.write_text("individuo_id,subpoblacion,sexo,edad,particion\n1,CEU,0,30,prueba\n", encoding="utf-8")
    fila = spine.cargar(ruta).filas[0]
    assert fila["origen"] == "simulada" and fila["muestra_1000g"] == ""


def test_preparar_panel_desde_el_formato_de_1000_genomes(tmp_path, panel):
    ruta = tmp_path / "panel.txt"
    lineas = ["sample\tpop\tsuper_pop\tgender"]
    lineas += [f"{f['muestra_1000g']}\t{f['subpoblacion']}\tEUR\t{'female' if f['sexo'] == 0 else 'male'}" for f in panel]
    lineas += ["NA18486\tYRI\tAFR\tmale", "HG00403\tCHS\tEAS\tfemale"]  # se descartan
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    destino = spine.preparar_panel(ruta, tmp_path / "muestras.csv")
    assert sorted(spine.cargar_panel(destino), key=lambda f: f["muestra_1000g"]) == \
        sorted(panel, key=lambda f: f["muestra_1000g"])


def test_preparar_panel_rechaza_conteos_distintos(tmp_path, panel):
    ruta = tmp_path / "panel.txt"
    lineas = ["sample\tpop\tsuper_pop\tgender"]
    lineas += [f"{f['muestra_1000g']}\t{f['subpoblacion']}\tEUR\tfemale" for f in panel[:-1]]
    ruta.write_text("\n".join(lineas) + "\n", encoding="utf-8")
    with pytest.raises(AssertionError):
        spine.preparar_panel(ruta, tmp_path / "muestras.csv")
