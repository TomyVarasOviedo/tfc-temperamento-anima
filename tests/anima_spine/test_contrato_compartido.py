"""Tests que los TRES módulos tienen que pasar, con el mismo código.

Esto lo escribe y mantiene una sola persona, y Rostro, Voz y Temperamento
lo corren igual. No reemplaza los tests propios de cada módulo: verifica
que lo que cada uno entrega sea comparable con lo de los otros dos.

Uso, desde la raíz del repo:
    pytest tests/test_contrato_compartido.py -q

Un módulo que todavía no exportó su CSV sale como "skipped", no como
error: así cada uno puede correrlo desde el primer día sin esperar a los
otros, y el día que los tres aparecen, el test de pareo se activa solo.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anima_spine import predictions, spine

RAIZ = Path(__file__).resolve().parent.parent
COHORTE = RAIZ / "data" / "cohorte_anima.csv"
OUTPUTS = RAIZ / "outputs"
TOLERANCIA = 0.05  # los ±5 pp de la idea directriz


@pytest.fixture(scope="module")
def sp():
    if not COHORTE.exists():
        pytest.skip("falta cohorte_anima.csv — correr python -m anima_spine.spine")
    return spine.cargar(COHORTE)


@pytest.fixture(scope="module")
def ids_prueba(sp):
    return set(sp.ids("prueba"))


def _ruta(dominio: str) -> Path:
    return OUTPUTS / f"predicciones_{dominio}.csv"


def _cargar_o_saltar(dominio: str):
    ruta = _ruta(dominio)
    if not ruta.exists():
        pytest.skip(f"{dominio} todavía no exportó sus predicciones")
    return predictions.cargar(ruta)


# --- La columna vertebral ---------------------------------------------------

def test_la_cohorte_es_valida(sp):
    spine.verificar(sp)


def test_las_particiones_no_se_solapan(sp):
    conjuntos = [set(sp.ids(p)) for p in ("entrenamiento", "calibracion", "prueba")]
    for i, a in enumerate(conjuntos):
        for b in conjuntos[i + 1:]:
            assert not (a & b), "las particiones se solapan: la cobertura saldría optimista"


def test_la_cohorte_es_reproducible(sp, tmp_path):
    """Misma semilla, misma cohorte: regenerar da un archivo con la misma huella.

    numpy no garantiza la misma secuencia aleatoria entre versiones. Si la
    versión instalada no es la que generó la cohorte y la huella no coincide,
    el test se saltea con aviso: la fuente de verdad es el CSV publicado, cuya
    huella se verifica igual al cargarlo.
    """
    if sp.meta.get("version") != spine.VERSION:
        pytest.skip(f"el CSV es de la versión {sp.meta.get('version')} y el código, de la {spine.VERSION}")
    if not spine.PANEL_PAQUETE.exists():
        pytest.skip("falta el listado de 1000 Genomes del paquete (datos/muestras_1000g_eur.csv)")
    regenerada = spine.generar(n=sp.meta["n"], semilla=sp.meta["semilla"])
    ruta_csv, _ = spine.exportar(regenerada, tmp_path)
    if spine.huella(ruta_csv) != sp.meta["sha256_csv"]:
        if np.__version__ != sp.meta.get("numpy_version"):
            pytest.skip(
                f"numpy {np.__version__} no reproduce la secuencia de numpy "
                f"{sp.meta.get('numpy_version')}: usar el CSV publicado"
            )
        pytest.fail("la cohorte regenerada no coincide con la publicada")


def test_los_metadatos_traen_la_huella(sp):
    assert "sha256_csv" in sp.meta, "los metadatos no traen la huella del CSV"
    assert spine.huella(COHORTE) == sp.meta["sha256_csv"]


# --- Lo que entrega cada módulo --------------------------------------------

@pytest.mark.parametrize("dominio", predictions.DOMINIOS)
def test_el_export_cumple_el_contrato(dominio, ids_prueba):
    predictions.verificar(_cargar_o_saltar(dominio), ids_prueba, dominio)


@pytest.mark.parametrize("dominio", predictions.DOMINIOS)
def test_reporta_las_dos_lineas_base(dominio):
    """Sin las dos, parte de la diferencia entre dominios es un artefacto."""
    filas = _cargar_o_saltar(dominio)
    presentes = {f["linea_base"] for f in filas}
    faltan = set(predictions.LINEAS_BASE) - presentes
    assert not faltan, f"{dominio} no reporta contra: {faltan}"


def _cargar_intervalos_o_saltar(dominio: str):
    ruta = OUTPUTS / f"intervalos_{dominio}.csv"
    if not ruta.exists():
        pytest.skip(f"{dominio} todavía no exportó sus intervalos por nivel")
    return predictions.cargar_intervalos(ruta)


@pytest.mark.parametrize("dominio", predictions.DOMINIOS)
def test_los_intervalos_cumplen_el_contrato(dominio, ids_prueba):
    preds = _cargar_o_saltar(dominio)
    intervalos = _cargar_intervalos_o_saltar(dominio)
    predictions.verificar_intervalos(intervalos, preds, ids_prueba, dominio)


def _cargar_barrido_o_saltar(dominio: str):
    ruta = OUTPUTS / f"barrido_{dominio}.csv"
    if not ruta.exists():
        pytest.skip(f"{dominio} todavía no exportó su barrido de réplicas")
    return predictions.cargar_barrido(ruta)


@pytest.mark.parametrize("dominio", predictions.DOMINIOS)
def test_el_barrido_cumple_el_contrato(dominio, ids_prueba):
    preds = _cargar_o_saltar(dominio)
    intervalos = _cargar_intervalos_o_saltar(dominio)
    barrido = _cargar_barrido_o_saltar(dominio)
    predictions.verificar_barrido(barrido, intervalos, preds, ids_prueba, dominio)


@pytest.mark.parametrize("dominio", predictions.DOMINIOS)
def test_la_cobertura_promedio_esta_dentro_de_la_tolerancia(dominio):
    """Criterio de calibración de la hipótesis (3.2), opción A.

    Cobertura promediada sobre las 20 réplicas, dentro de ±5 puntos del nivel
    nominal, en los seis niveles y rasgo por rasgo. Una sola corrida no se
    evalúa con este criterio: su variación por azar lo haría fallar aunque el
    modelo estuviera bien calibrado. Si este test falla, es un hallazgo del
    informe de validación, no necesariamente un bug.
    """
    barrido = _cargar_barrido_o_saltar(dominio)
    cob = predictions.cobertura_promedio(barrido)
    fuera = [
        f"{rasgo}@{nivel:.0%}={c:.3f}"
        for rasgo, por_nivel in cob.items()
        for nivel, c in por_nivel.items()
        if abs(c - nivel) > TOLERANCIA
    ]
    assert not fuera, f"{dominio}: cobertura promedio fuera de ±{TOLERANCIA:.0%} en {fuera}"


# --- Que los tres sean comparables -----------------------------------------

def test_los_tres_cubren_los_mismos_individuos(ids_prueba):
    """Sin esto el bootstrap pareado no es válido y no hay gradiente."""
    cubiertos = {}
    for dominio in predictions.DOMINIOS:
        if _ruta(dominio).exists():
            filas = predictions.cargar(_ruta(dominio))
            cubiertos[dominio] = {f["individuo_id"] for f in filas}

    if len(cubiertos) < 2:
        pytest.skip("hacen falta al menos dos módulos exportados")

    for dominio, ids in cubiertos.items():
        assert ids == ids_prueba, f"{dominio} no cubre la partición de prueba completa"
