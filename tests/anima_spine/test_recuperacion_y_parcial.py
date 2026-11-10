"""Lo que agrega la versión 2.1 del paquete.

Dos cosas: el ΔR² parcial junto con el total en el contraste entre
dominios, y el criterio de recuperación del nivel declarado, que es la
condición que la hipótesis reformulada del 3/11/2026 puede fallar.

Los tests no dependen de que ningún módulo haya exportado: arman filas
sintéticas con la forma del contrato.
"""

from __future__ import annotations

import numpy as np
import pytest

from anima_spine import compare, predictions

RASGOS = ("a", "b")
NIVEL = 0.07


def filas_de_recuperacion(desvio: float = 0.0, replicas: int = predictions.REPLICAS,
                          nivel_que_se_mueve: bool = False) -> list[dict]:
    """Un archivo de recuperación con un desvío conocido respecto del nivel declarado."""
    rng = np.random.default_rng(7)
    filas = []
    for r in range(replicas):
        for rasgo in RASGOS:
            for lb in predictions.LINEAS_BASE:
                nivel = NIVEL + (0.01 * r if nivel_que_se_mueve else 0.0)
                # ruido chico entre réplicas: al promediar queda el desvío pedido
                parcial = nivel + desvio + float(rng.normal(0, 0.002))
                filas.append(predictions.fila_recuperacion(
                    r, 20261019 + r, "voz", rasgo, lb, nivel, parcial * 0.5, parcial))
    return filas


# --- ΔR² parcial ------------------------------------------------------------

def test_sin_linea_base_el_parcial_coincide_con_el_total():
    rng = np.random.default_rng(1)
    y = rng.standard_normal(4000)
    mu = 0.3 * y + 0.1
    mu_base = np.full_like(y, y.mean())
    total, parcial = compare.delta_r2(y, mu, mu_base)
    assert parcial == pytest.approx(total, abs=2e-3)


def test_con_linea_base_fuerte_el_parcial_es_mayor_que_el_total():
    rng = np.random.default_rng(2)
    n = 20000
    cov = rng.standard_normal(n) * np.sqrt(0.75)      # la línea base explica el 75 %
    g = rng.standard_normal(n) * np.sqrt(0.07 * 0.25)
    e = rng.standard_normal(n) * np.sqrt(0.93 * 0.25)
    y = cov + g + e
    total, parcial = compare.delta_r2(y, cov + g, cov)
    assert total == pytest.approx(0.07 * 0.25, abs=0.01)
    assert parcial == pytest.approx(0.07, abs=0.01)
    assert parcial > total


def test_el_parcial_falla_fuerte_si_la_linea_base_explica_todo():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    with pytest.raises(AssertionError, match="parcial no está definido"):
        compare.delta_r2(y, y, y)


# --- las dos líneas base son obligatorias -----------------------------------

def _fila_prediccion(ident: int, linea_base: str) -> dict:
    return {
        "individuo_id": ident, "dominio": "voz", "rasgo": "a", "y_real": 1.0,
        "mu": 1.0, "sigma": 1.0, "lo": 0.0, "hi": 2.0,
        "mu_base": 0.5, "sigma_base": 1.5, "linea_base": linea_base,
    }


def test_verificar_rechaza_una_sola_linea_base():
    filas = [_fila_prediccion(1, "solo intercepto")]
    with pytest.raises(AssertionError, match="falta exportar contra la línea base"):
        predictions.verificar(filas, {1}, "voz")


def test_verificar_acepta_las_dos():
    filas = [_fila_prediccion(1, lb) for lb in predictions.LINEAS_BASE]
    predictions.verificar(filas, {1}, "voz")


# --- criterio de recuperación ----------------------------------------------

def test_una_recuperacion_exacta_cumple():
    r = predictions.criterio_recuperacion(filas_de_recuperacion(desvio=0.0))
    assert r["cumple"]
    assert r["nivel_declarado"] == pytest.approx(NIVEL)
    assert abs(r["desvio"]) < predictions.TOLERANCIA_RECUPERACION


def test_un_desvio_mayor_que_la_tolerancia_no_cumple():
    r = predictions.criterio_recuperacion(filas_de_recuperacion(desvio=0.04))
    assert not r["cumple"]
    assert r["desvio"] == pytest.approx(0.04, abs=0.002)


def test_el_criterio_distingue_el_borde():
    adentro = predictions.criterio_recuperacion(filas_de_recuperacion(desvio=0.012))
    afuera = predictions.criterio_recuperacion(filas_de_recuperacion(desvio=0.018))
    assert adentro["cumple"] and not afuera["cumple"]


def test_informa_el_peor_rasgo_y_las_replicas_usadas():
    r = predictions.criterio_recuperacion(filas_de_recuperacion())
    assert r["peor_rasgo"] in RASGOS
    assert all(v["replicas"] == predictions.REPLICAS for v in r["por_rasgo"].values())


# --- validación del archivo de recuperación ---------------------------------

def test_verificar_recuperacion_acepta_un_archivo_bien_armado():
    predictions.verificar_recuperacion(filas_de_recuperacion(), "voz")


def test_verificar_recuperacion_exige_las_veinte_replicas():
    with pytest.raises(AssertionError, match="réplicas 0 a"):
        predictions.verificar_recuperacion(filas_de_recuperacion(replicas=5), "voz")


def test_verificar_recuperacion_rechaza_un_nivel_que_se_mueve():
    with pytest.raises(AssertionError, match="cambia entre réplicas"):
        predictions.verificar_recuperacion(
            filas_de_recuperacion(nivel_que_se_mueve=True), "voz")


def test_verificar_recuperacion_exige_las_dos_lineas_base():
    filas = [f for f in filas_de_recuperacion() if f["linea_base"] == predictions.LINEAS_BASE[0]]
    with pytest.raises(AssertionError, match="falta la línea base"):
        predictions.verificar_recuperacion(filas, "voz")


def test_exportar_y_cargar_recuperacion_ida_y_vuelta(tmp_path):
    filas = filas_de_recuperacion()
    ruta = predictions.exportar_recuperacion(filas, "voz", tmp_path)
    leidas = predictions.cargar_recuperacion(ruta)
    assert len(leidas) == len(filas)
    predictions.verificar_recuperacion(leidas, "voz")
    a = predictions.criterio_recuperacion(filas)
    b = predictions.criterio_recuperacion(leidas)
    assert a["parcial_recuperado"] == pytest.approx(b["parcial_recuperado"])


def test_fila_recuperacion_rechaza_una_linea_base_inventada():
    with pytest.raises(ValueError, match="linea_base"):
        predictions.fila_recuperacion(0, 1, "voz", "a", "sexo nada mas", 0.07, 0.01, 0.07)


# --- el corrimiento por estimar la línea base (corrección del 9/11/2026) -----

def test_la_tolerancia_absorbe_el_corrimiento_por_estimar_la_linea_base():
    """Con línea base estimada el parcial sale por encima del nivel declarado.

    El corrimiento medido por simulación va de +0,0045 a +0,0087 según el
    dominio. La tolerancia tiene que absorberlo y seguir rechazando un desvío
    del orden de dos veces eso.
    """
    for desvio in (0.0045, 0.0065, 0.0087):
        assert predictions.criterio_recuperacion(filas_de_recuperacion(desvio))["cumple"]
    assert not predictions.criterio_recuperacion(filas_de_recuperacion(0.020))["cumple"]


def test_contrato_error_es_un_value_error():
    assert issubclass(predictions.ContratoError, ValueError)
