"""Pruebas de la simulación y de sus identidades (T-03 y T-08 a T-14)."""

from __future__ import annotations

import hashlib

import numpy as np
import pytest

from temperamento import parametros, simulacion

N_GRANDE = 200_000


def _hash_cohorte() -> str:
    return hashlib.sha256(parametros.COHORTE.read_bytes()).hexdigest()


def test_t03_la_cohorte_no_se_modifica():
    """T-03: leer la cohorte y simular no cambia su huella."""
    antes = _hash_cohorte()
    simulacion.simular_cohorte("base", replica=0)
    assert _hash_cohorte() == antes


def test_la_simulacion_de_la_cohorte_respeta_las_particiones():
    """La cohorte real se lee sin tocarla y conserva 1000/500/500."""
    sp, sim = simulacion.simular_cohorte("base", replica=0)
    assert len(sp.filas) == 2000
    assert sim.g.shape == (2000, 5)
    for particion, esperado in (("entrenamiento", 1000), ("calibracion", 500), ("prueba", 500)):
        assert int(simulacion.mascara_particion(sp, particion).sum()) == esperado


def test_t08_misma_semilla_da_resultados_identicos():
    """T-08: misma semilla implica los mismos arrays, byte a byte."""
    a = simulacion.simular(1000, "base", replica=7)
    b = simulacion.simular(1000, "base", replica=7)
    for nombre in ("g", "t", "y", "epsilon"):
        assert np.array_equal(getattr(a, nombre), getattr(b, nombre))


def test_t08_misma_semilla_da_la_misma_permutacion_en_nulo():
    """T-08: el nulo también es reproducible."""
    a = simulacion.simular(1000, "nulo", replica=2)
    b = simulacion.simular(1000, "nulo", replica=2)
    assert np.array_equal(a.permutacion, b.permutacion)
    assert np.array_equal(a.g, b.g)
    assert np.array_equal(a.y, b.y)


def test_t09_semillas_distintas_dan_g_distinto():
    """T-09: cambiar la réplica cambia el puntaje genético."""
    a = simulacion.simular(1000, "base", replica=0)
    b = simulacion.simular(1000, "base", replica=1)
    assert not np.array_equal(a.g, b.g)
    assert not np.array_equal(a.y, b.y)


@pytest.mark.parametrize("nivel", parametros.NIVELES)
def test_t10_varianza_observada_es_uno(nivel):
    """T-10: Var(y_k) = 1 ± 0.01."""
    sim = simulacion.simular(N_GRANDE, nivel)
    np.testing.assert_allclose(sim.y.var(axis=0, ddof=1), np.ones(5), atol=0.01)


@pytest.mark.parametrize("nivel", parametros.NIVELES)
def test_t11_correlacion_fenotipica(nivel):
    """T-11: Corr(y_i, y_j) = RP[i, j] ± 0.01."""
    sim = simulacion.simular(N_GRANDE, nivel)
    np.testing.assert_allclose(np.corrcoef(sim.y.T), parametros.RP, atol=0.01)


@pytest.mark.parametrize("nivel", parametros.NIVELES)
def test_t12_correlacion_genetica(nivel):
    """T-12: Corr(g_i, g_j) = RG[i, j] ± 0.02, también en el nulo."""
    sim = simulacion.simular(N_GRANDE, nivel)
    np.testing.assert_allclose(np.corrcoef(sim.g.T), parametros.RG, atol=0.02)


@pytest.mark.parametrize("nivel", ("base", "ldsc", "pgi"))
def test_t13_heredabilidad_observada(nivel):
    """T-13: Corr(g_k, y_k)² = alpha_k * h_k ± 0.005."""
    sim = simulacion.simular(N_GRANDE, nivel)
    observada = np.array(
        [np.corrcoef(sim.g[:, k], sim.y[:, k])[0, 1] ** 2 for k in range(5)]
    )
    np.testing.assert_allclose(observada, parametros.h2_observada(nivel), atol=0.005)


def test_t13_el_nulo_no_tiene_senal():
    """T-13 (control): tras permutar g, la h² observada es aproximadamente cero."""
    sim = simulacion.simular(N_GRANDE, "nulo")
    observada = np.array(
        [np.corrcoef(sim.g[:, k], sim.y[:, k])[0, 1] ** 2 for k in range(5)]
    )
    assert np.all(observada < 0.01)


def test_t14_confiabilidad_empirica():
    """T-14: dos aplicaciones del test con el mismo t correlacionan alpha."""
    sim = simulacion.simular(N_GRANDE, "base", replica=0)
    y2 = simulacion.aplicar_test(sim.t, np.random.default_rng(123))
    confiabilidad = np.array(
        [np.corrcoef(sim.y[:, k], y2[:, k])[0, 1] for k in range(5)]
    )
    np.testing.assert_allclose(confiabilidad, parametros.vector(parametros.ALPHA), atol=0.01)
