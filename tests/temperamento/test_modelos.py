"""Pruebas de los ajustes y de la ausencia de fuga de información (T-06)."""

from __future__ import annotations

import numpy as np
import pytest

from temperamento import modelos, parametros, simulacion


def test_ajuste_genotipo_recupera_una_recta_conocida():
    """El OLS de genotipo recupera intercepto y pendiente de datos exactos."""
    g = np.linspace(-2.0, 2.0, 50)
    y = 1.5 + 0.8 * g
    ajuste = modelos.ajustar_genotipo(y, g)
    np.testing.assert_allclose(ajuste.intercepto, 1.5, atol=1e-10)
    np.testing.assert_allclose(ajuste.pendiente, 0.8, atol=1e-10)
    np.testing.assert_allclose(ajuste.sigma, 0.0, atol=1e-10)


def test_sigma_del_genotipo_usa_ddof_dos():
    """El sigma de entrenamiento se calcula con ddof = DDOF_GENOTIPO."""
    g = np.arange(10, dtype=float)
    y = 2.0 + 0.5 * g + np.linspace(-1.0, 1.0, 10)
    ajuste = modelos.ajustar_genotipo(y, g)
    diseno = np.column_stack([np.ones_like(g), g])
    residuos = y - diseno @ np.array([ajuste.intercepto, ajuste.pendiente])
    esperado = np.sqrt(np.sum(residuos**2) / (len(y) - parametros.DDOF_GENOTIPO))
    np.testing.assert_allclose(ajuste.sigma, esperado)


def test_linea_base_solo_intercepto():
    """La línea base de solo intercepto usa media y desvío de entrenamiento."""
    y = np.array([1.0, 2.0, 4.0, 8.0])
    ajuste = modelos.ajustar_solo_intercepto(y)
    assert ajuste.mu == pytest.approx(float(np.mean(y)))
    assert ajuste.sigma == pytest.approx(float(np.std(y, ddof=parametros.DDOF_MEDIA)))
    np.testing.assert_allclose(ajuste.predecir(3), np.full(3, np.mean(y)))


def test_diseno_ancestria_omite_la_categoria_de_referencia():
    """El diseño usa intercepto, sexo, edad y una indicadora por categoría extra."""
    sexo = np.array([0, 1, 0, 1])
    edad = np.array([20, 30, 40, 50])
    subpoblacion = ["CEU", "GBR", "CEU", "TSI"]
    diseno = modelos.diseno_ancestria(sexo, edad, subpoblacion, ("CEU", "GBR", "TSI"))
    assert diseno.shape == (4, 5)
    np.testing.assert_allclose(diseno[:, 0], 1.0)
    np.testing.assert_allclose(diseno[:, 1], sexo)
    np.testing.assert_allclose(diseno[:, 2], edad)
    np.testing.assert_allclose(diseno[:, 3], [0.0, 1.0, 0.0, 0.0])
    np.testing.assert_allclose(diseno[:, 4], [0.0, 0.0, 0.0, 1.0])


def test_subpoblacion_desconocida_lanza_error():
    """Una categoría no declarada no se convierte en ceros en silencio."""
    with pytest.raises(ValueError, match="no declaradas"):
        modelos.diseno_ancestria(
            np.array([0, 1]), np.array([20, 30]), ["CEU", "XXX"], ("CEU", "GBR")
        )


def test_t06_sin_fuga_de_calibracion_ni_prueba():
    """T-06: alterar y fuera de entrenamiento no cambia ningún ajuste."""
    sp, sim = simulacion.simular_cohorte("base", replica=0)
    train = simulacion.mascara_particion(sp, parametros.PARTICION_ENTRENAMIENTO)
    no_train = ~train
    sexo = np.array([f["sexo"] for f in sp.filas])
    edad = np.array([f["edad"] for f in sp.filas])
    subpoblacion = np.array([f["subpoblacion"] for f in sp.filas])

    y_alterada = sim.y.copy()
    y_alterada[no_train] += 100.0

    for k, rasgo in enumerate(parametros.RASGOS):
        g_ent = sim.g[train, k]
        y_ent = sim.y[train, k]
        y_alt_ent = y_alterada[train, k]

        a1 = modelos.ajustar_genotipo(y_ent, g_ent)
        a2 = modelos.ajustar_genotipo(y_alt_ent, g_ent)
        assert (a1.intercepto, a1.pendiente, a1.sigma) == (
            a2.intercepto,
            a2.pendiente,
            a2.sigma,
        ), rasgo

        b1 = modelos.ajustar_solo_intercepto(y_ent)
        b2 = modelos.ajustar_solo_intercepto(y_alt_ent)
        assert (b1.mu, b1.sigma) == (b2.mu, b2.sigma), rasgo

        c1 = modelos.ajustar_ancestria_sexo_edad(
            y_ent, sexo[train], edad[train], subpoblacion[train]
        )
        c2 = modelos.ajustar_ancestria_sexo_edad(
            y_alt_ent, sexo[train], edad[train], subpoblacion[train]
        )
        np.testing.assert_array_equal(c1.coeficientes, c2.coeficientes)
        assert c1.sigma == c2.sigma

        # Las predicciones sobre calibración y prueba tampoco cambian.
        np.testing.assert_array_equal(a1.predecir(sim.g[:, k]), a2.predecir(sim.g[:, k]))
        np.testing.assert_array_equal(
            c1.predecir(sexo, edad, subpoblacion),
            c2.predecir(sexo, edad, subpoblacion),
        )
