"""Pruebas de las matrices de covarianza (T-01 y T-02)."""

from __future__ import annotations

import numpy as np
import pytest

from temperamento import matrices, parametros

AUTOVALORES_G_BASE = np.array([0.0695, 0.0801, 0.1104, 0.1690, 0.2211])
AUTOVALORES_E_BASE = np.array([0.4951, 0.5619, 0.6952, 0.9028, 1.6950])


@pytest.mark.parametrize("nivel", parametros.NIVELES)
def test_t01_matrices_simetricas_y_definidas_positivas(nivel):
    """T-01: G y Sigma_E son simétricas y definidas positivas en los 4 niveles."""
    m = matrices.construir(nivel)
    assert m.G.shape == (5, 5)
    assert m.Sigma_E.shape == (5, 5)
    assert np.allclose(m.G, m.G.T)
    assert np.allclose(m.Sigma_E, m.Sigma_E.T)
    assert np.all(np.linalg.eigvalsh(m.G) > 0.0)
    assert np.all(np.linalg.eigvalsh(m.Sigma_E) > 0.0)


def test_t01_autovalores_de_control_en_base():
    """T-01: los autovalores de base coinciden con MODEL.md, tolerancia 1e-3."""
    m = matrices.construir("base")
    np.testing.assert_allclose(np.linalg.eigvalsh(m.G), AUTOVALORES_G_BASE, atol=1e-3)
    np.testing.assert_allclose(
        np.linalg.eigvalsh(m.Sigma_E), AUTOVALORES_E_BASE, atol=1e-3
    )


def test_t01_correlacion_latente_de_control():
    """La corrección de atenuación de RP coincide con la matriz de control."""
    m = matrices.construir("base")
    esperada = np.array(
        [
            [1.000, 0.171, 0.179, -0.268, 0.330],
            [0.171, 1.000, 0.330, -0.273, 0.168],
            [0.179, 0.330, 1.000, -0.322, 0.094],
            [-0.268, -0.273, -0.322, 1.000, -0.094],
            [0.330, 0.168, 0.094, -0.094, 1.000],
        ]
    )
    np.testing.assert_allclose(m.correlacion_latente, esperada, atol=1e-3)


def test_t02_sigma_e_no_definida_positiva_lanza_error():
    """T-02: si Sigma_E deja de ser definida positiva, no se corrige en silencio."""
    rg = parametros.RG.copy()
    rg[0, 1] = rg[1, 0] = 0.18
    with pytest.raises(ValueError, match="Sigma_E"):
        matrices.construir("base", h2_latente=np.full(5, 0.9), rg=rg)


def test_t02_g_no_definida_positiva_lanza_error():
    """T-02: una G no definida positiva también lanza error."""
    rg = parametros.RG.copy()
    rg[0, 1] = rg[1, 0] = 3.0
    with pytest.raises(ValueError, match="G"):
        matrices.construir("base", rg=rg)


def test_nivel_desconocido_lanza_error():
    """Un nivel fuera de parametros.NIVELES no se acepta."""
    with pytest.raises(ValueError, match="nivel desconocido"):
        matrices.construir("otro")
