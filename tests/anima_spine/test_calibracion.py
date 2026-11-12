"""Tests de la capa de calibración conformal (CONTRACT.md v5.1)."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import anima_spine
from anima_spine import predictions as P
from anima_spine import spine
from anima_spine.calibracion import (
    ConformalCalibrator,
    control_permutacion,
    coverage_report,
)
from anima_spine.predictions import ContratoError

RASGOS = ("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8")
SEMILLAS = tuple(20261019 + r for r in range(P.REPLICAS))


def tolerancia(p: float, n_cal: int = 500, n_test: int = 500, R: int = 20) -> float:
    """Tolerancia de cobertura de CONTRACT.md §9.2."""
    return 4 * math.sqrt(p * (1 - p) * (1 / n_cal + 1 / n_test) / R)


def tabla_sintetica(
    semilla: int,
    n_cal: int = 500,
    n_test: int = 500,
    n_ent: int = 10,
    rasgos: tuple[str, ...] = RASGOS,
    factor_sigma=None,
    dominio: str = "temperamento",
) -> pd.DataFrame:
    """Genera una tabla larga sintética que cumple CONTRACT.md §5."""
    rng = np.random.default_rng(semilla)
    n_total = n_ent + n_cal + n_test
    ids = np.arange(1, n_total + 1)
    sexo = rng.integers(0, 2, size=n_total)
    particiones = (
        ["entrenamiento"] * n_ent
        + ["calibracion"] * n_cal
        + ["prueba"] * n_test
    )

    filas = []
    for posicion, individuo_id in enumerate(ids):
        particion = particiones[posicion]
        for rasgo in rasgos:
            mu = float(rng.normal())
            sigma = float(np.exp(rng.normal(0, 0.3)))
            y_real = mu + sigma * float(rng.normal())
            sigma_reportada = (
                sigma
                if factor_sigma is None
                else sigma * float(factor_sigma(int(sexo[posicion])))
            )
            for linea_base in P.LINEAS_BASE:
                filas.append(
                    {
                        "individuo_id": int(individuo_id),
                        "dominio": dominio,
                        "particion": particion,
                        "rasgo": rasgo,
                        "y_real": y_real,
                        "mu": mu,
                        "sigma": sigma_reportada,
                        "linea_base": linea_base,
                        "mu_base": mu,
                        "sigma_base": sigma_reportada,
                        "sexo": int(sexo[posicion]),
                    }
                )
    return pd.DataFrame(filas)


def _tabla_scores(scores: tuple[float, ...]) -> pd.DataFrame:
    filas = []
    for individuo_id, score in enumerate(scores, start=1):
        for linea_base in P.LINEAS_BASE:
            filas.append(
                {
                    "individuo_id": individuo_id,
                    "dominio": "temperamento",
                    "particion": "calibracion",
                    "rasgo": "r1",
                    "y_real": float(score),
                    "mu": 0.0,
                    "sigma": 1.0,
                    "linea_base": linea_base,
                    "mu_base": 0.0,
                    "sigma_base": 1.0,
                }
            )
    return pd.DataFrame(filas)


def _barrido(df: pd.DataFrame, cal: ConformalCalibrator, semilla: int) -> pd.DataFrame:
    barras = []
    for replica in range(P.REPLICAS):
        iv = cal.predict(df)
        barras.append(
            cal.contar(df, iv, replica=replica, semilla=semilla + replica)
        )
    return pd.concat(barras, ignore_index=True)


def test_T1_caso_a_mano() -> None:
    """T1: k y q̂ del caso chico; k de n = 500; y rechazo de k > n."""
    df = _tabla_scores(tuple(0.1 * i for i in range(1, 10)))
    cal = ConformalCalibrator(niveles=(0.80,)).fit(df)
    fila = cal.qhat_.iloc[0]
    assert fila["n"] == 9
    assert fila["k"] == 8
    assert fila["qhat"] == pytest.approx(0.8)

    with pytest.raises(ContratoError, match="nivel"):
        ConformalCalibrator(niveles=(0.95,)).fit(df)

    grande = tabla_sintetica(20261019)
    cal_grande = ConformalCalibrator().fit(grande)
    esperados = (251, 301, 351, 401, 451, 476)
    for _, grupo in cal_grande.qhat_.groupby("rasgo"):
        assert tuple(grupo.sort_values("nivel")["k"]) == esperados


def test_T2_modelo_perfecto() -> None:
    """T2: la cobertura promedio de 20 réplicas queda dentro de 4 desvíos."""
    barras = []
    for semilla in SEMILLAS:
        df = tabla_sintetica(semilla)
        cal = ConformalCalibrator().fit(df)
        iv = cal.predict(df)
        barras.append(
            cal.contar(df, iv, replica=semilla - 20261019, semilla=semilla)
        )
    barrido = pd.concat(barras, ignore_index=True)
    cobertura = P.cobertura_promedio(barrido.to_dict("records"))

    for rasgo, por_nivel in cobertura.items():
        for nivel, valor in por_nivel.items():
            assert abs(valor - nivel) <= tolerancia(nivel), (rasgo, nivel, valor)


def test_T3a_invariancia_ante_la_escala() -> None:
    """T3a: escalar sigma por una constante no cambia lo ni hi."""
    df = tabla_sintetica(20261019)
    cal = ConformalCalibrator().fit(df)
    iv = cal.predict(df)

    for factor in (0.7, 1.5):
        escalada = df.copy()
        escalada["sigma"] = escalada["sigma"] * factor
        escalada["sigma_base"] = escalada["sigma_base"] * factor
        cal_escalada = ConformalCalibrator().fit(escalada)
        iv_escalada = cal_escalada.predict(escalada)
        np.testing.assert_allclose(
            iv_escalada[["lo", "hi"]].to_numpy(),
            iv[["lo", "hi"]].to_numpy(),
            rtol=1e-9,
            atol=0,
        )


def test_T3b_error_de_sigma_por_sexo() -> None:
    """T3b: el error de sigma por sexo separa la cobertura por sexo."""
    factor = lambda sexo: 0.7 if sexo == 0 else 1.5
    barras = []
    por_sexo: dict[int, dict[float, list[float]]] = {
        0: {nivel: [] for nivel in P.NIVELES},
        1: {nivel: [] for nivel in P.NIVELES},
    }

    for semilla in SEMILLAS:
        df = tabla_sintetica(semilla, factor_sigma=factor)
        cal = ConformalCalibrator().fit(df)
        iv = cal.predict(df)
        barras.append(
            cal.contar(df, iv, replica=semilla - 20261019, semilla=semilla)
        )

        datos = df[
            (df["particion"] == "prueba")
            & (df["linea_base"] == "solo intercepto")
        ]
        iv_90 = iv[iv["nivel"] == 0.90]
        cruz = datos[["individuo_id", "rasgo", "y_real", "sexo"]].merge(
            iv_90[["individuo_id", "rasgo", "lo", "hi"]],
            on=["individuo_id", "rasgo"],
            how="inner",
            validate="one_to_one",
        )
        cruz["dentro"] = (
            (cruz["lo"] <= cruz["y_real"]) & (cruz["y_real"] <= cruz["hi"])
        )
        for sexo in (0, 1):
            cobertura = cruz[cruz["sexo"] == sexo].groupby("rasgo")["dentro"].mean()
            for rasgo, valor in cobertura.items():
                por_sexo[sexo][0.90].append(float(valor))

    barrido = pd.concat(barras, ignore_index=True)
    cobertura_total = P.cobertura_promedio(barrido.to_dict("records"))
    for rasgo, por_nivel in cobertura_total.items():
        for nivel, valor in por_nivel.items():
            assert abs(valor - nivel) <= tolerancia(nivel), (rasgo, nivel, valor)

    promedio_0 = float(np.mean(por_sexo[0][0.90]))
    promedio_1 = float(np.mean(por_sexo[1][0.90]))
    assert promedio_0 <= 0.85
    assert promedio_1 >= 0.95


def test_T4_anidamiento() -> None:
    """T4: al subir el nivel, lo no sube y hi no baja."""
    df = tabla_sintetica(20261019)
    cal = ConformalCalibrator().fit(df)
    iv = cal.predict(df)

    for _, grupo in iv.groupby(["individuo_id", "rasgo"]):
        grupo = grupo.sort_values("nivel")
        assert (np.diff(grupo["lo"].to_numpy()) <= 1e-12).all()
        assert (np.diff(grupo["hi"].to_numpy()) >= -1e-12).all()


def test_T5_rechazo_de_violaciones() -> None:
    """T5: cada chequeo V1 a V15 levanta ContratoError con el conteo correcto."""
    base = tabla_sintetica(11, n_ent=2, n_cal=20, n_test=10, rasgos=("r1", "r2"))

    df = base.drop(columns=["sigma"])
    with pytest.raises(ContratoError, match="sigma: columna obligatoria ausente"):
        ConformalCalibrator().fit(df)

    df = base.copy()
    df["individuo_id"] = df["individuo_id"].astype(str)
    with pytest.raises(
        ContratoError, match=f"individuo_id: {len(df)} filas con dtype no entero"
    ):
        ConformalCalibrator().fit(df)

    df = base.copy()
    df.loc[df.index[:3], "dominio"] = "voz"
    with pytest.raises(ContratoError, match="dominio: 3 filas"):
        ConformalCalibrator().fit(df)

    df = base.copy()
    df.loc[df.index[:2], "particion"] = "validacion"
    with pytest.raises(ContratoError, match="particion: 2 filas"):
        ConformalCalibrator().fit(df)

    df = base.copy()
    df.loc[df.index[:2], "linea_base"] = "otra"
    with pytest.raises(ContratoError, match="linea_base: 2 filas"):
        ConformalCalibrator().fit(df)

    df = base.drop(index=base.index[0])
    with pytest.raises(ContratoError, match="linea_base: 1 filas"):
        ConformalCalibrator().fit(df)

    df = pd.concat([base, base.iloc[[0]]], ignore_index=True)
    with pytest.raises(ContratoError, match="2 filas duplicadas"):
        ConformalCalibrator().fit(df)

    df = base.copy()
    df.loc[df.index[0], "y_real"] = np.nan
    with pytest.raises(ContratoError, match="y_real: 1 filas con valor no finito"):
        ConformalCalibrator().fit(df)

    df = base.copy()
    df.loc[df.index[0], "sigma"] = 0.0
    with pytest.raises(ContratoError, match="sigma: 1 filas con valor <= 0"):
        ConformalCalibrator().fit(df)

    df = base.copy()
    df.loc[df.index[0], "mu"] = df.loc[df.index[0], "mu"] + 0.1
    with pytest.raises(
        ContratoError, match="mu: 2 filas con valores distintos entre líneas base"
    ):
        ConformalCalibrator().fit(df)

    df = base.copy()
    grupo = df[(df["individuo_id"] == 1) & (df["rasgo"] == "r1")]
    df.loc[grupo.index, "sexo"] = 2
    with pytest.raises(ContratoError, match="sexo: 2 filas"):
        ConformalCalibrator().fit(df)

    df = base.copy()
    df["mu_oraculo"] = df["mu"]
    df["sigma_oraculo"] = df["sigma"]
    df = df.drop(columns=["sigma_oraculo"])
    with pytest.raises(ContratoError, match="mu_oraculo, sigma_oraculo"):
        ConformalCalibrator().fit(df)

    df = base[base["particion"] != "calibracion"]
    with pytest.raises(
        ContratoError, match="particion: 0 filas de 'calibracion'"
    ):
        ConformalCalibrator().fit(df)

    cal = ConformalCalibrator().fit(base)
    df = base[base["particion"] != "prueba"]
    with pytest.raises(ContratoError, match="particion: 0 filas de 'prueba'"):
        cal.predict(df)
    iv = cal.predict(base)
    with pytest.raises(ContratoError, match="particion: 0 filas de 'prueba'"):
        cal.contar(df, iv, replica=0, semilla=1)

    fila = base[
        (base["particion"] == "prueba")
        & (base["rasgo"] == "r1")
        & (base["linea_base"] == "solo intercepto")
    ].iloc[0]
    nuevas = [
        {**fila.to_dict(), "rasgo": "r3", "linea_base": linea_base}
        for linea_base in P.LINEAS_BASE
    ]
    df = pd.concat([base, pd.DataFrame(nuevas)], ignore_index=True)
    with pytest.raises(
        ContratoError, match="rasgo: 2 filas de prueba con rasgo no calibrado"
    ):
        cal.predict(df)

    with pytest.raises(ContratoError, match="nivel: 1 niveles fuera de P.NIVELES"):
        ConformalCalibrator(niveles=(0.50, 0.99))


def test_T6_determinismo() -> None:
    """T6: dos corridas idénticas; la semilla no cambia n ni dentro."""
    df = tabla_sintetica(20261019)
    cal_1 = ConformalCalibrator().fit(df)
    cal_2 = ConformalCalibrator().fit(df)
    pd.testing.assert_frame_equal(cal_1.qhat_, cal_2.qhat_)

    iv_1 = cal_1.predict(df)
    iv_2 = cal_2.predict(df)
    pd.testing.assert_frame_equal(iv_1, iv_2)

    barrido_1 = cal_1.contar(df, iv_1, replica=0, semilla=20261019)
    barrido_2 = cal_2.contar(df, iv_2, replica=0, semilla=20261019)
    pd.testing.assert_frame_equal(barrido_1, barrido_2)

    cal_base_1 = ConformalCalibrator().fit(
        df, columnas=("mu_base", "sigma_base")
    )
    cal_base_2 = ConformalCalibrator().fit(
        df, columnas=("mu_base", "sigma_base")
    )
    reporte_1 = coverage_report(df, cal_1, cal_base_1)
    reporte_2 = coverage_report(df, cal_2, cal_base_2)
    for clave in reporte_1:
        pd.testing.assert_frame_equal(reporte_1[clave], reporte_2[clave])

    barrido_otra_semilla = cal_1.contar(df, iv_1, replica=0, semilla=20261020)
    pd.testing.assert_frame_equal(
        barrido_otra_semilla[["n", "dentro"]], barrido_1[["n", "dentro"]]
    )


def test_T7_n_efectivo() -> None:
    """T7: en modo modelo n = 500; en modo base n = 500 por línea base."""
    df = tabla_sintetica(20261019)
    cal = ConformalCalibrator().fit(df)
    assert (cal.qhat_["n"] == 500).all()

    cal_base = ConformalCalibrator().fit(
        df, columnas=("mu_base", "sigma_base")
    )
    assert (cal_base.qhat_["n"] == 500).all()
    assert set(cal_base.qhat_["linea_base"]) == set(P.LINEAS_BASE)


def test_T8_sin_efectos_laterales() -> None:
    """T8: la tabla de entrada queda igual después de usar la capa."""
    df = tabla_sintetica(20261019, n_cal=100, n_test=100)
    original = df.copy(deep=True)
    cal = ConformalCalibrator().fit(df)
    cal_base = ConformalCalibrator().fit(
        df, columnas=("mu_base", "sigma_base")
    )
    iv = cal.predict(df)
    cal.contar(df, iv, replica=0, semilla=20261019)
    coverage_report(df, cal, cal_base)
    control_permutacion(df, cal_base)
    pd.testing.assert_frame_equal(df, original)


def test_T9_columnas_y_verificadores_del_paquete() -> None:
    """T9: columnas exactas y salidas aceptadas por los verificadores."""
    ruta = Path("data/cohorte_anima.csv")
    if not ruta.exists():
        pytest.skip("cohorte no disponible")
    cohorte = spine.cargar(ruta)
    df = tabla_sintetica(20261019, n_cal=100, n_test=100).copy()
    mapa = {}
    for particion, necesarios in (
        ("entrenamiento", 10),
        ("calibracion", 100),
        ("prueba", 100),
    ):
        reales = sorted(
            fila["individuo_id"]
            for fila in cohorte.filas
            if fila["particion"] == particion
        )[:necesarios]
        if len(reales) < necesarios:
            pytest.skip(f"la cohorte no tiene {necesarios} ids en {particion}")
        sinteticos = sorted(
            df.loc[df["particion"] == particion, "individuo_id"].unique()
        )
        mapa.update(zip(sinteticos, reales))
    df["individuo_id"] = df["individuo_id"].map(mapa)
    cal = ConformalCalibrator().fit(df)
    iv = cal.predict(df)
    assert list(iv.columns) == list(P.CAMPOS_INTERVALOS)

    barrido = cal.contar(df, iv, replica=0, semilla=20261019)
    assert list(barrido.columns) == list(P.CAMPOS_BARRIDO)

    datos = df[
        (df["particion"] == "prueba")
        & (df["linea_base"] == "solo intercepto")
    ]
    iv_90 = iv[iv["nivel"] == 0.90]
    predicciones = datos.merge(
        iv_90[["individuo_id", "rasgo", "lo", "hi"]],
        on=["individuo_id", "rasgo"],
        how="inner",
        validate="one_to_one",
    )
    ids_prueba = set(int(i) for i in datos["individuo_id"].unique())
    P.verificar_intervalos(
        iv.to_dict("records"),
        predicciones.to_dict("records"),
        ids_prueba,
        "temperamento",
    )

    barras = [
        cal.contar(df, iv, replica=r, semilla=20261019 + r)
        for r in range(P.REPLICAS)
    ]
    barrido_20 = pd.concat(barras, ignore_index=True)
    P.verificar_barrido(
        barrido_20.to_dict("records"),
        iv.to_dict("records"),
        predicciones.to_dict("records"),
        ids_prueba,
        "temperamento",
    )


def test_T10_filas_de_entrenamiento() -> None:
    """T10: agregar o quitar entrenamiento no cambia q̂ ni predict."""
    df = tabla_sintetica(20261019, n_cal=100, n_test=100)
    cal = ConformalCalibrator().fit(df)
    iv = cal.predict(df)

    sin_entrenamiento = df[df["particion"] != "entrenamiento"].copy()
    cal_sin = ConformalCalibrator().fit(sin_entrenamiento)
    iv_sin = cal_sin.predict(sin_entrenamiento)
    pd.testing.assert_frame_equal(cal.qhat_, cal_sin.qhat_)
    pd.testing.assert_frame_equal(iv, iv_sin)

    extra = df[df["particion"] == "entrenamiento"].copy()
    extra["individuo_id"] = extra["individuo_id"] + 100000
    con_extra = pd.concat([df, extra], ignore_index=True)
    cal_extra = ConformalCalibrator().fit(con_extra)
    iv_extra = cal_extra.predict(con_extra)
    pd.testing.assert_frame_equal(cal.qhat_, cal_extra.qhat_)
    pd.testing.assert_frame_equal(iv, iv_extra)


def test_T11_cruce_con_el_paquete() -> None:
    """T11: coverage_report cruza exacto con dentro / n del barrido."""
    df = tabla_sintetica(20261019, n_cal=100, n_test=100)
    cal = ConformalCalibrator().fit(df)
    cal_base = ConformalCalibrator().fit(
        df, columnas=("mu_base", "sigma_base")
    )
    iv = cal.predict(df)
    barrido = cal.contar(df, iv, replica=0, semilla=20261019)
    reporte = coverage_report(df, cal, cal_base)["cobertura"]

    unido = reporte.merge(
        barrido[["rasgo", "nivel", "n", "dentro"]],
        on=["rasgo", "nivel"],
        how="inner",
        suffixes=("", "_barrido"),
        validate="one_to_one",
    )
    assert (
        unido["cobertura"].to_numpy()
        == (unido["dentro_barrido"] / unido["n_barrido"]).to_numpy()
    ).all()

    promedio = P.cobertura_promedio(barrido.to_dict("records"))
    for fila in unido.itertuples(index=False):
        assert promedio[fila.rasgo][round(fila.nivel, 6)] == fila.cobertura


def test_T12_error_compartido() -> None:
    """T12: ContratoError es la clase de predictions y es un ValueError."""
    df = tabla_sintetica(11, n_ent=2, n_cal=20, n_test=10, rasgos=("r1",))
    with pytest.raises(ContratoError) as excinfo:
        ConformalCalibrator().fit(df.drop(columns=["sigma"]))
    assert isinstance(excinfo.value, ValueError)
    assert ContratoError is P.ContratoError
    if hasattr(anima_spine, "ContratoError"):
        assert anima_spine.ContratoError is P.ContratoError


def test_T13_tipo_de_entrada() -> None:
    """T13: dict o lista en lugar de DataFrame levanta ContratoError con df."""
    df = tabla_sintetica(20261019, n_cal=100, n_test=100)
    cal = ConformalCalibrator().fit(df)
    cal_base = ConformalCalibrator().fit(
        df, columnas=("mu_base", "sigma_base")
    )
    iv = cal.predict(df)

    with pytest.raises(ContratoError, match="df: se esperaba pandas.DataFrame"):
        cal.fit({})
    with pytest.raises(ContratoError, match="df: se esperaba pandas.DataFrame"):
        cal.predict([])
    with pytest.raises(ContratoError, match="df: se esperaba pandas.DataFrame"):
        cal.contar({}, iv, replica=0, semilla=1)
    with pytest.raises(ContratoError, match="df: se esperaba pandas.DataFrame"):
        coverage_report({}, cal, cal_base)
    with pytest.raises(ContratoError, match="df: se esperaba pandas.DataFrame"):
        control_permutacion([], cal_base)


def test_T14_forma_de_control_permutacion() -> None:
    """T14: claves, columnas y promedio simple de control_permutacion."""
    df = tabla_sintetica(20261019, n_cal=100, n_test=100)
    cal_base = ConformalCalibrator().fit(
        df, columnas=("mu_base", "sigma_base")
    )
    df_perm = df.copy()
    df_perm["mu"] = df_perm["mu"] + 0.1

    control = control_permutacion(df_perm, cal_base)
    assert set(control) == {"por_rasgo", "por_nivel"}
    assert list(control["por_rasgo"].columns) == [
        "rasgo",
        "nivel",
        "ancho_permutado",
        "ancho_base",
        "cociente",
    ]
    assert "pasa" not in control["por_rasgo"].columns
    assert list(control["por_nivel"].columns) == [
        "nivel",
        "n_rasgos",
        "cociente_medio",
        "pasa",
    ]
    assert len(control["por_nivel"]) == len(cal_base.niveles_)

    for _, grupo in control["por_rasgo"].groupby("nivel"):
        media = float(grupo["cociente"].mean())
        fila = control["por_nivel"].loc[
            control["por_nivel"]["nivel"] == grupo["nivel"].iloc[0]
        ].iloc[0]
        assert fila["cociente_medio"] == pytest.approx(media)
        assert fila["n_rasgos"] == grupo["rasgo"].nunique()
        assert fila["pasa"] == (0.95 <= media <= 1.05)
