"""Capa de calibración conformal normalizada del proyecto Anima.

Recibe una tabla larga por dominio con ``mu`` y ``sigma`` por rasgo y devuelve
intervalos calibrados, conteos y diagnósticos. No ajusta modelos, no hace I/O y
no usa generadores aleatorios.

@see CONTRACT.md
@see docs/Anima_capa_de_calibracion_contrato_v5.1_congelado.md
"""

from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import pandas as pd
from scipy import stats

from anima_spine import predictions as P
from anima_spine.predictions import ContratoError

_COLUMNAS_OBLIGATORIAS = (
    "individuo_id",
    "dominio",
    "particion",
    "rasgo",
    "y_real",
    "mu",
    "sigma",
    "linea_base",
    "mu_base",
    "sigma_base",
)
_COLUMNAS_ORACULO = ("mu_oraculo", "sigma_oraculo")
_COLUMNAS_FINITAS = ("y_real", "mu", "sigma", "mu_base", "sigma_base")
_COLUMNAS_POSITIVAS = ("sigma", "sigma_base")
_PARTICIONES = ("entrenamiento", "calibracion", "prueba")
_CLAVE = ("individuo_id", "rasgo", "linea_base")


def _exigir_dataframe(df: object) -> pd.DataFrame:
    """Comprueba que la entrada sea un DataFrame y devuelve la misma tabla."""
    if not isinstance(df, pd.DataFrame):
        raise ContratoError(
            f"df: se esperaba pandas.DataFrame, se recibió {type(df).__name__}"
        )
    return df


def _validar(df: pd.DataFrame) -> None:
    """Valida las columnas y los invariantes V1 a V12 de CONTRACT.md §5.3.

    @param  df  tabla larga por dominio, con todas las particiones.
    @raises ContratoError  ante la primera violación del contrato.
    @see    CONTRACT.md §5
    """

    for columna in _COLUMNAS_OBLIGATORIAS:
        if columna not in df.columns:
            raise ContratoError(f"{columna}: columna obligatoria ausente")

    if (
        not pd.api.types.is_integer_dtype(df["individuo_id"])
        or pd.api.types.is_bool_dtype(df["individuo_id"])
    ):
        raise ContratoError(
            f"individuo_id: {len(df)} filas con dtype no entero"
        )

    if len(df) == 0:
        raise ContratoError("dominio: 0 filas con valor no permitido")
    if df["dominio"].nunique(dropna=False) > 1:
        moda = df["dominio"].mode(dropna=False).iloc[0]
        n = int((df["dominio"] != moda).sum())
        raise ContratoError(
            f"dominio: {n} filas con valor distinto del más frecuente"
        )
    if df["dominio"].isna().any() or df["dominio"].iloc[0] not in P.DOMINIOS:
        raise ContratoError(f"dominio: {len(df)} filas con valor no permitido")

    malas = ~df["particion"].isin(_PARTICIONES)
    if malas.any():
        raise ContratoError(
            f"particion: {int(malas.sum())} filas con valor no permitido"
        )

    malas = ~df["linea_base"].isin(P.LINEAS_BASE)
    if malas.any():
        raise ContratoError(
            f"linea_base: {int(malas.sum())} filas con valor no permitido"
        )

    conteo = df.groupby(["individuo_id", "rasgo"], dropna=False)[
        "linea_base"
    ].nunique(dropna=False)
    malos = conteo != len(P.LINEAS_BASE)
    if malos.any():
        claves = conteo.index[malos]
        n = int(
            df.set_index(["individuo_id", "rasgo"]).index.isin(claves).sum()
        )
        raise ContratoError(
            f"linea_base: {n} filas en grupos con cantidad de líneas base distinta de 2"
        )

    duplicadas = df.duplicated(subset=list(_CLAVE), keep=False)
    if duplicadas.any():
        raise ContratoError(
            "individuo_id, rasgo, linea_base: "
            f"{int(duplicadas.sum())} filas duplicadas"
        )

    columnas_finitas = list(_COLUMNAS_FINITAS)
    columnas_finitas += [c for c in _COLUMNAS_ORACULO if c in df.columns]
    for columna in columnas_finitas:
        valores = pd.to_numeric(df[columna], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        malas = ~np.isfinite(valores)
        if malas.any():
            raise ContratoError(
                f"{columna}: {int(malas.sum())} filas con valor no finito"
            )

    columnas_positivas = list(_COLUMNAS_POSITIVAS)
    if "sigma_oraculo" in df.columns:
        columnas_positivas.append("sigma_oraculo")
    for columna in columnas_positivas:
        valores = pd.to_numeric(df[columna], errors="coerce").to_numpy(
            dtype=float, na_value=np.nan
        )
        malas = valores <= 0
        if malas.any():
            raise ContratoError(
                f"{columna}: {int(malas.sum())} filas con valor <= 0"
            )

    columnas_iguales = ["mu", "sigma", "y_real", "particion"]
    columnas_iguales += [c for c in _COLUMNAS_ORACULO if c in df.columns]
    if "sexo" in df.columns:
        columnas_iguales.append("sexo")

    for columna in columnas_iguales:
        pares = df.pivot(
            index=["individuo_id", "rasgo"],
            columns="linea_base",
            values=columna,
        )
        izquierda, derecha = P.LINEAS_BASE
        malas = pares[izquierda].astype(object) != pares[derecha].astype(object)
        if malas.any():
            claves = pares.index[malas]
            n = int(
                df.set_index(["individuo_id", "rasgo"]).index.isin(claves).sum()
            )
            raise ContratoError(
                f"{columna}: {n} filas con valores distintos entre líneas base"
            )

    if "sexo" in df.columns:
        malas = ~df["sexo"].isin((0, 1))
        if malas.any():
            raise ContratoError(
                f"sexo: {int(malas.sum())} filas con valor distinto de 0 o 1"
            )

    presentes = [c for c in _COLUMNAS_ORACULO if c in df.columns]
    if len(presentes) == 1:
        raise ContratoError(
            "mu_oraculo, sigma_oraculo: las columnas oráculo deben venir juntas"
        )


def _exigir_particion(
    df: pd.DataFrame, particion: str, contexto: str
) -> pd.DataFrame:
    """Devuelve una copia de las filas de la partición pedida."""
    seleccion = df[df["particion"] == particion]
    if seleccion.empty:
        raise ContratoError(
            f"particion: 0 filas de '{particion}' para {contexto}"
        )
    return seleccion.copy()


def _crps_normal(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """CRPS medio de la normal N(mu, sigma) evaluada en y."""
    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    w = (y - mu) / sigma
    valor = sigma * (
        w * (2 * stats.norm.cdf(w) - 1)
        + 2 * stats.norm.pdf(w)
        - 1 / np.sqrt(np.pi)
    )
    return float(np.mean(valor))


def _anchos(intervalos: pd.DataFrame) -> pd.DataFrame:
    """Ancho medio por rasgo y nivel."""
    tabla = intervalos.copy()
    tabla["ancho"] = tabla["hi"] - tabla["lo"]
    return (
        tabla.groupby(["rasgo", "nivel"], as_index=False)["ancho"]
        .mean()
        .rename(columns={"ancho": "ancho_medio"})
    )


class ConformalCalibrator:
    """Calibra intervalos por split conformal normalizado.

    @param    niveles  subconjunto de P.NIVELES.
    @raises   ContratoError  si algún nivel no está en P.NIVELES.
    @atributo modo_     "modelo" o "base", fijado por fit.
    @atributo dominio_  dominio único de la tabla ajustada.
    @atributo qhat_     cuantil por línea base, rasgo y nivel, con n y k.
    @see      CONTRACT.md §6.2
    """

    def __init__(self, niveles: Iterable[float] = P.NIVELES) -> None:
        try:
            pedidos = set(niveles)
        except TypeError:
            pedidos = {niveles}
        invalidos = [n for n in pedidos if n not in P.NIVELES]
        if invalidos:
            raise ContratoError(
                f"nivel: {len(invalidos)} niveles fuera de P.NIVELES"
            )
        self.niveles_ = tuple(n for n in P.NIVELES if n in pedidos)

    def _exigir_ajustado(self) -> None:
        if not hasattr(self, "qhat_"):
            raise ContratoError("fit: calibrador sin ajustar")

    def fit(
        self,
        df: pd.DataFrame,
        columnas: tuple[str, str] = ("mu", "sigma"),
    ) -> ConformalCalibrator:
        """Ajusta q̂ sobre las filas de calibración.

        @param  df        tabla válida según CONTRACT.md §5.
        @param  columnas  ("mu", "sigma") o ("mu_base", "sigma_base").
        @return           este calibrador, con modo_, dominio_ y qhat_.
        @raises ContratoError  si la tabla o las columnas violan el contrato.
        @see    CONTRACT.md §6.3
        """
        df = _exigir_dataframe(df)
        _validar(df)

        if columnas not in (("mu", "sigma"), ("mu_base", "sigma_base")):
            raise ContratoError(
                "columnas: se esperaba ('mu', 'sigma') o ('mu_base', 'sigma_base')"
            )

        calibracion = _exigir_particion(df, "calibracion", "fit")
        filas: list[dict] = []

        if columnas == ("mu", "sigma"):
            modo = "modelo"
            datos = calibracion[
                calibracion["linea_base"] == "solo intercepto"
            ].copy()
            grupos = [
                (None, rasgo, grupo)
                for rasgo, grupo in datos.groupby("rasgo", sort=True)
            ]
        else:
            modo = "base"
            grupos = [
                (linea, rasgo, grupo)
                for linea in P.LINEAS_BASE
                for rasgo, grupo in calibracion[
                    calibracion["linea_base"] == linea
                ].groupby("rasgo", sort=True)
            ]

        for linea, rasgo, grupo in grupos:
            mu = grupo[columnas[0]].to_numpy(dtype=float)
            sigma = grupo[columnas[1]].to_numpy(dtype=float)
            scores = np.abs(grupo["y_real"].to_numpy(dtype=float) - mu) / sigma
            ordenados = np.sort(scores)
            n = int(len(ordenados))
            for nivel in self.niveles_:
                k = math.ceil((n + 1) * nivel)
                if k > n:
                    raise ContratoError(
                        f"nivel: 1 filas con k > n para nivel {nivel}"
                    )
                filas.append(
                    {
                        "linea_base": linea,
                        "rasgo": rasgo,
                        "nivel": nivel,
                        "n": n,
                        "k": k,
                        "qhat": float(ordenados[k - 1]),
                    }
                )

        self.dominio_ = str(df["dominio"].iloc[0])
        self.modo_ = modo
        self.qhat_ = pd.DataFrame(
            filas,
            columns=["linea_base", "rasgo", "nivel", "n", "k", "qhat"],
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula los intervalos calibrados sobre la partición de prueba.

        @param  df  tabla válida según CONTRACT.md §5.
        @return     DataFrame con P.CAMPOS_INTERVALOS, y linea_base en modo base.
        @raises ContratoError  si la tabla no es válida o hay rasgos sin calibrar.
        @see    CONTRACT.md §6.4
        """
        df = _exigir_dataframe(df)
        _validar(df)
        self._exigir_ajustado()
        prueba = _exigir_particion(df, "prueba", "predict")

        rasgos_calibrados = set(self.qhat_["rasgo"])
        no_calibrados = set(prueba["rasgo"]) - rasgos_calibrados
        if no_calibrados:
            n = int(prueba["rasgo"].isin(no_calibrados).sum())
            raise ContratoError(
                f"rasgo: {n} filas de prueba con rasgo no calibrado"
            )

        if self.modo_ == "modelo":
            datos = prueba[prueba["linea_base"] == "solo intercepto"].copy()
            datos = datos.merge(
                self.qhat_[["rasgo", "nivel", "qhat"]],
                on="rasgo",
                how="inner",
                validate="many_to_many",
            )
            datos["lo"] = datos["mu"] - datos["qhat"] * datos["sigma"]
            datos["hi"] = datos["mu"] + datos["qhat"] * datos["sigma"]
            columnas = list(P.CAMPOS_INTERVALOS)
            orden = ["individuo_id", "rasgo", "nivel"]
        else:
            datos = prueba.merge(
                self.qhat_[["linea_base", "rasgo", "nivel", "qhat"]],
                on=["linea_base", "rasgo"],
                how="inner",
                validate="many_to_many",
            )
            datos["lo"] = datos["mu"] - datos["qhat"] * datos["sigma"]
            datos["hi"] = datos["mu"] + datos["qhat"] * datos["sigma"]
            columnas = [*P.CAMPOS_INTERVALOS, "linea_base"]
            orden = ["individuo_id", "rasgo", "linea_base", "nivel"]

        return (
            datos[columnas]
            .sort_values(orden, kind="mergesort")
            .reset_index(drop=True)
        )

    def contar(
        self,
        df: pd.DataFrame,
        iv: pd.DataFrame,
        replica: int,
        semilla: int,
    ) -> pd.DataFrame:
        """Cuenta personas de prueba dentro de su intervalo, por rasgo y nivel.

        @param  df       tabla válida según CONTRACT.md §5.
        @param  iv       salida de predict en modo modelo, sin linea_base.
        @param  replica  etiqueta entera de la réplica.
        @param  semilla  etiqueta entera de la semilla.
        @return          DataFrame con P.CAMPOS_BARRIDO.
        @raises ContratoError  si iv no cubre exactamente la prueba esperada.
        @see    CONTRACT.md §6.5
        """
        df = _exigir_dataframe(df)
        _validar(df)
        self._exigir_ajustado()
        if self.modo_ != "modelo":
            raise ContratoError("modo_: contar requiere un calibrador en modo modelo")
        if not isinstance(iv, pd.DataFrame):
            raise ContratoError(
                f"iv: se esperaba pandas.DataFrame, se recibió {type(iv).__name__}"
            )
        if "linea_base" in iv.columns:
            raise ContratoError("iv: no debe traer la columna linea_base")

        prueba = _exigir_particion(df, "prueba", "contar")
        datos = prueba[prueba["linea_base"] == "solo intercepto"].copy()

        esperadas = {
            (individuo_id, rasgo, nivel)
            for individuo_id, rasgo in datos[
                ["individuo_id", "rasgo"]
            ].itertuples(index=False, name=None)
            for nivel in self.niveles_
        }
        columnas_iv = ("individuo_id", "rasgo", "nivel", "lo", "hi")
        faltantes_iv = [c for c in columnas_iv if c not in iv.columns]
        if faltantes_iv:
            raise ContratoError(
                f"iv: {len(faltantes_iv)} columnas obligatorias ausentes"
            )
        claves_iv = list(
            zip(iv["individuo_id"], iv["rasgo"], iv["nivel"], strict=True)
        )
        sobrantes = len(set(claves_iv) - esperadas)
        faltantes = len(esperadas - set(claves_iv))
        duplicadas = len(claves_iv) - len(set(claves_iv))
        if sobrantes or faltantes or duplicadas:
            n = faltantes + len(claves_iv) - len(esperadas & set(claves_iv))
            raise ContratoError(
                f"iv: {n} filas faltantes, sobrantes o duplicadas"
            )

        cruz = datos[["individuo_id", "rasgo", "y_real"]].merge(
            iv[list(columnas_iv)],
            on=["individuo_id", "rasgo"],
            how="inner",
            validate="one_to_many",
        )
        cruz["dentro"] = (
            (cruz["lo"] <= cruz["y_real"]) & (cruz["y_real"] <= cruz["hi"])
        ).astype(int)

        barrido = (
            cruz.groupby(["rasgo", "nivel"], as_index=False)
            .agg(n=("dentro", "size"), dentro=("dentro", "sum"))
            .sort_values(["rasgo", "nivel"], kind="mergesort")
            .reset_index(drop=True)
        )
        barrido.insert(0, "dominio", self.dominio_)
        barrido.insert(0, "semilla", int(semilla))
        barrido.insert(0, "replica", int(replica))
        barrido["n"] = barrido["n"].astype(int)
        barrido["dentro"] = barrido["dentro"].astype(int)
        return barrido[list(P.CAMPOS_BARRIDO)]


def coverage_report(
    df: pd.DataFrame,
    cal: ConformalCalibrator,
    cal_base: ConformalCalibrator,
) -> dict[str, pd.DataFrame]:
    """Arma el diagnóstico de cobertura, anchos y CRPS de la réplica 0.

    @param  df        tabla válida según CONTRACT.md §5.
    @param  cal       calibrador en modo modelo.
    @param  cal_base  calibrador en modo base.
    @return           diccionario con las claves cobertura, crps y, si hay sexo,
                      cobertura_por_sexo.
    @raises ContratoError  si la tabla o los modos no son los esperados.
    @see    CONTRACT.md §6.6
    """
    df = _exigir_dataframe(df)
    _validar(df)
    if (
        not isinstance(cal, ConformalCalibrator)
        or getattr(cal, "modo_", None) != "modelo"
    ):
        raise ContratoError("cal: se esperaba un calibrador en modo modelo")
    if (
        not isinstance(cal_base, ConformalCalibrator)
        or getattr(cal_base, "modo_", None) != "base"
    ):
        raise ContratoError("cal_base: se esperaba un calibrador en modo base")

    iv = cal.predict(df)
    iv_base = cal_base.predict(df)
    prueba = df[df["particion"] == "prueba"].copy()
    datos = prueba[prueba["linea_base"] == "solo intercepto"].copy()

    cruz = datos[["individuo_id", "rasgo", "y_real"]].merge(
        iv, on=["individuo_id", "rasgo"], how="inner", validate="one_to_many"
    )
    cruz["dentro"] = (
        (cruz["lo"] <= cruz["y_real"]) & (cruz["y_real"] <= cruz["hi"])
    ).astype(int)
    cobertura = (
        cruz.groupby(["rasgo", "nivel"], as_index=False)
        .agg(n=("dentro", "size"), dentro=("dentro", "sum"))
        .sort_values(["rasgo", "nivel"], kind="mergesort")
        .reset_index(drop=True)
    )
    cobertura["cobertura"] = cobertura["dentro"] / cobertura["n"]
    cobertura["ic_inf"] = [
        0.0 if dentro == 0 else float(stats.beta.ppf(0.025, dentro, n - dentro + 1))
        for dentro, n in zip(cobertura["dentro"], cobertura["n"], strict=True)
    ]
    cobertura["ic_sup"] = [
        1.0 if dentro == n else float(stats.beta.ppf(0.975, dentro + 1, n - dentro))
        for dentro, n in zip(cobertura["dentro"], cobertura["n"], strict=True)
    ]
    cobertura["z"] = [
        float(stats.norm.ppf((1 + nivel) / 2)) for nivel in cobertura["nivel"]
    ]
    cobertura = cobertura.merge(
        cal.qhat_[["rasgo", "nivel", "qhat"]],
        on=["rasgo", "nivel"],
        how="left",
        validate="many_to_one",
    )
    cobertura["qhat_sobre_z"] = cobertura["qhat"] / cobertura["z"]

    anchos_modelo = _anchos(iv)
    cobertura = cobertura.merge(
        anchos_modelo, on=["rasgo", "nivel"], how="left", validate="one_to_one"
    )

    for linea, destino in (
        ("solo intercepto", "ancho_solo_intercepto"),
        ("ancestria sexo edad", "ancho_ancestria_sexo_edad"),
    ):
        anchos = _anchos(iv_base[iv_base["linea_base"] == linea]).rename(
            columns={"ancho_medio": destino}
        )
        cobertura = cobertura.merge(
            anchos, on=["rasgo", "nivel"], how="left", validate="one_to_one"
        )

    tiene_oraculo = all(c in df.columns for c in _COLUMNAS_ORACULO)
    if tiene_oraculo:
        oraculo = datos[["rasgo", "sigma_oraculo"]].copy()
        oraculo = oraculo.merge(
            cobertura[["rasgo", "nivel", "z"]],
            on="rasgo",
            how="inner",
            validate="many_to_many",
        )
        oraculo["ancho_oraculo"] = (
            2 * oraculo["z"] * oraculo["sigma_oraculo"]
        )
        anchos_oraculo = (
            oraculo.groupby(["rasgo", "nivel"], as_index=False)["ancho_oraculo"]
            .mean()
        )
        cobertura = cobertura.merge(
            anchos_oraculo,
            on=["rasgo", "nivel"],
            how="left",
            validate="one_to_one",
        )
    else:
        cobertura["ancho_oraculo"] = np.nan

    columnas_cobertura = [
        "rasgo",
        "nivel",
        "n",
        "dentro",
        "cobertura",
        "ic_inf",
        "ic_sup",
        "qhat",
        "z",
        "qhat_sobre_z",
        "ancho_medio",
        "ancho_solo_intercepto",
        "ancho_ancestria_sexo_edad",
        "ancho_oraculo",
    ]
    cobertura = cobertura[columnas_cobertura].sort_values(
        ["rasgo", "nivel"], kind="mergesort"
    ).reset_index(drop=True)

    filas_crps = []
    for rasgo in sorted(datos["rasgo"].unique()):
        modelo = datos[datos["rasgo"] == rasgo]
        fila = {
            "rasgo": rasgo,
            "crps_modelo": _crps_normal(
                modelo["y_real"], modelo["mu"], modelo["sigma"]
            ),
        }
        for linea, destino in (
            ("solo intercepto", "crps_solo_intercepto"),
            ("ancestria sexo edad", "crps_ancestria_sexo_edad"),
        ):
            base = prueba[
                (prueba["rasgo"] == rasgo) & (prueba["linea_base"] == linea)
            ]
            fila[destino] = _crps_normal(
                base["y_real"], base["mu_base"], base["sigma_base"]
            )
        if tiene_oraculo:
            fila["crps_oraculo"] = _crps_normal(
                modelo["y_real"], modelo["mu_oraculo"], modelo["sigma_oraculo"]
            )
        else:
            fila["crps_oraculo"] = np.nan
        filas_crps.append(fila)

    crps = pd.DataFrame(
        filas_crps,
        columns=[
            "rasgo",
            "crps_modelo",
            "crps_solo_intercepto",
            "crps_ancestria_sexo_edad",
            "crps_oraculo",
        ],
    )

    reporte: dict[str, pd.DataFrame] = {"cobertura": cobertura, "crps": crps}

    if "sexo" in df.columns:
        cruz_sexo = datos[
            ["individuo_id", "rasgo", "y_real", "sexo"]
        ].merge(iv, on=["individuo_id", "rasgo"], how="inner", validate="one_to_many")
        cruz_sexo["dentro"] = (
            (cruz_sexo["lo"] <= cruz_sexo["y_real"])
            & (cruz_sexo["y_real"] <= cruz_sexo["hi"])
        ).astype(int)
        por_sexo = (
            cruz_sexo.groupby(["rasgo", "nivel", "sexo"], as_index=False)
            .agg(n=("dentro", "size"), dentro=("dentro", "sum"))
            .sort_values(["rasgo", "nivel", "sexo"], kind="mergesort")
            .reset_index(drop=True)
        )
        por_sexo["cobertura"] = por_sexo["dentro"] / por_sexo["n"]
        reporte["cobertura_por_sexo"] = por_sexo[
            ["rasgo", "nivel", "sexo", "n", "dentro", "cobertura"]
        ]

    return reporte


def control_permutacion(
    df_perm: pd.DataFrame, cal_base: ConformalCalibrator
) -> dict[str, pd.DataFrame]:
    """Compara el ancho permutado con la línea base de ancestría, sexo y edad.

    @param  df_perm   tabla válida con mu y sigma del puntaje permutado.
    @param  cal_base  calibrador en modo base, ajustado sobre la tabla original.
    @return           diccionario con por_rasgo y por_nivel.
    @raises ContratoError  si la tabla o el modo no son los esperados.
    @see    CONTRACT.md §6.7
    """
    df_perm = _exigir_dataframe(df_perm)
    _validar(df_perm)
    if (
        not isinstance(cal_base, ConformalCalibrator)
        or getattr(cal_base, "modo_", None) != "base"
    ):
        raise ContratoError("cal_base: se esperaba un calibrador en modo base")

    cal_perm = ConformalCalibrator(niveles=cal_base.niveles_).fit(
        df_perm, columnas=("mu", "sigma")
    )
    iv_perm = cal_perm.predict(df_perm)
    iv_base = cal_base.predict(df_perm)

    anchos_perm = _anchos(iv_perm).rename(
        columns={"ancho_medio": "ancho_permutado"}
    )
    anchos_base = _anchos(
        iv_base[iv_base["linea_base"] == "ancestria sexo edad"]
    ).rename(columns={"ancho_medio": "ancho_base"})

    por_rasgo = anchos_perm.merge(
        anchos_base,
        on=["rasgo", "nivel"],
        how="inner",
        validate="one_to_one",
    )
    por_rasgo["cociente"] = por_rasgo["ancho_permutado"] / por_rasgo["ancho_base"]
    por_rasgo = por_rasgo[
        ["rasgo", "nivel", "ancho_permutado", "ancho_base", "cociente"]
    ].sort_values(["rasgo", "nivel"], kind="mergesort").reset_index(drop=True)

    por_nivel = (
        por_rasgo.groupby("nivel", as_index=False)
        .agg(
            n_rasgos=("rasgo", "nunique"),
            cociente_medio=("cociente", "mean"),
        )
        .sort_values("nivel", kind="mergesort")
        .reset_index(drop=True)
    )
    por_nivel["pasa"] = (por_nivel["cociente_medio"] >= 0.95) & (
        por_nivel["cociente_medio"] <= 1.05
    )
    por_nivel["n_rasgos"] = por_nivel["n_rasgos"].astype(int)
    por_nivel = por_nivel[["nivel", "n_rasgos", "cociente_medio", "pasa"]]

    return {"por_rasgo": por_rasgo, "por_nivel": por_nivel}
