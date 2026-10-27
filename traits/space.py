"""Fase 1 — Espacio de rasgos.

Los cinco dominios Big Five ya son un espacio de baja dimensión: a
diferencia de rostro, acá NO se aplica PCA. Lo que sí hay que modelar
es la correlación entre rasgos, porque los Big Five no son ortogonales.

Las constantes se cargan de traits_config.yaml, no se hardcodean acá:
cada valor tiene que poder rastrearse a su fuente en el informe.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

CONFIG_POR_DEFECTO = Path(__file__).parent / "traits_config.yaml"


@dataclass(frozen=True)
class TraitSpace:
    """Espacio de rasgos temperamentales.

    Attributes:
        names: nombres de los cinco dominios, en orden fijo.
        mu: medias poblacionales, forma (5,).
        sigma: desvíos poblacionales, forma (5,).
        corr: matriz de correlación entre rasgos, forma (5, 5).
        reliability: confiabilidad test-retest por escala, forma (5,).
            Impone el techo de varianza explicable: ningún modelo puede
            explicar más varianza que la que el instrumento mide de forma
            estable. Se declara como límite, no se corrige por atenuación.
        h2_snp: heredabilidad SNP por rasgo según Schwaba et al. (2025).
    """

    names: list[str]
    mu: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray
    reliability: np.ndarray
    h2_snp: np.ndarray

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Chequea las invariantes que rompen las fases siguientes en silencio.

        La más importante es que corr sea definida positiva: las matrices
        de correlación copiadas de papers suelen no serlo por redondeo, y
        el muestreo de la Fase 2 falla o devuelve basura sin avisar.
        """
        k = len(self.names)
        for nombre, arr, forma in (
            ("mu", self.mu, (k,)),
            ("sigma", self.sigma, (k,)),
            ("reliability", self.reliability, (k,)),
            ("h2_snp", self.h2_snp, (k,)),
            ("corr", self.corr, (k, k)),
        ):
            if arr.shape != forma:
                raise ValueError(f"{nombre} debería tener forma {forma}, tiene {arr.shape}")

        if not np.allclose(self.corr, self.corr.T, atol=1e-8):
            raise ValueError("corr no es simétrica")
        if not np.allclose(np.diag(self.corr), 1.0):
            raise ValueError("la diagonal de corr debe ser 1")
        try:
            np.linalg.cholesky(self.corr)
        except np.linalg.LinAlgError as exc:
            raise ValueError(
                "corr no es definida positiva. Revisar los valores del paper o "
                "proyectar a la matriz definida positiva más cercana, dejándolo "
                "documentado en el informe."
            ) from exc

        if np.any(self.h2_snp > self.reliability):
            raise ValueError(
                "hay rasgos con h2_snp por encima de su confiabilidad: eso es "
                "imposible, revisar las fuentes"
            )

    @property
    def cholesky(self) -> np.ndarray:
        """Factor L tal que L @ L.T == corr. Lo usa el generador de cohorte."""
        return np.linalg.cholesky(self.corr)


def load_space(path: str | Path = CONFIG_POR_DEFECTO) -> TraitSpace:
    """Construye el TraitSpace desde el YAML de configuración."""
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    rasgos = cfg["traits"]
    names = [r["name"] for r in rasgos]
    return TraitSpace(
        names=names,
        mu=np.array([r["mu"] for r in rasgos], dtype=float),
        sigma=np.array([r["sigma"] for r in rasgos], dtype=float),
        corr=np.array(cfg["correlation"], dtype=float),
        reliability=np.array([r["reliability"] for r in rasgos], dtype=float),
        h2_snp=np.array([r["h2_snp"] for r in rasgos], dtype=float),
    )
