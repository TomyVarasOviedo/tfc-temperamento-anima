"""Contrato de salida común a los tres módulos del proyecto Anima.

Rostro, Voz y Temperamento exportan una lista de TraitResult a JSON.
La tabla comparativa final y el revelado secuencial de la interfaz se
construyen leyendo esos tres archivos y ordenando por delta_r2.

No modificar sin acordarlo con los tres módulos.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

DOMINIOS = ("rostro", "voz", "temperamento")


@dataclass(frozen=True)
class TraitResult:
    """Resultado calibrado para un rasgo de un dominio.

    Attributes:
        domain: uno de DOMINIOS.
        trait: nombre del rasgo o componente (p. ej. "neuroticismo", "PC3", "F0").
        mu: estimación puntual (media de la distribución predictiva).
        lo: límite inferior del intervalo calibrado.
        hi: límite superior del intervalo calibrado.
        nominal: nivel nominal declarado del intervalo (p. ej. 0.90).
        delta_r2: R² del modelo completo menos R² de la línea base.
        delta_r2_ci: intervalo de confianza bootstrap de delta_r2.
        coverage: cobertura empírica observada en el conjunto de validación.
        crps_gain: mejora de CRPS respecto de la línea base (positivo = mejora).
        baseline: descripción de la línea base usada. En temperamento es
            "solo intercepto"; en rostro y voz, "ancestría, sexo y edad".
    """

    domain: str
    trait: str
    mu: float
    lo: float
    hi: float
    nominal: float
    delta_r2: float
    delta_r2_ci: tuple[float, float]
    coverage: float
    crps_gain: float
    baseline: str

    def __post_init__(self) -> None:
        if self.domain not in DOMINIOS:
            raise ValueError(f"domain debe ser uno de {DOMINIOS}, no {self.domain!r}")
        if not self.lo <= self.mu <= self.hi:
            raise ValueError(f"se esperaba lo <= mu <= hi, se recibió {self.lo}, {self.mu}, {self.hi}")
        if not 0.0 < self.nominal < 1.0:
            raise ValueError("nominal debe estar entre 0 y 1")

    @property
    def cobertura_dentro_de_tolerancia(self, tol: float = 0.05) -> bool:
        """True si la cobertura empírica cae dentro de ±5 pp del nivel nominal.

        Es el criterio de la idea directriz. No se cambia el valor de tol
        sin registrarlo en el informe de validación.
        """
        return abs(self.coverage - self.nominal) <= tol


def export(results: list[TraitResult], path: str | Path) -> Path:
    """Escribe los resultados al JSON que consume la interfaz y el informe."""
    path = Path(path)
    path.write_text(
        json.dumps([asdict(r) for r in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def load(path: str | Path) -> list[TraitResult]:
    """Lee un JSON exportado por cualquiera de los tres módulos."""
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [TraitResult(**{**r, "delta_r2_ci": tuple(r["delta_r2_ci"])}) for r in raw]
