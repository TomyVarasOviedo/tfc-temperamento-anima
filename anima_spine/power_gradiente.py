"""¿Qué N hace falta para demostrar el escalón Rostro - Voz?

El escalón de abajo del gradiente (cualquiera de los dos contra
Temperamento) se detecta con poco N, porque la diferencia es grande. El de
arriba es el problema: Rostro y Voz están cerca entre sí, y si no se puede
distinguir uno de otro, la hipótesis de TRES niveles queda demostrada a
medias. Este cálculo es el que llevó al grupo a formular la hipótesis en dos
niveles (22/10/2026): Rostro - Voz se informa, pero no se afirma.

Esto no es un bootstrap sobre un dataset: es un estudio de características
operativas. Para cada combinación de N y de brecha verdadera, simula muchos
experimentos completos y cuenta en qué fracción el intervalo de confianza
de la diferencia excluye el cero. Esa fracción es la potencia.

Uso:
    python -m anima_spine.power_gradiente
"""

from __future__ import annotations

import argparse

import numpy as np

# Techo publicado para morfología facial (Xiong et al., 2025: hasta 7,9 %
# de varianza por rasgo). Se usa como ΔR² verdadero del dominio Rostro.
# ATENCIÓN (decisión abierta 2 del documento integrado): en la fuente es el
# máximo de un rasgo, y acá se aplica a las 28 componentes a la vez. Si el
# valor real es menor, la brecha con Voz se achica todavía más.
H2_ROSTRO = 0.079

# h² de los 8 parámetros de Voz con la regla agregada de cinco vocales
# (Gisladottir et al., 2023, Tabla S6; jitter y hnr: 0,05 conservador).
# Promedio 0,0707: brecha esperada con Rostro de 0,0083.
H2_VOZ_REAL = (0.1027, 0.0410, 0.0619, 0.1353, 0.0644, 0.0606, 0.0500, 0.0500)

# Brechas a evaluar: cuánto menos informativo es Voz que Rostro. Incluye las
# brechas chicas (0,005 a 0,010), que son las compatibles con los valores
# publicados.
BRECHAS = (0.005, 0.008, 0.010, 0.015, 0.025, 0.035, 0.045)

# Tamaños de la partición de PRUEBA (no de la cohorte entera).
# Con 25 % de prueba, N_prueba = 500 implica una cohorte total de 2000.
N_PRUEBA = (500, 1000, 2000, 4000)


def _r2_boot(y: np.ndarray, pred: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """R² para cada fila de la matriz de remuestreos idx, forma (B, n)."""
    yb, pb = y[idx], pred[idx]
    sse = ((yb - pb) ** 2).sum(axis=1)
    sst = ((yb - yb.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return 1.0 - sse / sst


def _delta_promedio(
    n: int, h2: float | tuple, k: int, idx: np.ndarray, rng: np.random.Generator
) -> np.ndarray:
    """ΔR² promediado sobre los k rasgos del dominio, por remuestreo.

    Promediar es lo que hace factible el escalón Rostro - Voz: reduce la
    varianza del estimador sin tocar N. Supone que el ruido de estimación
    es independiente entre rasgos, lo cual es defendible para los
    componentes PCA de Rostro (ortogonales por construcción) y optimista
    para los parámetros de Voz (F1, F2 y F3 comparten el tracto vocal,
    así que el k efectivo es menor que 8). Documentarlo en el informe.
    """
    h2s = tuple(h2) if isinstance(h2, (tuple, list)) else (h2,) * k
    if len(h2s) != k:
        raise ValueError("la lista de h2 debe tener k elementos")
    acum = 0.0
    for h2_k in h2s:
        g = rng.standard_normal(n) * np.sqrt(h2_k)
        y = g + rng.standard_normal(n) * np.sqrt(1 - h2_k)
        base = np.full(n, y.mean())
        acum = acum + (_r2_boot(y, g, idx) - _r2_boot(y, base, idx))
    return acum / k


def potencia(
    n: int,
    brecha: float,
    k_a: int = 1,
    k_b: int = 1,
    n_rep: int = 200,
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Fracción de experimentos en que la diferencia sale significativa.

    Args:
        k_a, k_b: cuántos rasgos promedia cada dominio. k=1 corresponde a
            evaluar el gradiente rasgo por rasgo; k=28 y k=8 a evaluarlo a
            nivel dominio, que es la unidad que la hipótesis afirma.

    Modelo oráculo: la media predictiva ES el valor genético verdadero. Es
    el mejor caso posible, así que esta potencia es un TECHO; un modelo
    real, que estima la pendiente, tendrá menos.

    Returns:
        (potencia a dos colas, potencia a una cola). Una cola es legítima
        porque la hipótesis es direccional: Rostro > Voz > Temperamento.
    """
    rng = np.random.default_rng(seed)
    h2_b = H2_ROSTRO - brecha
    if h2_b <= 0:
        raise ValueError("la brecha no puede dejar h2 de voz en cero o menos")

    dos = una = 0
    for _ in range(n_rep):
        idx = rng.integers(0, n, size=(n_boot, n))
        dif = _delta_promedio(n, H2_ROSTRO, k_a, idx, rng) - _delta_promedio(
            n, h2_b, k_b, idx, rng
        )
        lo2, hi2 = np.quantile(dif, [alpha / 2, 1 - alpha / 2])
        dos += bool(lo2 > 0 or hi2 < 0)
        una += bool(np.quantile(dif, alpha) > 0)
    return dos / n_rep, una / n_rep


def potencia_h2(
    n: int,
    h2_a: tuple,
    h2_b: tuple,
    n_rep: int = 200,
    n_boot: int = 400,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float]:
    """Como `potencia`, pero con un h² por rasgo en cada dominio.

    Sirve para usar los valores publicados de cada parámetro (por ejemplo
    H2_VOZ_REAL) en lugar de un promedio con una brecha supuesta.
    """
    rng = np.random.default_rng(seed)
    dos = una = 0
    for _ in range(n_rep):
        idx = rng.integers(0, n, size=(n_boot, n))
        dif = _delta_promedio(n, tuple(h2_a), len(h2_a), idx, rng) - _delta_promedio(
            n, tuple(h2_b), len(h2_b), idx, rng
        )
        lo2, hi2 = np.quantile(dif, [alpha / 2, 1 - alpha / 2])
        dos += bool(lo2 > 0 or hi2 < 0)
        una += bool(np.quantile(dif, alpha) > 0)
    return dos / n_rep, una / n_rep


def main() -> None:
    p = argparse.ArgumentParser(description="Potencia del escalón Rostro - Voz")
    p.add_argument(
        "--unidad",
        choices=("rasgo", "dominio"),
        default="dominio",
        help="rasgo: compara un rasgo contra uno (k=1). dominio: promedia "
        "los 28 componentes de Rostro y los 8 parámetros de Voz.",
    )
    p.add_argument("--n-rep", type=int, default=150)
    p.add_argument("--n-boot", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--voz-real",
        action="store_true",
        help="compara Rostro (0,079 en sus 28 componentes) contra los 8 h² "
        "publicados de Voz, en vez de una grilla de brechas supuestas.",
    )
    args = p.parse_args()

    if args.voz_real:
        print("Rostro: 0,079 en 28 componentes. Voz: h² publicados de sus 8 parámetros")
        print(f"(promedio {np.mean(H2_VOZ_REAL):.4f}, brecha {H2_ROSTRO - np.mean(H2_VOZ_REAL):.4f})")
        print(f"{args.n_rep} experimentos por fila, {args.n_boot} remuestreos cada uno\n")
        print(f"{'N prueba':>9}  {'dos colas':>10}  {'una cola':>9}")
        for n in N_PRUEBA:
            dos, una = potencia_h2(
                n, (H2_ROSTRO,) * 28, H2_VOZ_REAL, args.n_rep, args.n_boot, seed=args.seed
            )
            print(f"{n:>9}  {dos:>10.2f}  {una:>9.2f}")
        return

    k_a, k_b = (28, 8) if args.unidad == "dominio" else (1, 1)

    print(f"ΔR² de Rostro fijado en {H2_ROSTRO:.3f} (Xiong et al., 2025)")
    print(f"Unidad de decisión: {args.unidad} (k_rostro={k_a}, k_voz={k_b})")
    print(f"{args.n_rep} experimentos por celda, {args.n_boot} remuestreos cada uno")
    print("Potencia a dos colas / a una cola. Objetivo convencional: 0,80\n")

    enc = [f"brecha {b:.3f}" for b in BRECHAS]
    print(f"{'N prueba':>9}  " + "  ".join(f"{e:>16}" for e in enc))
    print("-" * (9 + 2 + 18 * len(BRECHAS)))

    for n in N_PRUEBA:
        celdas = []
        for b in BRECHAS:
            dos, una = potencia(
                n, b, k_a, k_b, args.n_rep, args.n_boot, seed=args.seed
            )
            celdas.append(f"{dos:.2f} / {una:.2f}")
        print(f"{n:>9}  " + "  ".join(f"{c:>16}" for c in celdas))

    print(
        "\nDos advertencias que apuntan en direcciones opuestas:\n"
        "  · Optimista: el modelo es un oráculo, conoce el valor genético\n"
        "    verdadero. Un modelo que estima la pendiente tiene menos potencia.\n"
        "  · Optimista con unidad=dominio: supone ruido independiente entre\n"
        "    rasgos. Vale para los PCA de Rostro, no tanto para los\n"
        "    parámetros de Voz.\n"
        "La tabla sirve para decidir el orden de magnitud de N, no el valor\n"
        "exacto. Elegir con margen."
    )


if __name__ == "__main__":
    main()
