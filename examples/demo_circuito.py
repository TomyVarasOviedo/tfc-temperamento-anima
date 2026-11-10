"""Prueba del circuito completo con los tres módulos simulados.

No es parte del entregable: sirve para verificar que la columna vertebral,
el formato de predicciones y el bootstrap pareado funcionan juntos antes
de que los tres módulos reales se enganchen.

Simula tres dominios con niveles de señal declarados distintos, exporta los
cuatro archivos del contrato (predicciones, intervalos, barrido y
recuperación) contra las DOS líneas base, y evalúa el criterio de
recuperación del nivel declarado junto con el contraste entre dominios.
"""

from __future__ import annotations

import sys
from pathlib import Path
from statistics import NormalDist

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

from anima_spine import compare, predictions, spine

COHORTE = RAIZ / "data" / "cohorte_anima.csv"
OUTPUTS = RAIZ / "outputs"

# Nivel de señal declarado por dominio, y cuánto explica la línea base de
# ancestría, sexo y edad. En el dominio vocal la línea base explica mucho
# (el sexo), y es ahí donde el ΔR² total y el parcial se separan.
H2_SIMULADO = {"rostro": 0.060, "voz": 0.035, "temperamento": 0.010}
R2_BASE = {"rostro": 0.02, "voz": 0.55, "temperamento": 0.0}
RASGOS = {
    "rostro": ["PC1", "PC2", "PC3"],
    "voz": ["F0", "F1", "F2"],
    "temperamento": ["neuroticismo", "extraversion", "apertura"],
}


def modulo_simulado(dominio, ids, h2, seed):
    """Lo que haría un módulo real: generar, ajustar y exportar contra las dos líneas base."""
    rng = np.random.default_rng(seed)
    n = len(ids)
    r2b = R2_BASE[dominio]
    filas = []

    for rasgo in RASGOS[dominio]:
        # y = covariables + genotipo + ruido, con var(g) = h2 * (1 - R2 base),
        # de modo que el ΔR² parcial contra "ancestria sexo edad" da h2.
        cov = rng.standard_normal(n) * np.sqrt(r2b)
        g = rng.standard_normal(n) * np.sqrt(h2 * (1 - r2b))
        y = cov + g + rng.standard_normal(n) * np.sqrt((1 - h2) * (1 - r2b))

        mu, sigma = cov + g, np.sqrt((1 - h2) * (1 - r2b))
        z = NormalDist().inv_cdf(0.95)  # intervalo nominal del 90 %
        bases = {
            "solo intercepto": (np.full(n, y.mean()), y.std(ddof=1)),
            "ancestria sexo edad": (cov, float(np.std(y - cov, ddof=1))),
        }

        for linea_base, (mu_base, sigma_base) in bases.items():
            filas += [
                {
                    "individuo_id": int(i),
                    "dominio": dominio,
                    "rasgo": rasgo,
                    "y_real": float(yi),
                    "mu": float(m),
                    "sigma": float(sigma),
                    "lo": float(m - z * sigma),
                    "hi": float(m + z * sigma),
                    "mu_base": float(mb),
                    "sigma_base": float(sigma_base),
                    "linea_base": linea_base,
                }
                for i, yi, m, mb in zip(ids, y, mu, np.asarray(mu_base, dtype=float))
            ]
    return filas


def recuperacion_simulada(dominio, ids, h2, semilla_principal):
    """El ΔR² de cada réplica contra su nivel declarado: lo que evalúa el criterio."""
    filas = []
    for r in range(predictions.REPLICAS):
        semilla = semilla_principal if r == 0 else 10_000 * (semilla_principal + 1) + r
        pred = modulo_simulado(dominio, ids, h2, seed=semilla)
        for rasgo in RASGOS[dominio]:
            for linea_base in predictions.LINEAS_BASE:
                sel = [f for f in pred if f["rasgo"] == rasgo and f["linea_base"] == linea_base]
                y = np.array([f["y_real"] for f in sel])
                total, parcial = compare.delta_r2(
                    y,
                    np.array([f["mu"] for f in sel]),
                    np.array([f["mu_base"] for f in sel]),
                )
                filas.append(predictions.fila_recuperacion(
                    r, semilla, dominio, rasgo, linea_base, h2, total, parcial))
    return filas


def intervalos_simulados(filas_pred):
    """Intervalos en los seis niveles, con el mismo mu y sigma de las predicciones."""
    vistos = set()
    salida = []
    for f in filas_pred:
        clave = (f["rasgo"], f["individuo_id"])
        if clave in vistos:
            continue
        vistos.add(clave)
        for nivel in predictions.NIVELES:
            z = NormalDist().inv_cdf(0.5 + nivel / 2)
            if nivel == predictions.NIVEL_PRINCIPAL:
                lo, hi = f["lo"], f["hi"]
            else:
                lo, hi = f["mu"] - z * f["sigma"], f["mu"] + z * f["sigma"]
            salida.append({
                "individuo_id": f["individuo_id"], "dominio": f["dominio"],
                "rasgo": f["rasgo"], "nivel": nivel, "lo": lo, "hi": hi,
            })
    return salida


def barrido_simulado(dominio, ids, h2, semilla_principal):
    """Las REPLICAS corridas completas del módulo simulado, resumidas por rasgo y nivel.

    La réplica 0 usa la semilla principal: es la misma corrida que se exporta
    en predicciones_ e intervalos_.
    """
    filas = []
    for r in range(predictions.REPLICAS):
        semilla = semilla_principal if r == 0 else 10_000 * (semilla_principal + 1) + r
        pred = modulo_simulado(dominio, ids, h2, seed=semilla)
        y = {(f["rasgo"], f["individuo_id"]): f["y_real"] for f in pred}
        for rasgo in RASGOS[dominio]:
            ints = [f for f in intervalos_simulados(pred) if f["rasgo"] == rasgo]
            for nivel in predictions.NIVELES:
                sel = [f for f in ints if f["nivel"] == nivel]
                filas.append(predictions.fila_barrido(
                    r, semilla, dominio, rasgo, nivel,
                    [y[(rasgo, f["individuo_id"])] for f in sel],
                    [f["lo"] for f in sel], [f["hi"] for f in sel],
                ))
    return filas


def main():
    sp = spine.cargar(COHORTE)
    spine.verificar(sp)
    ids_prueba = sp.ids("prueba")
    print(f"Columna vertebral: {len(sp.filas)} individuos, "
          f"{len(ids_prueba)} en prueba\n")

    preds = {}
    recuperaciones = {}
    for i, (dominio, h2) in enumerate(H2_SIMULADO.items()):
        filas = modulo_simulado(dominio, ids_prueba, h2, seed=i)
        predictions.verificar(filas, set(ids_prueba), dominio)
        ruta = predictions.exportar(filas, dominio, directorio=OUTPUTS)
        preds[dominio] = predictions.cargar(ruta)
        intervalos = intervalos_simulados(filas)
        predictions.verificar_intervalos(intervalos, filas, set(ids_prueba), dominio)
        predictions.exportar_intervalos(intervalos, dominio, directorio=OUTPUTS)
        barrido = barrido_simulado(dominio, ids_prueba, h2, semilla_principal=i)
        predictions.verificar_barrido(barrido, intervalos, filas, set(ids_prueba), dominio)
        predictions.exportar_barrido(barrido, dominio, directorio=OUTPUTS)
        rec = recuperacion_simulada(dominio, ids_prueba, h2, semilla_principal=i)
        predictions.verificar_recuperacion(rec, dominio)
        predictions.exportar_recuperacion(rec, dominio, directorio=OUTPUTS)
        recuperaciones[dominio] = rec
        print(f"  {dominio:>14}: {len(filas)} filas -> {ruta.name}  [ok]")

    print()
    resultado = compare.comparar(preds, RASGOS, n_boot=1000, seed=0)
    compare.imprimir(resultado)

    print("\nCriterio de recuperación del nivel declarado "
          f"(tolerancia {predictions.TOLERANCIA_RECUPERACION})")
    print(f"{'dominio':>14}  {'declarado':>10}  {'recuperado':>11}  {'desvío':>8}  {'cumple':>7}")
    print("-" * 58)
    todos = True
    for dominio, rec in recuperaciones.items():
        c = predictions.criterio_recuperacion(rec)
        todos = todos and c["cumple"]
        print(f"{dominio:>14}  {c['nivel_declarado']:>10.4f}  "
              f"{c['parcial_recuperado']:>11.4f}  {c['desvio']:>+8.4f}  "
              f"{'sí' if c['cumple'] else 'no':>7}")
    print(f"\nLos tres dominios recuperan su nivel declarado: {'sí' if todos else 'NO'}")


if __name__ == "__main__":
    main()
