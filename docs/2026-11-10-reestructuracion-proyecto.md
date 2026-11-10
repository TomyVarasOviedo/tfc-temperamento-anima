# Minuta: Actualización a 2.2 y limpieza de estructura

| Campo | Valor |
|-------|-------|
| Fecha | 2026-11-10 |
| Tema | Actualización del spine a 2.2 y reestructuración del repo |
| Lenguajes | Python |
| Frameworks/Libs | numpy, pytest, pyyaml, scipy (solo legacy) |
| Herramientas | uv/venv, git, zsh |
| Versiones clave | Python 3.11+, anima_spine 2.0 → 2.2, cohorte 2.0 |
| Estado final | éxito |

## Contexto
El `anima_spine/` de la raíz estaba en 2.0 mientras la entrega 2.2 vivía como zip y como bundle descomprimido (`columna_vertebral_anima/`). Se pidió sincronizar el código a 2.2 y luego dejar una estructura limpia, sin duplicados y con artefactos consistentes.

## Decisiones
- Copiar los cuatro archivos que cambian (`__init__.py`, `compare.py`, `predictions.py`, `README.md`) desde el bundle 2.2: garantiza paridad exacta y evita reimplementar.
- Eliminar `columna_vertebral_anima/`: duplicaba spine, datos, tests y demo; el entregable se conserva en `dist/*.zip`.
- Organizar tests por componente (`tests/anima_spine/`, `tests/temperamento/`): separación clara entre spine y módulo.
- Dejar `MODEL.md` y `CONTRACT.md` en la raíz: son especificaciones canónicas visibles.
- Regenerar `outputs/` con la demo 2.2: los CSV de 2.0 tenían una sola línea base y rompían los tests.
- La demo añade la raíz a `sys.path` y usa rutas absolutas: `python examples/demo_circuito.py` funciona desde cualquier cwd.

## Complicaciones y soluciones
- **Problema**: ejecutar la demo daba `ModuleNotFoundError: anima_spine`.
  **Solución**: insertar `RAIZ` en `sys.path` antes del import del paquete.
- **Problema**: `RAIZ` de `test_contrato_compartido.py` quedó mal al mover el test a subcarpeta.
  **Solución**: cambiar a `Path(__file__).resolve().parents[2]`.
- **Problema**: 6 tests fallaban porque `outputs/` venía de 2.0 con una sola línea base.
  **Solución**: borrar esos CSV y regenerarlos con la demo 2.2 (dos líneas base).

## Cambios realizados
- `anima_spine/__init__.py`: versión `2.0` → `2.2`.
- `anima_spine/compare.py`: ΔR² total/parcial, `ordenamiento`, sin `hipotesis_dos_niveles`.
- `anima_spine/predictions.py`: `ContratoError`, criterio y archivo de recuperación, `TOLERANCIA_RECUPERACION = 0.015`.
- `anima_spine/README.md`: changelog 2.1/2.2.
- `examples/demo_circuito.py`: versión 2.2, lee `data/` y exporta a `outputs/`.
- `tests/anima_spine/test_spine.py` y `test_contrato_compartido.py`: movidos.
- `tests/anima_spine/test_recuperacion_y_parcial.py`: incorporado desde el bundle.
- `columna_vertebral_anima/`: eliminado por duplicación.
- `README.md`: árbol, componentes y salidas actualizados.
- `outputs/`: regenerado con 12 CSV (predicciones, intervalos, barrido y recuperación × 3 dominios).
- `dist/columna_vertebral_anima_v2.2.zip`: entregable conservado.

## Pendientes
- Hacer commit de los cambios (no se hizo en esta sesión).
- Decidir seguimiento en git de `CONTRACT.md` y `dist/columna_vertebral_anima_v2.2.zip` (hoy sin trackear).
- Reemplazar los datos simulados de `outputs/` cuando los módulos reales exporten.

## Notas técnicas
- Tests: `.venv/bin/python -m pytest -q` → **96 passed**.
- Demo: `.venv/bin/python examples/demo_circuito.py`.
- `outputs/` está en `.gitignore`; se regenera con la demo.
- API nueva relevante: `ContratoError`, `fila_recuperacion`, `exportar_recuperacion`, `cargar_recuperacion`, `verificar_recuperacion`, `criterio_recuperacion`, `compare.delta_r2`.
- La cohorte 2.0 y su huella SHA-256 no cambian entre 2.0 y 2.2.
