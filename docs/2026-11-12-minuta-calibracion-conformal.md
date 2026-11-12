# Minuta: capa de calibración conformal v1

| Campo | Valor |
|-------|-------|
| Fecha | 2026-11-12 |
| Tema | calibracion-conformal |
| Lenguajes | Python 3.11 |
| Frameworks/Libs | numpy, pandas 3.1.1, scipy 1.17.1, pytest |
| Herramientas | uv, pytest |
| Versiones clave | contrato v5.1 congelado; cohorte 2.0 |
| Estado final | exito |

## Contexto
Se implementó la capa de calibración conformal del contrato v5.1 congelado, sin conectar aún el modelo de Temperamento. La tarea se limitó a `anima_spine/calibracion.py` y sus tests.

## Decisiones
- Implementar en `anima_spine/calibracion.py`: el usuario pidió ignorar el desvío previo hacia `calibration/`.
- Usar `pandas` y `scipy`: necesarios para DataFrames y distribución normal/beta; solo se importan en `calibracion.py`.
- No tocar archivos de solo lectura: `predictions.py`, `spine.py`, `compare.py`, `power_gradiente.py`, `__init__.py`, cohorte, versión 2.2.
- Validar con datos sintéticos y, en T9, con ids reales de la cohorte: permite verificar el contrato sin depender de valores reales.
- No integrar todavía el modelo de Temperamento: queda fuera del alcance de la capa.

## Complicaciones y soluciones
- **Problema**: faltaba el import de `math`.
  **Solución**: se agregó.
- **Problema**: merges de `predict`, `contar` y `coverage_report` sin cardinalidad explícita.
  **Solución**: se agregaron validaciones `one_to_many`, `many_to_one` y `many_to_many`.
- **Problema**: la validación de positividad se aplicaba por error a `mu_oraculo`.
  **Solución**: se limitó a `sigma_oraculo`.
- **Problema**: T9 usaba ids sintéticos.
  **Solución**: se remapearon a ids reales por partición, con `pytest.skip` si la cohorte falta.
- **Problema**: `fit` podía dejar `modo_` cambiado si fallaba a mitad de camino.
  **Solución**: `modo_` se asigna recién al final, junto con `qhat_`.

## Cambios realizados
- `anima_spine/calibracion.py`: implementación completa de V1-V15, `ConformalCalibrator.fit`, `predict`, `contar`, `coverage_report` y `control_permutacion`.
- `tests/anima_spine/test_calibracion.py`: 15 tests T1-T14, incluidos V1-V15, determinismo, n efectivo, sin efectos laterales, columnas y verificadores del paquete.
- `pyproject.toml`: dependencias `pandas>=3.1.1` y `scipy>=1.17.1`.
- `uv.lock`: actualizado por `uv add pandas scipy`.

## Pendientes
- Ninguno bloqueante.
- La conexión de la capa con el flujo de Temperamento queda fuera de esta tarea.

## Notas tecnicas
- Verificación: `.venv/bin/python -m pytest -q -p no:cacheprovider` -> 111 passed.
- `fit` en modo modelo usa solo `linea_base == "solo intercepto"` y n = 500 por rasgo.
- `fit` en modo base calibra cada línea por separado, n = 500 por línea.
- `k = ceil((n + 1) * nivel)`; los niveles salen de `P.NIVELES`.
- `coverage_report` devuelve `cobertura`, `crps` y, si hay `sexo`, `cobertura_por_sexo`.
- `control_permutacion` devuelve `por_rasgo` completo y `por_nivel` con `pasa`.
- No se hizo commit.
