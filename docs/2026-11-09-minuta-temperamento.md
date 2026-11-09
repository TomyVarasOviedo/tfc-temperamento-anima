# Minuta · Estado del módulo Temperamento

**Fecha:** 2026-11-09
**Responsable:** Tomás Varas Oviedo
**Referencia:** `MODEL.md` (reformulación de noviembre de 2026)

## Contexto

`MODEL.md` redefine el módulo: cinco dominios del IPIP-NEO-120, genotipo
simulado, modelos sin calibración y ΔR² fuera de muestra. Esta minuta deja
registrado qué se implementó, qué decisiones se tomaron y qué falta para
poder exportar y ejecutar el barrido.

## Estado del repositorio al iniciar

- La cohorte compartida `data/cohorte_anima.csv` es la versión 2.0 y está
  validada por `anima_spine.spine.cargar`.
- `anima_spine/predictions.py` sigue el contrato anterior: exige `lo`, `hi`,
  `sigma_base`, archivos de intervalos y barrido de cobertura.
- `anima_spine/compare.py` solo calcula ΔR² promedio por dominio cuando los
  tres módulos están presentes. No ofrece ΔR² por rasgo ni ΔR² parcial.
- No existe en el repositorio el “contrato v2” con invariantes 1 a 6 que
  menciona `MODEL.md`.
- No existe un módulo Rostro ni una carpeta `entrega/` de referencia.
- El módulo `temperamento/` anterior era de la etapa heterocedástica y
  conformal, incompatible con el modelo nuevo.

## Decisiones tomadas

1. Implementar primero el núcleo estadístico y dejar la exportación para
   cuando esté disponible el contrato v2.
2. Archivar el código anterior en `temperamento/legacy/`, sin borrarlo.
3. Generar artefactos en `outputs/` cuando llegue la etapa de exportación.
4. Usar `SEMILLA_TEMPERAMENTO = 20261109` y el orden de niveles
   `base=0`, `ldsc=1`, `pgi=2`, `nulo=3`.
5. Mantener el orden canónico de rasgos `E, A, C, N, O` y documentación de
   funciones y clases con docstrings estilo Google.

## Implementado en esta sesión

### `temperamento/parametros.py`

Parámetros con fuente: `ALPHA`, `H2_LATENTE_BASE`, `H2_OBSERVADA_LDSC`,
`H2_OBSERVADA_PGI`, `RG`, `RP`, niveles, semilla y rutas. Incluye
`h2_observada(nivel)` y `h2_latente(nivel)`.

### `temperamento/matrices.py`

Construye `R_lat`, `G` y `Sigma_E` según `MODEL.md` §4.2. Verifica simetría
y definida positiva y lanza `ValueError` si falla. No proyecta ni corrige
matrices en silencio.

### `temperamento/simulacion.py`

Simula `g`, `t`, `y` y `epsilon`. Usa `SeedSequence([semilla, réplica,
índice_nivel])` y cuatro flujos `spawn` independientes para `g`, `e`,
`epsilon` y la permutación. En el nivel `nulo` simula con los parámetros de
`base` y permuta `g` entre personas.

### `temperamento/modelos.py`

- OLS `y ~ 1 + g` con `sigma` calculado con `ddof = 2`.
- Línea base de solo intercepto.
- Línea base `y ~ 1 + sexo + edad + subpoblación`, con indicadoras y
  categoría de referencia alfabética (la primera).
- Todas las funciones de ajuste reciben únicamente datos de entrenamiento.

### Tests

- `tests/temperamento/test_matrices.py`: T-01 y T-02, más valores de control.
- `tests/temperamento/test_simulacion.py`: T-03, T-08, T-09 y T-10 a T-14.
- `tests/temperamento/test_modelos.py`: ajustes, diseño de líneas base y T-06.

## Semillas y reproducibilidad

- La cohorte conserva sus semillas; el módulo no la regenera.
- Misma semilla y misma réplica producen arrays idénticos.
- Réplicas distintas producen `g` distinto.
- El nivel `nulo` usa su propio índice de semilla y conserva la misma
  permutación para los cinco rasgos.

## Verificación

```bash
.venv/bin/python -m pytest tests/temperamento -q
```

Resultado: **37 passed**.

La suite completa da **76 passed, 3 failed**. Los tres fallos son previos y
están documentados en el README: la demo exporta una sola línea base y por
eso falla `test_reporta_las_dos_lineas_base` para rostro, voz y temperamento.
No provienen de este cambio.

## Pendientes bloqueantes

1. **Contrato v2 e invariantes 1–6:** no están en el repositorio. Hay que
   esperar el archivo o la rama indicada por el responsable.
2. **Exportación:** `tabla_larga_temperamento.csv`, `predicciones_temperamento.csv`
   y `LEEME.md` no se generaron.
3. **Barrido ΔR²:** `compare.py` no ofrece la API necesaria para ΔR² por
   rasgo, línea base y nivel. Falta definir si se usa una versión 2.1 o una
   función propia verificada contra la fórmula de `compare.py`.
4. **Tests de contrato:** T-04, T-05 y T-07 quedan pendientes hasta tener el
   contrato v2 y la carpeta de salida definitiva.
5. **Calibración:** la función `calibrar` de `MODEL.md` §7 no se implementó;
   sigue siendo trabajo futuro.
6. **Compatibilidad del formato viejo:** `anima_spine/predictions.py` exige
   intervalos, mientras que `MODEL.md` prohíbe exportarlos. No se modificó
   infraestructura compartida.
7. **Código archivado:** `temperamento/legacy/` no se importa y sus rutas
   relativas pueden no resolver tras el traslado.

## Próximos pasos

1. Recibir el contrato v2 y su validador.
2. Implementar `exportar.py` con las columnas exactas del contrato.
3. Implementar `barrido.py` y el cálculo de ΔR² por rasgo, línea base y nivel.
4. Añadir T-04, T-05 y T-07.
5. Generar los artefactos en `outputs/` y actualizar esta minuta con los
   resultados esperados de `MODEL.md` §9.4.
