# MODEL.md · Módulo Temperamento (Proyecto Anima)

Instrucciones para un agente de codificación. Este archivo describe **qué** hay que construir, **por qué** cada decisión es así y **cómo** verificar que quedó bien. Leelo completo antes de escribir código.

Responsable del módulo: Tomás Varas Oviedo. Fecha de las decisiones: 09/11/2026.

---

## 0. Reglas para el agente

1. **No inventes nombres del repositorio.** Este documento no conoce la estructura exacta del repo. Antes de escribir código, buscá y leé:
   - el generador de cohortes y el archivo de cohorte (`cohorte_anima.csv`, versión 2.0, paquete `anima_spine`);
   - el contrato de entrada v2 y su validador (invariantes 1 a 6);
   - `predictions.py` y `compare.py` (versión 2.0);
   - el módulo Rostro, si está en el repo, como **referencia de formato** (`entrega/`, `tabla_larga_rostro.csv`, `predicciones_rostro.csv`, `LEEME.md`).

   Usá exactamente los nombres de columnas, de particiones, de líneas base y de funciones que encuentres ahí. Donde este documento dice `<…>`, es un dato que tenés que sacar del repo.
2. **No modifiques** el generador de cohortes, la cohorte, el contrato, `compare.py` ni el archivo de pruebas de contrato. Son infraestructura compartida por tres módulos. Si algo no encaja, detenete y dejá la pregunta escrita en `temperamento/PENDIENTES.md`.
3. **Sin calibración.** La capa conformal (`anima_spine/calibracion.py`) es trabajo pendiente del responsable. No calcules `q̂`, intervalos conformales ni cobertura. El módulo exporta `mu` y `sigma` sin calibrar y deja el punto de enganche preparado (sección 7).
4. **Todo número del modelo sale de `parametros.py`**, con su fuente al lado. Nada de constantes sueltas en el código.
5. Si un resultado no coincide con lo esperado en la sección 9, no "ajustes" parámetros para que coincida. Reportalo.

---

## 1. Qué es este módulo y qué tiene que demostrar

Anima reconstruye rasgos a partir de genotipo y cuantifica la incertidumbre. Tiene tres dominios con señal genética decreciente: Rostro > Voz > Temperamento. El instrumento estadístico es el **ΔR² fuera de muestra** del modelo con genotipo contra una línea base poblacional.

**Argumento de Temperamento (reformulado en noviembre de 2026):** la personalidad tiene **señal genética chica pero real**. No es "señal cero". El GWAS de referencia (Schwaba et al., Nature 2026) muestra que las variantes comunes explican entre 4,8 % y 9,3 % de la varianza de cada rasgo y que esa asociación casi no está confundida por el entorno familiar. El módulo tiene que mostrar:

- que el ΔR² de Temperamento es positivo pero chico (del orden del 10 % en el caso base, contra 23 % de Rostro);
- que con un puntaje poligénico realista (nivel de sensibilidad, R² de 1 a 4 %) la señal ya no se distingue de cero con 500 personas de prueba. Esto es un límite de **potencia estadística**, no ausencia de efecto, y así se tiene que reportar.

**Instrumento:** IPIP-NEO-120 (Johnson 2014). Se modelan sus **5 dominios**, no las 30 facetas. El genotipo es **simulado** (no hay datos públicos que emparejen genoma y personalidad para los mismos individuos).

---

## 2. Alcance

### Dentro
- Simular, para cada persona de la cohorte compartida, un puntaje genético y un puntaje observado de IPIP-NEO-120 en los 5 dominios, **correlacionados entre sí**.
- Ajustar el predictor con genotipo y las dos líneas base sobre la partición de entrenamiento.
- Exportar la tabla larga y las predicciones con el formato del contrato v2.
- Barrido de 20 réplicas en 4 niveles de señal y cálculo de ΔR² con `compare.py`.
- Pruebas con pytest.

### Fuera (trabajo futuro, no implementar)
- Calibración conformal, intervalos, cobertura, CRPS calibrado.
- Efecto de la ancestría (decisión del 09/11/2026: va a trabajo futuro). El puntaje poligénico real rinde mucho menos en ascendencia africana (R² medio 2,5 % EUR contra 0,3 % AFR, tabla S36), pero no se modela.
- Efecto del sexo y de la edad sobre los rasgos (efecto cero, igual que Rostro).
- Facetas, simulación ítem por ítem, normas de puntuación del IPIP en escala 24 a 120.
- Interfaz web, API, capa de LLM.

---

## 3. Parámetros (archivo `temperamento/parametros.py`)

Orden fijo de los rasgos en todo el código: `E, A, C, N, O`.

| Código | Dominio IPIP-NEO-120 | Nombre en inglés (Schwaba) |
|---|---|---|
| E | Extraversión | Extraversion |
| A | Amabilidad | Agreeableness |
| C | Responsabilidad | Conscientiousness |
| N | Neuroticismo | Neuroticism |
| O | Apertura a la experiencia | Openness to experience |

Si el contrato o Rostro usan una columna tipo `componente` o `rasgo` con identificadores, usá ese mismo formato (por ejemplo, `E` o `extraversion`). Elegí uno y respetalo en todo el módulo.

### 3.1 Confiabilidad del instrumento (α de Cronbach)

Fuente: Johnson, J. A. (2014), *Measuring thirty facets of the Five Factor Model with a 120-item public domain inventory*, J. Research in Personality, Tabla 2, columna "Internet sample (N = 619,150), IPIP 120". Es la única muestra que respondió directamente el inventario de 120 ítems.

```python
ALPHA = {"E": 0.89, "A": 0.86, "C": 0.90, "N": 0.90, "O": 0.81}
```

### 3.2 Heredabilidad latente (sin error de medición)

Fuente: Schwaba et al. (2026), Tabla 1, columna "Disattenuated for measurement instrument error", segundo valor ("all", α = 1). Coincide con la tabla suplementaria S21, columna "Meta-regression, internal consistency alpha = 1.0".

```python
H2_LATENTE_BASE = {"E": 0.158, "A": 0.108, "C": 0.112, "N": 0.120, "O": 0.152}
```

### 3.3 Niveles de señal del barrido

Cada nivel fija el **h² observado** (lo que vería el IPIP-NEO-120, con su error de medición). El código lo convierte a h² latente con `h2_latente = h2_observado / ALPHA[k]`.

| Nivel | Qué representa | Fuente | E | A | C | N | O |
|---|---|---|---|---|---|---|---|
| `base` | h²_SNP del rasgo medido con el IPIP-NEO-120 | Tabla 1 (α = 1) × α de Johnson | 0,1406 | 0,0929 | 0,1008 | 0,1080 | 0,1231 |
| `ldsc` | h²_SNP de LDSC sobre puntajes observados, GWAS combinado | Tabla S4, columna "LDSC h2", EUR-like | 0,0926 | 0,0480 | 0,0713 | 0,0832 | 0,0745 |
| `pgi` | R² que logra hoy un puntaje poligénico real | Tabla S36, "Population", EUR-like, cohortes HRS, GSOEP, TEDS, NTR, Add Health | 0,0355 | 0,0094 | 0,0269 | 0,0302 | 0,0302 |
| `nulo` | Control: genotipo permutado | (sección 5.4) | usa `base` y permuta | | | | |

Para `base`, el h² latente es exactamente `H2_LATENTE_BASE` (la fila de la tabla es α × h² latente). Guardá en `parametros.py` los valores **de origen** (α, h² latente, LDSC, PGI) y calculá el resto en código. No copies la fila `base` ya multiplicada.

**Importante sobre el nulo:** el escenario nulo **no** se construye con h² = 0. Con h² = 0 el genotipo vale idénticamente cero y el ΔR² da cero por construcción, lo que no prueba nada (lección de la Práctica 10). Se simula con el nivel `base` y se **permuta el puntaje genético entre personas** antes de ajustar el predictor.

### 3.4 Correlaciones entre los cinco rasgos

Fuente: Schwaba et al. (2026), Figura 1f. Debajo de la diagonal: correlaciones genéticas (LDSC, EUR-like). Arriba de la diagonal: correlaciones fenotípicas meta-analíticas (referencia 32 del paper). Orden `E, A, C, N, O`.

```python
RG = [  # correlación genética
    [ 1.00,  0.16,  0.19, -0.27,  0.22],
    [ 0.16,  1.00,  0.20, -0.36,  0.05],
    [ 0.19,  0.20,  1.00, -0.20, -0.20],
    [-0.27, -0.36, -0.20,  1.00, -0.05],
    [ 0.22,  0.05, -0.20, -0.05,  1.00],
]
RP = [  # correlación fenotípica observada
    [ 1.00,  0.15,  0.16, -0.24,  0.28],
    [ 0.15,  1.00,  0.29, -0.24,  0.14],
    [ 0.16,  0.29,  1.00, -0.29,  0.08],
    [-0.24, -0.24, -0.29,  1.00, -0.08],
    [ 0.28,  0.14,  0.08, -0.08,  1.00],
]
```

Limitación declarada: `RP` viene de un meta-análisis con varios instrumentos, no es específica del IPIP-NEO-120.

---

## 4. Modelo generativo (la "verdad" simulada)

### 4.1 Esquema

```
cohorte_anima.csv (id, subpoblación, sexo, edad, partición)   <- NO se modifica
        │
        ▼
genotipo   g ~ N(0, G)                    5 dimensiones, una por rasgo
ambiente   e ~ N(0, Σ_E)                  independiente de g
rasgo latente   t = g + e                 varianza 1 por rasgo
puntaje observado   y = √α · t + √(1−α) · ε     ε ~ N(0, I), independiente
        │
        ▼
y = lo que "responde" la persona en el IPIP-NEO-120 (escala z)
g = lo que el predictor puede ver como puntaje genético
```

### 4.2 Matrices

Sea `h` el vector de h² latente del nivel y `a` el vector de α.

1. **Correlación latente** (corrige la atenuación de `RP`):
   `R_lat[i][j] = RP[i][j] / sqrt(a[i] * a[j])` para i ≠ j; diagonal = 1.
2. **Covarianza genética:** `G[i][j] = RG[i][j] * sqrt(h[i] * h[j])`. Diagonal = `h`.
3. **Covarianza ambiental:** `Σ_E = R_lat − G`. Diagonal = `1 − h`.
4. **Chequeo obligatorio:** `G` y `Σ_E` tienen que ser simétricas y definidas positivas (todos los autovalores > 0). Si no lo son, lanzar error. **No** corregir la matriz con proyecciones al cono positivo: eso cambiaría los datos de Schwaba sin avisar.

Valores de control para el nivel `base` (para un test):

```
R_lat ≈
[[ 1.000  0.171  0.179 -0.268  0.330]
 [ 0.171  1.000  0.330 -0.273  0.168]
 [ 0.179  0.330  1.000 -0.322  0.094]
 [-0.268 -0.273 -0.322  1.000 -0.094]
 [ 0.330  0.168  0.094 -0.094  1.000]]
autovalores de G   ≈ [0.0695 0.0801 0.1104 0.1690 0.2211]
autovalores de Σ_E ≈ [0.4951 0.5619 0.6952 0.9028 1.6950]
```

### 4.3 Propiedades que resultan (y que hay que verificar)

- `Var(y_k) = 1`.
- Confiabilidad: `Var(√α·t) / Var(y) = α`.
- `Corr(y_i, y_j) = RP[i][j]` (la corrección del paso 1 se deshace).
- h² observado: `Cov(g_k, y_k)² / (Var(g_k)·Var(y_k)) = α_k · h_k`. Es el valor de la tabla 3.3.

### 4.4 Por qué rasgos correlacionados

Decisión del grupo: los cinco dominios se simulan **correlacionados** porque así se mueven en las personas reales, y eso hace más creíble la cohorte.

Advertencia para quien redacte resultados: la correlación **no** aumenta el ΔR² de cada rasgo. En esta simulación el mejor predictor de `y_k` es `g_k`, y los puntajes genéticos de los otros cuatro rasgos no agregan información. Además, como los cinco ΔR² quedan correlacionados entre sí, el intervalo del **promedio de dominio** sale algo más ancho que con rasgos independientes. Es correcto que sea así.

### 4.5 Sexo, edad y subpoblación

Efecto **cero** sobre `g`, `e` y `ε`. Esas columnas se usan solo para la segunda línea base (sección 5.2). No las uses para generar los rasgos.

### 4.6 Semillas

- La cohorte trae sus propias semillas. No las toques.
- Temperamento usa un generador propio: `numpy.random.default_rng(semilla)`, con `semilla` derivada de `(réplica, nivel)` de forma determinística y documentada (por ejemplo, `SeedSequence([SEMILLA_TEMPERAMENTO, replica, indice_nivel])`).
- Generá `g`, `e`, `ε` y la permutación del nulo con **flujos separados** (`SeedSequence.spawn`), para que agregar un paso no cambie los números de los otros.
- Misma semilla ⇒ archivos idénticos byte a byte.

---

## 5. Modelos de predicción (sin calibración)

Todos los ajustes usan **solo la partición de entrenamiento** (`<nombre de la partición de entrenamiento según la cohorte>`). Calibración y prueba no participan en ningún ajuste. Esto es lo que piden las invariantes 1 y 2 del contrato: `mu` y `sigma` no pueden depender de los `y` de calibración ni de prueba.

Para cada rasgo `k` por separado:

### 5.1 Modelo con genotipo ("solo genotipo")
- Entrada: `g_k` (el puntaje genético del propio rasgo).
- Regresión lineal por mínimos cuadrados con intercepto: `y_k ~ 1 + g_k`.
- `mu` = predicción para cada persona de la cohorte.
- `sigma` = desvío estándar de los residuos de entrenamiento (homocedástico, igual para todas las personas). Usar `ddof` = cantidad de parámetros ajustados (2) y documentarlo.

Esto es el **camino B** del contrato, el mismo que usa Rostro: el modelo usa solo el genotipo y se compara contra cada línea base por separado. **No** implementes como resultado principal el modelo anidado (línea base + genotipo).

### 5.2 Líneas base (dos)
Usá los identificadores exactos que define el contrato. Conceptualmente:

1. **Solo intercepto:** `mu_base` = media de `y_k` en entrenamiento.
2. **Ancestría, sexo y edad:** regresión lineal `y_k ~ 1 + sexo + edad + subpoblación` (subpoblación con variables indicadoras, una categoría de referencia), ajustada en entrenamiento.

Como el efecto declarado es cero, el R² fuera de muestra de la segunda línea base va a salir **levemente negativo** (Rostro obtuvo ≈ −0,008 con la misma cohorte). Es lo esperado por estimar coeficientes de más. **No** lo recortes a cero ni lo corrijas.

### 5.3 Regla de repetición
En la tabla larga, `mu` y `sigma` del modelo con genotipo **se repiten idénticos** en las dos filas de línea base de cada persona y rasgo. Solo cambia `mu_base`.

### 5.4 Nulo por permutación
Para el nivel `nulo`: simular igual que `base`, después permutar `g` entre las personas (la misma permutación para los 5 rasgos, para conservar su correlación interna) y recién entonces ajustar 5.1. El ΔR² esperado es ≈ 0.

---

## 6. Salidas

Seguí la estructura de `entrega/` de Rostro si está en el repo. Nombres esperados:

| Archivo | Contenido | Tamaño esperado | En git |
|---|---|---|---|
| `entrega/tabla_larga_temperamento.csv` | toda la cohorte × 5 rasgos × 2 líneas base, réplica 0, nivel `base` | 2.000 × 5 × 2 = 20.000 filas | no (se regenera) |
| `entrega/predicciones_temperamento.csv` | solo partición de prueba × 5 × 2 | n_prueba × 10 | no (se regenera) |
| `entrega/barrido_temperamento.csv` | ΔR² por réplica, rasgo, línea base y nivel (20 × 5 × 2 × 4) | 800 filas | sí |
| `entrega/LEEME.md` | versión de `anima_spine`, huella (hash) del CSV de cohorte, fecha, comando para regenerar, semillas | | sí |

Columnas: las del contrato v2 (como mínimo, identificador de persona, partición, rasgo, línea base, `y`, `mu`, `sigma`, `mu_base`). Si el contrato pide más, agregalas. Si pide nombres distintos, usá los del contrato. Recomendado como columnas extra, si el contrato lo permite: `g` (puntaje genético) y `nivel`.

No exportes intervalos ni `q̂`.

### 6.1 ΔR² del barrido
- Calcular con `compare.py` (versión 2.0) sobre `mu` y `mu_base` de la partición de prueba, con los mismos parámetros que Rostro: 2.000 remuestreos bootstrap, semilla 0, `colas = 1`.
- Reportar ΔR² contra las **dos** líneas base, por rasgo y promedio del dominio (promedio simple de los 5 rasgos).
- `compare.py` 2.0 no trae el ΔR² parcial (llega en la 2.1). Si hace falta, calcularlo con la misma fórmula sobre los mismos `mu` y `mu_base`, y verificar que el total coincida exacto con `compare.py`.
- No mezclar en una misma tabla ΔR² de camino A (anidado) y de camino B sin rotularlos. Por defecto, solo camino B.

---

## 7. Punto de enganche para la calibración (no implementar)

Dejá una función con esta forma y que lance `NotImplementedError`:

```python
def calibrar(tabla_larga, linea_base):
    """Pendiente: delegará en anima_spine/calibracion.py (capa común).
    Al implementarla: filtrar UNA sola línea base antes de calcular q̂
    (n de calibración, no el doble), porque mu y sigma se repiten en ambas filas."""
    raise NotImplementedError
```

---

## 8. Estructura sugerida

Adaptala a la convención del repo si ya existe una para módulos.

```
temperamento/
  __init__.py
  parametros.py      # sección 3, con fuente de cada número
  matrices.py        # sección 4.2 + chequeo de definida positiva
  simulacion.py      # sección 4: lee la cohorte, devuelve g, y por persona y rasgo
  modelos.py         # sección 5: genotipo, dos líneas base, nulo
  exportar.py        # sección 6: tabla larga, predicciones, LEEME
  barrido.py         # 20 réplicas × 4 niveles, ΔR² con compare.py
  calibracion_hook.py  # sección 7
  PENDIENTES.md      # dudas que el agente no pudo resolver con el repo
tests/temperamento/
  test_*.py
```

Punto de entrada único, por ejemplo `python -m temperamento.exportar` y `python -m temperamento.barrido`. Tiempo objetivo: menos de un minuto en una computadora común.

---

## 9. Pruebas (pytest)

### 9.1 Software e invariantes
- **T-01** `G` y `Σ_E` son simétricas y definidas positivas en los 4 niveles. Para `base`, los autovalores coinciden con 4.2 (tolerancia 1e-3).
- **T-02** Si `RG` se altera para que `Σ_E` no sea definida positiva, el código lanza error (no corrige en silencio).
- **T-03** La cohorte no se modifica: hash del CSV antes y después igual.
- **T-04** Forma de la tabla larga: 20.000 filas, sin nulos, 5 rasgos, 2 líneas base.
- **T-05** `mu` y `sigma` idénticos en las dos filas de línea base para cada persona y rasgo.
- **T-06** Sin fuga de información: alterar los `y` de calibración y de prueba no cambia ningún `mu`, `sigma` ni `mu_base`.
- **T-07** El archivo de pruebas de contrato del repo pasa sin modificarlo (Temperamento deja de salir como `skipped`).

### 9.2 Reproducibilidad
- **T-08** Misma semilla ⇒ tabla larga y barrido idénticos byte a byte.
- **T-09** Semillas distintas ⇒ `g` distinto.

### 9.3 Fidelidad de la simulación (N grande, por ejemplo 200.000 personas sintéticas, sin pasar por la cohorte)
- **T-10** `Var(y_k)` = 1 ± 0,01.
- **T-11** `Corr(y_i, y_j)` = `RP[i][j]` ± 0,01.
- **T-12** `Corr(g_i, g_j)` = `RG[i][j]` ± 0,02.
- **T-13** h² observado (`Corr(g_k, y_k)²`) = α_k · h_k ± 0,005, para cada nivel.
- **T-14** Confiabilidad empírica: `Corr(y_k, y'_k)` entre dos "aplicaciones" del test (mismo `t`, distinto `ε`) = α_k ± 0,01.

### 9.4 Resultados esperados (van al informe, no son pasa/no pasa estrictos)
Promedio de 20 réplicas, partición de prueba, contra "solo intercepto":

| Nivel | ΔR² esperado por rasgo | Promedio de dominio esperado |
|---|---|---|
| `base` | ≈ tabla 3.3 (0,09 a 0,14) | ≈ 0,113 |
| `ldsc` | ≈ 0,05 a 0,09 | ≈ 0,074 |
| `pgi` | ≈ 0,01 a 0,04 | ≈ 0,026 |
| `nulo` | ≈ 0 | ≈ 0 |

- Contra "ancestría, sexo y edad", el ΔR² debería quedar unas milésimas **por encima** del de "solo intercepto" (por el R² negativo de esa línea base). Si la diferencia es mayor que ~0,01, revisar.
- Con ~500 personas de prueba, el error estándar del R² de un rasgo en una réplica es ≈ 0,014 cuando R² = 0,025 y ≈ 0,025 a 0,03 cuando R² ≈ 0,10 a 0,14. Por eso en el nivel `pgi` es esperable que el intervalo de un rasgo individual incluya el cero. Eso **es** el resultado de "señal chica", no un error.
- Criterio de recuperación del grupo: ΔR² parcial a menos de ±0,005 del nivel declarado. Ojo: en el nivel `pgi`, el rasgo A tiene R² 0,0094, así que la tolerancia es más de la mitad del valor. Reportarlo como está, sin cambiar el criterio después de ver los resultados.

---

## 10. Limitaciones a dejar documentadas (en `LEEME.md` o en el informe)

1. Genotipo simulado como un puntaje genético por rasgo, no variante por variante.
2. Las correlaciones fenotípicas de la Fig. 1f no son específicas del IPIP-NEO-120.
3. En el IPIP, el dominio Amabilidad es genéticamente poco limpio: carga 0,49 en Extraversión y 0,51 en Amabilidad (Schwaba, tabla S22).
4. Los parámetros vienen de cohortes de ascendencia europea (EUR-like). No se modela la pérdida de rendimiento en otras ascendencias.
5. Efecto cero de sexo y edad, aunque fenotípicamente existen diferencias en los Cinco Grandes.
6. Los α de Johnson vienen de una muestra de internet, mayormente angloparlante.

---

## 11. Referencias

- Schwaba, T. et al. (2026). *Robust inference and correlates from genetic associations with personality*. Nature. https://doi.org/10.1038/s41586-026-10992-9. Tabla 1, Figura 1f, tablas suplementarias S4, S21, S22, S36.
- Johnson, J. A. (2014). *Measuring thirty facets of the Five Factor Model with a 120-item public domain inventory: Development of the IPIP-NEO-120*. Journal of Research in Personality, 51, 78-89. Tabla 2.
- Contrato de entrada v2 del grupo Anima y `compare.py` 2.0 (en el repositorio).
- Nota de Audrey al grupo, 09/11/2026: cálculo del ΔR² (camino B), R² negativo de la segunda línea base, formato de `entrega/`.
