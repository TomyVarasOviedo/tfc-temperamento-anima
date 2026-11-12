# CONTRACT.md: capa de calibración conformal de Anima

Guía para agentes de codificación que construyen o modifican `anima_spine/calibracion.py`.

- **Fuente de verdad:** «Anima: capa de calibración conformal · Requisitos y contrato · v5.1 · CONGELADO» (`Anima_capa_de_calibracion_contrato_v5.1_congelado.md`, 12/11/2026). Este archivo es su versión operativa. Si los dos no coinciden, manda el v5.1: detenete y avisá.
- **Responsable de la capa:** Tomás.
- **Modelo de Temperamento:** está especificado en [`./MODEL.md`](./MODEL.md) (ver sección 12).

---

## 0. Cambios v5.1 (aprobados por los tres el 12/11/2026)

Ya están incorporados en el documento v5.1 y en el resto de este archivo.

| Tema | v5 | v5.1 | Motivo y origen |
| --- | --- | --- | --- |
| RF6, `control_permutacion` | `pasa` por rasgo × nivel, con ventana [0,95; 1,05] | `pasa` sobre el **cociente medio por nivel** (6 banderas, no 48): el promedio simple de los cocientes de los rasgos del nivel (en Rostro, de las componentes). `por_rasgo` se sigue informando completa, sin bandera. Ver sección 7.6 | Simulación de Diego: con la bandera por celda, las 48 celdas de Voz pasan todas juntas solo el 61 % de las veces con un modelo correcto. Aprobado el 12/11 |
| Paquete de base | 2.2 | 2.2 o 2.2.1. La 2.2.1 solo corrige `power_gradiente.py` y no toca nada que use la capa | Diego |
| Importación de `ContratoError` | `from anima_spine import ContratoError` | `from anima_spine.predictions import ContratoError`, porque el `__init__.py` de la 2.2 todavía no la exporta. Es la misma clase | Aviso de Audrey |
| Tipo de entrada | Implícito | Si la tabla no es un `pandas.DataFrame`, la capa levanta `ContratoError`. La conversión, por ejemplo desde un dict de arreglos, la hace cada módulo | Nota de Audrey |
| CRPS | «Es la que usa la condición de la hipótesis» | Diagnóstico **sin umbral**, que **no forma parte de la condición de la hipótesis**. Ver sección 7.5 | Decisión del grupo del 12/11/2026, tomada antes de que la capa calcule ningún CRPS |
| Control de permutación en el módulo | Sin aclarar | A la capa no le importa cómo se permuta: solo exige una tabla con el mismo contrato. Permutar juntos todos los puntajes del dominio entre las personas de entrenamiento es válido | Pregunta de Audrey |

---

## 1. Reglas para el agente (leer antes de escribir código)

1. **El contrato está congelado.** No cambies nombres, firmas, columnas, tolerancias ni criterios. Si algo parece estar mal, no lo «arregles»: detenete y explicá el problema. Un cambio exige el acuerdo de Tomás, Audrey y Diego.
2. **Leé primero el código real del paquete.** Antes de escribir, abrí `anima_spine/predictions.py` (versión 2.2 o 2.2.1) y confirmá que existen, con estos nombres, `NIVELES`, `LINEAS_BASE`, `DOMINIOS`, `CAMPOS_INTERVALOS`, `CAMPOS_BARRIDO`, `ContratoError`, `exportar_intervalos`, `exportar_barrido`, `cobertura_promedio`, `verificar_intervalos` y `verificar_barrido`. Si alguno falta o tiene otra firma, detenete y avisá. No inventes un reemplazo.
3. **Solo tocás `anima_spine/calibracion.py` y sus tests.** `spine.py`, `predictions.py`, `compare.py` y `power_gradiente.py` son de Diego: no los modifiques. Tampoco cambies la versión del paquete ni `cohorte_anima.csv` (su huella SHA-256 es fija).
4. **No agregues `import anima_spine.calibracion` en `anima_spine/__init__.py`.** pandas y scipy se importan solo dentro de `calibracion.py`, y el resto del paquete usa solo numpy. Los módulos importan la capa así: `from anima_spine.calibracion import ...`.
5. **Fallá en voz alta.** Ante cualquier violación del contrato, levantá `ContratoError`. Nunca corrijas, imputes, recortes, redondees ni descartes filas en silencio.
6. **Sin azar y sin I/O.** La capa no usa generadores aleatorios, no lee ni escribe archivos y no imprime resultados.
7. **Los identificadores entre comillas invertidas son exactos.** No los traduzcas ni les cambies las mayúsculas.
8. **No ajustes tolerancias para que un test pase.** Si un test de cobertura falla, el problema está en el código o en los datos del test, no en el umbral.

---

## 2. Qué es la capa

Una librería Python que recibe, por rasgo, una predicción puntual `mu` y una escala de error `sigma`, y devuelve intervalos calibrados con **split conformal normalizado** en seis niveles: `NIVELES = (0.50, 0.60, 0.70, 0.80, 0.90, 0.95)`.

Cada módulo (Rostro, Voz, Temperamento) arma su tabla, llama a la capa en cada una de sus 20 réplicas y exporta los resultados con las funciones de `predictions`.

```
módulo (Rostro / Voz / Temperamento)
  └─ arma la tabla larga (mu, sigma, mu_base, sigma_base, y_real)
       └─ calibracion.py
            ├─ fit       ← solo filas de "calibracion"
            ├─ predict   ← solo filas de "prueba"  → intervalos
            ├─ contar    ← solo filas de "prueba"  → conteos de la réplica
            ├─ coverage_report     → diagnóstico (réplica 0)
            └─ control_permutacion → diagnóstico (réplica 0)
  └─ exporta con predictions.exportar_intervalos / exportar_barrido
```

**La capa NO:** ajusta modelos, genera la cohorte, construye líneas base, orquesta réplicas, calcula ΔR², promete cobertura por subgrupo ni calibra las reconstrucciones muestreadas.

---

## 3. Archivos

| Ruta | Acción |
| --- | --- |
| `anima_spine/calibracion.py` | Crear. Contiene toda la capa |
| Tests de la capa | Crear un archivo `test_calibracion.py` en la carpeta donde el paquete ya tiene sus tests, siguiendo su convención. Si no hay ninguna, usar `tests/test_calibracion.py` |
| Cualquier otro archivo del paquete | No tocar |

**API pública de `calibracion.py`:**
- `ConformalCalibrator`
- `coverage_report`
- `control_permutacion`

`ContratoError` **no** se define en este archivo: se importa de `predictions`.

```python
# anima_spine/calibracion.py (encabezado)
import math
import numpy as np
import pandas as pd
from scipy import stats

from anima_spine import predictions as P
from anima_spine.predictions import ContratoError   # subclase de ValueError
```

Se importa de `predictions` y no de `anima_spine` porque el `__init__.py` de la 2.2 todavía no la exporta. Cuando Diego la exporte, las dos rutas apuntan a la misma clase y este import sigue siendo válido. No modifiques `__init__.py` para agregarla.

---

## 4. Contrato de entrada

Una tabla larga **por dominio**, en un pandas DataFrame, con **una fila por `individuo_id` × `rasgo` × `linea_base`**. Puede traer filas de las tres particiones: las de `entrenamiento` se validan y se ignoran en todos los cálculos.

| Campo | Obligatorio | Tipo | Notas |
| --- | --- | --- | --- |
| `individuo_id` | sí | entero | dtype entero, no texto ni float. Ids de la cohorte 2.0, de 1 a 2.000 |
| `dominio` | sí | str | Uno de `P.DOMINIOS`: `"rostro"`, `"voz"`, `"temperamento"`. Un único valor en toda la tabla |
| `particion` | sí | str | `"entrenamiento"`, `"calibracion"` o `"prueba"` |
| `rasgo` | sí | str | Rostro `pc01`..`pc28`; Voz `f0_medio`, `f0_variabilidad`, `jitter`, `hnr`, `f1`, `f2`, `f3`, `vtl`; Temperamento `E`, `A`, `C`, `N`, `O`. **La capa no valida los nombres** |
| `y_real` | sí | float | Valor observado |
| `mu`, `sigma` | sí | float | Modelo con genotipo y covariables. `sigma > 0` |
| `linea_base` | sí | str | Uno de `P.LINEAS_BASE`: `"solo intercepto"` o `"ancestria sexo edad"` |
| `mu_base`, `sigma_base` | sí | float | Línea base de esa fila. `sigma_base > 0` |
| `mu_oraculo`, `sigma_oraculo` | no | float | Solo diagnóstico. **Nunca se usan en `fit`.** Vienen las dos o ninguna |
| `sexo` | no | int 0/1 | 0 = mujer, 1 = hombre. Solo diagnóstico |
| `edad` | no | numérico | Reservado. No se usa en esta versión |

Cualquier otra columna (por ejemplo `origen` o `muestra_1000g`) se ignora: no se valida ni se copia a la salida.

**Tipo de entrada.** `df` (y `df_perm` en `control_permutacion`) tiene que ser un `pandas.DataFrame`. Si no lo es, la capa levanta `ContratoError("df: se esperaba pandas.DataFrame, se recibió <tipo>")` antes de cualquier otro chequeo. La capa no convierte otros formatos: si un módulo tiene, por ejemplo, un dict de arreglos, lo convierte él antes de llamarla.

---

## 5. Validación: `_validar(df)`

Se ejecuta al comienzo de `fit`, `predict`, `contar`, `coverage_report` y `control_permutacion`, **sobre toda la tabla**, incluidas las filas de entrenamiento. Corré los chequeos en este orden, porque cada uno supone que los anteriores pasaron. Ante la primera falla, levantá `ContratoError`.

**Formato del mensaje:** `"<campo>: <cantidad> filas <descripción>"`, por ejemplo `"sigma: 3 filas con valor <= 0"`. En V1 se usa `"<columna>: columna obligatoria ausente"`.

| # | Chequeo | Implementación sugerida |
| --- | --- | --- |
| V1 | Están las columnas obligatorias | Diferencia de conjuntos con la lista de la sección 4 |
| V2 | `individuo_id` es de tipo entero | `pd.api.types.is_integer_dtype` y no `is_bool_dtype`. La cantidad informada es la de filas de la tabla |
| V3 | Hay un único `dominio` y está en `P.DOMINIOS` | Si hay más de un valor, informar las filas con valores distintos del más frecuente |
| V4 | `particion` es válida | `~isin({"entrenamiento", "calibracion", "prueba"})` |
| V5 | `linea_base` es válida | `~isin(P.LINEAS_BASE)` |
| V6 | Cada (`individuo_id`, `rasgo`) tiene las dos líneas base | Agrupar y contar valores únicos de `linea_base`. Si no son 2, contar las filas de esos grupos |
| V7 | La clave (`individuo_id`, `rasgo`, `linea_base`) es única | `duplicated(keep=False)` |
| V8 | Valores finitos en `y_real`, `mu`, `sigma`, `mu_base`, `sigma_base` y las columnas oráculo presentes | `~np.isfinite`, columna por columna. El campo es el nombre de la columna |
| V9 | `sigma > 0`, `sigma_base > 0` y `sigma_oraculo > 0` si está | Columna por columna |
| V10 | `mu`, `sigma`, `y_real` y `particion` (y `mu_oraculo`, `sigma_oraculo` y `sexo` si están) son **exactamente iguales** entre las dos líneas base | Pivotear por `linea_base` y comparar con `!=`, sin tolerancia |
| V11 | `sexo`, si está, toma solo los valores 0 y 1 | `~isin({0, 1})` |
| V12 | Las columnas oráculo vienen las dos o ninguna | Chequeo de presencia |
| V13 | `fit` tiene filas de `calibracion`; `predict` y `contar` tienen filas de `prueba` | Se chequea en cada método, después de `_validar` |
| V14 | Todo rasgo de `prueba` fue calibrado | En `predict`, comparar contra los rasgos de `qhat_` |
| V15 | Los niveles pedidos están en `P.NIVELES` | En `ConformalCalibrator.__init__` |

**La capa no valida** que los ids y las particiones coincidan con la cohorte: eso lo hacen `P.verificar`, `P.verificar_intervalos` y `P.verificar_barrido`.

---

## 6. Algoritmo

### 6.1 Cálculo de q̂ (RF1)

Para cada rasgo y nivel, y en el modo de las líneas base también para cada línea base:

```
s     = |y_real − m| / e        sobre las n personas de calibracion
         (m, e) = (mu, sigma)            en el modo del modelo
         (m, e) = (mu_base, sigma_base)  en el modo de las líneas base
k     = math.ceil((n + 1) * nivel)
si k > n:  ContratoError("nivel: ...")   # q̂ sería infinito; nunca devolver inf
q̂     = np.sort(s)[k − 1]
```

Con n = 500, los k exactos son **251, 301, 351, 401, 451 y 476** para los niveles 0,50 a 0,95. El test T1 los fija. `math.ceil((n + 1) * nivel)` no tiene problemas de punto flotante con n de 1 a 5.000 en los seis niveles.

### 6.2 Una persona cuenta una vez

- **Modo del modelo**, `columnas=("mu", "sigma")`: después de V10, usá solo las filas con `linea_base == "solo intercepto"`. Así n = 500, no 1.000. **Este es el error más importante que hay que evitar.**
- **Modo de las líneas base**, `columnas=("mu_base", "sigma_base")`: calibrá cada `linea_base` por separado, con n = 500 en cada una.
- Cualquier otro valor de `columnas` levanta `ContratoError("columnas: ...")`.

### 6.3 Intervalos (RF2)

- `lo = m − q̂ · e` y `hi = m + q̂ · e`.
- El valor de `nivel` que se escribe es el objeto float tomado de `P.NIVELES`, nunca uno recalculado.
- Los intervalos quedan anidados por construcción, porque k crece con el nivel. El test T4 lo verifica.

### 6.4 Conteo (RF3)

`dentro` cuenta las personas que cumplen `lo <= y_real <= hi`, **con los extremos incluidos**. Es la misma cuenta que usa el paquete.

---

## 7. Interfaz

### 7.1 `ConformalCalibrator(niveles=P.NIVELES)`

- Si `niveles` no es subconjunto de `P.NIVELES`, levanta `ContratoError` (V15). Guardá los niveles en el orden de `P.NIVELES`.
- Atributos después de `fit`:
  - `modo_`: `"modelo"` o `"base"`.
  - `qhat_`: DataFrame con las columnas `linea_base, rasgo, nivel, n, k, qhat`. En el modo del modelo `linea_base` es `None`; en el de las líneas base lleva el texto de la línea base.
  - Guardá también el `dominio`, para la salida.

### 7.2 `fit(df, columnas=("mu", "sigma")) -> self`

Valida, filtra `calibracion`, aplica la sección 6.2 y calcula q̂ (6.1). Devuelve `self`.

### 7.3 `predict(df) -> pd.DataFrame`

- Valida y filtra `prueba` (V13, V14).
- **Modo del modelo:** devuelve exactamente las columnas `list(P.CAMPOS_INTERVALOS)` = `individuo_id, dominio, rasgo, nivel, lo, hi`, con una fila por (`individuo_id`, `rasgo`, `nivel`), calculada sobre las filas de `"solo intercepto"`.
- **Modo de las líneas base:** las mismas columnas más `linea_base` al final, con una fila por (`individuo_id`, `rasgo`, `linea_base`, `nivel`). Estos intervalos son solo para el reporte y no se exportan.
- **Orden de las filas:** `individuo_id, rasgo, nivel`; en el modo de las líneas base, `individuo_id, rasgo, linea_base, nivel`. Usá `kind="mergesort"` y `reset_index(drop=True)`.

### 7.4 `contar(df, iv, replica, semilla) -> pd.DataFrame` (método)

- Requiere `modo_ == "modelo"`. Si `iv` trae la columna `linea_base`, levanta `ContratoError("iv: ...")`.
- Valida `df` y filtra `prueba`. Toma `y_real` de las filas de `"solo intercepto"`, una por (`individuo_id`, `rasgo`).
- Cruza `iv` con `y_real` por (`individuo_id`, `rasgo`). Si alguna persona de `prueba` no tiene intervalo para alguno de los niveles del calibrador, o si `iv` trae filas que no están en `prueba`, levanta `ContratoError("iv: <cantidad> filas ...")`. No descartes filas en silencio.
- Devuelve exactamente `list(P.CAMPOS_BARRIDO)` = `replica, semilla, dominio, rasgo, nivel, n, dentro`, con una fila por (`rasgo`, `nivel`), ordenada por `rasgo, nivel`. `n`, `dentro`, `replica` y `semilla` son `int`.
- `semilla` es una etiqueta y no cambia ningún cálculo.

### 7.5 `coverage_report(df, cal, cal_base) -> dict[str, pd.DataFrame]`

- `cal` tiene que estar en el modo del modelo y `cal_base` en el de las líneas base. Si no, levanta `ContratoError`.
- Llama a `cal.predict(df)` y `cal_base.predict(df)`. Es un diagnóstico de la réplica 0 y no se exporta.

| Clave | Una fila por | Columnas, en este orden |
| --- | --- | --- |
| `"cobertura"` | rasgo × nivel | `rasgo, nivel, n, dentro, cobertura, ic_inf, ic_sup, qhat, z, qhat_sobre_z, ancho_medio, ancho_solo_intercepto, ancho_ancestria_sexo_edad, ancho_oraculo` |
| `"crps"` | rasgo | `rasgo, crps_modelo, crps_solo_intercepto, crps_ancestria_sexo_edad, crps_oraculo` |
| `"cobertura_por_sexo"` | rasgo × nivel × sexo | `rasgo, nivel, sexo, n, dentro, cobertura`. Solo si `df` trae `sexo` |

Fórmulas:

```python
cobertura = dentro / n                       # mismo dentro que contar (T11)
ic_inf = 0.0 if dentro == 0 else stats.beta.ppf(0.025, dentro, n - dentro + 1)
ic_sup = 1.0 if dentro == n else stats.beta.ppf(0.975, dentro + 1, n - dentro)
z = stats.norm.ppf((1 + nivel) / 2)
qhat_sobre_z = qhat / z
ancho_* = mean(hi - lo)                      # modelo y cada línea base
ancho_oraculo = mean(2 * z * sigma_oraculo)  # NaN si no hay columnas oráculo

def crps_normal(y, mu, sigma):               # promedio sobre las personas de prueba
    w = (y - mu) / sigma
    return np.mean(sigma * (w * (2 * stats.norm.cdf(w) - 1)
                            + 2 * stats.norm.pdf(w) - 1 / np.sqrt(np.pi)))
# crps_modelo: (mu, sigma); crps_<base>: (mu_base, sigma_base) de esa línea base;
# crps_oraculo: (mu_oraculo, sigma_oraculo), o NaN si no hay columnas oráculo
```

`ancho_oraculo` y `crps_oraculo` son los **únicos** NaN permitidos en la salida.

**Qué es y qué no es el CRPS.** La cobertura se mide sobre los intervalos conformales `mu ± q̂·sigma`. El CRPS se calcula sobre la normal paramétrica N(`mu`, `sigma`), que es otra distribución predictiva y no depende de la calibración. El CRPS se reporta como **diagnóstico, sin umbral**, y **no forma parte de la condición de la hipótesis** (decisión del grupo del 12/11/2026). No lo uses como criterio de aprobación dentro de la capa ni en sus tests.

### 7.6 `control_permutacion(df_perm, cal_base) -> dict[str, pd.DataFrame]` (cambio v5.1)

- `df_perm` cumple el mismo contrato, con `mu` y `sigma` de un modelo ajustado con el puntaje poligénico permutado. Sus columnas de línea base son las originales: lo garantiza el módulo.
- A la capa no le importa cómo se hizo la permutación. Por ejemplo, permutar juntos todos los puntajes del dominio entre las personas de entrenamiento es válido: Rostro permuta sus 28 y Temperamento sus 5.
- La función ajusta un calibrador nuevo en el modo del modelo sobre `df_perm` y predice `prueba`.
- `ancho_base` sale de `cal_base.predict(df_perm)`, solo de `linea_base == "ancestria sexo edad"`.
- Devuelve un diccionario con dos claves:

| Clave | Una fila por | Columnas, en este orden |
| --- | --- | --- |
| `"por_rasgo"` | rasgo × nivel | `rasgo, nivel, ancho_permutado, ancho_base, cociente`. **Sin columna `pasa`** |
| `"por_nivel"` | nivel | `nivel, n_rasgos, cociente_medio, pasa` |

```python
cociente = ancho_permutado / ancho_base                  # por rasgo y nivel
cociente_medio = mean(cociente sobre los rasgos)         # promedio simple de los cocientes, por nivel
pasa = 0.95 <= cociente_medio <= 1.05                    # 6 banderas, una por nivel
```

- `cociente_medio` es el **promedio simple** de los cocientes de todos los rasgos del nivel (en Rostro, las 28 componentes), con el mismo peso para cada uno. No es el cociente entre los anchos promedio. `n_rasgos` es la cantidad de rasgos promediados.
- `"por_rasgo"` se devuelve siempre completa y va al informe, aunque no lleve bandera.

- **Por qué la bandera va sobre el promedio:** el control de permutación es una afirmación sobre el dominio, no sobre un rasgo suelto. Con la bandera por celda, un modelo correcto marca al menos un «no pasa» en 2 de cada 5 corridas de Voz (48 celdas). Sobre el promedio de rasgos, el desvío baja a alrededor de 0,008 y la ventana de ±0,05 queda a más de seis desvíos.
- El cociente se centra en alrededor de 0,997 y no en 1, porque los dos modelos estiman distinta cantidad de parámetros. Es esperable, no un error, y no se corrige.
- Es un diagnóstico para el informe, no un test de la capa.

### 7.7 Uso desde un módulo

```python
import pandas as pd
from anima_spine import predictions as P
from anima_spine.calibracion import ConformalCalibrator, coverage_report, control_permutacion

bars = []
for r in range(P.REPLICAS):                    # 20 réplicas, las orquesta el módulo
    s = 20261019 + r
    df = construir_tabla(semilla=s)            # responsabilidad del módulo (ver MODEL.md)
    cal = ConformalCalibrator(niveles=P.NIVELES).fit(df, columnas=("mu", "sigma"))
    iv = cal.predict(df)
    bars.append(cal.contar(df, iv, replica=r, semilla=s))
    if r == 0:
        cal_base = ConformalCalibrator(niveles=P.NIVELES).fit(df, columnas=("mu_base", "sigma_base"))
        rep = coverage_report(df, cal, cal_base)
        ctl = control_permutacion(construir_tabla_permutada(semilla=s), cal_base)
        P.exportar_intervalos(iv.to_dict("records"), dominio="temperamento")

barrido = pd.concat(bars, ignore_index=True)
P.exportar_barrido(barrido.to_dict("records"), dominio="temperamento")
cobertura = P.cobertura_promedio(barrido.to_dict("records"))   # cobertura de registro
```

- El `lo`/`hi` al 90 % de `predicciones_<dominio>.csv` sale de `iv[iv.nivel == 0.90]`, nunca de un cálculo aparte.
- El parámetro se llama siempre `nivel`. El nombre `alfa` no aparece en la capa.

---

## 8. Cobertura de registro

La cobertura que va al informe y se evalúa contra el criterio de ±5 pp es `P.cobertura_promedio(barrido)`, sobre las 20 réplicas. La de `coverage_report` es solo de la réplica 0 y es un diagnóstico. Las dos **no** tienen por qué coincidir. Lo que sí tiene que coincidir exactamente es la cobertura de `coverage_report` con `dentro / n` del barrido de la réplica 0 (test T11).

Hay dos tolerancias distintas que no hay que confundir:
- **4 desvíos** (sección 9.2): es la de los tests internos de la capa.
- **±5 pp:** es el criterio científico de cada módulo, que va al informe y no se evalúa en pytest.

---

## 9. Tests

### 9.1 Datos sintéticos (fixture)

Los tests generan sus propios datos con una semilla fija. **No dependen de archivos externos.**

```python
def tabla_sintetica(semilla, n_cal=500, n_test=500, n_ent=10,
                    rasgos=("r1", "r2", "r3", "r4", "r5", "r6", "r7", "r8"),
                    factor_sigma=None, dominio="temperamento"):
    rng = np.random.default_rng(semilla)
    # ids enteros 1..N; particiones: entrenamiento, calibracion, prueba
    # sexo 50/50; por rasgo: mu ~ N(0,1), sigma = exp(N(0, 0.3)), y_real = mu + sigma*Z
    # factor_sigma: función opcional (sexo -> factor) que multiplica la sigma REPORTADA
    #               (no la verdadera), para T3a y T3b
    # se duplica cada fila para las dos líneas base, con mu_base = mu y sigma_base = sigma
    # devuelve un DataFrame con todas las columnas obligatorias más sexo
```

**Réplicas:** semillas `20261019 + r`, con r de 0 a 19.

### 9.2 Tolerancia de cobertura

```python
def tolerancia(p, n_cal=500, n_test=500, R=20):
    return 4 * math.sqrt(p * (1 - p) * (1 / n_cal + 1 / n_test) / R)
# 0.50→0.0283, 0.60→0.0277, 0.70→0.0259, 0.80→0.0226, 0.90→0.0170, 0.95→0.0123
```

### 9.3 Lista de tests

Cada test lleva su id en el nombre, por ejemplo `test_T07_n_efectivo`.

| Id | Verifica | Criterio |
| --- | --- | --- |
| T1 | Caso a mano | Con 9 personas de calibración y scores conocidos, `niveles=(0.80,)` da `k == 8` y `qhat ==` el octavo score. `niveles=(0.95,)` levanta `ContratoError` en `fit`. Con n = 500, `qhat_.k` vale 251, 301, 351, 401, 451 y 476 |
| T2 | Modelo perfecto | En 20 réplicas sintéticas, `abs(cobertura_promedio − nivel) <= tolerancia(nivel)` en cada rasgo y nivel. El promedio se calcula con `P.cobertura_promedio` |
| T3a | Invariancia ante la escala | Con la sigma reportada × 0,7 y × 1,5 en todas las filas, `lo` y `hi` son iguales a los originales con `np.allclose(rtol=1e-9, atol=0)` |
| T3b | Error de sigma por sexo | Con × 0,7 si `sexo == 0` y × 1,5 si `sexo == 1`: la cobertura total cumple T2; en el nivel 0,90, la cobertura promedio de 20 réplicas es ≤ 0,85 para `sexo == 0` y ≥ 0,95 para `sexo == 1` (se esperan 0,805 y 0,995) |
| T4 | Anidamiento | Para cada (`individuo_id`, `rasgo`), `lo` no sube y `hi` no baja al aumentar el nivel |
| T5 | Rechazo de violaciones | Un caso por cada chequeo V1 a V15. Cada uno levanta `ContratoError`, y el mensaje contiene el campo y la cantidad exacta de filas afectadas |
| T6 | Determinismo | Dos corridas dan `qhat_`, `predict`, `contar` y `coverage_report` iguales con `DataFrame.equals`. `contar` con otra `semilla` da los mismos `n` y `dentro` |
| T7 | n efectivo | En el modo del modelo, `qhat_.n == 500` en todas las filas. En el modo de las líneas base, `n == 500` en cada `linea_base` |
| T8 | Sin efectos laterales | El `df` de entrada queda igual a una copia previa (`DataFrame.equals`) después de llamar a todos los métodos |
| T9 | Columnas de salida | `list(predict(df).columns) == list(P.CAMPOS_INTERVALOS)` y `list(contar(...).columns) == list(P.CAMPOS_BARRIDO)`. Con los ids reales de la cohorte, la salida pasa `P.verificar_intervalos` y `P.verificar_barrido`. Si la cohorte no está disponible, esta parte se marca como `skip` y no como error |
| T10 | Filas de entrenamiento | Agregar o quitar filas de `entrenamiento` no cambia `qhat_` ni `predict` |
| T11 | Cruce con el paquete | `coverage_report(...)["cobertura"].cobertura` es exactamente igual a `dentro / n` del barrido de la réplica 0, y `P.cobertura_promedio` sobre ese barrido de una sola réplica da el mismo valor |
| T12 | Error compartido | Los errores de la capa son instancias de `anima_spine.predictions.ContratoError` (la misma clase que levantan los validadores del paquete) y también se pueden capturar como `ValueError`. Si `anima_spine` exporta `ContratoError`, verificar además que `anima_spine.ContratoError is anima_spine.predictions.ContratoError` |
| T13 | Tipo de entrada (v5.1) | Pasar un dict o una lista en lugar de un DataFrame levanta `ContratoError` con el campo `df`, en `fit`, `predict`, `contar`, `coverage_report` y `control_permutacion` |
| T14 | Forma de `control_permutacion` (v5.1) | Con datos sintéticos, devuelve las claves `"por_rasgo"` y `"por_nivel"` con las columnas exactas de la sección 7.6. `"por_rasgo"` no tiene `pasa`, `"por_nivel"` tiene una fila por nivel, y `cociente_medio` es el promedio simple de `cociente` |

**Antes de escribir T9 y T11**, leé las firmas reales de `verificar_intervalos`, `verificar_barrido` y `cobertura_promedio` en `predictions.py`. Según Diego, `verificar_barrido(barrido, intervalos, predicciones, ids, dominio)` y `cobertura_promedio(barrido)` devuelve `{rasgo: {nivel: promedio}}`. Si no coinciden, detenete y avisá.

### 9.4 Datos oráculo de Voz (validación adicional, opcional en pytest)

El archivo de 20 réplicas sobre la cohorte 2.0 llega el 13/11. **El archivo anterior, de la cohorte 1.1, está obsoleto: no lo uses.** El nuevo no trae líneas base. Antes de pasarlo a la capa, adaptalo réplica por réplica:

1. Duplicá cada fila: una con `linea_base = "solo intercepto"` y otra con `"ancestria sexo edad"`.
2. Asigná `mu = mu_base = mu_oraculo` y `sigma = sigma_base = sigma_oraculo`.
3. Conservá `mu_oraculo`, `sigma_oraculo` y `sexo`.

Con esos datos corré los criterios de T2 y T3. La adaptación vive en los tests, nunca en la capa.

---

## 10. Plan de trabajo

**Fase v0 (prioridad máxima)**

1. Leer `predictions.py` 2.2 o 2.2.1 y verificar los nombres de la regla 2 de la sección 1.
2. Escribir la fixture `tabla_sintetica` y T1.
3. Implementar el chequeo de tipo de entrada y `_validar` (V1 a V15), con T5, T12 y T13.
4. Implementar `ConformalCalibrator.__init__`, `fit` y `qhat_`, con T1 y T7.
5. Implementar `predict`, con T4, T9 (columnas) y T10.
6. Implementar `contar`, con T9 (columnas) y T6.
7. Implementar T2 y T8.

La v0 está terminada cuando pasan T1, T2, T4, T5, T6, T7, T8, T9, T10, T12 y T13.

**Fase v1**

8. `coverage_report`, con T11.
9. T3a y T3b.
10. `control_permutacion`, con la salida v5.1 de la sección 7.6 y T14.
11. T9 completo contra la cohorte real.

La v1 está terminada cuando pasan todos los tests y también los 26 tests existentes del paquete. Con la v1, la capa entra en `anima_spine` 2.3, que publica Diego.

---

## 11. Lo que no se hace

- Conformalized quantile regression, regiones multivariadas, Mondrian por subgrupo, conformal adaptativo, jackknife+.
- El diagnóstico por tramo de edad.
- Endpoints de API, lectura o escritura de archivos y logs con resultados.
- Desempates al azar, semillas dentro de la capa.
- Calcular ΔR², CRPS como criterio de aprobación o promedios entre réplicas dentro de la capa.
- Lógica específica de un dominio dentro de `calibracion.py`: la capa no sabe qué es un rasgo de Temperamento, de Voz ni de Rostro.

---

## 12. Modelo de Temperamento: `./MODEL.md`

Las especificaciones del modelo de Temperamento (señal simulada, h², predictor, sigma, líneas base, permutación y el margen δ) están en **[`./MODEL.md`](./MODEL.md)**. Ese archivo es de Tomás.

**Cuándo leerlo:**
- **Si tu tarea es solo construir `calibracion.py`:** no lo necesitás. La capa no depende de ningún dominio, y no debés copiar lógica de Temperamento en ella.
- **Si tu tarea es conectar Temperamento con la capa** (armar la tabla, correr las 20 réplicas, exportar): leé `MODEL.md` para construir la tabla, y este archivo para todo lo que toca la capa.

**Lo que la salida del modelo de Temperamento DEBE cumplir para entrar a la capa**, diga lo que diga `MODEL.md`:

| Requisito | Valor |
| --- | --- |
| `dominio` | `"temperamento"` |
| `rasgo` | `E`, `A`, `C`, `N`, `O` (exactos) |
| Sigma | Constante por rasgo: el desvío residual sobre `entrenamiento` (decisión D4) |
| Líneas base | Las dos, `"solo intercepto"` y `"ancestria sexo edad"`, sin imponer efectos de sexo, edad ni ancestría (D5). Un R² fuera de muestra levemente negativo en la segunda es esperable y no se corrige |
| Invariantes del módulo | 1. `mu` y `sigma` ajustados solo con `entrenamiento`. 2. `sigma` no usa el `y_real` de la fila que predice. 3. `mu_base` y `sigma_base` sobre las mismas filas y partición. 4. Sin NaN ni infinitos. 5. Ids de la cohorte 2.0. 6. Una fila por (`individuo_id`, `rasgo`, `linea_base`) |
| Control de permutación | Una tabla con el mismo contrato, con `mu` y `sigma` del puntaje poligénico permutado y las columnas de línea base originales |
| Réplicas | 20, con semillas `20261019 + r`, sobre la misma cohorte y las mismas particiones |

Si `MODEL.md` contradice esta tabla o cualquier otra parte de la interfaz con la capa, manda este archivo y el contrato v5. Detenete y avisá a Tomás. Lo que `MODEL.md` decida sobre la señal, el h² o el δ queda fuera del contrato y no cambia la capa.
