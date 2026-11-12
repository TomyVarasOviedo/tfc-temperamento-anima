# Anima: capa de calibración conformal

**Requisitos y contrato · v5.1 · CONGELADO**

Autor: Tomás (responsable de `anima_spine/calibracion.py`) · Para: Audrey, Diego · Fecha: 12/11/2026
Reemplaza a: v5 del 10/11/2026 (y antes, v4 del 9/11 y v3 del 3/11).
Confirmado por: Tomás, Audrey y Diego. La v5 el 9/11 y los cambios de la v5.1 el 12/11. Ver sección 11.

> **Para quien implemente o use este documento, sea una persona o un agente:**
>
> - Este contrato está congelado. Un cambio se hace solo con acuerdo de Tomás, Audrey y Diego, y sube la versión del documento.
> - Las palabras **DEBE**, **NO DEBE** y **PUEDE** son obligatorias en ese sentido estricto.
> - No queda ningún punto pendiente de confirmación. Los únicos datos que todavía no llegaron (las 20 réplicas oráculo de Voz) no bloquean nada: la sección 7.3 dice qué se usa mientras tanto.
> - Los nombres entre comillas invertidas son identificadores exactos del código. No se traducen, no se cambian de mayúsculas y no se renombran.
> - Si este documento y el código de `anima_spine` no coinciden en algo que no esté marcado como pendiente, se detiene el trabajo y se avisa a los tres. No se resuelve en silencio para ninguno de los dos lados.

---

## 0. Resumen en una pantalla

- La capa es una librería Python, `anima_spine/calibracion.py`, dentro del paquete común `anima_spine` **2.3**.
- Calibra con **split conformal normalizado**, por rasgo y nivel, en los seis niveles de `predictions.NIVELES`.
- Recibe **una tabla larga por dominio, en memoria** (pandas DataFrame). No lee ni escribe archivos.
- `fit` usa solo `particion == "calibracion"`, `predict` y `contar` usan solo `particion == "prueba"`, y las filas de `entrenamiento` se validan y se ignoran.
- Con dos líneas base, cada persona cuenta **una vez** al calibrar el modelo: **n = 500, no 1.000**.
- Es determinista. No hay azar.
- Si encuentra un error, lo levanta. Nunca corrige ni imputa en silencio.

---

## 1. Propósito y alcance

La capa recibe, por rasgo, una predicción puntual (`mu`) y una escala de error (`sigma`). Devuelve intervalos con cobertura empírica verificable en seis niveles, de 0,50 a 0,95. Cada módulo (Rostro, Voz, Temperamento) la importa y la llama.

**Mantenimiento**

| Archivo | Responsable |
| --- | --- |
| `anima_spine/calibracion.py` | Tomás |
| `spine.py`, `predictions.py`, `compare.py`, `power_gradiente.py` | Diego |
| Cambios en el contrato (`predictions.py` y este documento) | Acuerdo de los tres |

**Versión.** La capa entra en el paquete **2.3**. Las versiones 2.1 (5/11) y 2.2 (9/11) ya salieron sin la capa: la 2.1 trae el ΔR² parcial y la exigencia de las dos líneas base en `predictions.verificar`, y la 2.2 trae `ContratoError`, `TOLERANCIA_RECUPERACION` y la corrección del comentario de los rasgos. La capa se desarrolla sobre la **2.2 o la 2.2.1** (la 2.2.1 solo corrige `power_gradiente.py` y no toca nada que use la capa). La cohorte (`cohorte_anima.csv`, versión 2.0) y su huella SHA-256 no cambian.

**Dependencias.** El resto del paquete usa solo numpy. `calibracion.py` PUEDE importar pandas y scipy, pero DEBE hacerlo **solo dentro de ese archivo**. Ningún otro archivo del paquete DEBE importarlas por causa de la capa.

### 1.1 Qué hace

- Calibra con split conformal normalizado, por rasgo y nivel (RF1).
- Devuelve intervalos y conteos de cobertura con las columnas de `anima_spine` (RF2, RF3).
- Reporta diagnósticos que no dependen solo de la cobertura: q̂ contra z, ancho, CRPS y la comparación con las líneas base y el oráculo (RF4, RF5).
- Evalúa el control de permutación por ancho (RF6).

### 1.2 Qué NO hace

- No ajusta los modelos de cada dominio, no genera la cohorte y no construye las líneas base.
- No orquesta las 20 réplicas: cada módulo las corre y llama a la capa en cada una.
- No calcula ni verifica ΔR². Eso vive en `compare.py` y en la infraestructura del OE5.
- No promete cobertura condicional por subgrupo.
- No calibra las reconstrucciones muestreadas del OE6.
- No hace I/O: no lee ni escribe archivos ni imprime resultados como salida.

---

## 2. Cambios

### 2.0 De la v5 a la v5.1 (12/11/2026)

| Tema | v5 | v5.1 | Motivo y origen |
| --- | --- | --- | --- |
| RF6, `control_permutacion` | `pasa` por rasgo × nivel (48 banderas en Voz), con ventana [0,95; 1,05] | `pasa` sobre el **cociente medio por nivel** (6 banderas), con la misma ventana. El cociente medio es el promedio simple de los cocientes de los rasgos de ese nivel (en Rostro, de las componentes). La tabla `por_rasgo` se sigue informando completa, sin bandera. Ver RF6 y 6.7 | Simulación de Diego (10/11), 1.500 corridas con datos parecidos a Voz: por celda la ventana falla entre el 0,4 % y el 2,3 % de las veces, pero las 48 celdas pasan todas juntas solo en el 61 % de las corridas con un modelo correcto. Sobre el promedio de rasgos el desvío baja a unos 0,008 y la ventana queda a más de seis desvíos. Además, el control de permutación es una afirmación sobre el dominio, no sobre un rasgo suelto. Propuso Diego, aceptaron Tomás y Audrey (12/11) |
| CRPS | «Es la que usa la condición de la hipótesis» | Se reporta **como diagnóstico, sin umbral**, y **no forma parte de la condición de la hipótesis**. Ver RF5 y 6.6 | Decisión del grupo del **12/11/2026**, tomada antes de que la capa calcule ningún CRPS. La cobertura se mide sobre los intervalos conformales y el CRPS sobre la normal paramétrica, que son distribuciones predictivas distintas; y lo que el CRPS agregaba (descartar un modelo inútil que igual cubre) ya lo resuelve el criterio de recuperación del ΔR² |
| Importación de `ContratoError` | `from anima_spine import ContratoError` | `from anima_spine.predictions import ContratoError`, porque el `__init__.py` de la 2.2 todavía no la exporta. Es la misma clase | Aviso de Audrey (10/11) |
| Tipo de entrada | Implícito | Si la tabla no es un `pandas.DataFrame`, la capa levanta `ContratoError`. La conversión la hace cada módulo | Nota de Audrey (10/11) |
| Paquete de base | 2.2 | 2.2 o 2.2.1 | Diego (10/11) |
| Tests | T1 a T12 | Se agregan T13 (tipo de entrada) y T14 (forma de `control_permutacion`) | Por los cambios anteriores |

### 2.1 De la v4 a la v5 (respuesta de Diego del 9/11)

| Tema | v4 | v5 |
| --- | --- | --- |
| Versión del paquete | 2.1 | **2.3**, porque la 2.1 y la 2.2 ya salieron |
| Función que exporta el barrido | A confirmar | `predictions.exportar_barrido(filas, dominio, directorio=".")`, que ya existe |
| `ContratoError` | Definida en `calibracion.py` | Definida en `predictions.py` (paquete 2.2) y exportada por `anima_spine`. La capa la importa y no la redefine |
| Cobertura de registro | Sin definir | La del criterio es `predictions.cobertura_promedio(barrido)`, sobre 20 réplicas. `coverage_report` es diagnóstico de la réplica 0. Se agrega el test T11, que cruza las dos (sección 7.4) |
| Datos oráculo de Voz | Sin fecha | 20 réplicas sobre la cohorte 2.0 para el 13/11, y cómo adaptarlas al contrato (sección 7.3). El archivo de la cohorte 1.1 queda obsoleto |
| Comentario `PC3` | A corregir | Corregido en la 2.2. El paquete no fija el formato del nombre del rasgo. Este contrato fija los de la sección 4 |
| Tolerancia de recuperación del ΔR² | Citada como ±0,005 | Fuera del contrato. Es `TOLERANCIA_RECUPERACION` del paquete (±0,015 desde la 2.2) |

### 2.2 De la v3 a la v4

| Tema | v3 | v4 | Origen |
| --- | --- | --- | --- |
| Test 2 | Cobertura promedio a ≤ 1 pp del nivel | Tolerancia por nivel de 4 desvíos esperados (sección 7.2). Con 1 pp, un modelo perfecto fallaba en el 92 % de las corridas | Simulación, aceptado por Tomás |
| Test 3 | Sigma × 0,7 y × 1,5 «empeora el ancho» | Dividido en 3a (invariancia: intervalos idénticos) y 3b (error distinto por sexo: la cobertura por sexo se separa). Escalar sigma por una constante deja los intervalos idénticos, así que el ancho no puede empeorar | Simulación, aceptado por Tomás |
| `coverage_report` | `coverage_report(df, iv)` | `coverage_report(df, cal, cal_base)`, porque necesita los q̂ y las líneas base | Aceptado por Tomás |
| CRPS (RF5) | Sin definir | CRPS de la normal N(`mu`, `sigma`), con fórmula exacta (sección 6.6) | Aceptado por Tomás |
| RF6 | Sin función ni tolerancia | `control_permutacion(df_perm, cal_base)`, que pasa si el cociente de anchos queda entre 0,95 y 1,05 | Aceptado por Tomás |
| `dentro` | Sin definir | `lo <= y_real <= hi`, con los extremos incluidos | Aceptado por Tomás |
| Intervalo binomial (RF4) | Sin definir | Clopper-Pearson al 95 % | Aceptado por Tomás |
| Filas de `entrenamiento` | Sin definir | Se aceptan, se validan y se ignoran en los cálculos | Nota de Audrey del 9/11 |
| Rasgos de Rostro | `pc01..pc28` | Confirmado `pc01..pc28`. El comentario `PC3` de `predictions.py` se corrige | Audrey |
| Sección 9, puntos de Audrey | Pendientes | Confirmados (sección 11) | Audrey, 9/11 |
| Responsable y cronograma | Punto de control del 8/11 | La capa sigue con Tomás, que la empieza el 9/11 | Tomás |
| Datos de los tests 2 y 3 | Solo datos oráculo de Voz | Datos sintéticos con semilla fija en pytest. Los datos oráculo de Voz son una validación adicional | Para no bloquear la implementación |

---

## 3. Requisitos funcionales

**RF1. Calibración.**
- El score de cada fila de calibración es `s = |y_real − mu| / sigma`.
- Para cada rasgo y cada nivel, sea n la cantidad de personas de `calibracion` de ese rasgo, contada **una vez por persona** (sección 6.2).
- El índice es `k = math.ceil((n + 1) * nivel)`.
- q̂ es el k-ésimo score ordenado de menor a mayor, empezando a contar en 1: `np.sort(s)[k - 1]`.
- Si `k > n`, la capa DEBE levantar `ContratoError` (q̂ sería infinito). Nunca devuelve límites infinitos.
- Con n = 500, los valores de k DEBEN ser: 0,50 → 251; 0,60 → 301; 0,70 → 351; 0,80 → 401; 0,90 → 451; 0,95 → 476.
- Se verificó que `math.ceil((n + 1) * nivel)` no tiene problemas de punto flotante para n de 1 a 5.000 con los seis niveles. Un test fija estos valores (T1).

**RF2. Intervalos.**
- `lo = mu − q̂ · sigma` y `hi = mu + q̂ · sigma`, para los seis niveles de `predictions.NIVELES` = `(0.50, 0.60, 0.70, 0.80, 0.90, 0.95)`.
- Los intervalos DEBEN quedar anidados: con un nivel mayor, `lo` no sube y `hi` no baja.
- El valor de `nivel` que se escribe DEBE ser el objeto float tomado de `NIVELES`, nunca uno recalculado.

**RF3. Conteos por réplica.** `contar` devuelve `n` y `dentro` por rasgo y nivel. El promedio de las 20 réplicas lo calcula `predictions.cobertura_promedio`, no la capa.

**Cobertura de registro.** La cobertura que se evalúa contra el criterio de ±5 pp y que va al informe como resultado es **`predictions.cobertura_promedio(barrido)`**, sobre las 20 réplicas. La cobertura de `coverage_report` es de una sola réplica (la 0) y es un diagnóstico: NO DEBE usarse para el criterio. Las dos no pueden coincidir, porque una promedia 20 réplicas y la otra mira una. Lo que sí DEBE coincidir exactamente es la cobertura de `coverage_report` con `dentro / n` del barrido filtrado en `replica == 0` (test T11).

**RF4. Diagnóstico de la réplica 0.** Por rasgo y nivel, el reporte DEBE incluir:
- la cobertura;
- su intervalo binomial Clopper-Pearson al 95 %;
- q̂;
- el cuantil gaussiano nominal `z = scipy.stats.norm.ppf((1 + nivel) / 2)`;
- el cociente q̂ / z. Si sigma está bien especificada, q̂ / z ≈ 1. El desvío es un diagnóstico, no un criterio de aprobación.

**RF5. Ancho y CRPS.**
- El reporte DEBE incluir el ancho medio del intervalo (`hi − lo`) del modelo y de cada línea base, y el CRPS medio del modelo y de cada línea base (sección 6.6).
- Si la tabla trae `mu_oraculo` y `sigma_oraculo`, DEBE incluir también el ancho del intervalo oráculo, `mu_oraculo ± z · sigma_oraculo`, sin calibrar, y su CRPS.
- `mu_oraculo` y `sigma_oraculo` NO DEBEN usarse nunca en `fit`.
- El CRPS es un **diagnóstico sin umbral**. No forma parte de la condición de la hipótesis (decisión del grupo del 12/11/2026) y NO DEBE usarse como criterio de aprobación en la capa ni en sus tests.

**RF6. Control de permutación (cambiado en v5.1).**
- El módulo produce una tabla con el puntaje poligénico permutado, con el mismo contrato. A la capa no le importa cómo se permuta: permutar juntos todos los puntajes del dominio entre las personas de entrenamiento es válido.
- `control_permutacion` calibra esa tabla y compara su ancho con el de la línea base `"ancestria sexo edad"`, por rasgo y nivel (sección 6.7).
- La bandera `pasa` se evalúa sobre el **cociente medio por nivel**: el promedio simple de los cocientes de todos los rasgos del dominio en ese nivel (en Rostro, de sus 28 componentes), con el mismo peso para cada uno. Son 6 banderas, una por nivel.
- Los cocientes por rasgo y nivel DEBEN seguir informándose completos, sin bandera.
- El ΔR² ≈ 0 de esa corrida lo verifica cada módulo, no la capa.

**RF7. Forma de sigma.** La capa DEBE aceptar tanto sigma constante por rasgo como sigma por individuo. No hay ningún modo distinto: la fórmula de RF1 cubre los dos casos.

**RF8. Validación.**
- La capa DEBE validar la entrada según la sección 5.3.
- Ante una violación DEBE levantar `ContratoError` con el campo y la cantidad de filas afectadas.
- NO DEBE corregir, imputar, recortar ni descartar filas en silencio.

**RF9. Diagnóstico por subgrupo.**
- Cobertura por `sexo` (0 = mujer, 1 = hombre), sin garantía condicional.
- Prioridad baja: el diagnóstico por tramo de edad queda para después de la v1 y no forma parte de esta versión.
- Si la tabla trae la columna `sexo`, `coverage_report` DEBE incluir la cobertura por sexo. Si no la trae, la omite sin error.

**RF10. Determinismo.**
- Con los mismos datos, q̂, los intervalos, los conteos y el reporte DEBEN salir idénticos bit a bit.
- No hay ningún desempate al azar ni ningún generador aleatorio en la capa.
- La `semilla` que recibe `contar` solo se escribe en la columna `semilla`.

**RF11. Sin efectos laterales.**
- La capa NO DEBE modificar el DataFrame de entrada: trabaja sobre copias o vistas de solo lectura.
- La capa NO DEBE hacer I/O.

---

## 4. Rasgos por dominio

| `dominio` | `rasgo` (identificadores exactos) | Cantidad |
| --- | --- | --- |
| `"rostro"` | `pc01`, `pc02`, …, `pc28` (dos dígitos, con cero a la izquierda) | 28 |
| `"voz"` | `f0_medio`, `f0_variabilidad`, `jitter`, `hnr`, `f1`, `f2`, `f3`, `vtl` | 8 |
| `"temperamento"` | `E`, `A`, `C`, `N`, `O` | 5 |

Total: 41 rasgos. La capa **no valida** que los nombres sean estos: es responsabilidad de cada módulo, y DEBEN ser iguales en predicciones, intervalos y barrido. La capa calibra los rasgos que encuentra.

---

## 5. Contrato de entrada

### 5.1 Forma

- Una tabla en formato largo **por dominio**, en memoria (pandas DataFrame), con **una fila por `individuo_id` × `rasgo` × `linea_base`**.
- PUEDE incluir filas de `entrenamiento`, `calibracion` y `prueba`. Las de `entrenamiento` se validan y se ignoran en todos los cálculos.
- No es el `predicciones_<dominio>.csv` exportado, que trae solo `prueba`.
- DEBE ser un `pandas.DataFrame`. Si no lo es, la capa levanta `ContratoError("df: se esperaba pandas.DataFrame, se recibió <tipo>")` antes de cualquier otro chequeo. La capa no convierte otros formatos: si un módulo tiene, por ejemplo, un dict de arreglos, lo convierte él.
- Las columnas que no figuran en 5.2 (por ejemplo `origen` y `muestra_1000g` de la cohorte 2.0) se ignoran: no se validan ni se copian a la salida.

### 5.2 Columnas

| Campo | Obligatorio | Tipo | Contenido |
| --- | --- | --- | --- |
| `individuo_id` | sí | entero (dtype entero de numpy o pandas, no texto ni float) | Id de `cohorte_anima.csv` 2.0, de 1 a 2.000 |
| `dominio` | sí | str | Exactamente uno de `predictions.DOMINIOS`: `"rostro"`, `"voz"`, `"temperamento"`. Un solo valor en toda la tabla |
| `particion` | sí | str | `"entrenamiento"`, `"calibracion"` o `"prueba"` |
| `rasgo` | sí | str | Ver sección 4 |
| `y_real` | sí | float | Valor simulado observado |
| `mu` | sí | float | Predicción con genotipo y covariables |
| `sigma` | sí | float | Escala de error de `mu`, estrictamente positiva |
| `linea_base` | sí | str | Exactamente uno de `predictions.LINEAS_BASE`: `"solo intercepto"` o `"ancestria sexo edad"` |
| `mu_base` | sí | float | Predicción de esa línea base, sin genotipo |
| `sigma_base` | sí | float | Escala de esa línea base, estrictamente positiva |
| `mu_oraculo` | no | float | Media verdadera. Solo para diagnóstico |
| `sigma_oraculo` | no | float | Desvío verdadero, estrictamente positivo. Solo para diagnóstico. Si viene una de las dos columnas oráculo, DEBE venir la otra |
| `sexo` | no | entero, 0 o 1 | 0 = mujer, 1 = hombre. Solo para RF9 y el test 3b |
| `edad` | no | numérico | Reservado para el diagnóstico por edad, que no está en esta versión |

### 5.3 Qué valida la capa (RF8)

Cada chequeo levanta `ContratoError` con el campo y la cantidad de filas afectadas. Se valida **toda la tabla**, incluidas las filas de `entrenamiento`.

| # | Chequeo | Campo que informa |
| --- | --- | --- |
| V1 | Están todas las columnas obligatorias | nombre de la columna que falta |
| V2 | `individuo_id` es de tipo entero | `individuo_id` |
| V3 | `dominio` tiene un único valor y está en `predictions.DOMINIOS` | `dominio` |
| V4 | `particion` está en las tres permitidas | `particion` |
| V5 | `linea_base` está en `predictions.LINEAS_BASE` | `linea_base` |
| V6 | Cada (`individuo_id`, `rasgo`) tiene exactamente las dos líneas base | `linea_base` |
| V7 | La clave (`individuo_id`, `rasgo`, `linea_base`) es única | `individuo_id, rasgo, linea_base` |
| V8 | `y_real`, `mu`, `sigma`, `mu_base`, `sigma_base` y las columnas oráculo presentes son finitas (sin NaN ni ±inf) | la columna |
| V9 | `sigma > 0`, `sigma_base > 0` y `sigma_oraculo > 0` si está | la columna |
| V10 | `mu`, `sigma`, `y_real` y `particion` son **exactamente iguales** (sin tolerancia) entre las dos líneas base de cada (`individuo_id`, `rasgo`). Lo mismo vale para las columnas oráculo y `sexo` si están | la columna |
| V11 | `sexo`, si está, toma solo los valores 0 y 1 | `sexo` |
| V12 | Si viene una columna oráculo, viene la otra | `mu_oraculo, sigma_oraculo` |
| V13 | `fit` encuentra filas de `calibracion` y `predict` y `contar` encuentran filas de `prueba` | `particion` |
| V14 | Todo rasgo de `prueba` que se pide en `predict` fue calibrado en `fit` | `rasgo` |
| V15 | Los niveles pedidos están en `predictions.NIVELES` | `nivel` |

**La capa NO valida** que los ids y las particiones coincidan con la cohorte. Eso lo hacen `predictions.verificar`, `verificar_intervalos`, `verificar_barrido` y el test de contrato del paquete.

### 5.4 Invariantes que garantiza cada módulo

1. `mu` y `sigma` de `calibracion` y `prueba` vienen de un modelo ajustado solo con filas de `entrenamiento`.
2. `sigma` no usa el `y_real` de la fila que predice.
3. `mu_base` y `sigma_base` están sobre exactamente las mismas filas y la misma partición que `mu` y `sigma`, para cada línea base.
4. Sin NaN ni infinitos. `sigma > 0` y `sigma_base > 0`.
5. Los `individuo_id` son los de la cohorte 2.0 (2.000 personas, semilla 20261019), iguales en los tres dominios.
6. Una sola fila por (`individuo_id`, `rasgo`, `linea_base`).

Las invariantes 1 a 3 y 5 no son verificables por la capa: son una garantía de cada módulo. La 4 y la 6 coinciden con V7, V8 y V9.

### 5.5 Estado de cumplimiento (9/11/2026)

| Módulo | Estado |
| --- | --- |
| Voz | Cumple las 6. Hasta fin de noviembre usa los datos oráculo como entrada |
| Rostro | Cumple las 6 (10/11, rama `rostro/contrato_v5`). Inv. 1 hecha (commits `d114adf` y `173b007`). Inv. 5: usa la cohorte 2.0 con el paquete 2.2, sin cambios en la huella. Inv. 3: exporta las dos líneas base. Con efecto 0 de sexo, edad y ancestría, `"ancestria sexo edad"` sale casi igual a `"solo intercepto"`, y está declarado como limitación |
| Temperamento | Se construye con esta forma desde el inicio |

---

## 6. Interfaz y contrato de salida

### 6.1 Importación y error

```python
from anima_spine import predictions as P
from anima_spine.predictions import ContratoError
from anima_spine.calibracion import (
    ConformalCalibrator, coverage_report, control_permutacion,
)
```

- `ContratoError` está definida en `anima_spine/predictions.py` (paquete 2.2) y es subclase de `ValueError`. Los validadores del paquete y la capa levantan la misma clase.
- Se importa de `anima_spine.predictions` porque el `__init__.py` de la 2.2 todavía no la exporta. Cuando Diego la exporte, `anima_spine.ContratoError` va a ser la misma clase y este import sigue siendo válido.
- `calibracion.py` DEBE importarla de `predictions` y NO DEBE definir otra clase con ese nombre.
- Los mensajes que levanta la capa DEBEN nombrar el campo y la cantidad de filas afectadas, por ejemplo: `ContratoError: sigma: 3 filas con valor <= 0`.

### 6.2 `ConformalCalibrator(niveles=P.NIVELES)`

- `niveles` DEBE ser un subconjunto de `P.NIVELES`. Si no, levanta `ContratoError` (V15).
- Atributo después de `fit`: **`qhat_`**, un DataFrame con las columnas `linea_base, rasgo, nivel, n, k, qhat`.
  - En el modo del modelo, `linea_base` es `None`.
  - En el modo de las líneas base, lleva el texto de la línea base.
- Atributo `modo_`: `"modelo"` o `"base"`.

### 6.3 `fit(df, columnas=("mu", "sigma")) -> self`

- Valida la tabla (5.3) y toma solo `particion == "calibracion"`.
- `columnas` DEBE ser exactamente `("mu", "sigma")` o `("mu_base", "sigma_base")`. Cualquier otro valor levanta `ContratoError`.
- **Modo del modelo**, `("mu", "sigma")`:
  - después de V10, usa **una fila por (`individuo_id`, `rasgo`)**, la de `linea_base == "solo intercepto"`;
  - n es la cantidad de personas, es decir 500 con la cohorte 2.0;
  - NO DEBE usar las filas de las dos líneas base juntas (n = 1.000 es el error que detectaron Audrey y Diego en la v2).
- **Modo de las líneas base**, `("mu_base", "sigma_base")`:
  - calibra por separado cada `linea_base`, con n = 500 en cada una.
- Calcula q̂ según RF1 para cada combinación (línea base, rasgo, nivel).

### 6.4 `predict(df) -> DataFrame`

- Valida la tabla y toma solo `particion == "prueba"`.
- En el modo del modelo devuelve exactamente las columnas `P.CAMPOS_INTERVALOS`, en ese orden: `individuo_id, dominio, rasgo, nivel, lo, hi`. Tiene una fila por (`individuo_id`, `rasgo`, `nivel`).
- En el modo de las líneas base devuelve las mismas columnas y agrega `linea_base` al final, con una fila por (`individuo_id`, `rasgo`, `linea_base`, `nivel`). Estos intervalos son solo para el reporte y **no se exportan**.
- Las filas salen ordenadas por `individuo_id, rasgo, nivel` (y `linea_base` antes de `nivel` en el modo de las líneas base), con un ordenamiento estable.

### 6.5 `contar(df, iv, replica, semilla) -> DataFrame` (método del calibrador)

- `iv` DEBE ser la salida de `predict` del modo del modelo, sin columna `linea_base`. Si no, levanta `ContratoError`.
- Toma `y_real` de las filas de `prueba`, una por (`individuo_id`, `rasgo`).
- Devuelve exactamente `P.CAMPOS_BARRIDO`, en ese orden: `replica, semilla, dominio, rasgo, nivel, n, dentro`.
  - `n` es la cantidad de personas de `prueba` del rasgo.
  - `dentro` es la cantidad que cumple `lo <= y_real <= hi`, con los extremos incluidos.
- `replica` y `semilla` son enteros que se copian tal cual. La semilla es solo una etiqueta (RF10).

### 6.6 `coverage_report(df, cal, cal_base) -> dict[str, DataFrame]`

- `cal` es el calibrador ajustado en el modo del modelo y `cal_base` el ajustado en el modo de las líneas base. Si alguno viene en el modo equivocado, levanta `ContratoError`.
- Llama a `predict` por dentro. Es para la réplica 0 y **no se exporta**.
- Devuelve un diccionario con estas claves:

| Clave | Una fila por | Columnas |
| --- | --- | --- |
| `"cobertura"` | rasgo × nivel | `rasgo, nivel, n, dentro, cobertura, ic_inf, ic_sup, qhat, z, qhat_sobre_z, ancho_medio, ancho_solo_intercepto, ancho_ancestria_sexo_edad, ancho_oraculo` |
| `"crps"` | rasgo | `rasgo, crps_modelo, crps_solo_intercepto, crps_ancestria_sexo_edad, crps_oraculo` |
| `"cobertura_por_sexo"` | rasgo × nivel × sexo | `rasgo, nivel, sexo, n, dentro, cobertura`. Solo si `df` trae `sexo` |

Definiciones:

- `ic_inf` e `ic_sup`: Clopper-Pearson al 95 % para `dentro` éxitos en `n`, con `scipy.stats.beta`. Si `dentro == 0`, `ic_inf = 0`; si `dentro == n`, `ic_sup = 1`.
- `z = scipy.stats.norm.ppf((1 + nivel) / 2)`.
- `ancho_oraculo` y `crps_oraculo`: con el intervalo `mu_oraculo ± z · sigma_oraculo`. Son NaN si la tabla no trae las columnas oráculo. Es el único NaN permitido en la salida.
- **CRPS:** el de la normal N(μ, σ) evaluada en y, promediado sobre las personas de `prueba` del rasgo:

  ```
  w = (y − μ) / σ
  CRPS = σ · ( w · (2·Φ(w) − 1) + 2·φ(w) − 1/√π )
  ```

  Φ y φ son la distribución y la densidad normal estándar. Se calcula con (`mu`, `sigma`) para el modelo, con (`mu_base`, `sigma_base`) para cada línea base y con (`mu_oraculo`, `sigma_oraculo`) para el oráculo. No depende de la calibración: es el CRPS de la distribución predictiva paramétrica, que es otra distribución que la de los intervalos conformales. **Se reporta como diagnóstico, sin umbral, y no forma parte de la condición de la hipótesis** (decisión del grupo del 12/11/2026).

### 6.7 `control_permutacion(df_perm, cal_base) -> dict[str, DataFrame]` (cambiado en v5.1)

- `df_perm` es una tabla con el mismo contrato (5.2), en la que `mu` y `sigma` vienen de un modelo ajustado con el puntaje poligénico permutado. Sus columnas de línea base son las originales: lo garantiza el módulo.
- `cal_base` es el calibrador del modo de las líneas base, ajustado sobre la tabla original.
- La función ajusta por dentro un calibrador nuevo en el modo del modelo sobre `df_perm` y predice `prueba`. `ancho_base` sale de `cal_base.predict(df_perm)`, solo de la línea base `"ancestria sexo edad"`.
- Devuelve un diccionario con dos claves:

| Clave | Una fila por | Columnas, en este orden |
| --- | --- | --- |
| `"por_rasgo"` | rasgo × nivel | `rasgo, nivel, ancho_permutado, ancho_base, cociente`. Se informa siempre, completa y **sin columna `pasa`** |
| `"por_nivel"` | nivel | `nivel, n_rasgos, cociente_medio, pasa` |

  - `cociente = ancho_permutado / ancho_base`, por rasgo y nivel;
  - `cociente_medio` es el **promedio simple** de `cociente` sobre todos los rasgos del nivel (en Rostro, las 28 componentes), con el mismo peso para cada uno. No es el cociente de los anchos promedio;
  - `n_rasgos` es la cantidad de rasgos promediados;
  - `pasa = 0.95 <= cociente_medio <= 1.05`.
- El cociente se centra en alrededor de 0,997 y no en 1, porque los dos modelos estiman distinta cantidad de parámetros. Es esperable y no se corrige.
- Es un diagnóstico que va al informe. No es un test de pytest de la capa.

### 6.8 Uso típico

```python
import pandas as pd

cal = ConformalCalibrator(niveles=P.NIVELES).fit(df, columnas=("mu", "sigma"))
cal_base = ConformalCalibrator(niveles=P.NIVELES).fit(df, columnas=("mu_base", "sigma_base"))

iv  = cal.predict(df)                                # CAMPOS_INTERVALOS, solo prueba
bar = cal.contar(df, iv, replica=r, semilla=s)       # CAMPOS_BARRIDO
rep = coverage_report(df, cal, cal_base)             # solo réplica 0, no se exporta
ctl = control_permutacion(df_perm, cal_base)         # solo réplica 0, va al informe

P.exportar_intervalos(iv.to_dict("records"), dominio="temperamento")

# Después de las 20 réplicas: concatenar los `bar` y exportar
barrido = pd.concat(bars, ignore_index=True)
P.exportar_barrido(barrido.to_dict("records"), dominio="temperamento")   # escribe barrido_temperamento.csv
cobertura = P.cobertura_promedio(barrido.to_dict("records"))              # cobertura de registro (RF3)
```

### 6.9 Reglas adicionales

- El `lo`/`hi` al 90 % de `predicciones_<dominio>.csv` se toma de la salida de `predict` con `nivel == 0.90`, nunca de un cálculo aparte. `verificar_intervalos` exige que coincidan.
- El parámetro se llama siempre `nivel`. El nombre `alfa` no aparece en la capa.
- Las funciones de `predictions` trabajan con listas de diccionarios: se convierte con `DataFrame.to_dict("records")`.
- Las 20 réplicas las orquesta cada módulo: regenera los rasgos, ajusta, llama a la capa y concatena los conteos.

---

## 7. Validación de la capa (tests)

### 7.1 Criterio científico (D3)

Cada módulo se corre 20 veces completas, con semillas `20261019 + r` para r = 0 a 19, regenerando los rasgos sobre la misma cohorte y las mismas particiones. El criterio de ±5 pp se aplica por rasgo y nivel sobre la cobertura promedio de las 20 réplicas, y se informa además cuántos chequeos quedan fuera contra lo esperable por azar. Este criterio es un resultado del informe, no un test de pytest.

### 7.2 Tolerancia de los tests de cobertura

Para un nivel p, con `n_cal` personas de calibración, `n_test` de prueba y R réplicas:

```
desvio(p) = sqrt( p · (1 − p) · (1/n_cal + 1/n_test) / R )
tolerancia(p) = 4 · desvio(p)
```

Con 500 / 500 / 20 da: 0,50 → 2,83 pp; 0,60 → 2,77; 0,70 → 2,59; 0,80 → 2,26; 0,90 → 1,70; 0,95 → 1,23.

En simulación, un modelo perfecto falla al menos un chequeo en el 0 % de las corridas con 8 rasgos y en el 0,5 % con 28. Con 3 desvíos fallaría en el 8 % y el 30 %.

### 7.3 Datos de prueba

Los tests T2 y T3 se corren en pytest con **datos sintéticos generados dentro del test**, con semilla fija:

- `y_real = mu + sigma · Z`, con Z normal estándar;
- `mu` normal;
- `sigma = exp(N(0; 0,3))` por individuo;
- `sexo` asignado 50/50;
- 500 personas de calibración, 500 de prueba, 8 rasgos y 20 réplicas.

Así no dependen de archivos externos.

**Datos oráculo de Voz (validación adicional, mismos criterios).**

- Llegan el 13/11: 20 réplicas sobre la cohorte 2.0, 160.000 filas, con `replica` (0 a 19), `semilla` (`20261019 + replica`), `dominio`, `sexo`, `individuo_id`, `particion`, `rasgo`, `y_real`, `mu_oraculo` y `sigma_oraculo`.
- El archivo anterior es de la cohorte 1.1: está obsoleto y NO DEBE usarse ni mezclarse, porque los ids cambiaron.
- El archivo trae una fila por (`individuo_id`, `rasgo`) y no tiene líneas base, así que **no cumple el contrato tal como viene** (fallaría V1 y V6). El test DEBE adaptarlo así antes de llamar a la capa, réplica por réplica:
  1. duplicar cada fila, una vez con `linea_base = "solo intercepto"` y otra con `"ancestria sexo edad"`;
  2. `mu = mu_base = mu_oraculo` y `sigma = sigma_base = sigma_oraculo`;
  3. conservar `mu_oraculo`, `sigma_oraculo` y `sexo`.
- Esta adaptación vive en el código de los tests, no en la capa.

### 7.4 Tests

| Id | Qué verifica | Criterio de aprobación |
| --- | --- | --- |
| T1 | Caso chico calculado a mano | Con n = 9 scores: nivel 0,80 → k = 8 y q̂ = octavo score; nivel 0,95 → k = 10 > 9 → `ContratoError`. Además, con n = 500 los k de RF1 son exactamente 251, 301, 351, 401, 451, 476 |
| T2 | Modelo perfecto | Con `mu` y `sigma` verdaderos, en cada rasgo y nivel, \|cobertura promedio de 20 réplicas − nivel\| ≤ tolerancia(nivel) de 7.2 |
| T3a | Invariancia ante la escala | Con `sigma` × 0,7 y con `sigma` × 1,5 en todas las filas, `lo` y `hi` son iguales a los de `sigma` sin escalar, con `numpy.allclose(rtol=1e-9, atol=0)` |
| T3b | Error de sigma distinto por sexo | Con `sigma` × 0,7 si `sexo == 0` y × 1,5 si `sexo == 1`: (a) la cobertura total promedio cumple el criterio de T2; (b) en el nivel 0,90, la cobertura promedio de `sexo == 0` es ≤ 0,85 y la de `sexo == 1` es ≥ 0,95. Valores esperados: 0,805 y 0,995 |
| T4 | Anidamiento | Para cada (`individuo_id`, `rasgo`), `lo` no sube y `hi` no baja al aumentar el nivel |
| T5 | Rechazo de violaciones | Cada chequeo V1 a V15 tiene al menos un caso que levanta `ContratoError`, y el mensaje contiene el campo y la cantidad correcta de filas |
| T6 | Determinismo | Dos corridas con los mismos datos dan `qhat_`, `predict`, `contar` y `coverage_report` idénticos (`DataFrame.equals`). `contar` con `semilla` distinta da los mismos `n` y `dentro` |
| T7 | n efectivo con dos líneas base | En el modo del modelo, `qhat_.n == 500` para todo rasgo y nivel, no 1.000. En el modo de las líneas base, `n == 500` para cada `linea_base` |
| T8 | Sin efectos laterales | El DataFrame de entrada queda igual después de llamar a `fit`, `predict`, `contar` y `coverage_report` (se compara contra una copia previa con `DataFrame.equals`) |
| T9 | Columnas y orden de salida | `predict` devuelve exactamente `P.CAMPOS_INTERVALOS` y `contar` exactamente `P.CAMPOS_BARRIDO`. La salida pasa `P.verificar_intervalos` y `P.verificar_barrido` cuando se usan los ids de la cohorte |
| T10 | Filas de entrenamiento | Agregar o quitar filas de `entrenamiento` no cambia `qhat_` ni `predict` |
| T11 | Cruce con el paquete | Para cada rasgo y nivel, `coverage_report(...)["cobertura"].cobertura` es exactamente igual a `dentro / n` del barrido de `contar` en `replica == 0`, y `P.cobertura_promedio` sobre un barrido de una sola réplica da ese mismo valor |
| T12 | Error compartido | Una violación detectada por la capa levanta una instancia de `anima_spine.predictions.ContratoError` (la misma clase que levantan los validadores del paquete) y también es capturable como `ValueError`. Si `anima_spine` exporta `ContratoError`, se verifica además que `anima_spine.ContratoError is anima_spine.predictions.ContratoError` |
| T13 | Tipo de entrada (v5.1) | Pasar un dict o una lista en lugar de un DataFrame levanta `ContratoError` con el campo `df`, en `fit`, `predict`, `contar`, `coverage_report` y `control_permutacion` |
| T14 | Forma de `control_permutacion` (v5.1) | Con datos sintéticos, devuelve las claves `"por_rasgo"` y `"por_nivel"` con las columnas exactas de 6.7. `"por_rasgo"` no tiene `pasa`, `"por_nivel"` tiene una fila por nivel, y `cociente_medio` es el promedio simple de `cociente` de ese nivel |

### 7.5 Alcance de la validación

Las réplicas regeneran los rasgos sobre la misma cohorte, así que miden la maquinaria de calibración sobre esta cohorte, no sobre poblaciones independientes. La cobertura sola no valida nada: split conformal cubre al nivel nominal aunque el modelo sea inútil, y solo da intervalos más anchos. Por eso existen T3 y los diagnósticos de RF4 y RF5. Nada de esto valida supuestos genéticos sobre población real, y el informe lo dice explícitamente.

---

## 8. Decisiones

| Decisión | Resolución |
| --- | --- |
| D1. Partición | Split fijo de la cohorte 2.0. El leave-one-out queda solo dentro de BRIM como chequeo de optimismo, así que no hace falta jackknife+ |
| D2. Proporciones | 1.000 / 500 / 500, ya fijadas en la cohorte 2.0 |
| D3. Criterio de ±5 pp | Por rasgo y nivel, sobre el promedio de 20 réplicas, con conteo de fallas contra el azar |
| D4. Sigma | Voz: por individuo, con log sigma dependiente de sexo y edad centrada. Rostro: constante por componente (desvío residual sobre `entrenamiento`). Temperamento: constante por rasgo |
| D5. Línea base | Los tres emiten las dos. En Rostro y Temperamento, con efecto 0 de sexo, edad y ancestría, `"ancestria sexo edad"` sale casi igual a `"solo intercepto"`, y su R² fuera de muestra puede dar levemente negativo. No es un error y no se corrige |
| D6. Rasgos de Rostro | Los 28 componentes por separado, `pc01..pc28` |
| D7. Garantía | Solo marginal. Los subgrupos reciben solo diagnósticos |
| D8. CRPS (12/11/2026) | Diagnóstico sin umbral. No forma parte de la condición de la hipótesis. Decidido antes de que la capa calcule ningún CRPS. El documento integrado y los tres escritos individuales se actualizan con la misma fecha |

**Fuera del contrato** (no bloquean la capa, que solo ve `mu` y `sigma`): la señal simulada de cada dominio, la reformulación de la hipótesis, el margen δ de Temperamento, el ΔR² parcial, la cola del intervalo del ΔR² y la tolerancia de recuperación (`TOLERANCIA_RECUPERACION` del paquete). Se resuelven en `compare.py`, `predictions.py` y el documento integrado.

**Dos tolerancias que no hay que confundir.** La tolerancia de 4 desvíos de la sección 7.2 es del test interno de la capa con un modelo perfecto: verifica la capa. El criterio del trabajo sigue siendo **±5 pp** sobre el promedio de 20 réplicas (D3): verifica cada módulo. Que la primera sea más estricta no cambia la segunda.

---

## 9. Cronograma

| Fecha | Hito |
| --- | --- |
| 10/11 | Contrato v5 congelado. Tomás desarrolla la v0 sobre el paquete 2.2 |
| 12/11 | Contrato v5.1 congelado |
| Lo antes posible | v0: RF1, RF2, RF3, RF8, RF10 y RF11, con T1, T2, T4, T5, T6, T7, T8, T9, T10, T12 y T13 en verde |
| Esta semana | Tomás le confirma a Diego si la capa llega al 1/12. Si no llega, se activa el plan B de Voz |
| 13/11 | Diego entrega las 20 réplicas oráculo de Voz sobre la cohorte 2.0 (si se traba, avisa el 11/11) |
| 9 al 15/11 | v1: `coverage_report`, `control_permutacion`, T3a, T3b, T11 y T14. Se aplica a Rostro y Temperamento |
| Con la v1 | La capa entra en `anima_spine` 2.3 |
| Segunda quincena de noviembre | Voz conecta la capa con los datos oráculo |
| Fines de noviembre | Modelo heterocedástico de Voz. Si todavía no hay capa, Voz escribe la provisoria (unas 6 horas) |
| 1 al 7/12 | Aplicación de la capa en Voz |
| Diciembre | Validación y prototipo, sin funcionalidades nuevas |

Voz aplica la calibración recién del 1 al 7/12, y su ventana de validación queda ajustada. Pedro y Sergio tienen que saberlo.

---

## 10. Fuera de alcance

Conformalized quantile regression, regiones conformales multivariadas, Mondrian por subgrupo, conformal adaptativo, diagnóstico por tramo de edad y cualquier endpoint de API. No aportan al OE4 y cada uno suma vocabulario que después hay que defender.

---

## 11. Confirmaciones

**Tomás (9/11):** de acuerdo con todo el documento. Responsable de `calibracion.py`.

**Audrey (9/11), según su respuesta a Diego y su nota al grupo:**

- Inv. 1: de acuerdo, hecho.
- Inv. 5: de acuerdo. Usa la cohorte 2.0, verificada, y sin objeciones al congelamiento.
- Inv. 3: de acuerdo. Exporta las dos líneas base.
- Rasgos: de acuerdo con `pc01..pc28`.

**Diego (9/11), en su respuesta al contrato v4:**

| Id | Punto | Resolución |
| --- | --- | --- |
| D1 | Dónde vive la capa y versión | En `anima_spine`, con Tomás como responsable. Entra en la **2.3** |
| D2 | Función que exporta el barrido | `exportar_barrido(filas, dominio, directorio=".")`, ya existente. `dentro` del paquete ya cuenta los extremos: `(lo <= y) & (y <= hi)` |
| D3 | 20 réplicas oráculo de Voz sobre la cohorte 2.0 | Sí, para el 13/11 |
| D4 | Firmas de `contar` y `coverage_report` | De acuerdo. La cobertura de registro queda fijada en RF3: es `cobertura_promedio` sobre 20 réplicas, y `coverage_report` es diagnóstico de la réplica 0, con el cruce exacto del test T11 |
| D5 | Comentario `PC3` | Corregido en la 2.2 |
| D6 | `ContratoError` | Subclase de `ValueError`, definida por Diego en `predictions.py` y exportada por `anima_spine`. La capa la importa |

**Cambios de la v5.1 (12/11/2026):**

| Cambio | Tomás | Audrey | Diego |
| --- | --- | --- | --- |
| RF6 con cociente medio por nivel | De acuerdo | De acuerdo, con la aclaración de que es un promedio simple y de que `por_rasgo` se sigue informando (incorporada). Rostro no cambia lo que entrega: la tabla permutada es la misma | Lo propuso (10/11) |
| CRPS como diagnóstico sin umbral, fuera de la hipótesis | De acuerdo (opción A) | De acuerdo (lo propuso el 10/11) | De acuerdo (lo propuso el 10/11) |
| `ContratoError` desde `predictions`, tipo DataFrame y paquete 2.2.1 | De acuerdo | Lo señaló | Lo informó |

El contrato queda **congelado** en su versión v5.1 desde el 12/11/2026. Cualquier cambio posterior se hace con acuerdo de los tres y sube la versión de este documento.
