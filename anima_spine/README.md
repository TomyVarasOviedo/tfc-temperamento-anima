# Columna vertebral — proyecto Anima

Una sola cohorte de individuos que los tres módulos comparten, para que el
ΔR² de Rostro, Voz y Temperamento se pueda comparar entre sí.

## Por qué

La hipótesis del TFC es un gradiente: el genotipo informa más sobre rostro
que sobre voz, y más sobre voz que sobre temperamento. Para demostrar un
gradiente hay que comparar los tres ΔR² entre sí, y eso exige un intervalo
de confianza sobre la *diferencia*. Un IC sobre una diferencia se calcula
con bootstrap pareado, y pareado significa que el individuo 1 de Rostro es
el mismo individuo 1 de Voz y de Temperamento.

Si cada módulo arma su propia cohorte, los tres números existen pero no se
pueden comparar.

## Qué es de quién

La columna vertebral tiene **solo** los individuos: id, subpoblación, sexo,
edad, a qué partición cae cada uno y, desde la 2.0, si es una persona real de
1000 Genomes o simulada. Nada de rostro, voz ni temperamento: los rasgos de
todas las personas, reales o simuladas, los simula cada módulo.

Cada módulo sigue siendo dueño de todo lo suyo:

| Módulo | Lo que sigue siendo propio |
|---|---|
| Rostro (Audrey) | el PCA de 28 componentes, el h² de Xiong, BRIM, su calibración |
| Voz (Diego) | los 8 parámetros source-filter, el h² de Gisladottir, su síntesis |
| Temperamento (Tomás) | los 5 rasgos Big Five, el h² de Schwaba, la correlación entre rasgos |

Lo único que cambia en cada módulo es de dónde salen los individuos: en vez
de generarlos, los lee.

## Versión 2.2 (9/11/2026): la tolerancia del criterio de recuperación

La nota de Rostro del 9/11/2026 encontró que el ΔR² parcial calculado contra
"ancestria sexo edad" sale sistemáticamente POR ENCIMA del nivel declarado, y
que con la tolerancia de 0,005 el margen quedaba justo. Lo verifiqué por
simulación y la causa está clara: la línea base no se conoce, se estima sobre
`entrenamiento` y se evalúa fuera de muestra. Estimar sus k coeficientes sobre
n personas cuesta alrededor de k/n en R² fuera de muestra, y ese costo queda
solo en el denominador de la línea base.

Corrimiento medido sobre el promedio de 20 réplicas, con k = 6 y n = 1.000:

| Línea base | Dominios | Corrimiento del parcial |
| --- | --- | --- |
| Efecto real cero | Facial, temperamental | +0,0045 a +0,0065 |
| Efecto real distinto de cero | Vocal | +0,0064 a +0,0087 |

El percentil 95 del desvío absoluto es 0,0091. Cambios:

- **`TOLERANCIA_RECUPERACION` pasa de 0,005 a 0,015.** Absorbe el corrimiento y
  su ruido, y sigue pudiendo fallar: un error en la línea base, en el bootstrap
  o en el generador se aparta mucho más que 1,5 puntos.
- **El criterio se evalúa sobre el caso base.** El segundo nivel declarado se
  informa con su intervalo y sin criterio de recuperación: con 500 personas de
  prueba un nivel del orden del 0,15 % no se distingue de cero, y una
  tolerancia absoluta de 0,015 sería diez veces el nivel que pretende verificar.
- **El corrimiento se declara, no se corrige.** Es una propiedad de estimar la
  línea base, no del circuito del genotipo, y recortar el R² negativo de la
  línea base sería maquillar el dato.
- **`ContratoError`, subclase de `ValueError`, se define en el paquete.** Así la
  capa de calibración y los validadores levantan el mismo tipo y un módulo
  puede capturar uno solo.
- **El comentario de `CAMPOS` que decía `"PC3"` ahora dice `"pc01"`**, y aclara
  que el contrato no fija el formato del nombre del rasgo.

La cohorte 2.0 y su huella no cambian con esta versión. El lugar de
`calibracion.py` queda reservado para la 2.3.

## Versión 2.1 (5/11/2026): la hipótesis reformulada

El 3/11/2026 el tutor aprobó la reformulación de la hipótesis (sección 6.8 del
documento integrado). Lo que cambia en el paquete:

**El ΔR² se calcula en sus dos formas.** `compare.delta_r2` devuelve el total y
el parcial:

    parcial = (R² completo - R² base) / (1 - R² base)

Los dos coinciden cuando la línea base no explica nada. En el dominio vocal el
sexo explica hasta el 74 % de algunos parámetros, y ahí el total queda
comprimido por el factor (1 - R² base) aunque la heredabilidad sea la misma.
Como las tres fuentes estiman sus heredabilidades netas de covariables, el
parcial es la medida comparable con el nivel declarado, y es la que usan los
criterios. El total se informa al lado.

**Ningún contraste entre dominios se afirma.** `comparar` ya no devuelve
`hipotesis_dos_niveles`: devuelve `ordenamiento` y los tres pares con rol
"informado". El código que leía esa clave va a fallar con KeyError, y está
bien que falle: el veredicto de la hipótesis ya no sale de ahí.

**Criterio nuevo: recuperación del nivel declarado.** Es la condición que la
hipótesis puede fallar. El ΔR² parcial recuperado tiene que quedar dentro de
`TOLERANCIA_RECUPERACION` (0,005) del nivel declarado, evaluado sobre el
promedio de las 20 réplicas. La tolerancia se fijó por simulación, con las
personas de prueba fijas entre réplicas y los fenotipos regenerados: el
desvío absoluto queda por debajo de 0,0043 en el 95 % de los experimentos.
Sobre una réplica individual la dispersión es cuatro veces mayor, hasta 0,016,
así que el criterio solo tiene sentido sobre el promedio. La corrida de
`demo_circuito.py` lo muestra: en el dominio vocal la réplica 0 da un parcial
de 0,057 y el promedio de las 20 da 0,035, que es el nivel declarado.

**Archivo nuevo del contrato: `recuperacion_<dominio>.csv`.** Una fila por
réplica, rasgo y línea base, con el nivel declarado del rasgo y el ΔR² total y
parcial de esa réplica. El nivel declarado sale de la configuración del módulo
y `verificar_recuperacion` rechaza un archivo donde cambie entre réplicas: si
cambia, se eligió mirando resultados. Funciones: `fila_recuperacion`,
`exportar_recuperacion`, `cargar_recuperacion`, `verificar_recuperacion` y
`criterio_recuperacion`.

**`predictions.verificar` ahora exige las dos líneas base.** Antes aceptaba un
archivo con una sola, y no debía: parte de la diferencia de ΔR² entre dominios
vendría de que cada módulo comparó contra otra cosa.

La cohorte 2.0 y su huella no cambian con esta versión.

## Versión 2.0 (1/11/2026): cohorte mixta

Decidida por el grupo el 1/11/2026. Es la cohorte base del trabajo, sin vuelta
atrás a la 1.1. Se congela el 6/11/2026.

| Cambio | 1.1 | 2.0 |
|---|---|---|
| Personas | 2.000 simuladas | **503 reales de 1000 Genomes** (fase 3, panel europeo) + 1.497 simuladas, con los mismos totales por subpoblación |
| Columnas | `individuo_id, subpoblacion, sexo, edad, particion` | Además `origen` (`1000g` o `simulada`) y `muestra_1000g` (id de la muestra, vacío en las simuladas) |
| Sexo | Sorteado para quedar 1.000 / 1.000 | Las reales con su sexo real; las simuladas completan hasta 1.000 / 1.000, balanceado dentro de cada subpoblación |
| Particiones | Estratificadas por subpoblación x sexo | Por subpoblación x sexo x origen: reales 253 / 125 / 125, simuladas 747 / 375 / 375 |
| Panel | — | `anima_spine/datos/muestras_1000g_eur.csv` (503 filas), preparado con `spine.preparar_panel` desde `integrated_call_samples_v3.20130502.ALL.panel` |

| Subpoblación | Reales | Simuladas | Total |
|---|---|---|---|
| CEU | 99 | 295 | 394 |
| FIN | 99 | 295 | 394 |
| GBR | 91 | 271 | 362 |
| IBS | 107 | 318 | 425 |
| TSI | 107 | 318 | 425 |

**Los rasgos siguen siendo simulados para todas las personas**: 1000 Genomes
no informa voz, cara ni personalidad. Lo real de las 503 es su subpoblación,
su sexo y su genoma. Cada módulo decide si usa variantes publicadas leídas del
genoma real (parte genética "conocida") y completa el resto simulado hasta la
heredabilidad del rasgo; Temperamento deja toda su parte genética simulada.

`spine.cargar` lee también los archivos de la 1.1 (sin las columnas nuevas,
todas las personas quedan como simuladas). La 1.1 queda en `historico/`.

## Versión 1.1 (22/10/2026)

| Cambio | Antes (1.0) | Ahora (1.1) |
|---|---|---|
| Subpoblaciones | Sorteadas con las proporciones del panel (GBR 376, IBS 413…) | Conteos exactos: CEU 394, FIN 394, GBR 362, IBS 425, TSI 425 |
| Sexo | Sorteado (1.020 / 980) y sin codificación declarada | 1.000 / 1.000, balanceado dentro de cada subpoblación. **0 = mujer, 1 = hombre** |
| Particiones | Al azar, sin estratificar | Estratificadas por subpoblación x sexo: cada estrato a ±1 persona de 50/25/25 |
| Reproducibilidad | Se comparaba regenerando con numpy | Huella SHA-256 del CSV y versión de numpy en los metadatos; `cargar` verifica la huella |
| Contrato | Solo el intervalo al 90 % | Además, `intervalos_<dominio>.csv` con los seis niveles de la hipótesis (50 % a 95 %) |
| Contraste | Dos colas | Una cola por defecto (sección 4.7 del documento integrado); el IC a dos colas se sigue informando |
| Calibración | Un intervalo, una corrida | `barrido_<dominio>.csv` con 20 réplicas; el criterio de ±5 puntos se evalúa sobre la cobertura promedio (opción A) |
| Potencia | Brechas desde 0,015 | Brechas desde 0,005, y `--voz-real` con los 8 h² publicados de Voz |

La versión 1.0 queda en `historico/` para trazabilidad. **Si algún módulo ya
calculó resultados sobre la 1.0, tiene que volver a correrlos sobre la 1.1.**

## Rango de edad (común a los tres módulos)

La edad es uniforme entera entre 18 y 80 años. Es un supuesto de diseño:

- **18**: los tres dominios ya tienen su forma adulta. La forma facial adulta
  aparece después de la pubertad (Kesterke et al., 2016); el estudio de voz
  empieza en 18 años (Gisladottir et al., 2023); los instrumentos de
  personalidad son de adultos.
- **80**: por encima, las poblaciones fuente tienen pocos datos y aparecen
  cambios de la vejez que ningún módulo modela.
- **Uniforme**: da la misma precisión en todas las edades para estimar el
  efecto de la edad. La media de la cohorte (≈49) es más joven que la de las
  fuentes (Voz: 54), y eso se declara.

Cada módulo modela el efecto de la edad sobre sus rasgos y supone que la
heredabilidad de su fuente vale en todo el rango.

## Cómo se usa

Generar la cohorte (una sola vez, ya está hecha):

```bash
python -m anima_spine.spine
```

Produce `cohorte_anima.csv` y `cohorte_anima.meta.json`. **No regenerarla**
sin avisar a los tres: todos los resultados calculados sobre la versión
anterior dejan de ser comparables.

Desde cada módulo:

```python
from anima_spine import spine, predictions

sp = spine.cargar("cohorte_anima.csv")
ids = sp.ids("entrenamiento")   # o "calibracion" / "prueba"

# ... acá va todo lo propio del módulo ...

predictions.exportar(filas, dominio="rostro")
```

Cada módulo exporta `predicciones_<dominio>.csv` con una fila por individuo
de la partición de prueba, por rasgo y por línea base. El formato está en
`predictions.CAMPOS` y se valida con `predictions.verificar`.

Además exporta `intervalos_<dominio>.csv`, con una fila por individuo de
prueba, por rasgo y por nivel nominal (`predictions.NIVELES`: 50, 60, 70, 80,
90 y 95 %). Se valida con `predictions.verificar_intervalos`, y la cobertura
la calcula `predictions.cobertura_por_nivel` contra el `y_real` de las
predicciones:

```python
predictions.exportar_intervalos(filas_intervalos, dominio="voz")
```

Y exporta `barrido_<dominio>.csv`: el módulo se corre completo
`predictions.REPLICAS` veces (20), cada vez con su semilla, y por réplica,
rasgo y nivel se guarda cuántas personas de prueba quedaron dentro de su
intervalo (`n` y `dentro`). La réplica 0 es la corrida principal, la misma de
los otros dos archivos, y `predictions.verificar_barrido` la recuenta de forma
central contra ellos. El criterio de calibración de la hipótesis (±5 puntos en
cada rasgo y cada nivel) se evalúa sobre `predictions.cobertura_promedio`, no
sobre una corrida suelta:

```python
filas = [predictions.fila_barrido(r, semilla, "voz", rasgo, nivel, y, lo, hi)
         for ...]   # una fila por réplica x rasgo x nivel
predictions.exportar_barrido(filas, dominio="voz")
```

Cada réplica tiene que repetir todo lo que depende del azar del módulo: la
generación de los rasgos y el ajuste. La cohorte y las particiones son
siempre las mismas.

Los textos de `linea_base` son exactamente `solo intercepto` y
`ancestria sexo edad` (con espacios). El Anexo C del documento integrado
tiene que usar los mismos.

Cuando los tres exportaron:

```python
from anima_spine import compare
resultados = compare.comparar_lineas_base(preds, rasgos)   # las dos líneas base
for r in resultados.values():
    compare.imprimir(r)
```

El contraste es a una cola por defecto, porque la hipótesis es direccional
(`colas=2` para el criterio a dos colas).

## Las dos líneas base

Los tres módulos reportan contra **las dos**: solo intercepto, y ancestría
+ sexo + edad. Es un número más por rasgo y evita que parte de la diferencia
entre dominios venga de que cada uno comparó contra otra cosa.

## Tests

```bash
pytest tests/test_contrato_compartido.py -q
```

Un módulo que todavía no exportó sale como `skipped`, no como error: cada
uno puede correrlo desde el primer día.

`demo_circuito.py` simula los tres módulos para verificar el circuito
completo. Exporta a propósito una sola línea base, así que hace fallar
`test_reporta_las_dos_lineas_base` — eso es el test funcionando, no un bug.

**Por qué el criterio va sobre el promedio (opción A, 22/10/2026).** En una
sola corrida, con 500 personas de calibración y 500 de prueba, la cobertura
al 50 % tiene un desvío de unos 3 puntos, y exigir ±5 puntos en cada rasgo y
cada nivel falla por azar aunque el modelo sea perfecto (simulado: 85 % de
las veces en Voz, 70 % en Temperamento y casi siempre en Rostro). Promediar
20 réplicas baja ese desvío a menos de 1 punto. En la demo, F0 al 50 % da
44,8 % en la corrida principal y 49,1 % en el promedio.

## Potencia

```bash
python -m anima_spine.power_gradiente            # grilla de brechas, por dominio
python -m anima_spine.power_gradiente --voz-real # Rostro contra los 8 h² publicados de Voz
```

Con la brecha que surge de los valores publicados (0,008), la potencia del
escalón Rostro–Voz es de 0,12 a 0,19 con 500 personas de prueba. Por eso la
hipótesis se formula en dos niveles (22/10/2026): Rostro y Voz por encima de
Temperamento son los contrastes que se afirman; la diferencia Rostro–Voz se
informa con su intervalo, pero no se afirma.

## Supuestos de diseño que hay que documentar en el informe

- 1000 Genomes no tiene edad. La edad se genera uniforme entre 18 y 80: es
  un supuesto de diseño, no un dato.
- El efecto de edad, sexo y ancestría sobre cada fenotipo lo define cada
  módulo y también es un supuesto.
- Los tamaños de subpoblación son los del panel europeo de 1000G (fase 3),
  usados como proporciones. Verificados contra Ensembl: 99, 91, 99, 107 y 107.
- Desde la 2.0, la cohorte incluye a las 503 personas europeas de 1000
  Genomes con su subpoblación, su sexo y el identificador de su muestra, que
  permite leer su genoma. Las 1.497 restantes son simuladas.
- El fenotipo de todas las personas, reales o simuladas, es simulado. El
  documento integrado tiene que decirlo explícitamente.
- Sexo balanceado por diseño (0 = mujer, 1 = hombre), contando el sexo real
  de las 503 personas de 1000 Genomes.
