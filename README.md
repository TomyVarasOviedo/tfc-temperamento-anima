# TFC Anima — Módulo Temperamento

Módulo de temperamento del proyecto **Anima**. Modela los cinco rasgos
Big Five (apertura, responsabilidad, extraversión, amabilidad y
neuroticismo) sobre una cohorte de individuos compartida con los módulos de
Rostro y Voz, para que sus ΔR² se puedan comparar entre sí.

El repositorio tiene dos componentes:

- `anima_spine/`: la **columna vertebral compartida**. Define la cohorte de
  individuos, el formato de predicciones/intervalos/barridos y las
  comparaciones con bootstrap pareado. No contiene rasgos de ningún dominio.
- `temperamento/`: el **módulo propio**. Parámetros, matrices de covarianza,
  simulación del genotipo/fenotipo y modelos predictivos de los cinco rasgos
  Big Five. El código de la etapa anterior (heterocedástica y conformal) se
  conserva sin uso en `temperamento/legacy/`.

Los datos de entrada viven en `data/`, los artefactos que generan los
scripts en `outputs/`, y las versiones entregables del spine en `dist/`.

## Estructura

```
.
├── anima_spine/          # columna vertebral compartida (v2.2)
│   ├── spine.py          # cohorte, particiones y verificación
│   ├── predictions.py    # contrato, recuperación y verificación
│   ├── compare.py        # ΔR² total/parcial y bootstrap pareado
│   ├── power_gradiente.py
│   └── datos/            # panel real de 1000 Genomes
├── temperamento/         # módulo de rasgos Big Five
│   ├── parametros.py     # parámetros y fuentes
│   ├── matrices.py       # matrices de covarianza
│   ├── simulacion.py     # genotipo y fenotipo simulados
│   ├── modelos.py        # modelos predictivos y líneas base
│   └── legacy/           # código archivado de la etapa anterior
├── data/                 # cohorte publicada e históricos
├── outputs/              # artefactos generados por la demo (ignorado)
├── tests/
│   ├── anima_spine/      # spine, contrato y recuperación
│   └── temperamento/     # parámetros, matrices, modelos y simulación
├── docs/                 # minutas y notas de sesión
├── examples/             # demo del circuito completo
├── dist/                 # versiones entregables del spine
├── CONTRACT.md           # contrato de la capa de calibración
└── MODEL.md              # especificación del módulo Temperamento
```

## Requisitos

- Python 3.11 o superior.
- [uv](https://docs.astral.sh/uv/) (recomendado) o `pip`.

## Ejecución en Linux / macOS

Con `uv`:

```bash
uv sync
uv run pytest -q
uv run python examples/demo_circuito.py
```

Con `venv` y `pip`:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install numpy pyyaml pytest
pytest -q
python examples/demo_circuito.py
```

## Ejecución en Windows

Con `uv` (PowerShell):

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
uv sync
uv run pytest -q
uv run python examples/demo_circuito.py
```

Con `venv` y `pip` (PowerShell):

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install numpy pyyaml pytest
pytest -q
python examples/demo_circuito.py
```

En `cmd.exe`, reemplazar la activación por:

```bat
.venv\Scripts\activate.bat
```

## Salidas

- `data/cohorte_anima.csv` y `data/cohorte_anima.meta.json`: cohorte
  publicada (versión 2.0) con su huella SHA-256.
- `outputs/predicciones_<dominio>.csv`: una fila por individuo de prueba,
  rasgo y línea base.
- `outputs/intervalos_<dominio>.csv`: intervalos calibrados en los seis
  niveles nominales.
- `outputs/barrido_<dominio>.csv`: cobertura por réplica y nivel.
- `outputs/recuperacion_<dominio>.csv`: ΔR² total y parcial por réplica,
  rasgo y línea base, con el nivel declarado.

La demo lee `data/cohorte_anima.csv` y escribe los cuatro archivos en
`outputs/` para los tres dominios simulados, contra las dos líneas base.
`outputs/` está ignorado por git: se regenera con la demo. Los tests omiten
los dominios que todavía no exportaron sus resultados.
