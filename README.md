# TFC Anima — Módulo Temperamento

Módulo de temperamento del proyecto **Anima**. Modela los cinco rasgos
Big Five (apertura, responsabilidad, extraversión, amabilidad y
neuroticismo) sobre una cohorte de individuos compartida con los módulos de
Rostro y Voz, para que sus ΔR² se puedan comparar entre sí.

El repositorio tiene dos componentes:

- `anima_spine/`: la **columna vertebral compartida**. Define la cohorte de
  individuos, el formato de predicciones/intervalos/barridos y las
  comparaciones con bootstrap pareado. No contiene rasgos de ningún dominio.
- `temperamento/`: el **módulo propio**. Espacio de rasgos Big Five, cohorte
  sintética, modelo heterocedástico, calibración conformal y reconstrucción
  individual.

Los datos de entrada viven en `data/`, los artefactos que generan los
scripts en `outputs/`, y las versiones entregables del spine en `dist/`.

## Estructura

```
.
├── anima_spine/          # columna vertebral compartida
│   ├── spine.py          # cohorte, particiones y verificación
│   ├── predictions.py    # contrato de predicciones, intervalos y barridos
│   ├── compare.py        # bootstrap pareado entre dominios
│   └── datos/            # panel real de 1000 Genomes
├── temperamento/         # módulo de rasgos Big Five
│   ├── contract.py       # contrato común de salida (TraitResult)
│   ├── traits/           # espacio de rasgos y constantes
│   ├── cohort/           # generador de cohorte sintética
│   ├── models/           # modelo heterocedástico
│   ├── calibration/      # calibración conformal
│   └── reconstruct.py    # reconstrucción individual
├── data/                 # cohorte publicada e históricos
├── outputs/              # predicciones, intervalos y barridos (generado)
├── tests/                # tests del spine y del contrato compartido
├── examples/             # demo del circuito completo
└── dist/                 # versiones entregables del spine
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

La demo escribe estos archivos en `outputs/`. Los tests omiten los dominios
que todavía no exportaron sus resultados.

> Si se corre la demo antes que los tests, `test_reporta_las_dos_lineas_base`
> falla para los tres dominios: la demo exporta una sola línea base a
> propósito. Es el test funcionando, no un error.
