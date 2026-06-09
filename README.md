# Frhodo

[![CI](https://github.com/Argonne-National-Laboratory/Frhodo/actions/workflows/ci.yml/badge.svg)](https://github.com/Argonne-National-Laboratory/Frhodo/actions/workflows/ci.yml)
[![License: BSD-3-Clause](https://img.shields.io/badge/License-BSD--3--Clause-blue.svg)](LICENSE.txt)
[![Python](https://img.shields.io/badge/python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](pyproject.toml)

<img src="https://raw.githubusercontent.com/Argonne-National-Laboratory/Frhodo/assets/Logo.png" alt="Frhodo logo" width="325">

Frhodo is an open-source, GUI-based Python application to simulate experimental
data and optimize chemical kinetics mechanisms using
[Cantera](https://cantera.org) as its chemistry solver.

<img src="https://raw.githubusercontent.com/Argonne-National-Laboratory/Frhodo/assets/Frhodo_screenshot_preview.png" alt="Frhodo screenshot" width="800">

## Features

- Intuitive, extensive GUI for shock-tube kinetics workflows
- Simulate chemical kinetics experiments with:
  - 0D closed, homogeneous, constant-volume reactor
  - 0D closed, homogeneous, constant-pressure reactor
  - Custom incident-shock reactor for reactions behind incident shock waves
- Import Cantera-valid mechanisms (CTML/XML input is not supported)
- Read an experimental directory to switch quickly between conditions and measured data
- Overlay simulated observables on experimental data
- Edit mechanisms in memory and re-simulate automatically
- Inspect non-observable simulation variables in the Sim Explorer
- Optimize mechanisms against observables, by hand or via an automated routine
  - The automated routine needs bounds on reaction rate constants
  - It can optimize all three Arrhenius parameters

## Installation

Frhodo runs on Windows, macOS, and Linux.

### Conda (recommended)

```bash
conda env create -f environment.yml
conda activate frhodo
pip install -e .
frhodo
```

### pip

```bash
pip install -e ".[gui]"
frhodo
```

A prebuilt Windows x64 installer is attached to each
[release](https://github.com/Argonne-National-Laboratory/Frhodo/releases).

## Documentation

See the [manual](doc/manual.pdf) for usage instructions and the underlying
derivations.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup, tests, and the
linting/type-checking workflow.

## Citing Frhodo

If you use Frhodo in your research, please cite:

> Sikes, T., & Tranter, R. S. (2023). Frhodo: A program for simulating chemical
> kinetic measurements and optimizing kinetic mechanisms. *Combustion and Flame*,
> 257, 112509. https://doi.org/10.1016/j.combustflame.2022.112509

## License

BSD-3-Clause. Copyright © 2020, UChicago Argonne, LLC. See [LICENSE.txt](LICENSE.txt).
