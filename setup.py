from setuptools import setup, find_packages
from pathlib import Path

# single source of truth for package version
version_ns = {}
with (Path("frhodo") / "version.py").open() as f:
    exec(f.read(), version_ns)
version = version_ns['__version__']

setup(
    name='frhodo',
    version='0.0.1',
    packages=find_packages(),
    description='Simulate experimental data and optimize chemical kinetics mechanisms with this GUI-based application',
    entry_points={
        'console_scripts': [
            'frhodo=frhodo.main:main'
        ]
    }
)
