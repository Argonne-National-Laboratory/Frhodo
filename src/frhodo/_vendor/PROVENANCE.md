# Vendored binary provenance

The optimizer's RBFOpt global-search backend shells out to the COIN-OR
MINLP/NLP solvers Bonmin and Ipopt. Prebuilt binaries are vendored
per platform and resolved at runtime by
`frhodo.optimize.algorithms._resolve_binary` (used in `rbfopt` as
`minlp_solver_path` / `nlp_solver_path`). `tests/test_binary_resolution.py`
pins the discovery contract and the SHA256s below.

## Bonmin

- Version: Bonmin 1.8.7 (using Cbc 2.10.3 and Ipopt 3.12.13), ASL(20161228)
- Project: https://github.com/coin-or/Bonmin
- License: Eclipse Public License v1.0 (`coin-license.txt` beside each binary)

| File | SHA256 |
| --- | --- |
| bonmin/bonmin-linux64/bonmin | e7b5a7faffea45ff0833e2458113227d5baff5e812015e41e6b52415fedb7ce9 |
| bonmin/bonmin-osx/bonmin | 4318915a6a5d4e255ec070b85e5ee3339a83d818066f48df19057357222769ad |
| bonmin/bonmin-win64/bonmin.exe | bf60d34f0a8c02810c6eb53e307ac5f704013150ece545de1d959961fd50c1dc |
| bonmin/bonmin-win64/libipoptfort.dll | 13efbf4defc057e7e0f5ab0ed85b914e436816e06fb17d09dbcacce2bba466d0 |

## Ipopt

- Version: Ipopt 3.12.13, ASL(20161228)
- Project: https://github.com/coin-or/Ipopt
- License: Eclipse Public License v1.0 (`coin-license.txt` beside each binary)

| File | SHA256 |
| --- | --- |
| ipopt/ipopt-linux64/ipopt | 5324be632e28cb99b5747003e2b174f10132935dea6ec52d2fc755d1d7e3fd45 |
| ipopt/ipopt-osx/ipopt | 42e813453aba45772a7a4e6793843ddad31de35e576dfc7982d053c281170ad1 |
| ipopt/ipopt-win64/ipopt.exe | fbfb59416249aeb64c2227712e105abdc96b845df1b685e7273b25fcc883b3b5 |
| ipopt/ipopt-win64/libipoptfort.dll | 7738c924592c6abca75a98be0b374cf280967e63483bdcfad9d1a1fa28c1f361 |

## opendsm

`_vendor/opendsm/` is the adaptive-loss source (Barron robust loss, OpenDSM,
Apache-2.0), pure Python — not a binary; provenance tracked in its own tree.
