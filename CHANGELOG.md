# Changelog

All notable changes to Frhodo are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Right-click a pressure-dependent reaction (Plog/Falloff/Chebyshev) in the
  mechanism tree to recast it in place, reversibly, to an Arrhenius-like form at
  a chosen pressure. Falloff/three-body reactions keep their `[M]` dependence and
  efficiencies; Plog/Chebyshev become pure Arrhenius. The dialog defaults to the
  active shock zone's pressure and unit.
- The optimization-tab scale selector and the observable plot's y-axis scale are
  tied together so changing one updates the other.
- CTML and XML mechanisms load, converted through Cantera's `ctml2yaml`.
- Chemkin transport files: `load_mechanism(transport=...)` and a transport-file
  checkbox and list in the mechanism panel.
- YAML mechanisms open from their own path, so files they include resolve
  against their folder.
- On first launch Frhodo opens a copy of the bundled example project, which now
  ships inside the package.

### Changed
- Uncertainty band: curvature-adaptive centerline that tracks sharp features, and
  a bounded-robust envelope for log-family scales.
- `load_mechanism` raises `MechanismLoadError` for a suffix outside the supported
  formats instead of treating the file as Chemkin.
- Mechanism suffixes match case-insensitively, and the loader, the API and the
  mechanism list share one format table.
- Converter output, warnings included, is captured into load errors and the log.
- Mechanisms serialize to YAML in memory, for saving and for worker processes,
  rather than through a temporary file.
- Directory files resolve relative folder entries against their own location.

### Fixed
- CI lint step targeted a nonexistent `src/` directory and errored; it now lints
  `frhodo` and `tests`.
- Optimized mechanisms whose names contain `(`, `)`, `+` or `[` overwrote
  `- Opt 1` on every run, and an unbalanced bracket or a hand-renamed Opt file
  stopped an optimization from starting.
- The pre-optimization `- PreOpt N` file renamed mechanisms whose names contain
  "Opt".
- Non-ASCII folders were garbled by a session restore and could not be saved on
  Windows. Directory files are now read and written as UTF-8.
- Selecting a YAML mechanism loaded the previously converted mechanism instead
  of the chosen file.
- The thermo-file checkbox reset itself on reload and could be enabled for
  mechanisms that cannot use a separate file.
- Conversion output (`generated_mech.*`, `<mech>.converted.yaml`) appeared in the
  mechanism list.
- Saving species aliases rewrote the whole directory file, so a session restore
  could replace its folders.

## [2.0.0]

Baseline release from which this changelog starts.
