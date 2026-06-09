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

### Changed
- Uncertainty band: curvature-adaptive centerline that tracks sharp features, and
  a bounded-robust envelope for log-family scales.

### Fixed
- CI lint step targeted a nonexistent `src/` directory and errored; it now lints
  `frhodo` and `tests`.

## [2.0.0]

Baseline release from which this changelog starts.
