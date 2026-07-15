# Changelog

All notable changes to `jlegroup`. Versions are tagged in git; entries list the
tagged commit. (Tags v0.1.0–v0.9.0 were created retroactively on 2026-07-14 from
the recorded version-bump commits; from v0.9.0 onward, tagging happens at release.)

Version scheme: `0.N.0` during development; `1.0.0` is reserved for the first
public release.

## [Unreleased]

## [0.10.0] — 2026-07-15 — The Great Constant Migration

**Two vintages, one switch — and modern values by default.**
Outputs of vintage-sensitive quantities change by up to **+2.6×10⁻⁴** (the
1986→2022 shift in G); everything else moves by ≲10⁻⁵.

- `physicalData` now carries **two immutable, test-pinned `ConstantSet`
  vintages**: `CODATA2022` (current values, several SI-exact; module-level
  names and `DEFAULT_CONSTANTS` point here) and `CODATA1986` (the Mathematica
  ``jleGroup`physicalData`` vintage of every validation reference, archived in
  `VINTAGE_1986`). Measured data (Table-9 refractivities, dispersion formulas,
  `BODIES`) do not participate in the switch.
- **Every model takes `constants=`** and defaults to `DEFAULT_CONSTANTS`:
  `EY92.ElliotYoung1992Model`, the three `CE97` Atmosphere builders (new), and
  all `EPQ03` entry points (defaults flipped from `CODATA1986`). The EY92
  traditional half-light classes are constants-free by construction.
- **CE97 Atmosphere migration**: the three builder classes' hardcoded R/G/k_B
  (2014–2018 vintages) are replaced by the injected set, and the internal
  refractivity conversion moves from a 1-bar reference state to the package's
  1-atm Loschmidt — the class `Refractivity` column now equals
  `physicalData.refractivity()` **exactly**, closing the long-documented
  ~1.3% "class-internal refractivity" gotcha. (Characterized migration:
  temperature columns unchanged, pressure/density ≤2×10⁻⁵, ν column ×0.98692.)
- **Validation is vintage-pinned**: every comparison against Mathematica
  references or paper tables passes `constants=CODATA1986`, so all published
  validation numbers (4.244×10⁻⁵ iso-clear, etc.) are unchanged and now
  vintage-explicit. New `tests/test_CE97_atmosphere.py`; both vintages pinned
  in `tests/test_physicalData.py`. 211 tests.

Earlier release engineering (unreleased at the time): CI matrix (Python
3.10–3.13 on Ubuntu + 3.13 on macOS), sdist ships `tests/data/` via
MANIFEST.in (a pytest run from the sdist was failing on missing reference
data), build/twine verified, metadata polish, retroactive tags v0.1.0–v0.9.0,
and this CHANGELOG.

## [0.9.0] — 2026-07-14 (`cac2f53`)

- **New module `shadowmap`**: occultation shadow maps — the Earth seen from the
  occulting body (orthographic/mercator/equirectangular) with Natural Earth
  1:110m coastlines (bundled, no runtime downloads), night shading at a chosen
  sun-depression angle, fundamental-plane track bands with prediction-error
  lines, on-Earth ground paths, and the `smDist`/`smOffset` prediction workflow.
  Implements the functionality of Mathematica ``jleGroup`shadowMap`` 4.1.4
  (no reference publication); astropy supersedes the original's time/solar
  machinery. Validation vs. reference output of the original documented two
  defects in the Mathematica package (night-cap sign; obliquity timescale) —
  see `tests/data/shadowmap-mathematica/README.md`.
- astropy is an **optional extra** (`pip install "jlegroup[shadowmap]"`); the
  module lazy-loads via PEP-562 `__getattr__`, so the light-curve modules stay
  astropy-free at install and import time.
- `physicalData.BODIES["Earth"]` (IAU-1976 equatorial radius, the value the
  original hardcoded).
- Tutorial `examples/04_shadowmap_basics.ipynb` (the 2015-06-29 Pluto event).

## [0.8.0] — 2026-07-12 (`0d3b611`)

- **EY92 traditional half-light parameterizations** — the parameter sets the
  group has always fit: `ElliotYoung1992ModelTraditional(radiusHalfLight,
  lambdaHalfLight, b, …)` (equivalent-isothermal `lambdaHi` convention;
  λ_true = λ_iso − (3a+5b)/2 applied internally) and
  `ElliotYoung1992ModelTraditionalScaleHeight(radiusHalfLight, scaleHeightH, …)`.
  Constants-free (ν₀ from the flux-level condition and geometry); anchor
  invariant under twoLimb/haze/surface options. Cross-validated bit-identically
  against EPQ03's independent parameterization.
- Tutorial section on fitting in half-light space (`01_EY92_basics.ipynb` §6).

## [0.7.0] — 2026-07-11 (`471f36a`)

- **Tutorial suite restructured**: the omnibus notebook is replaced by
  `00_jlegroup_overview` (EY92 ↔ CE97 cross-check + EPQ03 round trip),
  `01_EY92_basics`, `02_CE97_basics`, `03_EPQ03_basics`, plus an
  `examples/README.md` index. All notebooks ship executed.

## [0.6.0] — 2026-07-08 (`59106fc`)

- `physicalData` ports the remaining Mathematica ``jleGroup`physicalData``
  content: multi-gas STP refractivity dispersion (N₂, H₂, Ar, CO₂, He, 85/15
  H₂–He "Uranus") with a `reference=` selector, molar masses, c/h/σ_SB, and a
  `BODIES` registry (Pluto, Triton — the benchmark body — and the Sun) with
  source citations. Default wavelength stays 0.7 µm (validation vintage).

## [0.5.0] — 2026-07-08 (`43806e4`)

- **Constants consolidated** into `physicalData` as the single home:
  `Constant` float-subclass records (value/symbol/units/uncertainty/source),
  the EY92 Table-9 `Gas`/`GASES` registry, and the `ConstantSet`/`CODATA1986`
  injection mechanism (moved up from EPQ03, which re-exports the same objects).
  The CODATA-1986 vintage pins are now enforced by tests.

## [0.4.0] — 2026-07-08 (`ee9512d`)

- **EY92 two-limb / central flash**: `twoLimb=True` adds the far-limb
  contribution (`r_far_of_rho`, `phi_two_limb`); `surfaceRadius=` blocks rays
  below a hard surface (strongly recommended with twoLimb). Shared `r_of_rho`
  solver hardened (physical-branch floor + robust initial guess). Cross-
  validated per limb against CE97 at ±ρ.

## [0.3.0] — 2026-07-08 (`3c1fc5c`)

- **New module `EPQ03`** (Elliot, Person & Qu 2003): light-curve inversion and
  atmospheric retrieval — boundary fitting (`EY92PowerLawBoundary`), onion-peel
  inversion to ν/n/p/T profiles, full Sec.-4 error propagation, synthetic-event
  helpers, and plotting. Paper-as-oracle validation (Table 3, Fig. 6, Table 10
  misprint corrected and documented).

## [0.2.0] — 2026-07-07 (`0c5682a`)

- **New module `EY92`** (Elliot & Young 1992): analytic small-planet light
  curves — asymptotic series (orders 0–4; two published Appendix misprints
  corrected by default, `"as-printed"` available), power-law T/µ profiles, haze
  layer. Validated against digitized paper tables, Mathematica jleGroup
  references (order 1, ≤2.5×10⁻⁸), and CE97 (order 4, ≤1.6×10⁻⁵).

## [0.1.0] — 2026-07-06 (`441491e`)

- Initial package: `CE97` (Chamberlain & Elliot 1997 numerical forward model;
  original lineage code, verbatim) with `ray_crossing`, `physicalData`
  (CODATA-1986 vintage constants + Peck & Khanna 1966 N₂ refractivity), the
  three-case benchmark regression suite with bundled reference light curves,
  CI, and the first tutorial notebook.
