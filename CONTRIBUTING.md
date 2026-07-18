# Contributing to jlegroup

This file is the integration contract for work developed outside this repository —
including parallel development sessions (human or AI). The repo is the interface:
deliver code as a branch here, not as loose files.

## Workflow

1. Clone and branch:
   ```sh
   git clone https://github.com/mjperson/jlegroup.git   # gh auth login first if needed
   cd jlegroup && git switch -c <topic>                 # e.g. ey92
   ```
2. Develop; verify locally:
   ```sh
   pip install -e ".[test]"
   pytest -v
   ```
3. Push the branch (`git push -u origin <topic>`; without push access, fork on GitHub
   and open a pull request — same workflow, same expectations). CI runs pytest on
   every push, on every branch — a red X on your branch means it is not ready to merge.
4. Report back to the maintainer: branch name, what you validated against and the
   residuals achieved, any new dependencies, and the provenance of any ported code.
5. Review and merge happen on the maintainer's side.

Do not modify the validated modules (`CE97.py`, `ray_crossing.py`) or existing tests on
a feature branch. Additive changes to `physicalData.py` (new gases, new constants) are
fine if documented in the commit message.

## Package conventions

- **One module per method**, src-layout: your method lands as `src/jlegroup/<NAME>.py`
  and is registered with a `from . import <NAME>` line in `src/jlegroup/__init__.py`.
- **All physical constants come from `jlegroup.physicalData`, injected via
  `constants=` (a `ConstantSet`).** Never hardcode values. Two immutable vintages
  exist: `CODATA2022` (= `DEFAULT_CONSTANTS`, what every model uses unless told
  otherwise) and `CODATA1986` (the Mathematica jleGroup vintage — the verification
  mode). **Any test that compares against a Mathematica reference or a published
  table must pass `constants=CODATA1986`**: the dominant vintage difference is G at
  +2.6×10⁻⁴, far above reference-comparison tolerances. New models should accept and
  forward a `constants=` keyword defaulting to `physicalData.DEFAULT_CONSTANTS`, and
  state in their delivery which quantities are vintage-sensitive. Refractivity
  conversions use the 1-atm Loschmidt convention (`physicalData.refractivity`) — do
  not use polarizability-based (2πα·n) refractivity or 1-bar "STP" states. (History:
  a ~1% STP-convention mismatch once produced a spurious ~1.5×10⁻³ light-curve
  discrepancy, and was the #1 integration trap until the conversions were unified.)
- **Units** (match CE97 so methods are drop-in comparable): radii, distances, and
  observer-plane positions in **km**; pressures in **Pa**; number density in **m⁻³**;
  temperatures in **K**; molar mass in **kg/mol**; refractivity dimensionless; output is
  normalized flux (1 = unocculted).
- Dependencies: stay within `numpy>=2.0, scipy, pandas, numdifftools, matplotlib` if
  possible; flag anything new. Python ≥ 3.10. Match the existing code style.

## Tests and reference data

- Benchmarks are pytest files in `tests/`, reference data in `tests/data/<case>/` as
  `parameters.csv`, `atmosphere.csv`, `lightcurve.csv` (Mathematica float format like
  `1.086*10^22` is fine — see the `_mfloat` parser in `tests/test_benchmarks.py`).
- Tolerances encode *measured* residuals with modest headroom, and any reference-side
  limitation is documented next to the tolerance (see the steep-clear example).
- Known caveats when validating against the Mathematica jleGroup generator
  (`occLightCurves` / `olcOneLimb2`, EY92 family):
  - It evaluates the EY92 θ/dθ series at **first order in 1/λ** by default
    (`olcEYorderForOneOverLambda = 1`). A Python implementation at a different series
    order will differ by O(λ⁻²) terms *by design* — match orders or budget for it
    (the neglected dθ coefficient is (9 − 34b + 25b²)/128 for T ∝ r^b).
  - The package's series order 4 is parse-broken (a line-break bug makes it evaluate
    as order-3 plus a stray term); orders 2 and 3 are usable.
  - The Chamberlain (1996) b-series error in the published EY92 is FIXED in the
    occLightCurves 4.2.0 series bodies: they carry the corrected δ⁴ coefficients and
    match this package's "corrected" variant exactly (confirmed 2026-07-08). Only the
    package's usage-text warning is stale.
  - Clear-atmosphere references are one-limb and instantaneous; only bins containing
    haze cut-on/cut-off or surface events are ExpTime-integrated (none in clear cases).

## Provenance and license

The license is MIT, with permissions granted by all lineage authors
(W. Tubthong → W. Saunders → M. Person; complete 2026-07-11). Still state clearly in
your delivery whether your code is newly written or ports existing code (Mathematica
jleGroup, Tubthong's or Saunders' Python, published equations), so the copyright line
and the provenance record stay accurate.
