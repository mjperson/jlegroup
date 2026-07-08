# jlegroup

Python implementations of the MIT Elliot-group stellar-occultation light-curve methods.

| Module | Method | Status |
|---|---|---|
| `jlegroup.CE97` | Chamberlain & Elliot (1997), PASP 109, 1170 — numerical light curves from an arbitrary atmospheric model | ✅ implemented & validated |
| `jlegroup.EY92` | Elliot & Young (1992), AJ 103, 991 — analytic small-planet model with haze | ✅ implemented & validated |
| `jlegroup.EPQ03` | Elliot, Person & Qu (2003), AJ 126, 1041 — light-curve **inversion** & atmospheric retrieval with error propagation | ✅ implemented & validated |
| `jlegroup.physicalData` | constants mirroring the Mathematica ``jleGroup`physicalData`` (CODATA-1986 vintage) | ✅ |

**Naming convention:** all-lowercase `jlegroup` is this Python package; camelCase
`jleGroup` refers to the original Mathematica package family used within the group.

## Status

**Private, in development.** Intended license is MIT; permissions from the original
authors in the code lineage (Wata → W. Saunders → M. Person) are being secured.
Do not redistribute until this notice is removed.

## Install

```sh
# from GitHub (works for collaborators on the private repo, via gh/ssh auth)
pip install git+https://github.com/mjperson/jlegroup.git

# or, for development, from a checkout:
pip install -e ".[test]"
```

Requires Python ≥ 3.10 and numpy ≥ 2.0.

## Quickstart

```python
import numpy as np
from jlegroup import CE97, physicalData

# refractivity profile from a number-density profile n(r) [m^-3], radius in km
nu = physicalData.refractivity(n, gas="N2", wavelength_um=0.7)

model = CE97.ChamberlainElliot1997Model(
    refractivityProfile=nu,
    radialDistance=radius_km,          # increasing, km
    planetDistance=30 * physicalData.AU_KM,   # observer distance, km
    position=y_km,                     # observer-plane positions, km
)
model.main()
flux = model.focusedFlux               # normalized light curve at `position`
```

See `examples/isothermal_occultation.py` for a complete, runnable comparison against a
bundled reference light curve.

**📓 Tutorial:** [`examples/jlegroup_tutorial.ipynb`](examples/jlegroup_tutorial.ipynb) —
a guided tour (ships executed, with figures): `physicalData` constants, building
atmospheres (isothermal, power-law gradients, thermal inversion layers), observer-plane
and time-domain light curves with noise, and reproducing the bundled EY92 validation
cases — including the atmosphere-top clamp idiom.

## Validation

The CE97 implementation is validated against independently generated reference light
curves (Mathematica `jleGroup` `olcOneLimb2`, EY92 family; λ = 0.7 µm; N₂; 30 AU;
b = 900 km; v = 25 km/s). These run as the pytest regression suite (`tests/`).

| Case | Temperature profile | max \|model − ref\| |
|---|---|---|
| iso-clear | T ∝ r⁰ (isothermal, 114.5 K) | 4.2 × 10⁻⁵ |
| shallow-clear | T ∝ r⁻⁰·⁵ | 9.7 × 10⁻⁵ |
| steep-clear | T ∝ r⁻⁴·⁵ | 1.1 × 10⁻³ * |

\* Attributed to the *reference's* first-order-in-1/λ EY92 truncation, not to CE97: the
neglected O(λ⁻²) dθ term has coefficient (9 − 34b + 25b²)/128, which quantitatively
predicts the observed miss. For b = 0 and −0.5 the agreement meets the CE97 paper's
~10⁻⁴ accuracy claim.

**Method cross-validation (EY92 ↔ CE97 ↔ references):** the analytic `EY92` module
validates independently against the paper's own benchmark tables (print precision at all
series orders; two misprints in the published Appendix identified — see its module
docstring), reproduces the Mathematica references above to **≤ 2.5 × 10⁻⁸** at
`seriesOrder=1` (the generator's own truncation order), and agrees with CE97 pointwise
to ≤ 1.6 × 10⁻⁵ at its default `seriesOrder=4` — confirming that the steep-clear
deviation in the table is entirely reference truncation.

**Inversion (EPQ03):** validated against the paper as oracle — the noiseless standard
case reproduces Table 3's printed digits (mean/convergence temperature
79.99685 / 79.99797 K vs the printed 79.997 / 79.998; max residual 0.0046 K vs ~0.004),
the nonisothermal series stays within Fig. 6's envelope (worst case b = 9: 0.483% vs the
printed +0.48%), and the formal errors match Monte-Carlo scatter at (S/N)_H = 100.
Independently cross-checked at merge review against the Mathematica-generated iso-clear
reference (external to the module's development): the boundary fit recovers the
generator's parameters to machine precision (fit rms 8 × 10⁻¹⁴) and the inversion
returns the machine-exact input atmosphere to 0.02% in temperature in the converged
region, with pressure/density deviations collapsing from ~10⁻³ to ≲ 2 × 10⁻⁴ as the
inversion boundary moves below the chord's coarsely sampled region — the binning
sensitivity the paper's error analysis predicts.

## Known behavior / gotchas

- **Positions above the modeled atmosphere top are extrapolated.** The y→r spline only
  reaches y_top ≈ (top radius) + D·θ(top); beyond that the spline extrapolates and can
  produce spurious caustic spikes. Physically those rays pass through vacuum → flux = 1:
  clamp them (see `tests/test_benchmarks.py` or the example for the idiom). Ensure your
  atmosphere table extends well above the flux-recovery altitude (a truncated top also
  produces a spline edge artifact ~10⁻³ within ~2 scale heights of the boundary).
- The model sums **all** geometric images found by the y→r mapping; the bundled
  references are one-limb curves (far-limb flux is negligible for the bundled geometry).
- Constants in `physicalData` are deliberately CODATA-1986 to match the Mathematica
  package and the validation references — see the module docstring before "fixing" them.
- `EY92` defaults to `seriesOrder=4` with corrected Appendix coefficients: pass
  `seriesOrder=1` to reproduce Mathematica jleGroup curves, and
  `seriesVariant="as-printed"` for the journal-text coefficients (~10⁻⁶ flux effect).
  Its haze path is validated against the paper tables only, and the model is
  one-limb/instantaneous (no ExpTime event-bin integration) — extend before fitting
  real hazy events.
- `EPQ03` inverts one limb of a **clear** atmosphere (haze/extinction and large-body
  adaptations are out of scope; flux normalization is the caller's responsibility and
  dominates the systematics — EPQ03 Sec. 7.4). Deliberate deviations from the printed
  paper are documented in the module docstring: the half-light refractivity default
  follows EY92 Eq. (4.28) — the paper's own digits require it; pass
  `nu_h_method="epq03-66"` for the printed Eq. (66) — Table 10's Loschmidt exponent
  misprint is corrected, and boundary conditions with λ_h ≲ 6 are rejected (the
  half-light relation has no real solution there). On noisy data choose the inversion
  boundary `i_b` explicitly (Sec. 7.3).

## Lineage & citation

Original Python implementation by **WataThep** (2021), extended by **William Saunders**
(2021), packaged, validated, and maintained by **Michael J. Person** (2026). The method
is **Chamberlain & Elliot (1997)**, PASP 109, 1170 — please cite that paper (and this
package; see `CITATION.cff`) in work that uses it. Development history and the full
validation campaign live in the maintainer's research repository.

## License

MIT (see `LICENSE`; permissions from lineage authors in progress — see Status).
