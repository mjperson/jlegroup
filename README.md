# jlegroup

Python implementations of the MIT Elliot-group stellar-occultation light-curve methods.

*Named for James Ludlow Elliot (1943–2011) — author of the methods implemented here,
and mentor to this package's maintainer.*

| Module | Method | Status |
|---|---|---|
| `jlegroup.CE97` | Chamberlain & Elliot (1997), PASP 109, 1170 — numerical light curves from an arbitrary atmospheric model | ✅ implemented & validated |
| `jlegroup.EY92` | Elliot & Young (1992), AJ 103, 991 — analytic small-planet model with haze, two-limb/central flash, surface cutoff, and the traditional half-light fitting parameterizations (r_h, λ_iso) / (r_h, H) | ✅ implemented & validated |
| `jlegroup.EPQ03` | Elliot, Person & Qu (2003), AJ 126, 1041 — light-curve **inversion** & atmospheric retrieval with error propagation | ✅ implemented & validated |
| `jlegroup.physicalData` | constants mirroring the Mathematica ``jleGroup`physicalData`` (CODATA-1986 vintage, test-pinned, with provenance records) + the EY92 Table 9 gas registry, multi-gas dispersion formulas (N₂, H₂, Ar, CO₂, He, 85/15 H₂–He "Uranus"), and an occultation-target body registry (Earth, Pluto, Triton, Sun) | ✅ |
| `jlegroup.shadowmap` | occultation **shadow maps** — the Earth seen from the occulting body with coastlines, night shading, fundamental-plane track bands and ground paths, plus the `smDist`/`smOffset` prediction workflow (functionality of Mathematica ``jleGroup`shadowMap`` 4.1.4; no reference publication). Optional: `pip install "jlegroup[shadowmap]"` (astropy); loads lazily | ✅ implemented & validated |

**Naming convention:** all-lowercase `jlegroup` is this Python package; camelCase
`jleGroup` refers to the original Mathematica package family used within the group.

## Status

**Private, in development.** License is MIT, with permissions granted by **all** authors
in the code lineage (W. Tubthong → W. Saunders → M. Person):
**Wata (Chanita) Tubthong — granted, unrestricted (2026-07-10)**;
**William Saunders — granted, full (2026-07-11)**.
Public release / PyPI publication is now purely a maintainer decision;
until it happens, do not redistribute.

## Install

```sh
# from GitHub (works for collaborators on the private repo, via gh/ssh auth)
pip install git+https://github.com/mjperson/jlegroup.git

# or, for development, from a checkout:
pip install -e ".[test]"

# shadow maps are an optional extra (adds astropy; the light-curve
# modules work without it):
pip install "jlegroup[shadowmap] @ git+https://github.com/mjperson/jlegroup.git"
pip install -e ".[test,shadowmap]"        # development equivalent
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

**📓 Tutorials** (in [`examples/`](examples/); shipped executed, with figures):

| notebook | covers |
|---|---|
| [`00_jlegroup_overview.ipynb`](examples/00_jlegroup_overview.ipynb) | the package in one exercise: EY92 ↔ CE97 forward cross-check, then the EPQ03 inversion round trip |
| [`01_EY92_basics.ipynb`](examples/01_EY92_basics.ipynb) | analytic model: parameters, series order & misprint corrections, haze, two-limb central flash, traditional fitting parameterizations |
| [`02_CE97_basics.ipynb`](examples/02_CE97_basics.ipynb) | numerical model: atmosphere builders, time-domain curves + noise, validation vs bundled references, the atmosphere-top clamp idiom |
| [`03_EPQ03_basics.ipynb`](examples/03_EPQ03_basics.ipynb) | inversion: noiseless round trip, noisy retrieval, the EPQ03 error budget, thermal gradients |
| [`04_shadowmap_basics.ipynb`](examples/04_shadowmap_basics.ipynb) | shadow maps on the 2015-06-29 Pluto event: globe / mercator / equirectangular views, night shading, track bands + ground paths, `smDist`/`smOffset` (needs the `shadowmap` extra) |

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
deviation in the table is entirely reference truncation. Two-limb (central-flash)
fluxes are cross-validated per limb against CE97 evaluated at ±ρ on the same
refractivity profiles: ≤ 3 × 10⁻⁷ (λ ≈ 21) and ≤ 7 × 10⁻⁸ (the bundled
Mathematica-exact table, λ = 30, checked at merge review).

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
  Its haze path is validated against the paper tables only. The model is one-limb by
  default: pass `twoLimb=True` for the far limb / central flash, and set
  `surfaceRadius` with it — the transparent analytic atmosphere otherwise passes
  far-limb rays arbitrarily deep. Sampling is instantaneous (no ExpTime event-bin
  integration) — extend before fitting real events with haze cut-ons or surface
  contact. For **fitting**, use the traditional half-light classes
  (`ElliotYoung1992ModelTraditional`, `...TraditionalScaleHeight`): physical inputs
  enter the model only through (ν₀, λ_g0), so fitting P/T/M chases a degeneracy ridge —
  the half-light pair (r_h, λ_iso) [Mathematica `lambdaHi` convention; λ_true =
  λ_iso − (3a+5b)/2 applied internally] or (r_h, H) is what the curve constrains.
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

Original Python implementation by **Wata (Chanita) Tubthong**
([ORCID 0000-0002-7907-2634](https://orcid.org/0000-0002-7907-2634), 2021), extended by
**William Saunders** ([ORCID 0000-0002-8737-742X](https://orcid.org/0000-0002-8737-742X),
2021), packaged, validated, and maintained by **Michael J. Person**
([ORCID 0000-0003-0000-0572](https://orcid.org/0000-0003-0000-0572), 2026). The method
is **Chamberlain & Elliot (1997)**, PASP 109, 1170 — please cite that paper (and this
package; see `CITATION.cff`) in work that uses it. Development history and the full
validation campaign live in the maintainer's research repository.

## License

MIT (see `LICENSE`; all lineage-author permissions granted — see Status).
