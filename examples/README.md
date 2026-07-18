# jlegroup examples

Notebooks ship **executed** (figures embedded), so they read on GitHub without running
anything. Start with the overview, then the module you need:

| notebook | covers |
|---|---|
| [`00_jlegroup_overview.ipynb`](00_jlegroup_overview.ipynb) | the package in one exercise: EY92 analytic curve ↔ CE97 numerical curve cross-check, then the EPQ03 inversion round trip |
| [`01_EY92_basics.ipynb`](01_EY92_basics.ipynb) | the analytic model: physical parameters, series order & the appendix-misprint corrections, haze layers, two-limb **central flash**, the traditional half-light fitting parameterizations |
| [`02_CE97_basics.ipynb`](02_CE97_basics.ipynb) | the numerical model: atmosphere builders (power-law and arbitrary T(r)), time-domain curves + noise, validation against the bundled references, the above-atmosphere vacuum clamp |
| [`03_EPQ03_basics.ipynb`](03_EPQ03_basics.ipynb) | inversion: noiseless round trip, noisy retrieval with a deliberate boundary choice, the full error budget, thermal gradients, ratchet binning |
| [`04_shadowmap_basics.ipynb`](04_shadowmap_basics.ipynb) | occultation shadow maps on the 2015-06-29 Pluto event: the view-from-the-body globe, night shading, fundamental-plane tracks and re-projected ground paths, the `smDist`/`smOffset` workflow |

`isothermal_occultation.py` — script version of the CE97 benchmark comparison
(`python examples/isothermal_occultation.py iso-clear --plot`).

To re-run the notebooks: install the package with dev extras plus notebook tooling
(`pip install -e ".[test]" matplotlib jupyter`) and run from the repo root or this
directory — `02_CE97_basics.ipynb` reads reference data from `../tests/data/`.
