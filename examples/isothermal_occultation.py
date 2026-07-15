"""Run the CE97 forward model on a bundled clear-atmosphere benchmark case and compare
against its EY92-generated reference light curve.

Usage:
    python examples/isothermal_occultation.py [case] [--plot]

    case: iso-clear (default) | shallow-clear | steep-clear
    --plot: also write <case>-comparison.png next to this script
"""
import os
import sys

import numpy as np
import pandas as pd

from jlegroup import CE97, physicalData

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "..", "tests", "data")

case = next((a for a in sys.argv[1:] if not a.startswith("-")), "iso-clear")
do_plot = "--plot" in sys.argv

# ---- geometry (see tests/data/<case>/parameters.csv) -----------------------
b_km, v_kms = 900.0, 25.0
D_km = 30 * physicalData.AU_KM

# ---- load the case ----------------------------------------------------------
def mfloat(s):
    return float(str(s).replace("*10^", "e").replace("*^", "e"))

atm = pd.read_csv(os.path.join(DATA, case, "atmosphere.csv"))
atm.columns = [c.strip() for c in atm.columns]
radius = atm["Radius (km)"].apply(mfloat).to_numpy()
numdens = atm["Number Density (/m^3)"].apply(mfloat).to_numpy()

lc = pd.read_csv(os.path.join(DATA, case, "lightcurve.csv"))
t = lc["Time (seconds)"].to_numpy(dtype=float)
flux_ref = lc["Flux (normalized)"].to_numpy(dtype=float)

# ---- forward model ----------------------------------------------------------
# Validation vintage: the bundled references use the CODATA-1986 Loschmidt.
nu = physicalData.refractivity(numdens, gas="N2", wavelength_um=0.7,
                               constants=physicalData.CODATA1986)
position = np.sqrt(b_km**2 + (v_kms * t) ** 2)

model = CE97.ChamberlainElliot1997Model(
    refractivityProfile=nu,
    radialDistance=radius,
    planetDistance=D_km,
    position=position,
)
model.main()
flux = np.asarray(model.focusedFlux, dtype=float)

# rays above the mapped atmosphere top pass through vacuum -> flux = 1 exactly
x0 = np.arange(0, radius[-1], model.integrationBin)
y_top = radius.max() + D_km * 2 * np.trapezoid(model.integrandTheta(x0, radius.max()), x0)
flux[position > y_top] = 1.0

# ---- compare ----------------------------------------------------------------
resid = flux - flux_ref
print(f"=== {case}: CE97 vs EY92 reference ===")
print(f"points          : {len(t)}")
print(f"max |residual|  : {np.max(np.abs(resid)):.3e}")
print(f"RMS residual    : {np.sqrt(np.mean(resid**2)):.3e}")
print(f"minimum flux    : ref {flux_ref.min():.6f}  model {flux.min():.6f}")

if do_plot:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(9, 7), sharex=True,
                           gridspec_kw={"height_ratios": [3, 2]})
    ax[0].plot(t, flux_ref, "k-", lw=2, label="reference (EY92)")
    ax[0].plot(t, flux, "r--", lw=1.2, label="CE97 (jlegroup)")
    ax[0].set_ylabel("normalized flux")
    ax[0].legend()
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, resid, "b-", lw=1)
    ax[1].set_yscale("symlog", linthresh=1e-5)
    ax[1].set_xlabel("time (s)")
    ax[1].set_ylabel("model - ref")
    ax[1].grid(alpha=0.3)
    out = os.path.join(HERE, f"{case}-comparison.png")
    fig.tight_layout()
    fig.savefig(out, dpi=130)
    print(f"wrote {out}")
