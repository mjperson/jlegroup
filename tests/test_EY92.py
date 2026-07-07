"""EY92 validation: paper benchmark, series identities, and reference curves.

Three independent validation layers:

1. **Published-paper oracle** (tests/data/ey92-paper/): the EY92 benchmark
   Tables 3a/3b and the p. 1004 derived parameters, digitized from the
   journal. Independent of the Mathematica jleGroup lineage. Measured
   agreement: conversion exact to all 7 printed digits at series orders
   0/1; full pipeline at print precision for all orders (r to <= 1 m,
   fluxes to ~2e-7 at orders 0/1 and ~3e-6 at order 4).

2. **Series self-consistency**: B(delta,a,b) is determined by A(delta,a,b)
   through the derivative of Eq. (4.6); requiring the identity to hold
   order-by-order pins down two misprints in the published Appendix
   (consistent with the Chamberlain 1996 b-series error mentioned in the
   Mathematica jleGroup usage text). The "corrected" variant satisfies
   the identity exactly for all (a, b); "as-printed" fails only in the
   delta^4 b-terms.

3. **Repo reference light curves** (tests/data/<case>/, Mathematica
   jleGroup olcOneLimb2): at seriesOrder=1 — matching the generator's
   olcEYorderForOneOverLambda = 1 — EY92 reproduces all three cases to
   max|res| <= 2.5e-8 (measured 2026-07-06: iso 9.1e-9, shallow 1.0e-8,
   steep 2.5e-8). This confirms the steep-clear reference's known ~1e-3
   deviation is entirely the generator's first-order truncation: at
   seriesOrder=4 the EY92 residuals against the same references are
   4.24e-5 / 9.72e-5 / 1.12e-3 — the same values CE97 shows (see
   test_benchmarks.py), i.e. EY92(order 4) agrees with CE97 and both
   differ from the reference by the documented O(lambda^-2) budget.
"""
import os

import numpy as np
import pandas as pd
import pytest

from jlegroup import EY92, physicalData

DATA = os.path.join(os.path.dirname(__file__), "data")
PAPER = os.path.join(DATA, "ey92-paper")

# ---------------------------------------------------------------------------
# Paper benchmark configuration (EY92 Table 2; see ey92-paper/README.md)
# ---------------------------------------------------------------------------

R0_PAPER = 1250.0  # km, EY92's adopted reference radius
D_PAPER = 28.89675 * physicalData.AU_KM


def paper_data_params(b):
    return EY92.DataParams(
        s_background=630.0,
        s_background_slope=0.0,
        s_full_scale=3340.0,
        d=D_PAPER,
        v=18.5,
        delta_t=0.2,
        t_min=46.9,
        a=0.0,
        b=b,
        t_im=53.9,
        t_em=139.9,
        t_hiso=4.7,
        t_h1=4.2,
        t_h2=7.0,
        t_hr2=2.3,
    )


def _paper_table(name):
    return pd.read_csv(os.path.join(PAPER, name))


# ---------------------------------------------------------------------------
# 1a. Digitization integrity (the printed tables are over-determined)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name,b", [("table3a.csv", 0.0), ("table3b.csv", -0.6)])
def test_paper_tables_internal_consistency(name, b):
    tab = _paper_table(name)
    dp = paper_data_params(b)
    # rho(t) straight-line geometry [5.1]
    rho = np.hypot(dp.rho_min, dp.v * (tab.time_s - dp.t_mid))
    assert np.allclose(tab.rho_km, rho, rtol=6e-7)
    # phi = exp(-tau) * phi_ref
    assert np.allclose(tab.phi, np.exp(-tab.tau_obs) * tab.phi_ref, rtol=3e-6)
    # s_i = dt (s_b + s_* phi)
    s = dp.delta_t * (dp.s_background + dp.s_star * tab.phi)
    assert np.allclose(tab.s_i, s, atol=2.5e-4)
    # lambda_g and r columns share one lambda_g0 per order [3.9]
    for order, grp in tab.groupby("order"):
        implied = grp.lambda_g * (grp.r_km / R0_PAPER) ** (1.0 + b)
        assert implied.max() / implied.min() - 1 < 2e-5, f"order {order}"


# ---------------------------------------------------------------------------
# 1b. Data -> atmospheric conversion vs p. 1004
# ---------------------------------------------------------------------------


def test_conversion_matches_p1004():
    """Orders 0/1: all 7 printed digits. Order 4: documented misprint in
    the b=0 lambda_g0 entry (see ey92-paper/README.md), wider bounds."""
    ref = _paper_table("p1004_derived.csv")
    for row in ref.itertuples():
        ap = EY92.data_to_atmospheric(
            paper_data_params(row.b), r0=R0_PAPER, order=row.order
        )
        got = (ap.nu0 * 1e9, ap.lambda_g0, ap.r1, ap.kappa1 * 1e3, ap.h_tau1)
        exp = (row.nu0_1e9, row.lambda_g0, row.r1_km, row.kappa1_1e8_cm, row.h_tau1_km)
        rel = 5e-7 if row.order in (0, 1) else 5e-5
        for g, e, col in zip(got, exp, ("nu0", "lambda_g0", "r1", "kappa1", "h_tau1")):
            assert g == pytest.approx(e, rel=rel), (
                f"b={row.b} order={row.order} {col}"
            )


# ---------------------------------------------------------------------------
# 1c. Full pipeline vs Tables 3a/3b (per-column, per-order tolerances)
# ---------------------------------------------------------------------------

# ~2x the measured maxima over both tables; the nu/tau columns admit more
# because 7-digit printing of lambda_g0/r amplifies exponentially into nu(r).
_TOL = {
    0: dict(r_abs=1e-3, lg=9e-7, nu=2e-5, kappa=4e-5, tau=8e-5, pc=5e-7, pr=5e-7, phi=5e-7),
    1: dict(r_abs=1e-3, lg=9e-7, nu=2e-5, kappa=4e-5, tau=8e-5, pc=5e-7, pr=5e-7, phi=5e-7),
    4: dict(r_abs=2e-3, lg=3e-6, nu=4e-5, kappa=8e-5, tau=2e-4, pc=5e-6, pr=5e-6, phi=3e-5),
}


@pytest.mark.parametrize("name,b", [("table3a.csv", 0.0), ("table3b.csv", -0.6)])
@pytest.mark.parametrize("order", [0, 1, 4])
def test_paper_benchmark_pipeline(name, b, order):
    tab = _paper_table(name)
    tab = tab[tab.order == order]
    dp = paper_data_params(b)
    ap = EY92.data_to_atmospheric(dp, r0=R0_PAPER, order=order)
    tol = _TOL[order]
    args = (ap.d, ap.r0, ap.nu0, ap.lambda_g0, ap.a, ap.b, order)

    # Exact rho from the Eq. (5.1) geometry, not the 7-digit printed
    # column (whose rounding would be amplified ~3x into the mid-event
    # flux columns; test_paper_tables_internal_consistency ties the
    # printed column to this geometry).
    rho = np.hypot(dp.rho_min, dp.v * (tab.time_s.to_numpy() - dp.t_mid))
    r = EY92.r_of_rho(rho, *args)
    assert np.allclose(r, tab.r_km, atol=tol["r_abs"])

    # physical columns at the *printed* radius isolate each relation
    rp = tab.r_km.to_numpy()
    assert np.allclose(
        EY92.lambda_g(rp, ap.r0, ap.lambda_g0, ap.a, ap.b), tab.lambda_g, rtol=tol["lg"]
    )
    assert np.allclose(
        EY92.refractivity_profile(rp, ap.r0, ap.nu0, ap.lambda_g0, ap.a, ap.b) * 1e9,
        tab.nu_1e9,
        rtol=tol["nu"],
    )
    kappa = EY92.haze_absorption(rp, ap.r1, ap.kappa1, ap.h_tau1) * 1e3
    assert np.allclose(kappa, tab.kappa_1e8_cm, rtol=tol["kappa"], atol=1e-12)
    tau = EY92.tau_obs(rp, ap.r1, ap.kappa1, ap.h_tau1, order)
    assert np.allclose(tau, tab.tau_obs, rtol=tol["tau"], atol=1e-12)

    # flux columns at the model's own r (end-to-end quantities)
    assert np.allclose(EY92.phi_cyl(r, *args), tab.phi_cyl, rtol=tol["pc"])
    assert np.allclose(EY92.phi_ref(r, *args), tab.phi_ref, rtol=tol["pr"])
    full = EY92.phi(r, ap.d, ap.r0, ap.nu0, ap.lambda_g0, ap.a, ap.b,
                    ap.r1, ap.kappa1, ap.h_tau1, order)
    assert np.allclose(full, tab.phi, rtol=tol["phi"])


# ---------------------------------------------------------------------------
# 2. Series identities (pin the published-Appendix misprints)
# ---------------------------------------------------------------------------

_AB_CASES = [(0.0, 0.0), (0.0, -0.6), (1.0, 0.0), (0.0, 1.0), (0.7, -1.3), (2.0, 3.0)]


def _identity_rhs(a, b, a_c, k):
    """delta^k coefficient of (1 + (1+a+3b)/2 d) A - (1+a+b) d^2 dA/dd."""
    return a_c[k] + ((1 + a + 3 * b) / 2 - (1 + a + b) * (k - 1)) * a_c[k - 1]


@pytest.mark.parametrize("a,b", _AB_CASES)
def test_corrected_series_satisfy_derivative_identity(a, b):
    a_c = EY92._a_coefficients(a, b, "corrected")
    b_c = EY92._b_coefficients(a, b, "corrected")
    for k in range(1, EY92.MAX_ORDER + 1):
        assert b_c[k] == pytest.approx(_identity_rhs(a, b, a_c, k), rel=1e-12)


def test_as_printed_series_fail_identity_only_at_delta4_b_terms():
    for a, b in _AB_CASES:
        a_c = EY92._a_coefficients(a, b, "as-printed")
        b_c = EY92._b_coefficients(a, b, "as-printed")
        for k in range(1, EY92.MAX_ORDER):
            assert b_c[k] == pytest.approx(_identity_rhs(a, b, a_c, k), rel=1e-12)
        if b == 0.0:
            assert b_c[4] == pytest.approx(_identity_rhs(a, b, a_c, 4), rel=1e-12)
        else:
            assert b_c[4] != pytest.approx(_identity_rhs(a, b, a_c, 4), rel=1e-9)


def test_first_order_dtheta_coefficient_matches_truncation_budget():
    """The (9 - 34b + 25b^2)/128 budget quoted in test_benchmarks.py is
    the delta^2 B coefficient at a = 0 — the first term the order-1
    reference generator drops."""
    b = -4.5
    assert EY92._b_coefficients(0.0, b)[2] == pytest.approx(
        (9 - 34 * b + 25 * b**2) / 128
    )


# ---------------------------------------------------------------------------
# 3. Repo reference light curves (Mathematica jleGroup olcOneLimb2)
# ---------------------------------------------------------------------------

# geometry shared by the bundled cases (see tests/data/<case>/parameters.csv)
B_KM = 900.0
V_KMS = 25.0
D_KM = 30 * physicalData.AU_KM
MASS_KG = 2.1398e22
CASE_EXPONENT = {"iso-clear": 0.0, "shallow-clear": -0.5, "steep-clear": -4.5}


def _mfloat(s):
    """Parse Mathematica-formatted floats like 1.086*10^22."""
    return float(str(s).replace("*10^", "e").replace("*^", "e"))


def _ey92_model_for_case(case, order):
    d = os.path.join(DATA, case)
    atm = pd.read_csv(os.path.join(d, "atmosphere.csv"))
    atm.columns = [c.strip() for c in atm.columns]
    radius = atm["Radius (km)"].apply(_mfloat).to_numpy()
    temp = atm["Temperature (K)"].apply(_mfloat).to_numpy()
    press = atm["Pressure (microbar)"].apply(_mfloat).to_numpy() * 0.1  # ubar -> Pa
    lc = pd.read_csv(os.path.join(d, "lightcurve.csv"))
    t = lc["Time (seconds)"].to_numpy(dtype=float)
    ref = lc["Flux (normalized)"].to_numpy(dtype=float)
    # Any table row works as the reference level (the analytic profile is
    # the exact hydrostatic solution; closed form matches the table to
    # ~1e-7, its print precision). Use the middle row for conditioning.
    k = len(radius) // 2
    position = np.sqrt(B_KM**2 + (V_KMS * t) ** 2)
    model = EY92.ElliotYoung1992Model(
        referencePressure=press[k],
        referenceTemperature=temp[k],
        referenceRadius=radius[k],
        planetMass=MASS_KG,
        meanMolecularMass=physicalData.MOLAR_MASS["N2"],
        planetDistance=D_KM,
        position=position,
        temperatureExponent=CASE_EXPONENT[case],
        seriesOrder=order,
    )
    return model, ref


@pytest.mark.parametrize("case", ["iso-clear", "shallow-clear", "steep-clear"])
def test_reference_curves_at_generator_order(case):
    """seriesOrder=1 matches olcEYorderForOneOverLambda=1: measured
    max|res| 9.1e-9 / 1.0e-8 / 2.5e-8 (2026-07-06); tolerance has ~20x
    headroom. Note steep-clear needs no truncation allowance at this
    order — its ~1e-3 reference deviation is *entirely* the generator's
    first-order truncation, reproduced here by construction."""
    model, ref = _ey92_model_for_case(case, order=1)
    flux = model.main()
    maxres = float(np.max(np.abs(flux - ref)))
    print(f"EY92 order1 {case}: max|residual| = {maxres:.3e}")
    assert maxres < 5e-7


@pytest.mark.parametrize(
    "case,tol",
    [
        ("iso-clear", 1.0e-4),
        ("shallow-clear", 1.5e-4),
        ("steep-clear", 1.3e-3),  # reference truncation; see module docstring
    ],
)
def test_reference_curves_at_order4_within_truncation_budget(case, tol):
    """Documentary: order-4 EY92 differs from the first-order reference
    by the O(lambda^-2) budget — the same residuals CE97 shows
    (4.24e-5 / 9.72e-5 / 1.12e-3), confirming EY92(order 4) ~ CE97."""
    model, ref = _ey92_model_for_case(case, order=4)
    flux = model.main()
    maxres = float(np.max(np.abs(flux - ref)))
    print(f"EY92 order4 {case}: max|residual| = {maxres:.3e} (tol {tol:g})")
    assert maxres < tol
