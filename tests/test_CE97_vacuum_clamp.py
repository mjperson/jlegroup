"""The above-atmosphere vacuum clamp (v0.11.0).

Positions beyond yTop = r_top + D*theta(r_top) correspond to rays that
passed above the tabulated atmosphere: the model must return flux = 1 and
theta = 0 exactly (they were previously y->r spline extrapolation — the
source of the spurious caustic-like spikes documented since the first
CE97 validation run). Below yTop nothing may change; the deep end is
deliberately not clamped.

The test atmosphere is the transparent analytic profile of the two-limb
cross-validation (vintage-free: built from (nu0, lambda_g0) directly), on a
TALL table — ~16 scale heights above half-light, per the standing "extend
well above flux recovery" advice. (On short tables the theta integral at
r_top itself samples the d1-refractivity spline beyond the table, so the
map's top boundary is polluted km-scale: that is the documented truncated-
top edge artifact, cured by table construction, not by the clamp.)
"""
import numpy as np
import pytest

from jlegroup import CE97, EY92

R0, LG0, NU0, D = 1250.0, 21.0, 1.11e-9, 4.323e9
R_TOP = 2200.0


def clamp_model(position, **kw):
    radius = np.arange(950.0, R_TOP + 1.0, 1.0)  # top row exactly R_TOP
    nu = EY92.refractivity_profile(radius, R0, NU0, LG0, 0.0, 0.0)
    m = CE97.ChamberlainElliot1997Model(nu, radius, D, np.asarray(position), **kw)
    m.main()
    return m


POSITIONS = np.arange(1000.0, 2400.0, 10.0)  # spans well past yTop ~ R_TOP


def test_ytop_matches_documented_definition():
    m = clamp_model(POSITIONS)
    x0 = np.arange(0, m.radialDistance[-1], m.integrationBin)
    y_top_manual = m.radialDistance[-1] + D * 2 * np.trapezoid(
        m.integrandTheta(x0, m.radialDistance[-1]), x0)
    assert m.yTop == pytest.approx(y_top_manual, abs=1e-9)
    # on a tall table the bending at r_top is negligible: yTop ~ r_top
    assert abs(m.yTop - R_TOP) < 0.5


def test_vacuum_above_ytop_is_exact():
    m = clamp_model(POSITIONS)
    above = POSITIONS > m.yTop
    assert above.any() and (~above).any()
    assert np.all(m.focusedFlux[above] == 1.0)
    assert np.all(m.unfocusedFlux[above] == 1.0)
    assert np.all(m.theta[above] == 0.0)
    assert np.all(m.dtheta[above] == 0.0)


def test_no_extrapolation_spikes():
    """The pre-clamp failure mode was flux of order tens beyond the table
    (spline extrapolation). What legitimately remains below yTop is the
    documented truncated-top spline EDGE artifact, sub-1e-3 on this table
    — the clamp's contract is the extrapolation region only."""
    m = clamp_model(POSITIONS)
    assert np.max(m.focusedFlux) <= 1.001
    assert np.all(m.focusedFlux[POSITIONS > m.yTop] == 1.0)


def test_below_ytop_untouched_and_physical():
    m = clamp_model(POSITIONS)
    # away from the table-top edge zone (~2 local scale heights ~ 370 km
    # here), refraction-only near-limb flux is strictly < 1
    h_top = R_TOP / (LG0 * R0 / R_TOP)          # local H = r/lambda(r)
    clean = POSITIONS < m.yTop - 2.5 * h_top
    assert clean.sum() > 50
    assert np.all(m.focusedFlux[clean] < 1.0)
    assert np.all(m.focusedFlux[clean] > 0.0)
    # inside the edge zone the artifact stays at the documented <~1e-3 level
    edge = (POSITIONS >= m.yTop - 2.5 * h_top) & (POSITIONS <= m.yTop)
    assert np.all(np.abs(m.focusedFlux[edge] - 1.0) < 1e-3)
    # cross-check the deep end against the analytic model (same profile)
    deep = POSITIONS < 1500.0
    phi = EY92.phi_ref(EY92.r_of_rho(POSITIONS[deep], D, R0, NU0, LG0, 0.0, 0.0),
                       D, R0, NU0, LG0, 0.0, 0.0)
    # documented CE97 <-> EY92(order 4) cross-method budget is <= 1.6e-5;
    # measured here 9.5e-6
    assert m.focusedFlux[deep] == pytest.approx(phi, abs=2e-5)


def test_manual_clamp_idiom_is_now_a_noop():
    """Backward compatibility: code still applying the documented user-side
    clamp gets bit-identical results."""
    m = clamp_model(POSITIONS)
    f = m.focusedFlux.copy()
    f[POSITIONS > m.yTop] = 1.0
    assert np.array_equal(f, m.focusedFlux)


def test_noise_applies_over_the_clamped_baseline():
    """addNoise runs after the clamp: the vacuum region carries baseline
    noise around 1, like real data."""
    np.random.seed(3)
    m = clamp_model(POSITIONS, snrPerScaleHeight=50.0, scaleHeight=R0 / LG0,
                    observerPlaneSampling=10.0)
    above = POSITIONS > m.yTop
    noisy = np.asarray(m.fluxWithNoiseAdded)[above]
    assert not np.all(noisy == 1.0)
    assert noisy.mean() == pytest.approx(1.0, abs=0.02)


def test_negative_positions_unaffected():
    """Far-limb evaluation at negative positions (the two-limb cross-check
    path) must not trip the clamp."""
    rho = np.array([-300.0, -150.0, 150.0, 300.0])
    m = clamp_model(rho)
    assert np.all(m.focusedFlux > 0.0)
    assert np.all(m.focusedFlux < 1.0)   # all are deep rays, never vacuum
