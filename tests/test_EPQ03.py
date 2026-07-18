"""EPQ03 inversion benchmarks: the paper is the oracle (EPQ03 Secs. 4-6).

Consolidated from the dev original's suite (Inversion/tests/, 65 tests).
Tolerances encode measured residuals with modest headroom; the printed
targets are EPQ03 Tables 1/3, Figs. 6/9/10, and Sec. 6.3's Monte-Carlo
agreement claim.  Runtime ~10 s (the Monte-Carlo test dominates).
"""

import csv

import numpy as np
import pytest

from jlegroup import EY92
from jlegroup.EPQ03 import (
    CODATA1986,
    EPQ03_TABLE10,
    GASES,
    STANDARD_CASE,
    STANDARD_DELTA_Y,
    EY92PowerLawBoundary,
    add_noise,
    average_until_positive,
    bin_shells,
    bin_uniform,
    boundary_index,
    boundary_integrals,
    fit_boundary,
    generate_light_curve,
    initial_guess,
    invert_light_curve,
    lambda_from_mass,
    mass_from_lambda,
    model_flux,
    propagate_errors,
    radius_scale,
    ray_crossing_free,
    result_table,
    shadow_radius,
    shell_edges,
    sigma_from_snr_h,
    snr_h_from_sigma,
    split_immersion_emersion,
    standard_case_variant,
    write_csv,
)

ATM = STANDARD_CASE


# ---------------------------------------------------------------------------
# shared helpers (dev original: tests/conftest.py)
# ---------------------------------------------------------------------------


def standard_true_boundary():
    """Boundary condition at the standard case's exact parameters."""
    return EY92PowerLawBoundary(
        r_h=ATM.r_h, lambda_hi=ATM.lambda_hi, b=ATM.b, d=ATM.d, order=ATM.order
    )


def noisy_result(snr_h=100.0, seed=0, use_fit=True, y_bottom=440.0,
                 fit_params=("r_h", "lambda_hi")):
    """Noisy standard case.  The boundary position i_b is fixed from the
    noiseless curve (the analyst's deliberate choice, Sec. 7.3), so noise
    spikes near flux 0.5 don't move the boundary between samples.  The
    default boundary fit is isothermal (b fixed at 0), matching the
    isothermal test atmosphere (cf. the paper's Fig. 16 usage)."""
    y, phi = generate_light_curve(ATM, 1600.0, y_bottom)
    i_b = boundary_index(phi, 0.5)
    sigma = sigma_from_snr_h(snr_h, ATM.h_h, 0.5)
    rng = np.random.default_rng(seed)
    noisy, sig = add_noise(phi, sigma, rng)
    return invert_light_curve(
        y, noisy, sig, d=ATM.d, gas=ATM.gas, m_p=ATM.mass(CODATA1986), constants=CODATA1986,
        boundary=None if use_fit else standard_true_boundary(),
        order=ATM.order, i_b=i_b, fit_params=fit_params,
    )


#############################################################################
# test_constants (dev original: tests/test_constants.py)
#############################################################################
# Constants and gas data: pins against EPQ03 Tables 1/9/10/11.

def test_standard_case_mass_pins_table1():
    """Eq. (60) with EPQ03 Table 10 constants reproduces Table 1's derived
    mass 1.70833e22 kg for (lambda_h, r_h, T_h, mu) = (40, 1200, 80, 28.01)."""
    m = mass_from_lambda(40.0, 1200.0, 80.0, 28.01, EPQ03_TABLE10)
    assert m == pytest.approx(1.70833e22, rel=5e-6)


def test_mass_lambda_round_trip():
    m = mass_from_lambda(40.0, 1200.0, 80.0, 28.01, CODATA1986)
    assert lambda_from_mass(m, 1200.0, 80.0, 28.01, CODATA1986) == pytest.approx(
        40.0, rel=1e-14
    )


def test_constant_vintages_differ_only_slightly():
    """CODATA-1986 vs EPQ03 Table 10: G differs by ~9e-5 relative — the scale
    of temperature shifts one must expect when mixing constant sets."""
    rel = abs(CODATA1986.gravitational - EPQ03_TABLE10.gravitational) / (
        CODATA1986.gravitational
    )
    assert 5e-5 < rel < 2e-4


def test_loschmidt_exponent_erratum():
    """EPQ03 Table 10 prints L = 2.68684e24 m^-3.  Backing L out of the
    paper's own Pluto inversion rows (Table 11: n = L nu / nu_STP with
    nu_STP = 2.98e-4) shows the exponent must be e25:

        y = 1188.9 km: nu = 1.05e-9, n = 0.95e14 cm^-3
        y =  900.4 km: nu = 2.79e-9, n = 2.52e14 cm^-3
    """
    nu_stp = GASES["N2"].nu_stp
    for nu, n_cm3 in [(1.05e-9, 0.95e14), (2.79e-9, 2.52e14)]:
        loschmidt = n_cm3 * 1e6 * nu_stp / nu  # m^-3
        # consistent with the corrected exponent (print-precision rounding)...
        assert loschmidt == pytest.approx(EPQ03_TABLE10.loschmidt, rel=0.02)
        # ...and inconsistent (10x) with the exponent as printed.
        assert loschmidt / 2.68684e24 > 9.0
    # the corrected preset agrees with the CODATA-1986 value to ~1e-5
    assert EPQ03_TABLE10.loschmidt == pytest.approx(CODATA1986.loschmidt, rel=1e-4)


def test_gas_presets():
    """EY92 Table 9 pairs (mu amu, nu_STP)."""
    assert GASES["N2"].mu == 28.01 and GASES["N2"].nu_stp == 2.980e-4
    assert GASES["CH4"].mu == 16.04 and GASES["CH4"].nu_stp == 4.401e-4
    assert GASES["CO"].mu == 28.01 and GASES["CO"].nu_stp == 3.364e-4


def test_amu_consistent_with_avogadro():
    assert CODATA1986.amu == pytest.approx(1.6605402e-27, rel=1e-7)
    assert EPQ03_TABLE10.amu == 1.66030e-27


#############################################################################
# test_synth (dev original: tests/test_synth.py)
#############################################################################
# Synthetic light-curve generator: standard-case properties (EPQ03 Table 1, Sec. 6).

def half_light_y(atm):
    """Shadow radius of the half-light level: y_h = r_h (1 + D theta_h/r_h)."""
    from jlegroup import EY92

    y_ratio = EY92.D_theta_over_r_at_flux_level(
        0.5, atm.lambda_h, 0.0, atm.b, atm.order, atm.variant
    )
    return atm.r_h * (1.0 + y_ratio)


def test_standard_case_table1_values():
    atm = STANDARD_CASE
    assert atm.h_h == pytest.approx(30.0)
    assert atm.mass(EPQ03_TABLE10) == pytest.approx(1.70833e22, rel=5e-6)
    assert atm.lambda_hi == atm.lambda_h  # b = 0


def test_half_light_definition():
    """phi = 0.5 exactly at r = r_h (refractive flux, focusing included)."""
    atm = STANDARD_CASE
    y_h = half_light_y(atm)
    phi, r = model_flux(atm, y_h)
    assert float(phi) == pytest.approx(0.5, abs=1e-10)
    assert float(r) == pytest.approx(atm.r_h, abs=1e-6)


def test_half_light_definition_nonisothermal():
    atm = standard_case_variant(b=3.0)
    assert atm.lambda_hi == pytest.approx(47.5)  # Eq. 62
    y_h = half_light_y(atm)
    phi, r = model_flux(atm, y_h)
    assert float(phi) == pytest.approx(0.5, abs=1e-10)
    assert float(r) == pytest.approx(atm.r_h, abs=1e-6)


def test_generate_light_curve_grid_and_limits():
    atm = STANDARD_CASE
    y, phi = generate_light_curve(atm, y_top=1600.0, y_bottom=1100.0)
    assert y[0] == 1600.0 and y[-1] == pytest.approx(1100.0)
    assert np.all(np.diff(y) == pytest.approx(-STANDARD_DELTA_Y))
    # descending y, flux from ~1 down through half light
    assert phi[0] > 0.9999
    assert np.all(np.diff(phi) < 0.0)  # monotonic well above the focus minimum
    y_h = half_light_y(atm)
    idx = np.argmin(np.abs(y - y_h))
    assert phi[idx] == pytest.approx(0.5, abs=2e-3)  # grid point nearest y_h


def test_snr_h_relation():
    """Eq. 99 for the standard case: (S/N)_H = 100 -> sigma = sqrt(60)/100."""
    sigma = sigma_from_snr_h(100.0, 30.0, 0.5)
    assert sigma == pytest.approx(np.sqrt(60.0) / 100.0)
    assert snr_h_from_sigma(sigma, 30.0, 0.5) == pytest.approx(100.0)


def test_add_noise_statistics():
    rng = np.random.default_rng(42)
    phi = np.full(200_000, 0.5)
    noisy, sig = add_noise(phi, 0.01, rng)
    assert sig.shape == phi.shape and np.all(sig == 0.01)
    assert np.mean(noisy - phi) == pytest.approx(0.0, abs=1e-4)
    assert np.std(noisy - phi) == pytest.approx(0.01, rel=1e-2)


def test_isothermal_has_no_ray_crossing():
    """Eq. 100: for an isothermal atmosphere dtheta/dr >= 0 > -1/D everywhere."""
    atm = STANDARD_CASE
    r = np.linspace(1110.0, 1400.0, 50)
    assert np.all(ray_crossing_free(atm, r))
    assert np.all(atm.dtheta_dr_of_radius(r) > 0.0)


def test_flux_focusing_minimum_exists():
    """Small-planet curves pass through a flux minimum (EY92 Fig. 4); for the
    standard case it lies near y ~ 550 km, phi ~ 0.088."""
    atm = STANDARD_CASE
    y = np.arange(700.0, 400.0, -1.0)
    phi, _ = model_flux(atm, y)
    i = np.argmin(phi)
    assert 0 < i < len(y) - 1
    assert phi[i] == pytest.approx(0.088, abs=0.005)


#############################################################################
# test_grid (dev original: tests/test_grid.py)
#############################################################################
# Geometry, shell construction, averaging, and the radius scale (EPQ03 Eqs. 33, 43-48).

# ---------------------------------------------------------------------------
# geometry
# ---------------------------------------------------------------------------


def test_shadow_radius_chord():
    y = shadow_radius([90.0, 100.0, 110.0], rho_min=300.0, v=20.0, t_mid=100.0)
    assert y[1] == 300.0
    assert y[0] == y[2] == pytest.approx(np.hypot(300.0, 200.0))


def test_split_immersion_emersion_orders_y_descending():
    t = np.arange(0.0, 10.0)
    y = shadow_radius(t, rho_min=100.0, v=30.0, t_mid=4.5)
    (t_im, y_im), (t_em, y_em) = split_immersion_emersion(t, y, t_mid=4.5)
    assert np.all(np.diff(y_im) < 0) and np.all(np.diff(y_em) < 0)
    assert set(t_im) | set(t_em) == set(t)


# ---------------------------------------------------------------------------
# shell construction (Eq. 33)
# ---------------------------------------------------------------------------


def test_shell_edges_midpoints_and_boundary():
    y = np.array([10.0, 8.0, 7.0, 5.0, 2.0])
    y_b, y_mid, delta_y, y_lower = shell_edges(y, i_b=2)
    assert y_b == 7.5                       # midpoint across the divide
    np.testing.assert_allclose(y_mid, [7.0, 5.0])   # deepest point dropped
    np.testing.assert_allclose(y_lower, [6.0, 3.5])  # midpoints below
    np.testing.assert_allclose(delta_y, [-1.5, -2.5])
    assert np.all(delta_y < 0)


def test_boundary_index_first_at_or_below_level():
    flux = np.array([0.9, 0.8, 0.55, 0.45, 0.3])
    assert boundary_index(flux, 0.5) == 3
    with pytest.raises(ValueError):
        boundary_index(np.array([0.4, 0.3]), 0.5)  # no boundary region above


# ---------------------------------------------------------------------------
# averaging (paragraph after Eq. 32)
# ---------------------------------------------------------------------------


def test_bin_uniform():
    y = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0])
    flux = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    sigma = np.full(7, 0.3)
    yb, fb, sb = bin_uniform(y, flux, sigma, 3)
    np.testing.assert_allclose(yb, [9.0, 6.0])       # midpoints; remainder dropped
    np.testing.assert_allclose(fb, [2.0, 5.0])
    np.testing.assert_allclose(sb, 0.3 / np.sqrt(3.0))


def test_average_until_positive():
    y = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    flux = np.array([0.5, -0.2, 0.4, 0.3, -0.5, 0.1])
    sigma = np.full(6, 0.1)
    yb, fb, sb = average_until_positive(y, flux, sigma)
    assert np.all(fb > 0)
    # group structure: [0], [1,2], [3], then [4,5] sums to -0.4 -> dropped
    np.testing.assert_allclose(yb, [10.0, 8.5, 7.0])
    np.testing.assert_allclose(fb, [0.5, 0.1, 0.3])
    np.testing.assert_allclose(sb, [0.1, 0.1 * np.sqrt(2) / 2, 0.1])


def test_average_until_positive_all_positive_is_identity():
    y = np.array([3.0, 2.0, 1.0])
    flux = np.array([0.5, 0.4, 0.3])
    sigma = np.array([0.1, 0.2, 0.3])
    yb, fb, sb = average_until_positive(y, flux, sigma)
    np.testing.assert_array_equal(yb, y)
    np.testing.assert_array_equal(fb, flux)
    np.testing.assert_array_equal(sb, sigma)


# ---------------------------------------------------------------------------
# radius scale (Eqs. 43-48) against the analytic forward model
# ---------------------------------------------------------------------------


def standard_grid(y_top=1600.0, y_bottom=600.0, i_b=None):
    atm = STANDARD_CASE
    y, phi = generate_light_curve(atm, y_top, y_bottom)
    sigma = np.zeros_like(phi)
    if i_b is None:
        i_b = boundary_index(phi, 0.5)
    y_b, y_mid, delta_y, _ = shell_edges(y, i_b)
    r_b = float(atm.radius_of_y(y_b))
    grid = radius_scale(r_b, y_b, y_mid, delta_y, phi[i_b:-1], sigma[i_b:-1], atm.d)
    return atm, grid


def test_radius_scale_matches_forward_model():
    """Eq. (44)'s flux sums must reproduce the model's r(y) relation.  With
    0.5 km sampling the midpoint-rule error stays at the metre level over
    ~2.5 scale heights of depth."""
    atm, grid = standard_grid()
    r_model = atm.radius_of_y(grid.y_lower)
    err = np.abs(grid.r_lower - r_model)
    assert err.max() < 2e-3  # km
    assert np.all(np.diff(grid.r_lower) < 0)  # radii strictly decreasing
    assert np.all(grid.delta_r < 0) and np.all(grid.delta_theta < 0)


def test_radius_scale_theta_identities():
    """theta_{i+1/2} = (y_{i+1/2} - r_{i+1/2})/D  [Eq. 47] and
    Delta theta_i = (Delta y_i - Delta r_i)/D  [Eq. 48], and theta matches
    the model's bending angle at the recovered radii."""
    atm, grid = standard_grid()
    np.testing.assert_allclose(
        grid.theta_lower, (grid.y_lower - grid.r_lower) / grid.d, rtol=1e-12
    )
    edges = np.concatenate(([grid.theta_b], grid.theta_lower))
    np.testing.assert_allclose(grid.delta_theta, np.diff(edges), rtol=1e-9)
    theta_model = atm.theta_of_radius(grid.r_lower)
    assert np.max(np.abs(grid.theta_lower - theta_model)) < 5e-10  # rad
    assert grid.theta_b < 0 and np.all(grid.theta_lower < 0)


def test_radius_scale_rejects_overrun():
    """Integrating past the shadow center must raise, not return NaN."""
    y_mid = np.array([100.0, 60.0])
    delta_y = np.array([-40.0, -40.0])
    flux = np.ones(2)
    with pytest.raises(ValueError, match="radicand"):
        radius_scale(80.0, 120.0, y_mid, delta_y, flux, np.zeros(2), 1e9)


# ---------------------------------------------------------------------------
# shell binning (Sec. 3.2, paragraph after Eq. 48)
# ---------------------------------------------------------------------------


def test_bin_shells_thickness_and_consistency():
    atm, grid = standard_grid()
    y_mid, delta_y, flux, sigma = bin_shells(
        grid.y_mid, grid.delta_y, grid.flux, grid.sigma, grid.delta_r,
        min_shell=1.0,
    )
    # merged widths are the sums of member widths
    assert delta_y.sum() == pytest.approx(
        grid.delta_y.sum(), abs=abs(grid.delta_y[-1]) * 60
    )
    # recomputing the radius scale from the binned data keeps every shell
    # at (or just above) the requested thickness, bar midpoint-rule slack
    binned = radius_scale(grid.r_b, grid.y_b, y_mid, delta_y, flux, sigma, atm.d)
    assert np.all(-binned.delta_r > 0.98)
    # and the binned radii still track the model
    r_model = atm.radius_of_y(binned.y_lower)
    assert np.max(np.abs(binned.r_lower - r_model)) < 5e-3


def test_bin_shells_standard_case_shell_count():
    """The Table 1 configuration (0.5 km sampling, 1.0 km shells, boundary
    at flux 0.5) yields shells ~1 km thick spanning ~2.5 scale heights —
    the regime of the paper's 80-shell standard case."""
    atm, grid = standard_grid()
    y_mid, delta_y, flux, sigma = bin_shells(
        grid.y_mid, grid.delta_y, grid.flux, grid.sigma, grid.delta_r,
        min_shell=1.0,
    )
    binned = radius_scale(grid.r_b, grid.y_b, y_mid, delta_y, flux, sigma, atm.d)
    depth = grid.r_b - binned.r_lower[-1]
    # greedy merging stops at "the minimum amount larger" than 1.0 km
    # (Sec. 5), so mean thickness sits a few percent above the floor
    assert 1.0 < depth / y_mid.size < 1.10
    assert 70 <= y_mid.size <= 90


#############################################################################
# test_boundary_fit (dev original: tests/test_boundary_fit.py)
#############################################################################
# Boundary condition (EPQ03 Sec. 3.3) and the boundary least-squares fit.

def standard_boundary(**kw):
    atm = STANDARD_CASE
    defaults = dict(
        r_h=atm.r_h, lambda_hi=atm.lambda_hi, b=atm.b, d=atm.d, order=atm.order
    )
    defaults.update(kw)
    return EY92PowerLawBoundary(**defaults)


# ---------------------------------------------------------------------------
# boundary condition
# ---------------------------------------------------------------------------


def test_boundary_matches_generator():
    """The boundary condition at the true parameters reproduces the
    generating model exactly (same equations, same series)."""
    atm = STANDARD_CASE
    bc = standard_boundary()
    r = np.array([1200.0, 1230.0, 1300.0, 1450.0])
    np.testing.assert_allclose(bc.theta(r), atm.theta_of_radius(r), rtol=1e-14)
    np.testing.assert_allclose(
        bc.dtheta_dr(r), atm.dtheta_dr_of_radius(r), rtol=1e-14
    )
    assert bc.nu_h == pytest.approx(atm.nu_h, rel=1e-14)
    assert bc.lambda_h == 40.0


def test_r_boundary_solves_eq68():
    """r_b = y_b - D theta_b(r_b)  [Eq. 68]."""
    bc = standard_boundary()
    y_b = 1168.7956
    r_b = bc.r_boundary(y_b)
    assert r_b == pytest.approx(y_b - bc.d * bc.theta(r_b), abs=1e-9)
    assert r_b == pytest.approx(1200.0, abs=0.05)


def test_lambda_hi_conversion():
    """lambda_h = lambda_hi - 5b/2  [Eq. 62]."""
    bc = standard_boundary(lambda_hi=47.5, b=3.0)
    assert bc.lambda_h == 40.0


def test_nu_h_methods():
    """EPQ03 Eq. (66) as printed vs EY92 Eq. (4.28): agree to leading
    order, differ by ~5e-4 relative for the standard case (FIDELITY.md)."""
    ey = standard_boundary(nu_h_method="ey92-4.28").nu_h
    pq = standard_boundary(nu_h_method="epq03-66").nu_h
    assert ey == pytest.approx(pq, rel=2e-3)
    assert abs(ey - pq) / ey > 1e-4


def test_theta_consistency_with_flux():
    """Eq. (5) consistency check the paper prescribes for r_b: the model
    flux and the radius-flux relation agree along the curve."""
    bc = standard_boundary()
    y = np.array([1500.0, 1300.0, 1200.0, 1169.0])
    phi = bc.flux(y)
    assert np.all((phi > 0.4) & (phi < 1.0))
    # r from Eq. 68 at each y, then flux at that r via theta relations
    r = np.array([bc.r_boundary(v) for v in y])
    np.testing.assert_allclose(r + bc.d * bc.theta(r), y, atol=1e-8)


# ---------------------------------------------------------------------------
# boundary fit
# ---------------------------------------------------------------------------


def boundary_region_data():
    """Standard-case data above the inversion boundary (flux >= 0.5)."""
    atm = STANDARD_CASE
    y, phi = generate_light_curve(atm, 1600.0, 1100.0)
    i_b = boundary_index(phi, 0.5)
    return y[:i_b], phi[:i_b]


def test_initial_guess_lands_in_basin():
    y, phi = boundary_region_data()
    r_h0, lambda_hi0 = initial_guess(y, phi)
    assert r_h0 == pytest.approx(1200.0, rel=0.05)
    assert lambda_hi0 == pytest.approx(40.0, rel=0.5)


def test_noiseless_fit_recovers_parameters():
    y, phi = boundary_region_data()
    res = fit_boundary(y, phi, d=STANDARD_CASE.d, fit=("r_h", "lambda_hi", "b"))
    assert res.values["r_h"] == pytest.approx(1200.0, abs=1e-4)
    assert res.values["lambda_hi"] == pytest.approx(40.0, abs=1e-4)
    assert res.values["b"] == pytest.approx(0.0, abs=1e-4)
    assert res.rms < 1e-9
    names, vals, errs, corr = res.atmospheric_subset()
    assert names == ("r_h", "lambda_hi", "b")
    assert corr.shape == (3, 3)
    np.testing.assert_allclose(np.diag(corr), 1.0, rtol=1e-12)
    np.testing.assert_allclose(corr, corr.T, rtol=1e-10)


def test_noiseless_isothermal_fit():
    y, phi = boundary_region_data()
    res = fit_boundary(y, phi, d=STANDARD_CASE.d, fit=("r_h", "lambda_hi"))
    assert res.values["b"] == 0.0  # fixed
    assert res.values["r_h"] == pytest.approx(1200.0, abs=1e-6)
    assert res.values["lambda_hi"] == pytest.approx(40.0, abs=1e-6)


def test_noiseless_nonisothermal_fit():
    atm = standard_case_variant(b=-3.0)
    y, phi = generate_light_curve(atm, 1600.0, 1100.0)
    i_b = boundary_index(phi, 0.5)
    res = fit_boundary(y[:i_b], phi[:i_b], d=atm.d)
    assert res.values["r_h"] == pytest.approx(1200.0, abs=1e-3)
    assert res.values["lambda_hi"] == pytest.approx(atm.lambda_hi, abs=1e-3)
    assert res.values["b"] == pytest.approx(-3.0, abs=1e-3)


def test_noisy_fit_recovers_within_errors():
    """(S/N)_H = 200: recovered parameters within 4 formal sigmas, and the
    formal errors have sensible magnitudes (cf. EPQ03 Table 8's Triton fit
    at (S/N)_H ~ 600-760 with sub-km r_h errors)."""
    atm = STANDARD_CASE
    y, phi = generate_light_curve(atm, 1600.0, 1100.0)
    i_b = boundary_index(phi, 0.5)
    sigma = sigma_from_snr_h(200.0, atm.h_h, 0.5)
    rng = np.random.default_rng(7)
    noisy, sig = add_noise(phi[:i_b], sigma, rng)
    res = fit_boundary(y[:i_b], noisy, sig, d=atm.d)
    for name, truth in [("r_h", 1200.0), ("lambda_hi", 40.0), ("b", 0.0)]:
        assert abs(res.values[name] - truth) < 4.0 * res.errors[name]
    assert 0.1 < res.errors["r_h"] < 10.0
    assert res.chi2 / res.dof == pytest.approx(1.0, abs=0.25)
    corr = res.correlation
    assert np.all(np.abs(corr) <= 1.0 + 1e-12)


#############################################################################
# test_inversion (dev original: tests/test_inversion.py)
#############################################################################
# Inversion acceptance tests against EPQ03 Sec. 6 (Tables 1-3, Figs. 4-6).
#
# The oracle is the paper itself: the noiseless standard test case must
# reproduce Table 3's average temperature (79.997 K), convergence
# temperature (79.998 K), and maximum residual (~0.004-0.005 K), and the
# nonisothermal series must stay within Fig. 6's tolerances (worst case
# b = 9: +0.48%).

def model_boundary(atm, **kw):
    args = dict(r_h=atm.r_h, lambda_hi=atm.lambda_hi, b=atm.b, d=atm.d,
                order=atm.order)
    args.update(kw)
    return EY92PowerLawBoundary(**args)


def run_standard(atm=STANDARD_CASE, y_bottom=440.0, use_fit=False, **kw):
    y, phi = generate_light_curve(atm, 1600.0, y_bottom)
    return invert_light_curve(
        y, phi, d=atm.d, gas=atm.gas, m_p=atm.mass(CODATA1986), constants=CODATA1986,
        boundary=None if use_fit else model_boundary(atm),
        order=atm.order, **kw,
    )


# ---------------------------------------------------------------------------
# closure of the integral pair (Eqs. 15/63): quadrature-level checks
# ---------------------------------------------------------------------------


def test_abel_closure_recovers_model_refractivity():
    """B_nu evaluated with r_b = r is the full Abel inverse [Eq. 15] of the
    series bending angle; it must return the model's nu(r) to within the
    series-truncation consistency (~1e-6 at order 2, ~3e-9 at order 4)."""
    atm = STANDARD_CASE
    for order, tol in [(2, 5e-6), (4, 5e-8)]:
        bc = model_boundary(atm, order=order)
        for r in (1150.0, 1200.0, 1250.0):
            b_nu, _ = boundary_integrals(bc, r, [r])
            assert b_nu[0] == pytest.approx(float(bc.nu_of(r)), rel=tol)


def test_abel_closure_temperature():
    """The B_p/B_nu ratio at r_b = r gives the model temperature  [Eq. 57]."""
    atm = STANDARD_CASE
    gm = (CODATA1986.gravitational * atm.mass(CODATA1986)
          * atm.gas.mu * CODATA1986.amu)
    bc = model_boundary(atm)
    b_nu, b_p = boundary_integrals(bc, 1200.0, [1200.0])
    t = gm / CODATA1986.boltzmann * (b_p[0] * 1e-3) / b_nu[0]
    assert t == pytest.approx(80.0, abs=2e-4)


# ---------------------------------------------------------------------------
# the standard test case (Table 1 -> Table 3, first row)
# ---------------------------------------------------------------------------


def test_standard_case():
    res = run_standard()
    t = res.temperature
    assert t.size == 80                       # Table 1: inversion points
    assert res.i_b == pytest.approx(860, abs=5)  # Table 1: boundary points
    assert t.mean() == pytest.approx(79.997, abs=1.5e-3)   # Table 3 average
    assert t[-1] == pytest.approx(79.998, abs=1.5e-3)      # Table 3 convergence
    assert np.abs(t - 80.0).max() < 6e-3                   # Table 3 max residual
    assert np.abs(t / 80.0 - 1.0).max() < 5e-4             # abstract's bound


def test_standard_case_profile_quantities():
    atm = STANDARD_CASE
    res = run_standard()
    p = res.profiles
    grid = res.grid
    # scale height: H = r/lambda(r) for the isothermal model  [Eq. 31]
    bc = res.boundary
    np.testing.assert_allclose(
        p.scale_height, grid.r_lower / bc.lambda_of(grid.r_lower), rtol=2e-4
    )
    # refractivity against the generating model  [Eq. 65]
    nu_model = bc.nu_of(grid.r_lower)
    np.testing.assert_allclose(p.refractivity, nu_model, rtol=3e-4)
    # number density is a fixed multiple of refractivity  [Eq. 55]
    np.testing.assert_allclose(
        p.number_density,
        CODATA1986.loschmidt / atm.gas.nu_stp * p.refractivity,
        rtol=1e-12,
    )
    # pressure: positive, increasing downward, p = n k T  [Eqs. 56, 30]
    assert np.all(p.pressure > 0) and np.all(np.diff(p.pressure) > 0)
    np.testing.assert_allclose(
        p.pressure,
        p.number_density * CODATA1986.boltzmann * p.temperature,
        rtol=1e-10,
    )
    # microbar display: ~1 ubar near 1e-1 Pa
    np.testing.assert_allclose(p.pressure_microbar, p.pressure * 10.0)


def test_standard_case_with_fitted_boundary():
    """Full pipeline including the boundary least-squares fit (noiseless:
    the fit recovers the generator, so Table 3 must still be reproduced)."""
    res = run_standard(use_fit=True)
    assert res.fit is not None
    assert res.fit.values["r_h"] == pytest.approx(1200.0, abs=1e-3)
    assert res.fit.values["lambda_hi"] == pytest.approx(40.0, abs=1e-3)
    t = res.temperature
    assert t.mean() == pytest.approx(79.997, abs=2e-3)
    assert t[-1] == pytest.approx(79.998, abs=2e-3)
    assert np.abs(t - 80.0).max() < 6e-3


def test_standard_case_isothermal_boundary():
    """Isothermal boundary condition (b fixed at 0) — same result for the
    isothermal test case."""
    res = run_standard(use_fit=True, fit_params=("r_h", "lambda_hi"))
    assert res.fit.values["b"] == 0.0
    assert res.temperature.mean() == pytest.approx(79.997, abs=2e-3)


# ---------------------------------------------------------------------------
# Table 3 sensitivity trials
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "flux_level, avg, conv",
    [(0.3, 79.997, 79.998), (0.6, 79.997, 79.998), (0.8, 79.998, 79.999)],
)
def test_boundary_flux_trials(flux_level, avg, conv):
    """Table 3, 'Boundary radius flux' series 1/4/6: the choice of boundary
    flux level moves the noiseless temperatures by < 0.01% (Fig. 5 top)."""
    res = run_standard(boundary_flux=flux_level)
    t = res.temperature
    assert t.mean() == pytest.approx(avg, abs=2e-3)
    assert t[-1] == pytest.approx(conv, abs=2e-3)
    assert np.abs(t - 80.0).max() < 6e-3


@pytest.mark.parametrize("n_per_bin, max_residual", [(2, 0.012), (6, 0.02)])
def test_data_averaging_fine(n_per_bin, max_residual):
    """Averaging to 1 and 3 km in the observer plane keeps the noiseless
    temperature errors at the 0.01-0.05 K level (Table 3 'Data resolution';
    see FIDELITY.md on the residual implementation sensitivity)."""
    res = run_standard(n_per_bin=n_per_bin)
    t = res.temperature
    assert np.abs(t - 80.0).max() < max_residual


def test_data_averaging_extreme():
    """One point per scale height (30 km bins): Sec. 6.2.2's claim is a
    maximum temperature error under ~0.5% even for this extreme case."""
    res = run_standard(n_per_bin=60)
    t = res.temperature
    assert np.abs(t / 80.0 - 1.0).max() < 7e-3


@pytest.mark.parametrize("min_shell, rel_bound", [(4.0, 1.5e-3), (15.0, 7e-3)])
def test_shell_binning_trials(min_shell, rel_bound):
    """Binning the atmospheric shells degrades gracefully; the paper quotes
    < 0.7% error for shells of half a scale height (15 km)  [Sec. 6.2.2,
    Fig. 5 bottom]."""
    res = run_standard(min_shell=min_shell)
    t = res.temperature
    assert np.abs(t / 80.0 - 1.0).max() < rel_bound


# ---------------------------------------------------------------------------
# nonisothermal test cases (Sec. 6.2.3, Fig. 6)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "b, y_bottom, pct_bound",
    [(-6.0, 440.0, 0.05), (-3.0, 440.0, 0.05), (3.0, 600.0, 0.05),
     (6.0, 800.0, 0.15), (9.0, 950.0, 0.55)],
)
def test_nonisothermal(b, y_bottom, pct_bound):
    """Thermal-gradient inversions with lambda_h held at 40 (Eq. 62): the
    profile must track T = 80 (r/1200)^b within Fig. 6's envelope (worst
    case b = 9 reaches 0.48%; all cases within 0.5%)."""
    atm = standard_case_variant(b=b)
    res = run_standard(atm, y_bottom=y_bottom)
    t = res.temperature
    t_model = 80.0 * (res.grid.r_lower / 1200.0) ** b
    pct = 100.0 * np.abs(t / t_model - 1.0)
    assert pct.max() < pct_bound


def test_nonisothermal_worst_case_pins_fig6():
    """b = 9: the divergence is largest just below the boundary and the
    agreement improves with depth (Sec. 6.2.3); Fig. 6 quotes a maximum
    error of +0.48% in the model-minus-inverted sense (we get +0.483%)."""
    atm = standard_case_variant(b=9.0)
    res = run_standard(atm, y_bottom=950.0)
    t = res.temperature
    t_model = 80.0 * (res.grid.r_lower / 1200.0) ** 9.0
    pct_model_minus_inv = 100.0 * (t_model / t - 1.0)
    assert pct_model_minus_inv.max() == pytest.approx(0.48, abs=0.05)
    # one-sided: the inverted profile runs low of the model...
    assert pct_model_minus_inv.min() > -0.1
    # ...worst at the top, improving with depth
    assert np.argmax(pct_model_minus_inv) == 0
    assert pct_model_minus_inv[-1] < 0.5 * pct_model_minus_inv[0]


#############################################################################
# test_errors (dev original: tests/test_errors.py)
#############################################################################
# Error propagation (EPQ03 Sec. 4) and its Monte-Carlo validation (Sec. 6.3).

# ---------------------------------------------------------------------------
# structure of the two Eq.-72 terms
# ---------------------------------------------------------------------------


def test_noiseless_boundary_uncertainty_only():
    """With noiseless fluxes, only the boundary term contributes."""
    y, phi = generate_light_curve(ATM, 1600.0, 440.0)
    res = invert_light_curve(
        y, phi, d=ATM.d, gas=ATM.gas, m_p=ATM.mass(CODATA1986), constants=CODATA1986,
        boundary=standard_true_boundary(), order=ATM.order,
    )
    budget = propagate_errors(
        res,
        boundary_uncertainty=(("r_h", "lambda_hi"), [1.0, 0.5], np.eye(2)),
    )
    assert budget.sigma_r_b > 0.0
    for q in (budget.refractivity, budget.pressure, budget.temperature,
              budget.scale_height, budget.radius):
        assert np.all(q.summation_only == 0.0)
        assert np.all(q.boundary_only > 0.0)
        np.testing.assert_allclose(q.total, q.boundary_only, rtol=1e-12)
    # 1 km on r_h with sigma(lambda) maps to ~1 km on r_b
    assert budget.sigma_r_b == pytest.approx(1.0, abs=0.6)


def test_noisy_summation_only():
    """With an exactly-known boundary, only the flux term contributes."""
    res = noisy_result(use_fit=False)
    budget = propagate_errors(res)
    assert budget.sigma_r_b == 0.0
    for q in (budget.refractivity, budget.pressure, budget.temperature):
        assert np.all(q.boundary_only == 0.0)
        assert np.all(q.summation_only[1:] > 0.0)
        np.testing.assert_allclose(q.total, q.summation_only, rtol=1e-12)


def test_decomposition_in_quadrature():
    res = noisy_result(use_fit=True)
    budget = propagate_errors(res)
    for q in (budget.radius, budget.refractivity, budget.number_density,
              budget.pressure, budget.temperature, budget.scale_height):
        np.testing.assert_allclose(
            q.total, np.hypot(q.boundary_only, q.summation_only), rtol=1e-10
        )


def test_number_density_error_scales_with_refractivity():
    """sigma(n) = (L/nu_STP) sigma(nu)  [Eq. 81]."""
    res = noisy_result()
    budget = propagate_errors(res)
    np.testing.assert_allclose(
        budget.number_density.total,
        CODATA1986.loschmidt / ATM.gas.nu_stp * budget.refractivity.total,
        rtol=1e-12,
    )


# ---------------------------------------------------------------------------
# radius-scale derivatives: closed form vs stepped  [Eqs. 74-79]
# ---------------------------------------------------------------------------


def test_radius_derivatives_match_closed_form():
    """d r_{i+1/2}/d phi_k = y_k dy_k / r_{i+1/2} for k <= i (0 above), and
    d r_{i+1/2}/d r_b = r_b / r_{i+1/2}  [Eqs. 74-75]."""
    res = noisy_result(use_fit=False)
    g = res.grid
    h = 1e-6
    k = 5
    flux_k = g.flux.copy()
    flux_k[k] += h
    stepped = radius_scale(g.r_b, g.y_b, g.y_mid, g.delta_y, flux_k,
                           g.sigma, g.d)
    deriv = (stepped.r_lower - g.r_lower) / h
    expected = np.where(
        np.arange(g.r_lower.size) >= k, g.y_mid[k] * g.delta_y[k] / g.r_lower, 0.0
    )
    np.testing.assert_allclose(deriv, expected, rtol=1e-4, atol=1e-12)

    stepped_b = radius_scale(g.r_b + h, g.y_b, g.y_mid, g.delta_y, g.flux,
                             g.sigma, g.d)
    deriv_b = (stepped_b.r_lower - g.r_lower) / h
    np.testing.assert_allclose(deriv_b, g.r_b / g.r_lower, rtol=1e-4)


def test_radius_budget_matches_stepped_operator():
    """The closed-form radius errors [Eq. 76] agree with brute stepping of
    the whole radius scale against each flux."""
    res = noisy_result(use_fit=False)
    g = res.grid
    budget = propagate_errors(res)
    var = np.zeros_like(g.r_lower)
    for k in range(g.flux.size):
        h = g.sigma[k] * 0.1
        flux_k = g.flux.copy()
        flux_k[k] += h
        stepped = radius_scale(g.r_b, g.y_b, g.y_mid, g.delta_y, flux_k,
                               g.sigma, g.d)
        var += ((stepped.r_lower - g.r_lower) / h) ** 2 * g.sigma[k] ** 2
    np.testing.assert_allclose(budget.radius.total, np.sqrt(var), rtol=1e-3)


# ---------------------------------------------------------------------------
# Monte-Carlo validation  (Sec. 6.3)
# ---------------------------------------------------------------------------


def test_monte_carlo_temperature_errors():
    """(S/N)_H = 100 standard case: the formal temperature errors must
    match the scatter of independently-inverted noise samples (the paper
    found agreement within 18% for 25 samples), the envelope must grow
    with depth (Fig. 9), and the summation contribution must vanish
    toward the boundary (Fig. 10)."""
    base = noisy_result(seed=0)
    budget = propagate_errors(base)
    sigma_t = budget.temperature.total
    r_base = base.grid.r_lower

    samples = []
    for seed in range(1, 26):
        res = noisy_result(seed=seed)
        r = res.grid.r_lower
        # interpolate this sample's profile onto the base radii (grids
        # differ slightly through the noise-dependent binning)
        t = np.interp(r_base[::-1], r[::-1], res.temperature[::-1])[::-1]
        inside = (r_base <= r[0]) & (r_base >= r[-1])
        samples.append(np.where(inside, t, np.nan))
    scatter = np.nanstd(np.vstack(samples), axis=0, ddof=1)

    ok = np.isfinite(scatter) & (scatter > 0)
    ratio = scatter[ok] / sigma_t[ok]
    assert 0.7 < np.median(ratio) < 1.35
    assert np.mean((ratio > 0.5) & (ratio < 2.0)) > 0.8

    # Fig. 10 morphology: the boundary term dominates at the top and decays
    # downward; the summation term starts near zero and grows, the two
    # crossing ~2.5-3 scale heights below half-light — leaving the total
    # with an interior minimum (Sec. 7.2's "minimum percentage error").
    b = budget.temperature.boundary_only
    s = budget.temperature.summation_only
    assert s[0] < 0.2 * s[-1] and s[0] < b[0]
    assert b[-1] < 0.3 * b[0]
    assert b[-1] < s[-1]
    assert sigma_t.min() < 0.6 * sigma_t[0]


def test_formal_errors_scale_inversely_with_snr():
    """sigma(T) is proportional to the light-curve noise level  [Sec. 6.1,
    Eq. 72]: halving the noise halves the errors."""
    y, phi = generate_light_curve(ATM, 1600.0, 440.0)
    i_b = boundary_index(phi, 0.5)
    rng = np.random.default_rng(3)
    unit = rng.normal(0.0, 1.0, phi.shape)
    out = {}
    for snr in (100.0, 200.0):
        sigma = sigma_from_snr_h(snr, ATM.h_h, 0.5)
        res = invert_light_curve(
            y, phi + sigma * unit, np.full_like(phi, sigma),
            d=ATM.d, gas=ATM.gas, m_p=ATM.mass(CODATA1986), constants=CODATA1986, order=ATM.order,
            i_b=i_b,
        )
        out[snr] = propagate_errors(res)
    t100 = out[100.0].temperature.total
    t200 = out[200.0].temperature.total
    n = min(t100.size, t200.size)
    ratio = np.median(t200[:n] / t100[:n])
    assert ratio == pytest.approx(0.5, abs=0.12)


#############################################################################
# test_export (dev original: tests/test_export.py)
#############################################################################
# CSV export round trip.

def test_write_csv_round_trip(tmp_path):
    res = noisy_result(seed=1)
    budget = propagate_errors(res)
    table = result_table(res, budget)
    path = tmp_path / "out.csv"
    names = write_csv(path, res, budget)
    assert names == list(table)

    with open(path) as f:
        rows = list(csv.reader(f))
    assert rows[0] == names
    data = np.array(rows[1:], dtype=float)
    assert data.shape == (res.temperature.size, len(names))
    col = names.index("temperature_K")
    np.testing.assert_allclose(data[:, col], res.temperature, rtol=1e-12)
    col = names.index("sigma_temperature_K")
    np.testing.assert_allclose(data[:, col], budget.temperature.total,
                               rtol=1e-12)
    # microbar column really is 10x the Pa pressure
    col = names.index("pressure_microbar")
    np.testing.assert_allclose(data[:, col], res.profiles.pressure * 10.0,
                               rtol=1e-12)


#############################################################################
# ratchet binning (concept: W. Saunders)
#############################################################################
# The published positivity averaging lets the noise realization set the
# binning, which biases the deep profile hot and suppresses its scatter;
# the ratchet makes the bin size monotone non-decreasing so binning tracks
# the S/N envelope.  The statistical acceptance values below are measured
# (15 paired seeds, this implementation) with headroom.


def test_ratchet_no_op_on_all_positive_data():
    """With no non-positive fluxes the ratchet never engages: output is
    identical to the published scheme (and to the input)."""
    y = np.array([10.0, 9.0, 8.0, 7.0, 6.0])
    flux = np.array([0.9, 0.7, 0.5, 0.3, 0.1])
    sigma = np.full(5, 0.05)
    plain = average_until_positive(y, flux, sigma)
    ratch = average_until_positive(y, flux, sigma, ratchet=True,
                                   return_counts=True)
    for a, b in zip(plain, ratch[:3]):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(ratch[3], np.ones(5, dtype=int))


def test_ratchet_monotone_levels_and_conservation():
    """Once positivity forces a k-point merge, every later bin has >= k
    points; the published scheme instead resets to single points.  Flux
    sums over consumed points are conserved in both."""
    y = np.arange(20.0, 0.0, -1.0)
    flux = np.array([0.5, 0.4, -0.2, 0.5, 0.3, 0.2, 0.1, 0.2, -0.3, 0.5,
                     0.2, 0.1, 0.3, -0.1, 0.2, 0.1, 0.2, 0.1, -0.2, 0.4])
    sigma = np.full(20, 0.1)

    yb, fb, sb, kb = average_until_positive(y, flux, sigma, ratchet=True,
                                            return_counts=True)
    assert np.all(np.diff(kb) >= 0)          # the ratchet never loosens
    assert kb[0] == 1 and kb.max() >= 2      # engages at the first negative
    consumed = kb.sum()
    assert np.isclose((fb * kb).sum(), flux[:consumed].sum())

    yp, fp, sp, kp = average_until_positive(y, flux, sigma, return_counts=True)
    assert np.any(np.diff(kp) < 0)           # published scheme un-bins again
    assert np.isclose((fp * kp).sum(), flux[:kp.sum()].sum())
    assert np.all(fb > 0) and np.all(fp > 0)


def test_ratchet_trailing_underfull_group_dropped():
    """A trailing group thinner than the current level is dropped even if
    positive (strict resolution guarantee); the published scheme keeps it."""
    y = np.arange(8.0, 0.0, -1.0)
    flux = np.array([0.5, -0.4, 0.2, 0.3, 0.1, 0.2, 0.4, 0.4])
    sigma = np.full(8, 0.1)
    # ratchet: [0](1pt) then [-0.4,0.2,0.3](3pts, level->3), then [0.1,0.2,0.4]
    # (3pts), leaving one positive point 0.4 < level -> dropped
    yb, fb, sb, kb = average_until_positive(y, flux, sigma, ratchet=True,
                                            return_counts=True)
    np.testing.assert_array_equal(kb, [1, 3, 3])
    assert kb.sum() == 7                     # the 8th point was dropped
    yp, fp, sp, kp = average_until_positive(y, flux, sigma, return_counts=True)
    assert kp.sum() == 8                     # published scheme keeps it


def test_ratchet_pipeline_switch_and_bin_counts():
    """invert_light_curve(ratchet_binning=True) reports monotone bin counts;
    the default reproduces the published scheme (counts reset to 1)."""
    y, phi = generate_light_curve(ATM, 1600.0, 440.0)
    i_b = boundary_index(phi, 0.5)
    sigma = sigma_from_snr_h(100.0, ATM.h_h, 0.5)
    noisy, sig = add_noise(phi, sigma, np.random.default_rng(4))
    kw = dict(d=ATM.d, gas=ATM.gas, m_p=ATM.mass(CODATA1986),
              constants=CODATA1986, boundary=standard_true_boundary(),
              order=ATM.order, i_b=i_b)
    res_r = invert_light_curve(y, noisy, sig, ratchet_binning=True, **kw)
    res_p = invert_light_curve(y, noisy, sig, **kw)
    assert res_r.bin_counts.max() > 1
    assert np.all(np.diff(res_r.bin_counts) >= 0)
    after_first = np.argmax(res_p.bin_counts > 1)
    assert np.any(res_p.bin_counts[after_first:] == 1)  # published un-bins
    # noiseless: both are the identity, counts all one
    res_0 = invert_light_curve(y, phi, ratchet_binning=True, **kw)
    assert np.all(res_0.bin_counts == 1)
    np.testing.assert_allclose(
        res_0.temperature.mean(), 79.997, atol=2e-3
    )


def _ratchet_mc_arm(ratchet, seeds=range(12), snr_h=20.0):
    """Deep-profile z = (T - 80)/sigma_T statistics with an exactly-known
    boundary (isolates the flux-summation term).  i_b is chosen by shadow
    radius on the averaged arrays (a flux-level trigger is meaningless at
    this noise level)."""
    y_b_ref = 1168.75  # the noiseless standard-case boundary shell edge
    bc = standard_true_boundary()
    y, phi = generate_light_curve(ATM, 1600.0, 440.0)
    sigma = sigma_from_snr_h(snr_h, ATM.h_h, 0.5)
    deep, shells = [], []
    for seed in seeds:
        noisy, sig = add_noise(phi, sigma, np.random.default_rng(seed))
        ya, fa, sa = average_until_positive(y, noisy, sig, ratchet=ratchet)
        i_b = int(np.argmax(ya <= y_b_ref))
        res = invert_light_curve(
            ya, fa, sa, d=ATM.d, gas=ATM.gas, m_p=ATM.mass(CODATA1986),
            constants=CODATA1986, boundary=bc, order=ATM.order, i_b=i_b,
        )
        budget = propagate_errors(res)
        z = (res.temperature - 80.0) / budget.temperature.total
        deep.append(z[2 * z.size // 3:])
        shells.append(z.size)
    return np.concatenate(deep), np.array(shells)


def test_ratchet_restores_honest_deep_statistics():
    """The feature's claim, measured at (S/N)_H = 20 with paired noise
    seeds.  Published scheme: deep temperatures biased hot by several
    formal sigmas (measured mean z ~ +5.2) with suppressed scatter
    (std ~ 0.84) — the spurious stability ratchet binning was invented
    for.  Ratchet: mean z ~ +0.2, std ~ 1.1 — honest statistics, at the
    price of deep resolution (fewer, coarser shells)."""
    z_pub, shells_pub = _ratchet_mc_arm(ratchet=False)
    z_rat, shells_rat = _ratchet_mc_arm(ratchet=True)

    # the artifact, documented: hot bias and under-dispersion
    assert 3.0 < z_pub.mean() < 8.0
    assert z_pub.std() < 0.95
    # the fix: unbiased at the fraction-of-sigma level, scatter ~ formal
    assert abs(z_rat.mean()) < 0.8
    assert 0.85 < z_rat.std() < 1.35
    # the price: coarser deep resolution
    assert shells_rat.mean() < 0.8 * shells_pub.mean()
