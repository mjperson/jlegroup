# -*- coding: utf-8 -*-
"""EY92 — analytic small-planet occultation light-curve model.

Implementation of

    Elliot, J. L., & Young, L. A. 1992, AJ, 103, 991 ("EY92").
    "Analysis of Stellar Occultation Data for Planetary Atmospheres.
     I. Model Fitting, with Application to Pluto."

Forward model for a small, spherically symmetric planet whose upper
atmosphere follows power-law (or isothermal) temperature and molecular-
weight profiles, optionally over a sharp extinction (haze) layer.  The
small-planet corrections are carried by the Appendix power series
A(delta,a,b), B(delta,a,b), C(delta_r) with expansion parameter
delta = 1/lambda_g = H/r; truncating at order 0 recovers the Baum &
Code (1953) large-planet limit, and order 1 matches the Mathematica
jleGroup occLightCurves default (olcEYorderForOneOverLambda = 1).

Provenance
----------
Newly written from the published equations (not ported from the
Mathematica jleGroup).  Developed and
validated standalone against the EY92 paper benchmark (Tables 2, 3a,
3b: print-precision agreement at all series orders), then integrated
here; see tests/test_EY92.py and tests/data/ey92-paper/.

Errata in the published paper (established during validation; the
"Chamberlain 1996" b-series error noted in the Mathematica jleGroup
usage text is consistent with these):

  * Eq. (A5): the printed factor 1/2 on the dA/d-delta term is
    spurious; B = (1 + (1+a+3b)/2 delta) A - (1+a+b) delta^2 dA/ddelta.
  * Eq. (A2), delta^4 b^3 term: printed /8192 must read /32768.
  * Eq. (A6), delta^4 b^2 term: printed 10555 must read 1055.
  * p. 1004 derived-parameter table, 4th-order lambda_g0 (b = 0):
    printed 20.95567 is inconsistent with the paper's own Table 3a
    (~20.9567).

The series default to the corrected coefficients; pass
variant="as-printed" for the journal text (differences are O(1e-6)
in flux for the Pluto regime, below the paper's print precision).

Units (package convention, matching CE97): radii/distances/positions in
km; pressure Pa; number density m^-3; temperature K; molar mass kg/mol;
refractivity and flux dimensionless (flux normalized to 1 unocculted);
haze linear absorption coefficient kappa in km^-1; bending angle rad.
All physical constants and refractivity conversions come from
jlegroup.physicalData (CODATA-1986 vintage; see that module's warning).

Scope: one near limb, geometric optics (no diffraction), spherical
symmetry, constant-velocity geometry left to the caller (position array
in the observer plane, as in CE97).  Least-squares fitting and the
EY92 Sec. 6 error propagation are not implemented here.

Equation numbers in docstrings refer to EY92.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import physicalData

__all__ = [
    "MAX_ORDER",
    "a_series",
    "b_series",
    "c_series",
    "lambda_g",
    "refractivity_profile",
    "haze_absorption",
    "bending_angle",
    "D_dtheta_dr",
    "tau_obs",
    "kappa1_from_unit_optical_depth",
    "rho_of_r",
    "r_of_rho",
    "r_far_of_rho",
    "phi_cyl",
    "phi_ref",
    "phi",
    "phi_two_limb",
    "twoLimb",
    "D_theta_over_r_at_flux_level",
    "refractivity_at_flux_level",
    "AtmosphericParams",
    "DataParams",
    "data_to_atmospheric",
    "ElliotYoung1992Model",
    "ElliotYoung1992ModelTraditional",
    "ElliotYoung1992ModelTraditionalScaleHeight",
]

#: Highest power-series truncation order printed in EY92 (Appendix).
MAX_ORDER = 4

_VARIANTS = ("corrected", "as-printed")

#: Ideal-gas constant J mol^-1 K^-1 (from the package's CODATA-1986 pair).

_erf = np.vectorize(math.erf, otypes=[float])


# ===========================================================================
# Appendix power series  (A2), (A6), (A7)
# ===========================================================================


def _check_variant(variant):
    if variant not in _VARIANTS:
        raise ValueError(f"variant must be one of {_VARIANTS}, got {variant!r}")


def _a_coefficients(a, b, variant="corrected"):
    """Coefficients of A(delta,a,b) in powers of delta  [Eq. A2]."""
    _check_variant(variant)
    # delta^4 b^3 term: journal prints /8192; consistency of (A2)/(A6)
    # under the derivative identity requires /32768 (module docstring).
    b3_denominator = 8192 if variant == "as-printed" else 32768
    return [
        1.0,
        -(3 + a) / 8 + (3 / 8) * b,
        -(15 + 26 * a + 7 * a**2) / 128 + ((7 + 5 * a) / 64) * b + b**2 / 128,
        (
            -(105 + 425 * a + 355 * a**2 + 75 * a**3) / 1024
            + ((27 + 50 * a + 35 * a**2) / 1024) * b
            + ((69 + 55 * a) / 1024) * b**2
            + (9 / 1024) * b**3
        ),
        (
            -(4725 + 35196 * a + 57134 * a**2 + 31836 * a**3 + 5509 * a**4) / 32768
            - ((1059 + 4907 * a + 3857 * a**2 + 609 * a**3) / 8192) * b
            + ((2353 + 4326 * a + 2233 * a**2) / 16384) * b**2
            + ((3764 + 3164 * a) / b3_denominator) * b**3
            + (491 / 32768) * b**4
        ),
    ]


def _b_coefficients(a, b, variant="corrected"):
    """Coefficients of B(delta,a,b) in powers of delta  [Eq. A6].

    Note the first-order coefficient's a = 0 form (1 + 15 b)/8 and the
    second-order (9 - 34 b + 25 b^2)/128 quoted in tests/test_benchmarks.py
    for the reference-generator truncation budget.
    """
    _check_variant(variant)
    # delta^4 b^2 term: journal prints 10555; consistency requires 1055.
    b2_lead = 10555 if variant == "as-printed" else 1055
    return [
        1.0,
        (1 + 3 * a) / 8 + (15 / 8) * b,
        (9 + 6 * a + a**2) / 128 - ((17 + 11 * a) / 64) * b + (25 / 128) * b**2,
        (
            (75 + 67 * a + 41 * a**2 + 9 * a**3) / 1024
            - ((81 + 134 * a + 57 * a**2) / 1024) * b
            + ((1 + 3 * a) / 1024) * b**2
            + (5 / 1024) * b**3
        ),
        (
            (3675 + 7204 * a + 5266 * a**2 + 2564 * a**3 + 491 * a**4) / 32768
            - ((339 + 1347 * a + 1297 * a**2 + 409 * a**3) / 8192) * b
            - ((b2_lead + 1834 * a + 807 * a**2) / 16384) * b**2
            - ((67 + 49 * a) / 8192) * b**3
            + (59 / 32768) * b**4
        ),
    ]


#: Coefficients of C(delta_r) in powers of delta_r  [Eq. A7].
_C_COEFFICIENTS = [1.0, 9 / 8, 345 / 128, 9555 / 1024, 1371195 / 32768]


def _truncated_polynomial(coeffs, x, order):
    if not 0 <= order <= MAX_ORDER:
        raise ValueError(f"order must be in [0, {MAX_ORDER}], got {order}")
    x = np.asarray(x)
    result = np.zeros_like(x, dtype=float) + coeffs[order]
    for k in range(order - 1, -1, -1):
        result = result * x + coeffs[k]
    return result


def a_series(delta, a, b, order=MAX_ORDER, variant="corrected"):
    """A(delta,a,b)  [Eq. A2], truncated after delta^order."""
    return _truncated_polynomial(_a_coefficients(a, b, variant), delta, order)


def b_series(delta, a, b, order=MAX_ORDER, variant="corrected"):
    """B(delta,a,b)  [Eq. A6], truncated after delta^order."""
    return _truncated_polynomial(_b_coefficients(a, b, variant), delta, order)


def c_series(delta_r, order=MAX_ORDER):
    """C(delta_r)  [Eq. A7], truncated after delta_r^order."""
    return _truncated_polynomial(_C_COEFFICIENTS, delta_r, order)


# ===========================================================================
# Empirical model atmosphere  (Sec. 3)
# ===========================================================================


def lambda_g(r, r0, lambda_g0, a, b):
    """Gravitational-to-thermal energy ratio  [Eq. 3.9].

    lambda_g(r) = G M mu m_amu/(k T r) = lambda_g0 (r/r0)^-(1+a+b),
    with a, b the mu and T power-law exponents [3.1, 3.3].
    """
    return lambda_g0 * (np.asarray(r) / r0) ** (-(1.0 + a + b))


def _gravity_exponent_term(r, r0, lambda_g0, a, b):
    """(lambda_g(r) - lambda_g0)/(1+a+b), with the a+b = -1 limit.

    The gravity term of the hydrostatic exponent in Eqs. (3.14)/(3.16);
    the singular case is evaluated by its analytic limit
    -lambda_g0 ln(r/r0).  (EY92's "except when a+b = -2 or 1" appears
    to be a typesetting slip; the calculus gives a+b = -1, matching the
    limit instruction in the same paragraph.)
    """
    x = np.asarray(r) / r0
    e = 1.0 + a + b
    if abs(e) < 1e-12:
        return -lambda_g0 * np.log(x)
    return (lambda_g0 / e) * (x ** (-e) - 1.0)


def refractivity_profile(r, r0, nu0, lambda_g0, a, b):
    """Refractivity nu(r) of the hydrostatic power-law atmosphere  [Eq. 3.20].

    nu(r) = nu0 (r/r0)^-b exp[(lambda_g(r) - lambda_g0)/(1+a+b)].
    The exponent is capped at 300 so root finders probing far below the
    domain of validity do not overflow float64 downstream (physical
    exponents are < ~15); nu stays monotonic there.
    """
    x = np.asarray(r) / r0
    exponent = np.minimum(_gravity_exponent_term(r, r0, lambda_g0, a, b), 300.0)
    return nu0 * x ** (-b) * np.exp(exponent)


def temperature_profile(r, r0, t0, b):
    """T(r) = T0 (r/r0)^b  [Eq. 3.3], K."""
    return t0 * (np.asarray(r) / r0) ** b


def pressure_profile(r, r0, p0, lambda_g0, a, b):
    """Hydrostatic pressure p(r)  [Eq. 3.14, non-rotating], Pa."""
    return p0 * np.exp(_gravity_exponent_term(r, r0, lambda_g0, a, b))


def number_density_profile(r, r0, n0, lambda_g0, a, b):
    """Number density n(r)  [Eq. 3.17, non-rotating], m^-3."""
    x = np.asarray(r) / r0
    return n0 * x ** (-b) * np.exp(_gravity_exponent_term(r, r0, lambda_g0, a, b))


def haze_absorption(r, r1, kappa1, h_tau1):
    """Haze linear absorption coefficient kappa(r), km^-1  [Eq. 3.23].

    Zero above the onset radius r1; below it the local e-folding length
    is H_tau(r) = h_tau1 (r/r1)^2 (same radial dependence as the
    gravitational scale height) [cf. Eq. 5.15].
    """
    r = np.asarray(r, dtype=float)
    exponent = -(r - r1) / (h_tau1 * (r / r1))
    return np.where(r > r1, 0.0, kappa1 * np.exp(np.minimum(exponent, 700.0)))


# ===========================================================================
# Refraction, extinction, and observer-plane flux  (Sec. 4)
# ===========================================================================


def bending_angle(r, r0, nu0, lambda_g0, a, b, order=MAX_ORDER, variant="corrected"):
    """Refraction angle theta(r), rad (negative: bent toward the planet)  [Eq. 4.6]."""
    lg = lambda_g(r, r0, lambda_g0, a, b)
    nu = refractivity_profile(r, r0, nu0, lambda_g0, a, b)
    return -np.sqrt(2.0 * np.pi * lg) * nu * a_series(1.0 / lg, a, b, order, variant)


def D_dtheta_dr(r, d, r0, nu0, lambda_g0, a, b, order=MAX_ORDER, variant="corrected"):
    """D * dtheta/dr (dimensionless shadow-stretching term)  [Eqs. 4.9/4.10]."""
    r = np.asarray(r, dtype=float)
    lg = lambda_g(r, r0, lambda_g0, a, b)
    nu = refractivity_profile(r, r0, nu0, lambda_g0, a, b)
    return (
        d
        * np.sqrt(2.0 * np.pi * lg**3)
        * (nu / r)
        * b_series(1.0 / lg, a, b, order, variant)
    )


def tau_obs(r, r1, kappa1, h_tau1, order=MAX_ORDER):
    """Line-of-sight haze optical depth  [Eq. 4.20].

    Zero above r1; below, kappa(r) (r/r1) sqrt(2 pi h_tau1 r)
    * erf[(r1/r) sqrt((r1^2 - r^2)/(2 h_tau1 r))] * C(delta_r), with
    delta_r = h_tau1 r/r1^2.  kappa1 in km^-1.
    """
    r = np.asarray(r, dtype=float)
    below = r <= r1
    rc = np.where(below, r, r1)  # keep sqrt/erf arguments valid where masked
    kappa = haze_absorption(rc, r1, kappa1, h_tau1)
    erf_arg = (r1 / rc) * np.sqrt((r1**2 - rc**2) / (2.0 * h_tau1 * rc))
    delta_r = h_tau1 * rc / r1**2
    tau = (
        kappa
        * (rc / r1)
        * np.sqrt(2.0 * np.pi * h_tau1 * rc)
        * _erf(erf_arg)
        * c_series(delta_r, order)
    )
    return np.where(below, tau, 0.0)


def kappa1_from_unit_optical_depth(r1, r2, h_tau1, order=MAX_ORDER):
    """kappa1 (km^-1) such that tau_obs(r2) = 1  [Eq. 4.21]; requires r2 < r1."""
    if not r2 < r1:
        raise ValueError(f"require r2 < r1, got r1={r1}, r2={r2}")
    return 1.0 / float(tau_obs(r2, r1, 1.0, h_tau1, order))  # tau linear in kappa1


def rho_of_r(r, d, r0, nu0, lambda_g0, a, b, order=MAX_ORDER, variant="corrected"):
    """Observer-plane radius rho(r) = r + D theta(r), km  [Eq. 4.23]."""
    theta = bending_angle(r, r0, nu0, lambda_g0, a, b, order, variant)
    return np.asarray(r, dtype=float) + d * theta


def _min_valid_radius(r0, lambda_g0, a, b):
    """Deep boundary of the physical branch of rho(r).

    For steep gradients (1 + a + b < 0) lambda_g *decreases* downward
    and |D theta| peaks where lambda_g = -b - (1+a+b)/2 (where
    d(nu sqrt(lambda))/dr = 0); below that the analytic continuation
    "un-bends" and spawns spurious rho(r) roots.  Root finding is
    restricted to r above this radius.  Returns 0 when the profile has
    no turnover (1 + a + b >= 0, the usual case).
    """
    e = 1.0 + a + b
    if e >= 0.0:
        return 0.0
    lambda_peak = -b - e / 2.0
    if lambda_peak <= 0.0:
        return 0.0
    return r0 * (lambda_peak / lambda_g0) ** (1.0 / (-e))


def r_of_rho(
    rho,
    d,
    r0,
    nu0,
    lambda_g0,
    a,
    b,
    order=MAX_ORDER,
    variant="corrected",
    rtol=1e-12,
    max_iterations=200,
):
    """Invert Eq. (4.23) for r(rho) by safeguarded Newton (EY92 prescribe
    Newton; rho(r) is monotonic on the physical branch).

    Uses the analytic derivative d rho/dr = 1 + D dtheta/dr.  Pure
    Newton can 2-cycle for steep thermal gradients (e.g. T ~ r^-4.5):
    starting from r = rho, deep in the (unphysical) analytic
    continuation, the first step overshoots above the atmosphere and
    the next lands back.  Since theta < 0 makes the residual at the
    lower bracket negative and any overshoot positive, the root is
    bracketed as soon as both signs are seen; Newton steps leaving the
    bracket are replaced by bisection (per element).  The search is
    confined above _min_valid_radius, where rho(r) is monotonic
    (d rho/dr = 1 + D dtheta/dr > 0), so the bracket logic is sound.
    """
    rho = np.asarray(rho, dtype=float)
    scalar = rho.ndim == 0
    rho = np.atleast_1d(rho)
    r_floor = _min_valid_radius(r0, lambda_g0, a, b)
    # Initial guess: the isothermal large-planet inversion
    # r ~ r0 - (r0/lambda_g0) ln[(r0 - rho) lambda_g0 / r0] where the
    # atmosphere dominates (rho < r0 - r0/lambda_g0), else r = rho.
    # Starting from r = rho alone stalls for small rho: Newton steps
    # there scale as r/lambda per iteration.  (Same strategy as the
    # Mathematica jleGroup findR.)
    shallow = rho > r0 - r0 / lambda_g0
    with np.errstate(invalid="ignore", divide="ignore"):
        deep_guess = r0 - (r0 / lambda_g0) * np.log(
            np.maximum((r0 - rho) * lambda_g0 / r0, 1e-300)
        )
    # Lower bracket: r = rho has residual D*theta <= 0 on the physical
    # branch; where the branch floor lies above rho, the floor itself
    # over-bends (residual < 0) and brackets from below instead.
    lo = np.maximum(rho, r_floor)
    r = np.maximum(np.where(shallow, rho, deep_guess), lo)
    hi = np.full_like(rho, np.inf)  # upper bracket found on first overshoot
    for _ in range(max_iterations):
        residual = rho_of_r(r, d, r0, nu0, lambda_g0, a, b, order, variant) - rho
        lo = np.where(residual < 0.0, np.maximum(lo, r), lo)
        hi = np.where(residual >= 0.0, np.minimum(hi, r), hi)
        slope = 1.0 + D_dtheta_dr(r, d, r0, nu0, lambda_g0, a, b, order, variant)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = r - residual / slope
        outside = (newton <= lo) | (newton >= hi) | ~np.isfinite(newton)
        bisect = np.where(np.isfinite(hi), 0.5 * (lo + hi), lo * 2.0)
        r_next = np.where(outside, bisect, newton)
        step = r_next - r
        r = r_next
        if np.all(np.abs(step) <= rtol * np.abs(r)):
            break
    else:
        raise RuntimeError(
            f"r_of_rho: iteration did not converge in {max_iterations} steps"
        )
    return float(r[0]) if scalar else r


def r_far_of_rho(
    rho,
    d,
    r0,
    nu0,
    lambda_g0,
    a,
    b,
    order=MAX_ORDER,
    variant="corrected",
    rtol=1e-12,
    max_iterations=200,
):
    """Far-limb periapsis radius: solve r + D theta(r) = -rho  (rho > 0).

    A far-limb ray is bent so strongly it crosses the shadow axis and
    arrives at observer-plane distance rho on the opposite side, i.e.
    its signed observer coordinate is -rho [EY92 Eq. 2.2 with the
    absolute value retained; cf. the Mathematica jleGroup olcTwoLimb4,
    which evaluates the one-limb model at -rho].

    rho(r) = r + D theta(r) is monotonic (d rho/dr = 1 + D dtheta/dr > 0)
    on the physical branch, decreasing through zero at the focal radius
    and negative below it, so the far root is bracketed between a
    descent below the focal radius and the near-limb solution.  The
    same bisection-safeguarded Newton as r_of_rho then refines it.
    """
    rho = np.asarray(rho, dtype=float)
    scalar = rho.ndim == 0
    rho = np.atleast_1d(rho)
    if not np.all(rho > 0.0):
        raise ValueError("r_far_of_rho: rho must be positive (use signed "
                         "targets only through the near/far pair)")
    args = (d, r0, nu0, lambda_g0, a, b, order, variant)

    # Upper bracket: the near-limb radius (residual there is +2 rho > 0).
    hi = np.atleast_1d(np.asarray(r_of_rho(rho, *args, rtol=rtol), dtype=float))
    # Lower bracket: descend one local refractivity scale height
    # H_n = r/(lambda_g + b) at a time until rho(r) < -rho.  |D theta|
    # grows ~e-fold per H_n, so ~ln((r+rho)/(r-rho)) steps suffice.
    # The descent floor is the physical-branch boundary
    # (_min_valid_radius): if the bending maximum there still cannot
    # reach -rho, no physical far-limb ray exists for these parameters.
    r_floor = _min_valid_radius(r0, lambda_g0, a, b) * (1.0 + 1e-9) + 1e-9
    lo = hi.copy()
    need = np.full(rho.shape, True)
    for _ in range(200):
        h_n = np.abs(lo / (lambda_g(lo, r0, lambda_g0, a, b) + b))
        trial = np.where(need, lo - np.maximum(h_n, 1e-3 * lo), lo)
        trial = np.maximum(trial, r_floor)
        residual = rho_of_r(trial, *args) + rho
        lo = np.where(need, trial, lo)
        need = need & (residual >= 0.0)
        if not np.any(need):
            break
        if np.all(lo[need] <= r_floor):
            raise RuntimeError(
                "r_far_of_rho: the atmosphere's maximum bending cannot "
                "carry rays across the shadow axis to these rho values "
                "(no physical far-limb ray)"
            )
    else:
        raise RuntimeError(
            "r_far_of_rho: could not bracket the far-limb root within "
            "the descent budget"
        )

    r = 0.5 * (lo + hi)
    for _ in range(max_iterations):
        residual = rho_of_r(r, *args) + rho  # target is -rho
        lo = np.where(residual < 0.0, np.maximum(lo, r), lo)
        hi = np.where(residual >= 0.0, np.minimum(hi, r), hi)
        slope = 1.0 + D_dtheta_dr(r, *args)
        with np.errstate(divide="ignore", invalid="ignore"):
            newton = r - residual / slope
        outside = (newton <= lo) | (newton >= hi) | ~np.isfinite(newton)
        r_next = np.where(outside, 0.5 * (lo + hi), newton)
        step = r_next - r
        r = r_next
        if np.all(np.abs(step) <= rtol * np.abs(r)):
            break
    else:
        raise RuntimeError(
            f"r_far_of_rho: iteration did not converge in {max_iterations} steps"
        )
    return float(r[0]) if scalar else r


def _limb_flux(
    r,
    d,
    r0,
    nu0,
    lambda_g0,
    a,
    b,
    r1=None,
    kappa1=0.0,
    h_tau1=1.0,
    surface_radius=None,
    order=MAX_ORDER,
    variant="corrected",
):
    """Flux contribution of one limb from its periapsis radius r.

    zeta(r) = exp(-tau) / (|1 + D theta/r| |1 + D dtheta/dr|)
    [EY92 Eq. 2.1]; valid for either limb because the absolute values
    are retained (on the far limb 1 + D theta/r = -rho/r < 0).  Rays
    with periapsis below surface_radius are blocked (instantaneous
    cutoff; ExpTime bin integration of the surface event is a separate,
    planned extension).
    """
    r = np.asarray(r, dtype=float)
    theta = bending_angle(r, r0, nu0, lambda_g0, a, b, order, variant)
    focus = np.abs(1.0 + d * theta / r)
    stretch = np.abs(
        1.0 + D_dtheta_dr(r, d, r0, nu0, lambda_g0, a, b, order, variant)
    )
    with np.errstate(divide="ignore"):
        result = 1.0 / (focus * stretch)
    if r1 is not None and kappa1 != 0.0:
        result = result * np.exp(-tau_obs(r, r1, kappa1, h_tau1, order))
    if surface_radius is not None:
        result = np.where(r < surface_radius, 0.0, result)
    return result


def phi_two_limb(
    rho,
    d,
    r0,
    nu0,
    lambda_g0,
    a,
    b,
    r1=None,
    kappa1=0.0,
    h_tau1=1.0,
    surface_radius=None,
    order=MAX_ORDER,
    variant="corrected",
):
    """Two-limb normalized flux at observer-plane distance rho > 0.

    Sums the near- and far-limb contributions [EY92 Eqs. 2.2/2.7 with
    the absolute value retained; the Mathematica jleGroup olcTwoLimb4
    equivalent].  The far limb produces the central flash: both terms
    scale as r/rho near the shadow center, so the flux diverges at
    rho = 0 exactly -- the geometric-optics focal singularity (EY92
    Sec. 2); finite stellar diameter/diffraction, not modeled here,
    bound it physically.

    For small-lambda bodies the transparent analytic atmosphere admits
    far-limb rays at all rho (passing arbitrarily deep), so a realistic
    body should set surface_radius (and/or haze) to block them; without
    it the model is a gas sphere with no surface.

    Returns the total; use r_far_of_rho/_limb_flux for the pieces.
    """
    args = (d, r0, nu0, lambda_g0, a, b, order, variant)
    extinction = dict(
        r1=r1, kappa1=kappa1, h_tau1=h_tau1, surface_radius=surface_radius,
        order=order, variant=variant,
    )
    r_near = r_of_rho(rho, *args)
    r_far = r_far_of_rho(rho, *args)
    near = _limb_flux(r_near, d, r0, nu0, lambda_g0, a, b, **extinction)
    far = _limb_flux(r_far, d, r0, nu0, lambda_g0, a, b, **extinction)
    return near + far


#: Alias honoring the Mathematica jleGroup lineage name (olcTwoLimb4).
twoLimb = phi_two_limb


def phi_cyl(r, d, r0, nu0, lambda_g0, a, b, order=MAX_ORDER, variant="corrected"):
    """Unfocused (cylindrical-planet) flux 1/(1 + D dtheta/dr)  [Sec. 7]."""
    return 1.0 / (1.0 + D_dtheta_dr(r, d, r0, nu0, lambda_g0, a, b, order, variant))


def phi_ref(r, d, r0, nu0, lambda_g0, a, b, order=MAX_ORDER, variant="corrected"):
    """Refraction-only flux 1/[(1 + D theta/r)(1 + D dtheta/dr)]  [Eq. 4.25, tau = 0]."""
    r = np.asarray(r, dtype=float)
    theta = bending_angle(r, r0, nu0, lambda_g0, a, b, order, variant)
    focus = 1.0 + d * theta / r
    stretch = 1.0 + D_dtheta_dr(r, d, r0, nu0, lambda_g0, a, b, order, variant)
    return 1.0 / (focus * stretch)


def phi(
    r,
    d,
    r0,
    nu0,
    lambda_g0,
    a,
    b,
    r1=None,
    kappa1=0.0,
    h_tau1=1.0,
    order=MAX_ORDER,
    variant="corrected",
):
    """Normalized flux exp(-tau_obs)/[(1 + D theta/r)(1 + D dtheta/dr)]  [Eq. 4.25]."""
    result = phi_ref(r, d, r0, nu0, lambda_g0, a, b, order, variant)
    if r1 is not None and kappa1 != 0.0:
        result = result * np.exp(-tau_obs(r, r1, kappa1, h_tau1, order))
    return result


# ---------------------------------------------------------------------------
# Flux-level (half-light) relations  [Eqs. 4.26-4.28]
# ---------------------------------------------------------------------------


def _x_ratio(lambda_gf, a, b, order, variant):
    """X = A/(lambda_gf B) at delta_f = 1/lambda_gf."""
    delta_f = 1.0 / lambda_gf
    return float(
        a_series(delta_f, a, b, order, variant)
        / (lambda_gf * b_series(delta_f, a, b, order, variant))
    )


def D_theta_over_r_at_flux_level(f, lambda_gf, a, b, order=MAX_ORDER, variant="corrected"):
    """D theta(r_f)/r_f where the refractive flux equals f  [Eq. 4.27].

    Positive root of y^2 + (1 - X) y + X (1/f - 1) = 0 (inverting
    Eq. 4.26); large-planet limit for f = 1/2 is -delta_gf.
    """
    if not 0.0 < f < 1.0:
        raise ValueError(f"flux level f must be in (0, 1), got {f}")
    x = _x_ratio(lambda_gf, a, b, order, variant)
    return 0.5 * (x - 1.0 + np.sqrt(1.0 + (2.0 - 4.0 / f) * x + x * x))


def refractivity_at_flux_level(
    r_f, f, d, lambda_gf, a, b, order=MAX_ORDER, variant="corrected"
):
    """Refractivity nu(r_f) at the flux-level radius  [Eq. 4.28]."""
    delta_f = 1.0 / lambda_gf
    y = D_theta_over_r_at_flux_level(f, lambda_gf, a, b, order, variant)
    a_val = float(a_series(delta_f, a, b, order, variant))
    return -y * r_f / (d * np.sqrt(2.0 * np.pi * lambda_gf) * a_val)


# ===========================================================================
# Parameter sets and the data -> atmospheric conversion  (Secs. 5-6)
# ===========================================================================


@dataclass(frozen=True)
class AtmosphericParams:
    """EY92 "atmospheric" parameter set (Table 1, left column).

    Distances km; kappa1 km^-1; signal levels in counts/s (only used
    when reproducing the paper's signal-domain benchmark).
    """

    s_background: float
    s_background_slope: float
    s_star: float
    d: float
    v: float
    delta_t: float
    rho_min: float
    t_mid: float
    a: float
    b: float
    r0: float
    lambda_g0: float
    nu0: float
    r1: float
    kappa1: float
    h_tau1: float

    @property
    def s_full_scale(self):
        """s_f = s_* + s_b  [Eq. 5.4]."""
        return self.s_star + self.s_background


@dataclass(frozen=True)
class DataParams:
    """EY92 "data" parameter set (Table 1, right column; Table 2 values).

    The set EY92 actually fit: signal levels, event times, and time
    intervals.  Units: s, km, km/s; signal levels counts/s.
    """

    s_background: float
    s_background_slope: float
    s_full_scale: float
    d: float
    v: float
    delta_t: float
    t_min: float
    a: float
    b: float
    t_im: float
    t_em: float
    t_hiso: float
    f: float = 0.5
    t_h1: float = 0.0
    t_h2: float = 0.0
    t_hr2: float = 0.0

    @property
    def t_mid(self):
        """Event midtime  [Eq. 6.1]."""
        return 0.5 * (self.t_im + self.t_em)

    @property
    def rho_min(self):
        """Minimum observer radius  [Eq. 5.8], km."""
        return self.t_min * self.v

    @property
    def s_star(self):
        """s_* = s_f - s_b  [Eq. 5.4]."""
        return self.s_full_scale - self.s_background


def data_to_atmospheric(
    dp, r0, order=MAX_ORDER, variant="corrected", rtol=1e-12, max_iterations=200
):
    """Convert the data parameter set to the atmospheric set  [Sec. 6].

    Implements Eqs. (6.1)-(6.8) with (4.27)/(4.28) and (4.21): geometry
    at the flux-level radius, successive substitution for r_h and
    lambda_gh, referral of nu and lambda_g to r0, and the haze
    parameters from the time intervals.  Validated digit-for-digit
    against the EY92 p. 1004 derived-parameter table at series orders
    0 and 1 (see tests/test_EY92.py).
    """
    rho_min = dp.rho_min
    half_chord = 0.5 * dp.v * (dp.t_em - dp.t_im)
    rho_h = math.hypot(rho_min, half_chord)  # [6.2]
    v_perp_h = dp.v * math.sqrt(rho_h**2 - rho_min**2) / rho_h  # [5.9]
    h_iso = dp.t_hiso * v_perp_h  # [5.14]

    r_h = rho_h + h_iso  # large-planet initial guess [6.3]
    for _ in range(max_iterations):
        lambda_gh = r_h / h_iso - (3.0 * dp.a + 5.0 * dp.b) / 2.0  # [6.4]
        y = D_theta_over_r_at_flux_level(dp.f, lambda_gh, dp.a, dp.b, order, variant)
        r_new = rho_h - y * r_h  # [6.5]
        converged = abs(r_new - r_h) <= rtol * r_h
        r_h = r_new
        if converged:
            break
    else:
        raise RuntimeError(
            f"data_to_atmospheric: r_h iteration did not converge in "
            f"{max_iterations} steps"
        )
    lambda_gh = r_h / h_iso - (3.0 * dp.a + 5.0 * dp.b) / 2.0

    nu_h = refractivity_at_flux_level(r_h, dp.f, dp.d, lambda_gh, dp.a, dp.b, order, variant)
    lambda_g0_val = lambda_gh * (r0 / r_h) ** (-(1.0 + dp.a + dp.b))  # [6.6]
    nu0 = (
        nu_h
        * (r_h / r0) ** dp.b
        / math.exp(float(_gravity_exponent_term(r_h, r0, lambda_g0_val, dp.a, dp.b)))
    )  # [6.7]

    def planet_radius_at(t):
        rho_t = math.hypot(rho_min, dp.v * (t - dp.t_mid))  # [5.1]
        return float(
            r_of_rho(rho_t, dp.d, r0, nu0, lambda_g0_val, dp.a, dp.b, order, variant)
        )

    # Immersion branch: haze onset (r1) and unit optical depth (r2) are
    # reached t_h1 and t_h2 after the flux-level time [5.5-5.7].
    r1 = planet_radius_at(dp.t_im + dp.t_h1)
    r2 = planet_radius_at(dp.t_im + dp.t_h2)
    h_tau1 = dp.t_hr2 * v_perp_h * (r1 / r2) ** 2  # [6.8]
    kappa1 = kappa1_from_unit_optical_depth(r1, r2, h_tau1, order)

    return AtmosphericParams(
        s_background=dp.s_background,
        s_background_slope=dp.s_background_slope,
        s_star=dp.s_star,
        d=dp.d,
        v=dp.v,
        delta_t=dp.delta_t,
        rho_min=rho_min,
        t_mid=dp.t_mid,
        a=dp.a,
        b=dp.b,
        r0=r0,
        lambda_g0=lambda_g0_val,
        nu0=nu0,
        r1=r1,
        kappa1=kappa1,
        h_tau1=h_tau1,
    )


# ===========================================================================
# CE97-style facade
# ===========================================================================


class ElliotYoung1992Model:
    """Analytic EY92 light curve from physical atmosphere parameters.

    Drop-in comparable with CE97.ChamberlainElliot1997Model: give the
    observer-plane positions (km) and call main(); read focusedFlux,
    unfocusedFlux, theta, dtheta (and transmission/tauObs when a haze
    layer is present).  One near limb, instantaneous sampling.

    Where CE97 takes a tabulated refractivity profile, EY92 is analytic:
    the atmosphere is specified by conditions at a reference radius and
    power-law exponents, and converted to the EY92 (nu0, lambda_g0)
    parameters using jlegroup.physicalData:

        lambda_g0 = G M_p m_mol / (R_gas T0 r0)      [Eq. 3.10]
        nu0 = nu_STP(gas, lambda) p0 / (k T0 L)      [Eqs. 3.15, 3.19]

    Parameters
    ----------
    referencePressure : float, Pa at referenceRadius.
    referenceTemperature : float, K at referenceRadius.
    referenceRadius : float, km.
    planetMass : float, kg.
    meanMolecularMass : float, kg/mol.
    planetDistance : float, observer-planet distance, km.
    position : array, radial coordinates in the observer plane, km.
    gas, wavelength_um : passed to physicalData.refractivitySTP
        (default N2 at 0.7 um), unless refractivityAtSTP is given.
    temperatureExponent : float, b in T ~ r^b  [Eq. 3.3]; default 0.
    molecularWeightExponent : float, a in mu ~ r^-a  [Eq. 3.1]; default 0.
    hazeOnsetRadius, hazeKappa1, hazeScaleHeight : optional haze layer
        [Eq. 3.23]: top radius km, absorption at onset km^-1, scale
        height at onset km.
    twoLimb : bool, default False.  When True, add the far-limb
        contribution (rays crossing the shadow axis; EY92 Eq. 2.7 with
        the absolute value retained -- the olcTwoLimb4 equivalent),
        producing the central flash near the shadow center.
    surfaceRadius : float km, optional.  Rays with periapsis below this
        are blocked (instantaneous cutoff).  Strongly recommended with
        twoLimb for small-lambda bodies: the transparent analytic
        atmosphere otherwise passes far-limb rays arbitrarily deep.
    seriesOrder : int 0-4; 1 matches the Mathematica jleGroup default,
        0 is the Baum & Code large-planet limit.  Default 4.
    seriesVariant : "corrected" (default) or "as-printed"; see module
        docstring errata notes.
    constants : physicalData.ConstantSet, default DEFAULT_CONSTANTS
        (CODATA-2022). Pass physicalData.CODATA1986 when reproducing the
        Mathematica references or published tables (their vintage).
    """

    def __init__(
        self,
        referencePressure,
        referenceTemperature,
        referenceRadius,
        planetMass,
        meanMolecularMass,
        planetDistance,
        position,
        gas="N2",
        wavelength_um=0.7,
        refractivityAtSTP=None,
        temperatureExponent=0.0,
        molecularWeightExponent=0.0,
        hazeOnsetRadius=None,
        hazeKappa1=None,
        hazeScaleHeight=None,
        twoLimb=False,
        surfaceRadius=None,
        seriesOrder=MAX_ORDER,
        seriesVariant="corrected",
        constants=None,
    ):
        self.referencePressure = referencePressure
        self.referenceTemperature = referenceTemperature
        self.referenceRadius = referenceRadius
        self.planetMass = planetMass
        self.meanMolecularMass = meanMolecularMass
        self.planetDistance = planetDistance
        self.position = np.asarray(position, dtype=float)
        self.a = molecularWeightExponent
        self.b = temperatureExponent
        self.hazeOnsetRadius = hazeOnsetRadius
        self.hazeKappa1 = hazeKappa1
        self.hazeScaleHeight = hazeScaleHeight
        self.twoLimb = twoLimb
        self.surfaceRadius = surfaceRadius
        self.seriesOrder = seriesOrder
        self.seriesVariant = seriesVariant

        if constants is None:
            constants = physicalData.DEFAULT_CONSTANTS
        self.constants = constants

        if refractivityAtSTP is None:
            refractivityAtSTP = physicalData.refractivitySTP(gas, wavelength_um)
        self.refractivityAtSTP = refractivityAtSTP

        # EY92 (nu0, lambda_g0) from the physical inputs [3.10, 3.15, 3.19].
        n0 = referencePressure / (constants.boltzmann * referenceTemperature)
        self.referenceNumberDensity = n0
        self.nu0 = refractivityAtSTP * n0 / constants.loschmidt
        self.lambda_g0 = (
            constants.gravitational
            * planetMass
            * meanMolecularMass
            / (constants.gas_constant * referenceTemperature
               * (referenceRadius * 1e3))
        )

        self.planetRadius_solution = None  # near-limb r(rho) per position, km
        self.farPlanetRadius_solution = None  # far-limb r, km (twoLimb only)
        self.theta = None  # near-limb bending angle, rad
        self.dtheta = None  # near-limb dtheta/dr, rad/km
        self.tauObs = None  # near-limb haze optical depth
        self.transmission = None  # near-limb exp(-tau)
        self.unfocusedFlux = None  # near-limb cylindrical flux
        self.nearLimbFlux = None
        self.farLimbFlux = None  # twoLimb only
        self.focusedFlux = None  # total model flux

    def main(self):
        """Compute the light curve at the given observer-plane positions."""
        args = (
            self.planetDistance,
            self.referenceRadius,
            self.nu0,
            self.lambda_g0,
            self.a,
            self.b,
            self.seriesOrder,
            self.seriesVariant,
        )
        extinction = dict(
            r1=self.hazeOnsetRadius,
            kappa1=self.hazeKappa1 or 0.0,
            h_tau1=self.hazeScaleHeight if self.hazeScaleHeight else 1.0,
            surface_radius=self.surfaceRadius,
            order=self.seriesOrder,
            variant=self.seriesVariant,
        )
        r = r_of_rho(self.position, *args)
        self.planetRadius_solution = r
        self.theta = bending_angle(
            r, self.referenceRadius, self.nu0, self.lambda_g0,
            self.a, self.b, self.seriesOrder, self.seriesVariant,
        )
        self.dtheta = (
            D_dtheta_dr(r, *args) / self.planetDistance
        )
        self.unfocusedFlux = phi_cyl(r, *args)
        if self.hazeOnsetRadius is not None and self.hazeKappa1:
            self.tauObs = tau_obs(
                r, self.hazeOnsetRadius, self.hazeKappa1,
                self.hazeScaleHeight, self.seriesOrder,
            )
            self.transmission = np.exp(-self.tauObs)
        self.nearLimbFlux = _limb_flux(
            r, self.planetDistance, self.referenceRadius, self.nu0,
            self.lambda_g0, self.a, self.b, **extinction,
        )
        self.focusedFlux = self.nearLimbFlux
        if self.twoLimb:
            r_far = r_far_of_rho(self.position, *args)
            self.farPlanetRadius_solution = r_far
            self.farLimbFlux = _limb_flux(
                r_far, self.planetDistance, self.referenceRadius, self.nu0,
                self.lambda_g0, self.a, self.b, **extinction,
            )
            self.focusedFlux = self.nearLimbFlux + self.farLimbFlux
        return self.focusedFlux


class ElliotYoung1992ModelTraditional(ElliotYoung1992Model):
    """EY92 light curve from the traditional half-light parameterization.

    The parameter set the Elliot group has always fit -- and the one the
    light curve actually constrains: the physical inputs of
    ElliotYoung1992Model (pressure, temperature, planet mass, molecular
    mass) enter the model only through (nu0, lambda_g0), so fitting in
    physical space chases a degeneracy ridge.  Here the atmosphere is
    specified by the refraction-only half-light point instead:

        radiusHalfLight  r_h : where the refractive flux equals
            referenceFluxLevel (default 1/2), one limb, no extinction.
        lambdaHalfLight      : the EQUIVALENT-ISOTHERMAL energy ratio
            r_h/H_iso at half-light -- the Mathematica jleGroup
            "lambdaHi" convention, so historical fit values drop in
            unchanged.  The true energy ratio follows EY92 Eq. (5.13):
                lambda_true = lambdaHalfLight - (3 a + 5 b)/2
            (the Mathematica realLam0/realLam0A relation; the two
            coincide for an isothermal, constant-mu atmosphere).
        b, a : temperature and molecular-weight exponents [3.3, 3.1].

    nu0 then follows from the flux-level condition [Eqs. 4.26-4.28] and
    the geometry alone -- no jlegroup.physicalData constants enter.
    The anchor is always the refraction-only, near-limb flux level, so
    (r_h, lambdaHalfLight) mean the same thing whether or not twoLimb,
    haze, or surfaceRadius options are active.

    Derived attributes:
        lambdaTrueHalfLight    lambda_g(r_h), as above
        scaleHeight            r_h / lambda_true  (pressure scale height
                               due to gravity at half-light; the
                               maintainer's working definition
                               r_h/(lambdaHiso - 5b/2) at a = 0)
        isothermalScaleHeight  r_h / lambdaHalfLight  (= H_iso)

    All remaining options (position, planetDistance, haze, twoLimb,
    surfaceRadius, seriesOrder, seriesVariant) behave exactly as in
    ElliotYoung1992Model; main() is inherited unchanged.
    """

    def __init__(
        self,
        radiusHalfLight,
        lambdaHalfLight,
        b,
        planetDistance,
        position,
        a=0.0,
        referenceFluxLevel=0.5,
        hazeOnsetRadius=None,
        hazeKappa1=None,
        hazeScaleHeight=None,
        twoLimb=False,
        surfaceRadius=None,
        seriesOrder=MAX_ORDER,
        seriesVariant="corrected",
    ):
        # Geometry, options, and outputs (parent __init__ is bypassed:
        # it requires the physical inputs this parameterization replaces).
        self.planetDistance = planetDistance
        self.position = np.asarray(position, dtype=float)
        self.a = a
        self.b = b
        self.hazeOnsetRadius = hazeOnsetRadius
        self.hazeKappa1 = hazeKappa1
        self.hazeScaleHeight = hazeScaleHeight
        self.twoLimb = twoLimb
        self.surfaceRadius = surfaceRadius
        self.seriesOrder = seriesOrder
        self.seriesVariant = seriesVariant

        # Half-light parameterization -> EY92 (nu0, lambda_g0) at r0 = r_h.
        # Deliberately constants-free: nu0 comes from the flux-level condition
        # and geometry alone, so no ConstantSet enters (vintage-independent).
        self.constants = None
        self.radiusHalfLight = radiusHalfLight
        self.lambdaHalfLight = lambdaHalfLight
        self.referenceFluxLevel = referenceFluxLevel
        lambda_true = lambdaHalfLight - (3.0 * a + 5.0 * b) / 2.0  # [5.13]
        if lambda_true <= 0.0:
            raise ValueError(
                "lambdaHalfLight - (3a+5b)/2 must be positive; got "
                f"{lambda_true} (lambdaHalfLight={lambdaHalfLight}, a={a}, b={b})"
            )
        self.lambdaTrueHalfLight = lambda_true
        self.scaleHeight = radiusHalfLight / lambda_true
        self.isothermalScaleHeight = radiusHalfLight / lambdaHalfLight
        self.referenceRadius = radiusHalfLight
        self.lambda_g0 = lambda_true
        self.nu0 = refractivity_at_flux_level(
            radiusHalfLight, referenceFluxLevel, planetDistance,
            lambda_true, a, b, seriesOrder, seriesVariant,
        )

        self.planetRadius_solution = None  # near-limb r(rho) per position, km
        self.farPlanetRadius_solution = None  # far-limb r, km (twoLimb only)
        self.theta = None  # near-limb bending angle, rad
        self.dtheta = None  # near-limb dtheta/dr, rad/km
        self.tauObs = None  # near-limb haze optical depth
        self.transmission = None  # near-limb exp(-tau)
        self.unfocusedFlux = None  # near-limb cylindrical flux
        self.nearLimbFlux = None
        self.farLimbFlux = None  # twoLimb only
        self.focusedFlux = None  # total model flux


class ElliotYoung1992ModelTraditionalScaleHeight(ElliotYoung1992ModelTraditional):
    """Traditional half-light parameterization with the scale height as
    the atmosphere-strength parameter instead of lambdaHalfLight.

        scaleHeightH : the pressure(-gravity) scale height at the
            half-light radius, km:
                scaleHeightH = r_h / lambda_true,
                lambda_true  = lambdaHalfLight - (3a+5b)/2
            (so lambdaHalfLight = r_h/scaleHeightH + (3a+5b)/2 is
            reconstructed internally and everything else follows
            ElliotYoung1992ModelTraditional unchanged).

    Useful when fitting: the scale height is often the physically
    stable quantity, so one holds scaleHeightH fixed while
    radiusHalfLight floats against the data -- in (rH, lambdaHiso)
    space that constraint couples both parameters, here it is a single
    frozen one.  The inherited `scaleHeight` attribute equals
    scaleHeightH by construction.
    """

    def __init__(
        self,
        radiusHalfLight,
        scaleHeightH,
        b,
        planetDistance,
        position,
        a=0.0,
        referenceFluxLevel=0.5,
        hazeOnsetRadius=None,
        hazeKappa1=None,
        hazeScaleHeight=None,
        twoLimb=False,
        surfaceRadius=None,
        seriesOrder=MAX_ORDER,
        seriesVariant="corrected",
    ):
        if scaleHeightH <= 0.0:
            raise ValueError(f"scaleHeightH must be positive, got {scaleHeightH}")
        self.scaleHeightH = scaleHeightH
        lambda_iso = radiusHalfLight / scaleHeightH + (3.0 * a + 5.0 * b) / 2.0
        super().__init__(
            radiusHalfLight=radiusHalfLight,
            lambdaHalfLight=lambda_iso,
            b=b,
            planetDistance=planetDistance,
            position=position,
            a=a,
            referenceFluxLevel=referenceFluxLevel,
            hazeOnsetRadius=hazeOnsetRadius,
            hazeKappa1=hazeKappa1,
            hazeScaleHeight=hazeScaleHeight,
            twoLimb=twoLimb,
            surfaceRadius=surfaceRadius,
            seriesOrder=seriesOrder,
            seriesVariant=seriesVariant,
        )
