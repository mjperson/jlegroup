# -*- coding: utf-8 -*-
"""EPQ03 — stellar-occultation light-curve inversion with error propagation.

Implementation of

    Elliot, J. L., Person, M. J., & Qu, S. 2003, AJ, 126, 1041 ("EPQ03").
    "Analysis of Stellar Occultation Data. II. Inversion, with
     Application to Pluto and Triton."

Inverts a normalized occultation light curve for the atmospheric
refractivity, number-density, pressure, temperature, and scale-height
profiles of a small-body atmosphere — no thin-atmosphere approximation:
the scale height need not be small compared with the body radius, local
gravity varies as 1/r^2, and limb curvature focusing is included — with
the paper's complete random-error propagation (Sec. 4): boundary-
condition and flux-summation contributions, separately reportable.

The boundary condition above the inversion region is modular (any object
with r_boundary/theta/dtheta_dr/flux); the paper's worked example — the
EY92 power-law thermal model fitted to the upper light curve
(Sec. 3.3) — is provided, built on this package's EY92 module.
Equation numbers in docstrings refer to EPQ03 unless prefixed EY92.

Typical use
-----------
    from jlegroup import EPQ03

    result = EPQ03.invert_light_curve(
        y, flux, sigma,              # shadow radii (km, decreasing), flux
        d=distance_km, gas=EPQ03.GASES["N2"], m_p=mass_kg,
        fit_params=("r_h", "lambda_hi", "b"),   # boundary fit (Sec. 3.3)
        boundary_flux=0.5, min_shell=1.0,
    )
    budget = EPQ03.propagate_errors(result)     # Sec. 4
    EPQ03.write_csv("inversion.csv", result, budget)

Provenance
----------
Newly written from the published equations (EPQ03 + EY92) — nothing
ported from the Mathematica jleGroup occInversions package, which serves
only as a planned cross-validation oracle.  Developed and validated
standalone (the dev original lives in the maintainer's Inversion/
project folder with its full FIDELITY.md); validated against the paper's
own printed test results:

* Standard test case (Table 1): inverted mean temperature 79.99685 K,
  convergence 79.99797 K, max residual 0.0046 K vs the paper's printed
  79.997 / 79.998 / 0.004 (Table 3), with the paper's 860 boundary
  points and 80 inversion shells.
* Boundary-flux trials (Table 3) at print precision; nonisothermal
  b = -6..+9 within Fig. 6's envelope, worst case b = 9: 0.483% vs the
  paper's "+0.48%".
* Formal errors validated against the scatter of 25 independently
  inverted noise samples at (S/N)_H = 100 (median ratio 0.76; the paper
  quotes agreement within 18%), with Fig. 10's boundary/summation error
  morphology and 1/(S/N)_H scaling.

Fidelity notes (implementation decisions with evidence)
-------------------------------------------------------
1. EPQ03 Eq. (66) as printed drops the A-series factors of EY92
   Eq. (4.28); reproducing the paper's own Table 3 digits requires the
   (4.28) form, so ``nu_h_method`` defaults to "ey92-4.28"
   ("epq03-66" preserves the printed equation; they differ by ~5e-4).
2. The asymptotic series default to the corrected coefficients (EPQ03
   Eqs. 64/67 = corrected EY92 A2/A6); "as-printed" reproduces the EY92
   journal text.  At series order <= 3 (and always for b = 0) the two
   are identical.
3. EPQ03 Table 10 prints Loschmidt's number as 2.68684e24 m^-3; the
   exponent is a misprint (the paper's own Table 11 rows imply e25) —
   ``EPQ03_TABLE10`` carries the corrected value.
4. The EY92 half-light relation (4.27) has no real solution for
   lambda_h <~ 5.8; the boundary condition raises below lambda_h = 6 and
   the fit walls Levenberg-Marquardt off at 6.2.
5. Extreme data/shell binning (>= half a scale height) reproduces the
   paper's headline error bounds but not its 2-decimal Table 3 rows —
   those are sensitive to binning bookkeeping; see the dev original's
   FIDELITY.md.
6. On noisy data, choose the inversion boundary index ``i_b`` explicitly
   (Sec. 7.3) rather than letting a noise spike trigger the flux-level
   test early.

Scope caveats (from the paper's assumptions, Secs. 2.1/7.4/7.5): clear
atmosphere in the inversion region, spherical symmetry, single near
limb, geometric optics, no ray crossing (Eq. 100 helper provided); the
flux normalization is the user's responsibility and its systematic
errors dominate everything else (Sec. 7.4.1); large-body adaptation
(Sec. 7.8) not implemented.

Units (package convention): radii/distances km; pressure Pa
(``pressure_microbar`` mirrors the papers' display unit); number density
m^-3; temperature K; bending angle rad; flux normalized to 1.  Physical
constants come from jlegroup.physicalData (CODATA-1986 vintage); the gas
data (Table 9 mean molecular weights and bandpass-integrated nu_STP)
are method-specific values from the papers and live here.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, replace as dataclass_replace

import numpy as np
from scipy.integrate import quad_vec
from scipy.optimize import least_squares

from . import EY92, physicalData
from .physicalData import CODATA1986, ConstantSet, Gas, GASES

__all__ = [
    "ConstantSet",
    "CODATA1986",
    "EPQ03_TABLE10",
    "Gas",
    "GASES",
    "AU_KM",
    "mass_from_lambda",
    "lambda_from_mass",
    "shadow_radius",
    "split_immersion_emersion",
    "bin_uniform",
    "average_until_positive",
    "boundary_index",
    "shell_edges",
    "bin_shells",
    "EY92PowerLawBoundary",
    "BoundaryFitResult",
    "fit_boundary",
    "PARAMETER_NAMES",
    "ShellGrid",
    "radius_scale",
    "summation_terms",
    "boundary_integrals",
    "InversionProfiles",
    "invert",
    "QuantityErrors",
    "ErrorBudget",
    "propagate_errors",
    "InversionResult",
    "invert_light_curve",
    "ModelAtmosphere",
    "STANDARD_CASE",
    "standard_case_variant",
    "model_flux",
    "generate_light_curve",
    "sigma_from_snr_h",
    "snr_h_from_sigma",
    "add_noise",
    "ray_crossing_free",
    "result_table",
    "write_csv",
    "plot_light_curve",
    "plot_temperature",
    "plot_error_decomposition",
]


#############################################################################
# constants
#############################################################################
# Physical constants and gas data for EPQ03 inversions.
#
# The shared pieces — the ``ConstantSet`` injection mechanism, the
# package-default ``CODATA1986`` set, and the EY92/EPQ03 Table 9 gas
# registry (``Gas``, ``GASES``) — live in ``jlegroup.physicalData``
# (consolidated 2026-07-08) and are re-exported here unchanged for the
# established EPQ03 API.  Method-specific to this module:
#
# * ``EPQ03_TABLE10`` — the values printed in EPQ03 Table 10, for
#   digit-level reproduction of the paper's tables.  Note one erratum:
#   Table 10 prints Loschmidt's number as 2.68684e24 m^-3; the exponent
#   is a misprint (the paper's own inversion tables imply ~2.687e25; see
#   tests/test_EPQ03.py), so the value here carries the corrected
#   exponent.  Provenance decode (2026-07-08, from the reference Mathematica
#   physicalData_2.3 package): Table 10's constants match that package's
#   older reference-[1]/Allen vintage at printed precision — G and k are
#   pdGravitationalConstant[1] and pdBoltzmannConstant[1] (Taylor et al.
#   1969), amu is pdAtomicMassUnit[1] (Allen 2nd ed.) — and the misprinted
#   Loschmidt is exactly pdLoschmidtNumber[2] (Allen 3rd ed.) with an
#   exponent typo.
#
# The temperature retrieved by the inversion depends on the constants only
# through mu * m_amu * G * M_p / k  [Eqs. 57, 60], so tests that must hit
# a temperature target to ~1e-4 derive the body mass from Eq. (60) with
# the same constant set used by the inversion (see ``mass_from_lambda``).

#: km per astronomical unit (jlegroup.physicalData).
AU_KM = physicalData.AU_KM

#: EPQ03 Table 10 as printed, except the Loschmidt exponent erratum (e24 -> e25).
EPQ03_TABLE10 = ConstantSet(
    name="EPQ03 Table 10 (Loschmidt exponent corrected)",
    gravitational=6.67320e-11,
    boltzmann=1.38062e-23,
    loschmidt=2.68684e25,
    amu=1.66030e-27,
)


def mass_from_lambda(lambda_h, r_h_km, t_h, mu, constants=CODATA1986):
    """Body mass M_p (kg) from the half-light energy ratio  [Eq. 60 inverted].

    lambda_h = G M_p mu m_amu / (k T_h r_h)  =>  M_p = lambda_h k T_h r_h / (G mu m_amu).
    r_h in km.
    """
    return (
        lambda_h
        * constants.boltzmann
        * t_h
        * (r_h_km * 1e3)
        / (constants.gravitational * mu * constants.amu)
    )


def lambda_from_mass(m_p, r_h_km, t_h, mu, constants=CODATA1986):
    """Half-light energy ratio lambda_h from the body mass  [Eq. 60]."""
    return (
        constants.gravitational
        * m_p
        * mu
        * constants.amu
        / (constants.boltzmann * t_h * (r_h_km * 1e3))
    )


#############################################################################
# geometry
#############################################################################
# Observer-plane geometry: shadow radii y(t) for the occultation chord.
#
# EPQ03 consumes shadow radii y_i in a plane through the shadow center
# perpendicular to the star direction (Sec. 3).  For a straight chord at
# constant shadow velocity the radius is  [EY92 Eq. 5.1]
#
#     y(t) = sqrt(rho_min^2 + v^2 (t - t_mid)^2).
#
# Anything more elaborate (full astrometric solutions) should be done
# upstream; every EPQ03 entry point also accepts y arrays directly.

def shadow_radius(t, rho_min, v, t_mid):
    """Shadow radius y(t), km  [EY92 Eq. 5.1].

    t : times, s.  rho_min : closest approach to shadow center, km.
    v : shadow velocity, km/s.  t_mid : time of closest approach, s.
    """
    t = np.asarray(t, dtype=float)
    return np.hypot(rho_min, v * (t - t_mid))


def split_immersion_emersion(t, *arrays, t_mid):
    """Split time-ordered arrays into immersion (t < t_mid) and emersion
    (t >= t_mid) halves, each ordered so that y decreases with index
    (EPQ03 Sec. 3 indexing: deeper shells have larger i).

    Immersion keeps time order (y already decreasing); emersion is
    reversed.  Returns (immersion_tuple, emersion_tuple), each holding
    (t, *arrays) subsets.
    """
    t = np.asarray(t, dtype=float)
    order = np.argsort(t, kind="stable")
    t_sorted = t[order]
    arrays_sorted = [np.asarray(a)[order] for a in arrays]
    before = t_sorted < t_mid
    imm = (t_sorted[before],) + tuple(a[before] for a in arrays_sorted)
    em = (t_sorted[~before][::-1],) + tuple(a[~before][::-1] for a in arrays_sorted)
    return imm, em


#############################################################################
# binning
#############################################################################
# Data averaging and shell construction (EPQ03 Sec. 3).
#
# Three distinct averaging operations, in the order the paper applies them:
#
# 1. ``bin_uniform`` — optional resolution reduction in the observer plane
#    (paragraph after Eq. 32; also Sec. 5's "reduce the resolution first").
# 2. ``average_until_positive`` — merge adjacent integration intervals until
#    every averaged flux is positive (paragraph after Eq. 32), so that the
#    radius scale of Eq. (44) is monotonic.
# 3. ``bin_shells`` — after a full-resolution radius scale has been computed,
#    merge adjacent atmospheric shells until each merged shell is at least
#    the requested thickness (Sec. 3.2 and the paragraph after Eq. 48).
#
# Plus the bookkeeping that turns a flux/shadow-radius series into the
# shell data set of Eq. (34): ``boundary_index`` and ``shell_edges``
# (Eq. 33 with the inversion-boundary convention of Sec. 3.1).
#
# Conventions: input series are ordered with y strictly decreasing
# (immersion order; Sec. 3), so delta-y and delta-r are negative.  Flux
# errors combine as independent Gaussians: an average of k points has
# sigma = sqrt(sum sigma_j^2)/k.

def _check_descending(y):
    y = np.asarray(y, dtype=float)
    if y.ndim != 1 or y.size < 2:
        raise ValueError("y must be a 1-D array with at least 2 points")
    if not np.all(np.diff(y) < 0.0):
        raise ValueError("y must be strictly decreasing (immersion order)")
    return y


def bin_uniform(y, flux, sigma, n_per_bin):
    """Average every ``n_per_bin`` consecutive points (trailing remainder
    dropped).  y of the bin is the midpoint of its first and last members;
    flux is the plain mean; sigma = sqrt(sum sigma^2)/n.

    [Paragraph after Eq. 32: an arbitrary number of adjacent points may be
    averaged.]
    """
    y = _check_descending(y)
    flux = np.asarray(flux, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    n = int(n_per_bin)
    if n < 1:
        raise ValueError("n_per_bin must be >= 1")
    if n == 1:
        return y.copy(), flux.copy(), sigma.copy()
    m = (y.size // n) * n
    yk = y[:m].reshape(-1, n)
    fk = flux[:m].reshape(-1, n)
    sk = sigma[:m].reshape(-1, n)
    return (
        0.5 * (yk[:, 0] + yk[:, -1]),
        fk.mean(axis=1),
        np.sqrt((sk**2).sum(axis=1)) / n,
    )


def average_until_positive(y, flux, sigma):
    """Merge each non-positive flux with the following point(s) until the
    average is positive  [paragraph after Eq. 32].

    Greedy forward pass: starting at each output bin, points are
    accumulated until the running mean is positive; the bin's y is the
    midpoint of its first and last members.  A trailing group that never
    turns positive is dropped (a non-positive shell flux has no valid
    radius and cannot be inverted).
    """
    y = _check_descending(y)
    flux = np.asarray(flux, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    out_y, out_flux, out_sigma = [], [], []
    i = 0
    n = y.size
    while i < n:
        j = i + 1
        total = flux[i]
        var = sigma[i] ** 2
        while total <= 0.0 and j < n:
            total += flux[j]
            var += sigma[j] ** 2
            j += 1
        if total <= 0.0:
            break  # trailing non-positive group: drop
        k = j - i
        out_y.append(0.5 * (y[i] + y[j - 1]))
        out_flux.append(total / k)
        out_sigma.append(np.sqrt(var) / k)
        i = j
    return np.asarray(out_y), np.asarray(out_flux), np.asarray(out_sigma)


def boundary_index(flux, level):
    """First index whose flux is at or below ``level`` — the first shell of
    the inversion region, i_b (Sec. 3.1).  Data above it (indices < i_b)
    belong to the boundary region and feed the boundary fit."""
    flux = np.asarray(flux, dtype=float)
    below = np.nonzero(flux <= level)[0]
    if below.size == 0:
        raise ValueError(f"no flux at or below the boundary level {level}")
    i_b = int(below[0])
    if i_b == 0:
        raise ValueError("boundary level reached at the first data point; "
                         "no boundary region remains for the fit")
    return i_b


def shell_edges(y, i_b):
    """Shell mid-values and widths for the inversion region  [Eq. 33].

    Shell boundaries are midpoints of adjacent data y's,
    y_{i+1/2} = (y_i + y_{i+1})/2; the top boundary of the first
    inversion shell is y_b = y_{i_b-1/2}.  The deepest data point has no
    lower boundary and is dropped (it seeds no complete shell).

    Returns (y_b, y_mid, delta_y, y_lower):
        y_b : shadow radius at the top of the inversion region.
        y_mid : data y of each complete shell  (y_i, i = i_b ... i_max-1).
        delta_y : shell widths Delta y_i = y_{i+1/2} - y_{i-1/2}  (negative).
        y_lower : lower shell-boundary radii y_{i+1/2}.
    """
    y = _check_descending(y)
    if not 1 <= i_b < y.size - 1:
        raise ValueError("i_b must leave at least one boundary point above "
                         "and one complete shell below")
    mid = 0.5 * (y[:-1] + y[1:])          # y_{i+1/2} between data i and i+1
    y_b = mid[i_b - 1]
    y_lower = mid[i_b:]                   # lower boundaries of shells i_b...
    edges = np.concatenate(([y_b], y_lower))
    delta_y = np.diff(edges)
    y_mid = y[i_b:-1]
    return float(y_b), y_mid, delta_y, y_lower


def bin_shells(y_mid, delta_y, flux, sigma, delta_r, min_shell):
    """Merge adjacent shells until each merged shell is at least
    ``min_shell`` thick in radius  [Sec. 3.2; paragraph after Eq. 48].

    ``delta_r`` is the full-resolution shell thickness (negative) from the
    radius scale; it is used only to decide the grouping.  The merged data
    set is returned as (y_mid, delta_y, flux, sigma): delta-y's sum, fluxes
    average (sigma = sqrt(sum sigma^2)/k), and the merged y_mid is the
    midpoint of the group's first and last y_mid.  A trailing group that
    does not reach ``min_shell`` is dropped.  Radii and thetas for the
    merged set are recomputed downstream from these arrays (Sec. 5's
    calculation flow).
    """
    y_mid = np.asarray(y_mid, dtype=float)
    delta_y = np.asarray(delta_y, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    delta_r = np.asarray(delta_r, dtype=float)
    if min_shell <= 0.0:
        raise ValueError("min_shell must be positive")

    out = []
    i = 0
    n = y_mid.size
    while i < n:
        j = i
        thickness = 0.0
        while j < n:
            thickness += -delta_r[j]
            j += 1
            if thickness >= min_shell:
                break
        if thickness < min_shell:
            break  # trailing under-thick group: drop
        k = j - i
        out.append((
            0.5 * (y_mid[i] + y_mid[j - 1]),
            delta_y[i:j].sum(),
            flux[i:j].mean(),
            np.sqrt((sigma[i:j] ** 2).sum()) / k,
        ))
        i = j
    if not out:
        raise ValueError("no shell reached the requested minimum thickness")
    y_out, dy_out, f_out, s_out = (np.asarray(c) for c in zip(*out))
    return y_out, dy_out, f_out, s_out


#############################################################################
# boundary
#############################################################################
# Boundary condition for the atmosphere above the inversion region.
#
# EPQ03's inversion integrals need the structure of the atmosphere from the
# boundary radius r_b out to (formally) infinite radius, supplied as a
# *boundary condition* (Sec. 3.1): the boundary radius r_b for a given
# shadow radius y_b, and the refraction angle theta_b(r) with its radial
# derivative for r >= r_b.  Any object with the four methods
#
#     r_boundary(y_b), theta(r), dtheta_dr(r), flux(y)
#
# can serve; ``EY92PowerLawBoundary`` implements the paper's worked example
# (Sec. 3.3): the EY92 power-law thermal model with parameters
# (r_h, lambda_hi, b), using the corrected asymptotic series of EPQ03
# Eqs. (64)/(67) through ``jlegroup.EY92``.
#
# Half-light refractivity: EPQ03 Eq. (66) as printed differs from EY92
# Eq. (4.28) evaluated at f = 1/2 by dropping the A-series factors — a
# relative difference of ~5e-4 for the standard test case, i.e. exactly at
# the accuracy the method itself achieves.  Both are available via
# ``nu_h_method``; the default is EY92 (4.28), which is self-consistent
# with light curves generated from the EY92 model (see FIDELITY.md).
#
# Equation numbers refer to EPQ03 unless prefixed EY92.

@dataclass(frozen=True)
class EY92PowerLawBoundary:
    """EY92 power-law thermal model as the boundary condition  [Sec. 3.3].

    r_h : half-light radius, km.
    lambda_hi : equivalent-isothermal energy ratio at half light (the
        quantity EY92 recommend fitting; Eq. 62 converts to lambda_h).
    b : temperature power-law exponent  [Eq. 59].  For an isothermal
        boundary fix b = 0.
    d : observer-body distance, km.
    order : asymptotic-series truncation order (paper standard: 2).
    variant : "corrected" (EPQ03 Eqs. 64/67) or "as-printed" (EY92
        journal coefficients).
    nu_h_method : "ey92-4.28" (default) or "epq03-66"; see module
        docstring.

    Note (Sec. 7.7): the model carries no atmospheric mass, so its far
    tail is unphysical; for b <= -1 theta_b(r) does not vanish at large
    r in the analytic form.  The boundary integrals handle this with a
    finite outer cutoff where the integrand has decayed to nothing.
    """

    r_h: float
    lambda_hi: float
    b: float
    d: float
    order: int = 2
    variant: str = "corrected"
    nu_h_method: str = "ey92-4.28"

    @property
    def lambda_h(self):
        """Energy ratio at half light  [Eq. 62]:  lambda_h = lambda_hi - 5b/2."""
        return self.lambda_hi - 2.5 * self.b

    @property
    def nu_h(self):
        """Refractivity at half light  [Eq. 66 or EY92 Eq. 4.28]."""
        if self.lambda_h <= 6.0:
            raise ValueError(
                f"lambda_h = {self.lambda_h:.3g}: the EY92 half-light "
                "relation (Eq. 4.27 discriminant) requires lambda_h > ~5.8; "
                "such a weakly bound atmosphere cannot serve as this "
                "boundary condition"
            )
        if self.nu_h_method == "ey92-4.28":
            return float(
                EY92.refractivity_at_flux_level(
                    self.r_h, 0.5, self.d, self.lambda_h, 0.0, self.b,
                    self.order, self.variant,
                )
            )
        if self.nu_h_method == "epq03-66":
            lam = self.lambda_h
            b_series = float(
                EY92.b_series(1.0 / lam, 0.0, self.b, self.order, self.variant)
            )
            x = 1.0 / (lam * b_series)
            bracket = 1.0 - x - np.sqrt(1.0 - 6.0 * x + x * x)
            return float(
                self.r_h / (2.0 * self.d * np.sqrt(2.0 * np.pi * lam)) * bracket
            )
        raise ValueError(f"unknown nu_h_method {self.nu_h_method!r}")

    # -- the four methods the inversion consumes ---------------------------

    def _args(self):
        return (self.r_h, self.nu_h, self.lambda_h, 0.0, self.b,
                self.order, self.variant)

    def lambda_of(self, r):
        """lambda(r) = lambda_h (r/r_h)^-(1+b)  [Eq. 61]."""
        return EY92.lambda_g(r, self.r_h, self.lambda_h, 0.0, self.b)

    def nu_of(self, r):
        """nu(r) = nu_h (r/r_h)^b exp[(lambda(r) - lambda_h)/(1+b)]  [Eq. 65]."""
        return EY92.refractivity_profile(r, *self._args()[:5])

    def theta(self, r):
        """theta_b(r) = -sqrt(2 pi lambda(r)) nu(r) A(r, b)  [Eq. 63], rad."""
        return EY92.bending_angle(r, *self._args())

    def dtheta_dr(self, r):
        """dtheta_b/dr = sqrt(2 pi lambda(r)^3) nu(r)/r B(r, b)  [Eq. 69], rad/km."""
        return EY92.D_dtheta_dr(r, self.d, *self._args()) / self.d

    def r_boundary(self, y_b):
        """Boundary radius: solve r_b = y_b - D theta_b(r_b)  [Eq. 68], km."""
        return float(EY92.r_of_rho(y_b, self.d, *self._args()))

    def flux(self, y):
        """Refraction-only model flux at shadow radius y (for the boundary
        fit and for checking Eq. (5) consistency)  [EY92 Eq. 4.25, tau=0]."""
        r = EY92.r_of_rho(np.asarray(y, dtype=float), self.d, *self._args())
        return EY92.phi_ref(r, self.d, *self._args())


#############################################################################
# fit
#############################################################################
# Least-squares fit of the boundary model to the upper light curve.
#
# EPQ03 Sec. 3.3 establishes the boundary condition by fitting the EY92
# light-curve model to the portion of the light curve above the inversion
# boundary; Sec. 4.1 then needs the fitted parameter values, their formal
# errors, and their correlation matrix.  This module supplies that fit:
# a weighted nonlinear least-squares (scipy Levenberg-Marquardt) of
#
#     m(y) = background + slope (y - y_ref) + (full_scale - background) phi(y)
#
# to the boundary-region fluxes, where phi(y) is the refraction-only EY92
# model flux of ``EY92PowerLawBoundary`` with atmospheric parameters
# (r_h, lambda_hi, b)  [Eqs. 59-69].  Signal parameters follow EY92
# Eqs. (5.3)-(5.4) with the y-domain standing in for time; y_ref is the
# mean y of the fit region (the analog of EY92's t_av, chosen to
# decorrelate the slope).
#
# Any subset of the six parameters may be fitted, the rest held fixed
# (e.g. fix b = 0 for an isothermal boundary, fix the signal parameters
# for pre-normalized fluxes).  Formal errors and correlations come from
# the weighted Jacobian at the solution, (J^T J)^-1 in units of the
# supplied flux sigmas (Bevington & Robinson conventions, as the papers
# use); if no sigmas are given the covariance is scaled by the reduced
# chi-square.

#: Canonical parameter order.
PARAMETER_NAMES = ("background", "slope", "full_scale", "r_h", "lambda_hi", "b")

#: Parameters that define the boundary condition itself (EPQ03 Sec. 4.1's
#: q_j for the worked example).
ATMOSPHERIC_NAMES = ("r_h", "lambda_hi", "b")


@dataclass(frozen=True)
class BoundaryFitResult:
    """Fitted boundary model: values, formal errors, correlations  [Sec. 4.1]."""

    values: dict            # all six parameters at the solution
    fitted: tuple           # names of the fitted subset, canonical order
    errors: dict            # formal errors of the fitted parameters
    correlation: np.ndarray  # correlation matrix of the fitted subset
    covariance: np.ndarray   # covariance matrix of the fitted subset
    rms: float              # rms flux residual of the fit
    chi2: float             # sum of squared weighted residuals
    dof: int
    y_ref: float
    d: float
    order: int = 2
    variant: str = "corrected"
    nu_h_method: str = "ey92-4.28"
    scaled_by_reduced_chi2: bool = False

    @property
    def boundary(self):
        """EY92PowerLawBoundary at the best-fit atmospheric parameters."""
        return EY92PowerLawBoundary(
            r_h=self.values["r_h"], lambda_hi=self.values["lambda_hi"],
            b=self.values["b"], d=self.d, order=self.order,
            variant=self.variant, nu_h_method=self.nu_h_method,
        )

    def atmospheric_subset(self):
        """(names, values, errors, correlation) restricted to the fitted
        atmospheric parameters (r_h, lambda_hi, b) — the boundary-parameter
        set whose covariance feeds the inversion error propagation."""
        names = tuple(n for n in self.fitted if n in ATMOSPHERIC_NAMES)
        idx = [self.fitted.index(n) for n in names]
        corr = self.correlation[np.ix_(idx, idx)]
        return (
            names,
            np.array([self.values[n] for n in names]),
            np.array([self.errors[n] for n in names]),
            corr,
        )


def _model_flux(y, params, y_ref, d, order, variant, nu_h_method):
    bc = EY92PowerLawBoundary(
        r_h=params["r_h"], lambda_hi=params["lambda_hi"], b=params["b"],
        d=d, order=order, variant=variant, nu_h_method=nu_h_method,
    )
    stellar = bc.flux(y)
    return (
        params["background"]
        + params["slope"] * (y - y_ref)
        + (params["full_scale"] - params["background"]) * stellar
    )


def initial_guess(y, flux):
    """Starting (r_h, lambda_hi) from the curve shape: the half-light
    shadow radius by interpolation and the slope there via EY92 Eq. (5.10)
    large-planet limit dphi/dy ~ 1/(8 H_iso), then r_h ~ y_h + H_iso
    [EY92 Eq. 6.3].  The curve is smoothed first so noisy data yield a
    usable starting point (this is only a basin guess for the fit)."""
    y = np.asarray(y, dtype=float)
    flux = np.asarray(flux, dtype=float)
    window = max(1, flux.size // 60)
    if window > 1:
        # edge-replicated running mean (plain convolution's zero padding
        # would fake a half-light crossing at the array ends)
        padded = np.concatenate(
            (np.full(window, flux[:window].mean()), flux,
             np.full(window, flux[-window:].mean()))
        )
        smooth = np.convolve(padded, np.ones(window) / window, mode="same")
        smooth = smooth[window:-window]
    else:
        smooth = flux
    # y decreasing, smoothed flux ~decreasing: interpolate on flux ascending
    order = np.argsort(smooth)
    levels = np.interp([0.5, 0.6, 0.8], smooth[order], y[order])
    y_h, lo, hi = (float(v) for v in levels)
    slope = 0.20 / max(hi - lo, 1e-3)
    h_iso = min(max(1.0 / (8.0 * slope), 1e-3 * y_h), 0.5 * y_h)
    r_h = y_h + h_iso
    return r_h, float(np.clip(r_h / h_iso, 3.0, 1e4))


def fit_boundary(
    y,
    flux,
    sigma=None,
    *,
    d,
    fit=("r_h", "lambda_hi", "b"),
    initial=None,
    order=2,
    variant="corrected",
    nu_h_method="ey92-4.28",
    **least_squares_kwargs,
):
    """Fit the boundary model to the boundary-region light curve.

    y, flux, sigma : boundary-region data (above the inversion boundary).
        sigma None (or all zero) means unweighted; the covariance is then
        scaled by the reduced chi-square.
    d : observer-body distance, km.
    fit : names of parameters to fit, from PARAMETER_NAMES.
    initial : dict of starting/fixed values.  Defaults: background 0,
        slope 0, full_scale 1, b 0, and (r_h, lambda_hi) auto-estimated
        from the curve shape.
    """
    y = np.asarray(y, dtype=float)
    flux = np.asarray(flux, dtype=float)
    fitted = tuple(n for n in PARAMETER_NAMES if n in fit)
    if set(fit) - set(fitted):
        raise ValueError(f"unknown parameter(s) {set(fit) - set(fitted)}")

    r_h0, lambda_hi0 = initial_guess(y, flux)
    params = {
        "background": 0.0, "slope": 0.0, "full_scale": 1.0,
        "r_h": r_h0, "lambda_hi": lambda_hi0, "b": 0.0,
    }
    if initial:
        params.update(initial)

    unweighted = sigma is None or not np.any(np.asarray(sigma) > 0)
    weights = (
        np.ones_like(flux) if unweighted else 1.0 / np.asarray(sigma, dtype=float)
    )
    y_ref = float(y.mean())

    def residuals(x):
        p = dict(params)
        p.update(zip(fitted, x))
        # barrier against unphysical excursions of the LM steps: below
        # lambda_h ~ 5.8 the half-light relation (EY92 Eq. 4.27) has no
        # real solution (its discriminant 1 - 6X + X^2 goes negative), and
        # the r(rho) solver has nothing to converge to
        lambda_h = p["lambda_hi"] - 2.5 * p["b"]  # [Eq. 62]
        if not (0.0 < p["r_h"] and p["lambda_hi"] > 6.2 and lambda_h > 6.2
                and -25.0 < p["b"] < 25.0):
            return np.full_like(flux, 1e6)
        try:
            model = _model_flux(y, p, y_ref, d, order, variant, nu_h_method)
        except (RuntimeError, FloatingPointError):
            return np.full_like(flux, 1e6)
        if not np.all(np.isfinite(model)):
            return np.full_like(flux, 1e6)
        return (model - flux) * weights

    # Stage the fit when the weakly-constrained thermal exponent b is free
    # (EY92 Sec. 9: b has formal errors of order unity even for clean
    # data): first solve the better-conditioned problem with b held at its
    # starting value, then release b from that solution.
    if "b" in fitted and len(fitted) > 1:
        pre = tuple(n for n in fitted if n != "b")

        def pre_residuals(x):
            return residuals(
                np.array([x[pre.index(n)] if n in pre else params[n]
                          for n in fitted])
            )

        x_pre = np.array([params[n] for n in pre])
        stage = least_squares(pre_residuals, x_pre, method="lm")
        if stage.success:
            params.update(zip(pre, stage.x))

    x0 = np.array([params[n] for n in fitted])
    result = least_squares(residuals, x0, method="lm", **least_squares_kwargs)
    if not result.success:
        raise RuntimeError(f"boundary fit did not converge: {result.message}")

    values = dict(params)
    values.update(zip(fitted, result.x))
    dof = y.size - len(fitted)
    chi2 = float(2.0 * result.cost)

    jtj = result.jac.T @ result.jac
    covariance = np.linalg.inv(jtj)
    if unweighted and dof > 0:
        covariance = covariance * (chi2 / dof)
    err = np.sqrt(np.diag(covariance))
    correlation = covariance / np.outer(err, err)

    model = _model_flux(y, values, y_ref, d, order, variant, nu_h_method)
    return BoundaryFitResult(
        values=values,
        fitted=fitted,
        errors=dict(zip(fitted, err)),
        correlation=correlation,
        covariance=covariance,
        rms=float(np.sqrt(np.mean((model - flux) ** 2))),
        chi2=chi2,
        dof=dof,
        y_ref=y_ref,
        d=d,
        order=order,
        variant=variant,
        nu_h_method=nu_h_method,
        scaled_by_reduced_chi2=unweighted and dof > 0,
    )


#############################################################################
# inversion
#############################################################################
# Core inversion: radius scale, summation terms, boundary integrals, profiles.
#
# Implements EPQ03 Sec. 3: the radius scale within the atmosphere from the
# cumulative flux sums (Eqs. 43-48), the refractivity and pressure summations
# over the inversion shells (Eqs. 49-53), the boundary integrals over the
# atmosphere above the inversion region (Eqs. 35-36), and the atmospheric
# profiles at the lower boundary of each shell (Eqs. 54-58).
#
# Conventions (Sec. 3): series ordered deeper-ward, y strictly decreasing,
# so delta_y, delta_r, delta_theta are negative and theta itself is negative
# (rays bend toward the body).  Distances km, angles rad.

@dataclass(frozen=True)
class ShellGrid:
    """Radius scale and refraction angles on the inversion shells.

    All arrays run over the inversion shells i = i_b ... (deepest complete
    shell); "lower" quantities are at the shell's lower boundary i+1/2.
    """

    y_b: float            # shadow radius at the top of the inversion region
    r_b: float            # closest-approach radius of the y_b ray  [Eq. 68]
    theta_b: float        # refraction angle of the boundary ray  [Eq. 1]
    d: float              # observer-body distance, km
    y_mid: np.ndarray     # y_i
    delta_y: np.ndarray   # Delta y_i  (negative)
    flux: np.ndarray      # phi_i
    sigma: np.ndarray     # sigma(phi_i)
    y_lower: np.ndarray   # y_{i+1/2}
    r_lower: np.ndarray   # r_{i+1/2}  [Eq. 44]
    r_mid: np.ndarray     # r_i        [Eq. 46]
    delta_r: np.ndarray   # Delta r_i  [Eq. 45]  (negative)
    theta_lower: np.ndarray  # theta_{i+1/2}  [Eq. 47]
    delta_theta: np.ndarray  # Delta theta_i  [Eq. 48]  (negative)


def radius_scale(r_b, y_b, y_mid, delta_y, flux, sigma, d):
    """Radius scale and refraction angles from the flux sums  [Eqs. 43-48].

    r_b, y_b : boundary radius (body plane) and shadow radius (observer
        plane) at the top of the inversion region.
    y_mid, delta_y, flux, sigma : the shell data set of Eq. (34) below the
        boundary (from ``binning.shell_edges`` / ``binning.bin_shells``).
    d : observer-body distance, km.
    """
    y_mid = np.asarray(y_mid, dtype=float)
    delta_y = np.asarray(delta_y, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    radicand = r_b**2 + 2.0 * np.cumsum(flux * y_mid * delta_y)  # [Eq. 44]
    if np.any(radicand <= 0.0):
        raise ValueError(
            "Eq. (44) radicand went non-positive: the light curve integrates "
            "past the shadow center; end the inversion region higher"
        )
    r_lower = np.sqrt(radicand)
    r_edges = np.concatenate(([r_b], r_lower))
    delta_r = np.diff(r_edges)                    # [Eq. 45]
    r_mid = r_edges[:-1] + 0.5 * delta_r          # [Eq. 46]
    y_lower = y_b + np.cumsum(delta_y)            # y_{i+1/2}
    theta_lower = (y_lower - r_lower) / d         # [Eq. 47]
    theta_b = (y_b - r_b) / d                     # [Eq. 1]
    delta_theta = (delta_y - delta_r) / d         # [Eq. 48]

    return ShellGrid(
        y_b=float(y_b), r_b=float(r_b), theta_b=float(theta_b), d=float(d),
        y_mid=y_mid, delta_y=delta_y, flux=flux, sigma=sigma,
        y_lower=y_lower, r_lower=r_lower, r_mid=r_mid, delta_r=delta_r,
        theta_lower=theta_lower, delta_theta=delta_theta,
    )


# ===========================================================================
# Summation terms over the inversion shells  (Eqs. 49-53)
# ===========================================================================
#
# The refractivity and pressure integrals over the inversion region are sums
# of Delta theta_j against kernels of r'/r  [Eqs. 49-50].  Following French
# et al. (1978), the kernel's r-dependence is integrated exactly across each
# shell before summing (the kernels change quickly near r'/r = 1), which
# amounts to replacing the kernel by its shell average  [Eqs. 51-53]:
#
#   <K>_j = [F(z_{j+}) - F(z_{j-})] / (z_{j+} - z_{j-}),   z = r'/r_{i+1/2},
#
# with F an antiderivative of the kernel:
#
#   K_nu(z) = arccosh(z)                     F_nu(z) = z arccosh z - sqrt(z^2-1)
#   K_p(z)  = arccosh(z) - sqrt(1 - z^-2)    F_p(z)  = z arccosh z
#                                                      - 2 sqrt(z^2-1) + arccos(1/z)
#
# (both antiderivatives vanish at z = 1, so the innermost shell needs no
# special casing; constants of integration cancel in the differences).


def _f_nu(z):
    """Antiderivative of arccosh(z); zero at z = 1."""
    zc = np.maximum(z, 1.0)
    return zc * np.arccosh(zc) - np.sqrt(zc * zc - 1.0)


def _f_p(z):
    """Antiderivative of arccosh(z) - sqrt(1 - z^-2); zero at z = 1."""
    zc = np.maximum(z, 1.0)
    return (
        zc * np.arccosh(zc)
        - 2.0 * np.sqrt(zc * zc - 1.0)
        + np.arccos(np.clip(1.0 / zc, -1.0, 1.0))
    )


def summation_terms(grid: ShellGrid):
    """Refractivity and pressure summations S_nu, S_p at every shell lower
    boundary  [Eqs. 52, 53].  Returns (s_nu, s_p), each shaped like
    grid.r_lower; s_p carries its 1/r prefactor (units 1/km)."""
    r = grid.r_lower
    edges = np.concatenate(([grid.r_b], r))
    n = r.size
    # z_{j -/+} = r_{j -/+ 1/2} / r_{i+1/2}; rows are targets i, columns shells j
    z_minus = edges[:-1][None, :] / r[:, None]
    z_plus = edges[1:][None, :] / r[:, None]
    include = np.tril(np.ones((n, n), dtype=bool))  # shells j <= i
    dz = z_plus - z_minus  # negative (radii decrease); never zero
    mean_nu = (_f_nu(z_plus) - _f_nu(z_minus)) / dz
    mean_p = (_f_p(z_plus) - _f_p(z_minus)) / dz
    weights = np.where(include, grid.delta_theta[None, :], 0.0)
    s_nu = -(1.0 / np.pi) * (mean_nu * weights).sum(axis=1)
    s_p = -(1.0 / np.pi) / r * (mean_p * weights).sum(axis=1)
    return s_nu, s_p


# ===========================================================================
# Boundary integrals over the atmosphere above r_b  (Eqs. 35-36)
# ===========================================================================


def outer_cutoff(bc, r_b, drop=1e-16, max_ratio=100.0):
    """Outer radius for the boundary integrals.

    The formal upper limit is infinite, but the EY92 boundary model carries
    no atmospheric mass and its far tail is unphysical (Sec. 7.7): nu(r)
    plateaus at nu_h e^{-lambda_h/(1+b)} instead of vanishing.  The
    integrand d theta_b/dr decays by ~e^-1 per local scale height for many
    tens of scale heights before that plateau matters, so we cut off where
    it has dropped by ``drop`` relative to its value at r_b (capped at
    ``max_ratio`` r_b).  The neglected tail is below ``drop`` x (cutoff
    span / scale height) relative — utterly negligible at 1e-16.
    """
    scale0 = abs(float(bc.dtheta_dr(r_b)))
    if not np.isfinite(scale0) or scale0 == 0.0:
        raise ValueError("boundary d theta/dr is not finite at r_b")
    h_b = r_b / float(bc.lambda_of(r_b))
    r = r_b
    while r < max_ratio * r_b:
        r += h_b
        if abs(float(bc.dtheta_dr(r))) < drop * scale0:
            return r
    return max_ratio * r_b


def boundary_integrals(bc, r_b, r_targets, epsrel=1e-11, r_cut=None):
    """Boundary integrals B_nu(r_b, r) and B_p(r_b, r)  [Eqs. 35, 36] for
    each target radius (r <= r_b), integrating the boundary model's
    d theta_b/dr from r_b outward.

    With d theta_b = (d theta_b/dr') dr' and theta_b(infinity) = 0, the
    theta-integrals of Eqs. (35)-(36) become radial integrals

        B_nu = (1/pi)     int_{r_b}  arccosh(r'/r) (d theta_b/dr') dr'
        B_p  = (1/(pi r)) int_{r_b} [arccosh(r'/r) - sqrt(1-(r/r')^2)]
                                     (d theta_b/dr') dr'

    (positive; d theta_b/dr > 0).  B_p carries its 1/r prefactor (1/km).
    """
    r_targets = np.atleast_1d(np.asarray(r_targets, dtype=float))
    if np.any(r_targets > r_b * (1.0 + 1e-12)):
        raise ValueError("boundary integrals need target radii r <= r_b")
    if r_cut is None:
        r_cut = outer_cutoff(bc, r_b)

    def integrands(rp):
        """Stacked [nu-kernels, p-kernels] x d theta/dr at radius rp."""
        weight = bc.dtheta_dr(rp)
        z = rp / r_targets
        k_nu = np.arccosh(z)
        k_p = k_nu - np.sqrt(1.0 - 1.0 / (z * z))
        return np.concatenate((k_nu, k_p)) * weight

    stacked, _ = quad_vec(integrands, r_b, r_cut, epsabs=0.0, epsrel=epsrel,
                          limit=400)
    n = r_targets.size
    b_nu = stacked[:n] / np.pi
    b_p = stacked[n:] / (np.pi * r_targets)
    return b_nu, b_p


# ===========================================================================
# Atmospheric profiles  (Eqs. 54-58)
# ===========================================================================


@dataclass(frozen=True)
class InversionProfiles:
    """Atmospheric profiles at the shell lower boundaries  [Eqs. 54-58].

    Units: number_density m^-3, pressure Pa, temperature K,
    scale_height km; b_p/s_p in km^-1.
    """

    grid: ShellGrid
    b_nu: np.ndarray
    s_nu: np.ndarray
    b_p: np.ndarray
    s_p: np.ndarray
    refractivity: np.ndarray
    number_density: np.ndarray
    pressure: np.ndarray
    temperature: np.ndarray
    scale_height: np.ndarray

    @property
    def pressure_microbar(self):
        """Pressure in microbar (1 ubar = 0.1 Pa), the papers' display unit."""
        return self.pressure / 0.1


def invert(grid: ShellGrid, bc, gas, m_p, constants=CODATA1986, epsrel=1e-11,
           r_cut=None):
    """Atmospheric profiles from the shell grid and boundary condition.

    grid : ShellGrid from ``radius_scale`` (possibly after shell binning).
    bc : boundary condition (theta/dtheta_dr/lambda_of over r >= r_b).
    gas : constants.Gas (mu, nu_STP).
    m_p : body mass, kg.
    constants : ConstantSet (G, k, L, amu).
    """
    s_nu, s_p = summation_terms(grid)
    b_nu, b_p = boundary_integrals(bc, grid.r_b, grid.r_lower, epsrel=epsrel,
                                   r_cut=r_cut)
    nu = b_nu + s_nu                                             # [Eq. 54]
    n = constants.loschmidt / gas.nu_stp * nu                    # [Eq. 55]
    ip_si = (b_p + s_p) * 1e-3                                   # km^-1 -> m^-1
    gm_mu = constants.gravitational * m_p * gas.mu * constants.amu
    pressure = (constants.loschmidt / gas.nu_stp) * gm_mu * ip_si  # [Eq. 56]
    temperature = (gm_mu / constants.boltzmann) * ip_si / nu       # [Eq. 57]
    scale_height = grid.r_lower**2 * (b_p + s_p) / nu              # [Eq. 58]
    return InversionProfiles(
        grid=grid, b_nu=b_nu, s_nu=s_nu, b_p=b_p, s_p=s_p,
        refractivity=nu, number_density=n, pressure=pressure,
        temperature=temperature, scale_height=scale_height,
    )


#############################################################################
# errors
#############################################################################
# Random-error propagation for the inversion profiles (EPQ03 Sec. 4).
#
# Every inverted quantity is a function of (1) the fitted boundary
# parameters q_j — with their formal errors and correlation matrix from the
# boundary fit — and (2) the (averaged) fluxes phi_k in the inversion
# region, with their individual standard deviations.  The standard
# deviation of a shell quantity x_i is the zeta operator  [Eq. 72]:
#
#     zeta(x_i)^2 = sum_jk (dx_i/dq_j)(dx_i/dq_k) rho_jk sigma(q_j) sigma(q_k)
#                 + sum_k  (dx_i/dphi_k)^2 sigma(phi_k)^2
#
# Derivatives are one-sided numerical differences, each fitted parameter
# and each flux stepped by one tenth of its error (Sec. 5), with *all*
# intermediate quantities — the boundary radius, the radius scale, the
# refraction angles, the summations, and the boundary integrals at the
# shifted radii — recalculated from the stepped values.  Stepping flux
# phi_k perturbs only shells at and below k, so the summation term is
# triangular (Sec. 4.2's derivative structure emerges automatically).
#
# Radius-scale errors are also computed in the paper's closed form
# [Eqs. 73-79]; the stepped operator reproduces them (see tests), and the
# closed form is what ``ErrorBudget.radius`` reports.
#
# The boundary-only / summation-only decompositions (the two terms of
# Eq. 72 separately) are reported for every quantity, as used in the
# paper's Figs. 10 and 17.

@dataclass(frozen=True)
class QuantityErrors:
    """sigma per shell: total and the two Eq.-72 contributions separately."""

    total: np.ndarray
    boundary_only: np.ndarray
    summation_only: np.ndarray


@dataclass(frozen=True)
class ErrorBudget:
    """Random errors of the inversion outputs  [Eqs. 76-98]."""

    sigma_r_b: float               # boundary-radius error  [via Eq. 71]
    radius: QuantityErrors         # sigma(r_{i+1/2})  [Eqs. 76-79]
    refractivity: QuantityErrors   # [Eq. 84]
    number_density: QuantityErrors  # [Eq. 81]
    pressure: QuantityErrors       # [Eq. 88]
    temperature: QuantityErrors    # [Eq. 92]
    scale_height: QuantityErrors   # [Eq. 98]
    step_fraction: float


# rows of the stacked core-output array
_R, _NU, _IP, _TCORE, _HCORE = range(5)


def _core_outputs(bc, y_b, y_mid, delta_y, flux, d, epsrel):
    """Recompute every flux/boundary-dependent intermediate  [Sec. 5]:
    radius scale -> summations -> boundary integrals at the (possibly
    shifted) shell radii.  Rows: r, (B+S)_nu, (B+S)_p, T-core, H-core."""
    r_b = bc.r_boundary(y_b)
    grid = radius_scale(
        r_b, y_b, y_mid, delta_y, flux, np.zeros_like(flux), d
    )
    s_nu, s_p = summation_terms(grid)
    b_nu, b_p = boundary_integrals(bc, r_b, grid.r_lower, epsrel=epsrel)
    nu = b_nu + s_nu
    ip = b_p + s_p
    return np.vstack((grid.r_lower, nu, ip, ip / nu, grid.r_lower**2 * ip / nu))


def _step_boundary(bc, name, delta):
    return dataclass_replace(bc, **{name: getattr(bc, name) + delta})


def propagate_errors(result, step_fraction=0.1, boundary_uncertainty=None,
                     epsrel=1e-11):
    """Error budget for an ``InversionResult``  [Sec. 4].

    result : InversionResult.
    step_fraction : numerical-derivative step as a fraction of each
        parameter's/flux's sigma (the paper uses 1/10).
    boundary_uncertainty : optional (names, sigmas, correlation) for the
        boundary parameters, overriding the fit's.  Names must be fields
        of the boundary condition (r_h, lambda_hi, b).  When neither a
        fit nor an override is available the boundary contribution is
        zero (e.g. an exactly-known boundary in numerical tests).
    """
    grid = result.grid
    bc = result.boundary
    base_args = (grid.y_b, grid.y_mid, grid.delta_y, grid.flux, grid.d)

    if boundary_uncertainty is not None:
        q_names, q_sigmas, q_corr = boundary_uncertainty
        q_sigmas = np.asarray(q_sigmas, dtype=float)
        q_corr = np.asarray(q_corr, dtype=float)
    elif result.fit is not None:
        q_names, _, q_sigmas, q_corr = result.fit.atmospheric_subset()
    else:
        q_names, q_sigmas = (), np.zeros(0)
        q_corr = np.zeros((0, 0))

    base = _core_outputs(bc, *base_args, epsrel)
    n_shells = base.shape[1]

    # --- boundary-parameter term of Eq. 72 (Eqs. 82/86/90/94) -------------
    live = [j for j, s in enumerate(q_sigmas) if s > 0.0]
    derivs = np.zeros((len(live), 5, n_shells))
    r_b_derivs = np.zeros(len(live))
    for row, j in enumerate(live):
        h = q_sigmas[j] * step_fraction
        bc_j = _step_boundary(bc, q_names[j], h)
        derivs[row] = (_core_outputs(bc_j, *base_args, epsrel) - base) / h
        r_b_derivs[row] = (bc_j.r_boundary(grid.y_b) - grid.r_b) / h
    if live:
        cov = (q_corr[np.ix_(live, live)]
               * np.outer(q_sigmas[live], q_sigmas[live]))
        var_q = np.einsum("jqn,kqn,jk->qn", derivs, derivs, cov)
        sigma_r_b = float(np.sqrt(r_b_derivs @ cov @ r_b_derivs))  # [Eq. 71]
    else:
        var_q = np.zeros((5, n_shells))
        sigma_r_b = 0.0

    # --- flux-summation term of Eq. 72 (Eqs. 83/87/91/95) -----------------
    var_phi = np.zeros((5, n_shells))
    for k in range(n_shells):
        s_k = grid.sigma[k]
        if s_k <= 0.0:
            continue
        h = s_k * step_fraction
        flux_k = grid.flux.copy()
        flux_k[k] += h
        d_k = (
            _core_outputs(bc, grid.y_b, grid.y_mid, grid.delta_y, flux_k,
                          grid.d, epsrel)
            - base
        ) / h
        var_phi += d_k**2 * s_k**2

    def budget(row, scale=1.0):
        return QuantityErrors(
            total=scale * np.sqrt(var_q[row] + var_phi[row]),
            boundary_only=scale * np.sqrt(var_q[row]),
            summation_only=scale * np.sqrt(var_phi[row]),
        )

    # --- radius errors in the paper's closed form  [Eqs. 76-79] -----------
    r_lower = grid.r_lower
    sq_q = (grid.r_b * sigma_r_b / r_lower) ** 2                  # [Eq. 77]
    sq_phi = (
        np.cumsum((grid.y_mid * grid.delta_y * grid.sigma) ** 2) / r_lower**2
    )                                                             # [Eq. 78]
    radius = QuantityErrors(
        total=np.sqrt(sq_q + sq_phi),                             # [Eq. 79]
        boundary_only=np.sqrt(sq_q),
        summation_only=np.sqrt(sq_phi),
    )

    gas, constants = result.gas, result.constants
    gm_mu = (constants.gravitational * result.m_p * gas.mu * constants.amu)
    density_scale = constants.loschmidt / gas.nu_stp             # [Eq. 81]
    pressure_scale = density_scale * gm_mu * 1e-3                # [Eq. 85]
    temperature_scale = gm_mu / constants.boltzmann * 1e-3       # [Eq. 92]

    return ErrorBudget(
        sigma_r_b=sigma_r_b,
        radius=radius,
        refractivity=budget(_NU),                                # [Eq. 84]
        number_density=budget(_NU, density_scale),               # [Eq. 81]
        pressure=budget(_IP, pressure_scale),                    # [Eq. 88]
        temperature=budget(_TCORE, temperature_scale),           # [Eq. 92]
        scale_height=budget(_HCORE),                             # [Eq. 98]
        step_fraction=step_fraction,
    )


#############################################################################
# pipeline
#############################################################################
# End-to-end inversion pipeline (EPQ03 Sec. 5, Fig. 3).
#
# The calculation flow the paper's Mathematica template performed, as one
# function: optional data averaging (resolution first, then positivity —
# Sec. 5), boundary selection, boundary least-squares fit (or a supplied
# boundary condition), radius scale, shell binning, and the profile
# calculation.  Error propagation attaches via ``propagate_errors``.

@dataclass(frozen=True)
class InversionResult:
    """Everything the inversion produced (Fig. 3's right-hand column)."""

    profiles: InversionProfiles   # nu, n, p, T, H on the final shells
    grid: ShellGrid               # final (shell-binned) grid
    grid_full: ShellGrid          # pre-shell-binning grid
    fit: BoundaryFitResult | None  # boundary fit (None if boundary supplied)
    boundary: object              # the boundary condition used
    i_b: int                      # first inversion index in the averaged data
    y: np.ndarray                 # averaged data actually used
    flux: np.ndarray
    sigma: np.ndarray
    gas: object                   # constants.Gas
    m_p: float                    # body mass, kg
    constants: object             # constants.ConstantSet

    @property
    def temperature(self):
        return self.profiles.temperature


def invert_light_curve(
    y,
    flux,
    sigma=None,
    *,
    d,
    gas,
    m_p,
    boundary=None,
    fit_params=("r_h", "lambda_hi", "b"),
    fit_initial=None,
    boundary_flux=0.5,
    i_b=None,
    n_per_bin=1,
    min_shell=1.0,
    order=2,
    variant="corrected",
    nu_h_method="ey92-4.28",
    constants=CODATA1986,
    epsrel=1e-11,
):
    """Invert a normalized occultation light curve.

    y, flux, sigma : shadow radii (km, strictly decreasing — immersion
        order), normalized fluxes, and their standard deviations (sigma
        None or all-zero for noiseless data).
    d : observer-body distance, km.
    gas : constants.Gas.  m_p : body mass, kg.
    boundary : a boundary condition (e.g. EY92PowerLawBoundary) to use
        directly; if None, one is fitted to the data above the boundary
        with ``fit_params``/``fit_initial``  [Sec. 3.3].
    boundary_flux : flux level starting the inversion region (i_b is the
        first point at or below it) unless ``i_b`` is given  [Sec. 7.3].
    n_per_bin : optional observer-plane averaging factor, applied before
        the positivity averaging  [Sec. 5].
    min_shell : minimum atmospheric shell thickness, km  [Sec. 3.2];
        None or 0 skips shell binning.
    order, variant, nu_h_method : EY92 series controls for a fitted
        boundary (ignored when ``boundary`` is supplied).
    """
    y = np.asarray(y, dtype=float)
    flux = np.asarray(flux, dtype=float)
    sigma = (
        np.zeros_like(flux) if sigma is None else np.asarray(sigma, dtype=float)
    )

    if n_per_bin > 1:
        y, flux, sigma = bin_uniform(y, flux, sigma, n_per_bin)
    y, flux, sigma = average_until_positive(y, flux, sigma)

    if i_b is None:
        i_b = boundary_index(flux, boundary_flux)

    fit_result = None
    if boundary is None:
        sigma_fit = sigma[:i_b] if np.any(sigma[:i_b] > 0.0) else None
        fit_result = fit_boundary(
            y[:i_b], flux[:i_b], sigma_fit, d=d, fit=fit_params,
            initial=fit_initial, order=order, variant=variant,
            nu_h_method=nu_h_method,
        )
        boundary = fit_result.boundary

    y_b, y_mid, delta_y, _ = shell_edges(y, i_b)
    r_b = boundary.r_boundary(y_b)
    grid_full = radius_scale(
        r_b, y_b, y_mid, delta_y, flux[i_b:-1], sigma[i_b:-1], d
    )

    if min_shell:
        ym, dy, f, s = bin_shells(
            grid_full.y_mid, grid_full.delta_y, grid_full.flux,
            grid_full.sigma, grid_full.delta_r, min_shell,
        )
        grid = radius_scale(r_b, y_b, ym, dy, f, s, d)
    else:
        grid = grid_full

    profiles = invert(grid, boundary, gas, m_p, constants, epsrel=epsrel)
    return InversionResult(
        profiles=profiles, grid=grid, grid_full=grid_full, fit=fit_result,
        boundary=boundary, i_b=int(i_b), y=y, flux=flux, sigma=sigma,
        gas=gas, m_p=m_p, constants=constants,
    )


#############################################################################
# synth
#############################################################################
# Synthetic occultation light curves for testing the inversion (EPQ03 Sec. 6).
#
# Generates observer-plane light curves phi(y) from the EY92 power-law
# small-body model (via ``jlegroup.EY92``), parameterized the EPQ03 way:
# half-light radius r_h, half-light energy ratio lambda_h, thermal-gradient
# exponent b  [Sec. 3.3].  Includes the paper's standard test case (Table 1),
# background-limited Gaussian noise and the per-scale-height signal-to-noise
# ratio (S/N)_H  [Sec. 6.3, Eq. 99], and the ray-crossing criterion  [Eq. 100].
#
# Equation numbers refer to EPQ03 unless prefixed EY92.

@dataclass(frozen=True)
class ModelAtmosphere:
    """EY92 power-law atmosphere in EPQ03's half-light parameterization.

    r_h : half-light radius, km (refractive flux 0.5, focusing included).
    lambda_h : energy ratio at r_h  [Eq. 60].  (For b = 0 this equals the
        "equivalent isothermal" lambda_hi; otherwise lambda_h =
        lambda_hi - 5 b / 2  [Eq. 62].)
    b : temperature power-law index  [Eq. 59].
    d : observer-body distance, km.
    t_h : temperature at r_h, K (only fixes the body mass via Eq. 60).
    gas : Gas (mu, nu_STP).
    order : truncation order of the EY92 asymptotic series (paper
        standard test case: 2).
    variant : "corrected" (EPQ03 Eqs. 64/67) or "as-printed" (EY92
        journal text) series coefficients.
    """

    r_h: float
    lambda_h: float
    b: float
    d: float
    t_h: float = 80.0
    gas: Gas = GASES["N2"]
    order: int = 2
    variant: str = "corrected"

    @property
    def lambda_hi(self):
        """Equivalent isothermal energy ratio  [Eq. 62]."""
        return self.lambda_h + 2.5 * self.b

    @property
    def h_h(self):
        """Pressure scale height at half-light, km: H_h = r_h/lambda_h."""
        return self.r_h / self.lambda_h

    @property
    def nu_h(self):
        """Refractivity at half-light  [Eq. 66 / EY92 Eq. 4.28]."""
        return float(
            EY92.refractivity_at_flux_level(
                self.r_h, 0.5, self.d, self.lambda_h, 0.0, self.b,
                self.order, self.variant,
            )
        )

    def mass(self, constants: ConstantSet = CODATA1986):
        """Body mass consistent with (lambda_h, r_h, t_h, mu)  [Eq. 60]."""
        return mass_from_lambda(
            self.lambda_h, self.r_h, self.t_h, self.gas.mu, constants
        )

    def _ey92_args(self):
        return (self.d, self.r_h, self.nu_h, self.lambda_h, 0.0, self.b,
                self.order, self.variant)

    def radius_of_y(self, y):
        """Body-plane radius r for shadow radius y  [EY92 Eq. 4.23 inverted]."""
        return EY92.r_of_rho(y, *self._ey92_args())

    def flux_of_radius(self, r):
        """Refraction-only normalized flux at body radius r  [EY92 Eq. 4.25, tau=0]."""
        return EY92.phi_ref(r, *self._ey92_args())

    def theta_of_radius(self, r):
        """Bending angle theta(r), rad (negative)  [Eq. 63 / EY92 Eq. 4.6]."""
        return EY92.bending_angle(
            r, self.r_h, self.nu_h, self.lambda_h, 0.0, self.b,
            self.order, self.variant,
        )

    def dtheta_dr_of_radius(self, r):
        """dtheta/dr, rad/km  [Eq. 69 / EY92 Eq. 4.9]."""
        return EY92.D_dtheta_dr(r, *self._ey92_args()) / self.d


#: EPQ03 Table 1 standard test case: isothermal 80 K, N2, 30 AU.
STANDARD_CASE = ModelAtmosphere(
    r_h=1200.0,
    lambda_h=40.0,
    b=0.0,
    d=30.0 * AU_KM,
    t_h=80.0,
    gas=GASES["N2"],
    order=2,
)

#: Standard-case sampling (Table 1): observer-plane resolution and shell floor, km.
STANDARD_DELTA_Y = 0.5
STANDARD_MIN_SHELL = 1.0
STANDARD_BOUNDARY_FLUX = 0.5


def standard_case_variant(**changes):
    """Standard case with fields replaced (e.g. b=3.0 keeps lambda_h fixed,
    matching Sec. 6.2.3's constant-binding-ratio trials)."""
    return dataclass_replace(STANDARD_CASE, **changes)


def model_flux(atm: ModelAtmosphere, y):
    """Normalized flux phi(y) at shadow radii y (km).  Returns (phi, r)."""
    r = atm.radius_of_y(np.asarray(y, dtype=float))
    return atm.flux_of_radius(r), r


def generate_light_curve(
    atm: ModelAtmosphere,
    y_top,
    y_bottom,
    delta_y=STANDARD_DELTA_Y,
):
    """Noiseless light curve on a uniform descending y-grid (immersion order:
    y decreasing with index, as EPQ03 Sec. 3 requires).  Returns (y, phi).

    Fluxes are instantaneous samples at the grid points; EPQ03's tests used
    0.5 km sampling of the model (Table 1, "radial resolution").
    """
    if y_top <= y_bottom:
        raise ValueError("require y_top > y_bottom")
    y = np.arange(y_top, y_bottom - 0.5 * delta_y, -delta_y)
    phi, _ = model_flux(atm, y)
    return y, phi


def sigma_from_snr_h(snr_h, h, delta_y):
    """Per-point flux rms for a given (S/N)_H  [Eq. 99]:
    (S/N)_H = sigma^-1 sqrt(H/Delta y)."""
    return np.sqrt(h / delta_y) / snr_h


def snr_h_from_sigma(sigma, h, delta_y):
    """(S/N)_H from the per-point flux rms  [Eq. 99]."""
    return np.sqrt(h / delta_y) / sigma


def add_noise(phi, sigma, rng):
    """Background-limited white Gaussian noise: constant rms, independent of
    the stellar flux  [Sec. 6.3].  Returns (noisy_phi, sigma_array)."""
    phi = np.asarray(phi, dtype=float)
    sigma_array = np.full_like(phi, float(sigma))
    return phi + rng.normal(0.0, sigma, phi.shape), sigma_array


def ray_crossing_free(atm: ModelAtmosphere, r):
    """True where no ray crossing occurs at the observer distance  [Eq. 100]:
    dtheta/dr >= -1/D."""
    return atm.dtheta_dr_of_radius(np.asarray(r, dtype=float)) >= -1.0 / atm.d


#############################################################################
# export
#############################################################################
# Tabular export of inversion results (the paper's Tables 11-14 layout).

def result_table(result, budget=None):
    """Column dict for the per-shell results, in the layout of EPQ03
    Tables 11-14: shadow radius, flux, radius, refractivity, number
    density, pressure, temperature, scale height — with errors when an
    ``ErrorBudget`` is supplied.

    Units mirror the paper's tables: km, normalized flux, refractivity
    dimensionless, number density cm^-3, pressure microbar, K, km.
    """
    g = result.grid
    p = result.profiles
    table = {
        "y_km": g.y_mid,
        "delta_y_km": g.delta_y,
        "flux": g.flux,
        "sigma_flux": g.sigma,
        "r_km": g.r_lower,
        "refractivity": p.refractivity,
        "number_density_cm3": p.number_density * 1e-6,
        "pressure_microbar": p.pressure_microbar,
        "temperature_K": p.temperature,
        "scale_height_km": p.scale_height,
    }
    if budget is not None:
        table["sigma_r_km"] = budget.radius.total
        table["sigma_refractivity"] = budget.refractivity.total
        table["sigma_number_density_cm3"] = budget.number_density.total * 1e-6
        table["sigma_pressure_microbar"] = budget.pressure.total / 0.1
        table["sigma_temperature_K"] = budget.temperature.total
        table["sigma_scale_height_km"] = budget.scale_height.total
        table["sigma_temperature_boundary_K"] = budget.temperature.boundary_only
        table["sigma_temperature_summation_K"] = (
            budget.temperature.summation_only
        )
    return table


def write_csv(path, result, budget=None):
    """Write the result table to CSV; returns the column names."""
    table = result_table(result, budget)
    names = list(table)
    rows = np.column_stack([np.asarray(table[k], dtype=float) for k in names])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(names)
        writer.writerows(rows.tolist())
    return names


#############################################################################
# plots
#############################################################################
# Standard plots for inversion results (optional; needs matplotlib).
#
# Three figures the papers use to present inversions: the light curve with
# the boundary-model fit, the temperature profile with error bars
# (Figs. 7/14/20 style), and the boundary/summation error decomposition
# (Figs. 10/17 style).

def _axes(ax):
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    return ax


def plot_light_curve(result, ax=None):
    """Data (with the boundary/inversion split) and the boundary model."""
    ax = _axes(ax)
    ax.plot(result.y, result.flux, ".", ms=2, color="0.6", label="data")
    if result.sigma.max() == 0.0:
        ax.plot(result.y, result.flux, "-", lw=0.5, color="0.6")
    y_fit = result.y[: result.i_b]
    ax.plot(y_fit, result.boundary.flux(y_fit), "-", color="C1", lw=1.2,
            label="boundary model")
    ax.axvline(result.grid.y_b, color="C3", lw=0.8, ls="--",
               label="inversion boundary")
    ax.set_xlabel("shadow radius y (km)")
    ax.set_ylabel("normalized stellar flux")
    ax.invert_xaxis()  # time runs left to right for immersion
    ax.legend(frameon=False, fontsize=8)
    return ax


def plot_temperature(result, budget=None, ax=None, **errorbar_kw):
    """Temperature vs radius with error bars (temperature on the abscissa,
    radius on the ordinate, as the papers plot profiles)."""
    ax = _axes(ax)
    r = result.grid.r_lower
    t = result.profiles.temperature
    if budget is not None:
        kw = dict(fmt=".", ms=3, lw=0.7, capsize=0, color="C0")
        kw.update(errorbar_kw)
        ax.errorbar(t, r, xerr=budget.temperature.total, **kw)
    else:
        ax.plot(t, r, ".-", ms=3, lw=0.7, color="C0")
    ax.set_xlabel("temperature (K)")
    ax.set_ylabel("radius (km)")
    return ax


def plot_error_decomposition(result, budget, quantity="temperature", ax=None):
    """Boundary vs summation error contributions vs radius (Fig. 10 style)."""
    ax = _axes(ax)
    q = getattr(budget, quantity)
    r = result.grid.r_lower
    ax.plot(q.boundary_only, r, "-", color="C1", label="boundary")
    ax.plot(q.summation_only, r, "-", color="C0", label="summation")
    ax.plot(q.total, r, "-", color="k", lw=1.4, label="total")
    ax.set_xlabel(f"{quantity} error contribution")
    ax.set_ylabel("radius (km)")
    ax.legend(frameon=False, fontsize=8)
    return ax
