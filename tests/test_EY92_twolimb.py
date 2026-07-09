"""Two-limb (central-flash) EY92: far-limb solver, flux, and CE97 cross-check.

The far limb is the set of rays bent across the shadow axis: their
signed observer-plane coordinate is -rho, so the far-limb periapsis
solves r + D theta(r) = -rho [EY92 Eq. 2.2 with the absolute value
retained; the Mathematica jleGroup olcTwoLimb4 evaluates its one-limb
model at -rho in exactly this way].

Primary validation is a method-independent cross-check against CE97,
which integrates the *same analytic refractivity profile* numerically
and whose ray_crossing map covers negative observer coordinates: CE97
evaluated at -rho returns the far-limb contribution (both its flux
factors carry np.abs).  Measured agreement (2026-07-07, iso profile,
lambda_g0 = 21):

    near limb  max rel diff = 2.6e-7    (rho = 150..1400 km)
    far limb   max rel diff = 1.2e-7

Tolerances below carry ~10x headroom.  Physics checks cover the 1/rho
central-flash divergence, far-limb decay, surface blocking, and haze
ordering (the far ray passes deeper, so haze extinguishes it first).
"""
import numpy as np
import pytest

from jlegroup import CE97, EY92

# Pluto-benchmark-like transparent isothermal atmosphere (a = b = 0).
R0, LG0, NU0, D = 1250.0, 21.0, 1.11e-9, 4.323e9
ARGS = (D, R0, NU0, LG0, 0.0, 0.0)


def test_far_limb_root_satisfies_signed_geometry():
    rho = np.array([50.0, 150.0, 300.0, 800.0, 1400.0, 2400.0])
    r_far = EY92.r_far_of_rho(rho, *ARGS)
    assert np.max(np.abs(EY92.rho_of_r(r_far, *ARGS) + rho)) < 1e-6  # km
    # Far rays pass deeper than near rays, and both above the focal radius
    # ordering: r_focal < r_far < r_near.
    r_near = EY92.r_of_rho(rho, *ARGS)
    assert np.all(r_far < r_near)


def test_far_limb_root_solvable_for_steep_gradient():
    """T ~ r^-4.5 (steep-clear regime): the analytic profile turns over
    far below the physical branch; the bracketed descent must still
    find the far root above the turnover."""
    lg0_steep, nu0_steep, r0_steep = 109.42, 7.634e-18, 1850.0
    args = (30 * 1.49597870691e8, r0_steep, nu0_steep, lg0_steep, 0.0, -4.5)
    rho = np.array([300.0, 900.0])
    r_far = EY92.r_far_of_rho(rho, *args)
    assert np.max(np.abs(EY92.rho_of_r(r_far, *args) + rho)) < 1e-6


def test_two_limb_cross_validates_against_ce97():
    """CE97 (numerical, same analytic nu(r) table) at +rho and -rho must
    reproduce the analytic near- and far-limb fluxes."""
    radius = np.arange(950.0, 2500.0, 1.0)
    nu = EY92.refractivity_profile(radius, R0, NU0, LG0, 0.0, 0.0)
    rho = np.array([150.0, 200.0, 300.0, 500.0, 800.0, 1100.0, 1400.0])

    model = CE97.ChamberlainElliot1997Model(
        refractivityProfile=nu,
        radialDistance=radius,
        planetDistance=D,
        position=np.concatenate([rho, -rho]),
    )
    model.main()
    ce = np.asarray(model.focusedFlux, dtype=float)
    ce_near, ce_far = ce[: len(rho)], ce[len(rho):]

    ey_near = EY92._limb_flux(EY92.r_of_rho(rho, *ARGS), *ARGS)
    ey_far = EY92._limb_flux(EY92.r_far_of_rho(rho, *ARGS), *ARGS)

    near_res = float(np.max(np.abs(ey_near / ce_near - 1)))
    far_res = float(np.max(np.abs(ey_far / ce_far - 1)))
    print(f"two-limb vs CE97: near max rel = {near_res:.2e}, "
          f"far max rel = {far_res:.2e}")
    assert near_res < 3e-6
    assert far_res < 3e-6
    # And the public two-limb total is exactly their sum.
    total = EY92.phi_two_limb(rho, *ARGS)
    assert total == pytest.approx(ey_near + ey_far, rel=1e-14)


def test_central_flash_diverges_as_inverse_rho():
    """Both limb radii converge to the focal radius as rho -> 0 and the
    total flux scales as 1/rho (geometric-optics focal line)."""
    products = [
        float(EY92.phi_two_limb(np.array([p]), *ARGS)[0]) * p
        for p in (3.0, 1.0, 0.1)
    ]
    assert products[0] == pytest.approx(products[2], rel=1e-4)
    assert products[1] == pytest.approx(products[2], rel=1e-5)
    r_near = float(EY92.r_of_rho(np.array([0.01]), *ARGS)[0])
    r_far = float(EY92.r_far_of_rho(np.array([0.01]), *ARGS)[0])
    assert r_near == pytest.approx(r_far, abs=0.03)  # both -> focal radius


def test_two_limb_reduces_to_near_limb_far_from_center():
    """Far contribution decays with rho; near limb -> 1 unocculted.
    (In a transparent gas sphere the far limb never reaches zero --
    that is what surface_radius is for.)"""
    rho = np.array([1400.0, 1800.0, 2400.0, 3000.0])
    near = EY92._limb_flux(EY92.r_of_rho(rho, *ARGS), *ARGS)
    far = EY92._limb_flux(EY92.r_far_of_rho(rho, *ARGS), *ARGS)
    assert np.all(np.diff(far) < 0)
    assert far[0] < 0.013
    assert near[-1] == pytest.approx(1.0, abs=1e-6)


def test_surface_radius_blocks_far_then_both_limbs():
    rho = np.array([300.0])
    r_near = float(EY92.r_of_rho(rho, *ARGS)[0])  # ~1120 km
    r_far = float(EY92.r_far_of_rho(rho, *ARGS)[0])  # ~1065 km
    near_only = EY92._limb_flux(np.array([r_near]), *ARGS)
    # Surface between the two periapses: far limb blocked exactly.
    surf = 0.5 * (r_far + r_near)
    two = EY92.phi_two_limb(rho, *ARGS, surface_radius=surf)
    assert two == pytest.approx(near_only, rel=1e-14)
    # Surface above both: fully occulted.
    assert float(
        EY92.phi_two_limb(rho, *ARGS, surface_radius=r_near + 10.0)[0]
    ) == 0.0


def test_haze_extinguishes_far_limb_first():
    """The far ray's periapsis is deeper, so its haze optical depth is
    larger and its transmission lower."""
    rho = np.array([400.0])
    r1, kappa1, h1 = 1219.0, 2.27e-3, 29.3
    clear_near = EY92._limb_flux(EY92.r_of_rho(rho, *ARGS), *ARGS)
    clear_far = EY92._limb_flux(EY92.r_far_of_rho(rho, *ARGS), *ARGS)
    hazy_near = EY92._limb_flux(
        EY92.r_of_rho(rho, *ARGS), *ARGS, r1=r1, kappa1=kappa1, h_tau1=h1
    )
    hazy_far = EY92._limb_flux(
        EY92.r_far_of_rho(rho, *ARGS), *ARGS, r1=r1, kappa1=kappa1, h_tau1=h1
    )
    assert hazy_far / clear_far < hazy_near / clear_near < 1.0


def test_facade_two_limb_wiring():
    from jlegroup import physicalData

    position = np.array([200.0, 500.0, 1000.0, 1600.0])
    common = dict(
        referencePressure=1.0,
        referenceTemperature=100.0,
        referenceRadius=1250.0,
        planetMass=1.3e22,
        meanMolecularMass=physicalData.MOLAR_MASS["N2"],
        planetDistance=D,
        position=position,
    )
    one = EY92.ElliotYoung1992Model(**common)
    two = EY92.ElliotYoung1992Model(**common, twoLimb=True)
    f1 = one.main()
    f2 = two.main()
    assert two.farLimbFlux is not None
    assert f2 == pytest.approx(two.nearLimbFlux + two.farLimbFlux, rel=1e-14)
    assert two.nearLimbFlux == pytest.approx(f1, rel=1e-14)
    assert np.all(f2 > f1)  # transparent sphere: far limb adds light
    assert np.all(two.farPlanetRadius_solution < two.planetRadius_solution)
