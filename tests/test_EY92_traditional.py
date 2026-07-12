"""ElliotYoung1992ModelTraditional: half-light (rH, lambdaHiso, b) inputs.

Conventions under test (maintainer-specified, 2026-07-07):
  * lambdaHalfLight is the EQUIVALENT-ISOTHERMAL ratio r_h/H_iso
    (Mathematica jleGroup "lambdaHi"); the true energy ratio is
    lambda_true = lambdaHalfLight - (3a+5b)/2  [EY92 Eq. 5.13].
  * scaleHeight attribute = r_h/lambda_true, i.e. r_h/(lambdaHiso-5b/2)
    at a = 0.
  * The anchor is the refraction-only near-limb flux level (default
    1/2), independent of twoLimb/haze/surface options.

The equivalence test drives both parameterizations of the same
atmosphere through the full option set (twoLimb + haze + surface) and
requires identical light curves.
"""
import numpy as np
import pytest

from jlegroup import EY92, physicalData

D = 30 * physicalData.AU_KM


def _physical_model(b, a=0.0, position=None, **options):
    """A physical-input model (Pluto-like N2) used as ground truth."""
    return EY92.ElliotYoung1992Model(
        referencePressure=0.14,
        referenceTemperature=104.0,
        referenceRadius=1250.0,
        planetMass=1.3e22,
        meanMolecularMass=physicalData.MOLAR_MASS["N2"],
        planetDistance=D,
        position=position if position is not None else np.array([1200.0]),
        temperatureExponent=b,
        molecularWeightExponent=a,
        **options,
    )


def _half_light_of(model, f=0.5):
    """Refraction-only half-light radius and equivalent-isothermal lambda."""
    lo, hi = 0.3 * model.referenceRadius, 4.0 * model.referenceRadius
    args = (model.planetDistance, model.referenceRadius, model.nu0,
            model.lambda_g0, model.a, model.b)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if EY92.phi_ref(mid, *args) < f:
            lo = mid
        else:
            hi = mid
    r_h = 0.5 * (lo + hi)
    lambda_true = float(
        EY92.lambda_g(r_h, model.referenceRadius, model.lambda_g0, model.a, model.b)
    )
    lambda_iso = lambda_true + (3.0 * model.a + 5.0 * model.b) / 2.0
    return r_h, lambda_iso


@pytest.mark.parametrize("a,b", [(0.0, 0.0), (0.0, -0.6), (0.3, -0.4)])
def test_traditional_reproduces_physical_model_light_curve(a, b):
    """Same atmosphere, both parameterizations, full options on:
    identical curves to near machine precision."""
    t = np.arange(-200.0, 200.0, 2.0)
    position = np.sqrt(60.0**2 + (18.5 * t) ** 2)
    options = dict(
        twoLimb=True,
        surfaceRadius=1000.0,
        hazeOnsetRadius=1219.0,
        hazeKappa1=2.27e-3,
        hazeScaleHeight=29.3,
    )
    physical = _physical_model(b, a, position, **options)
    r_h, lambda_iso = _half_light_of(physical)

    traditional = EY92.ElliotYoung1992ModelTraditional(
        radiusHalfLight=r_h,
        lambdaHalfLight=lambda_iso,
        b=b,
        a=a,
        planetDistance=D,
        position=position,
        **options,
    )
    # The conversion reproduces (nu0, lambda_g0) at a different reference
    # radius (r_h vs 1250 km); compare the physics, not the labels.
    f_phys = physical.main()
    f_trad = traditional.main()
    assert f_trad == pytest.approx(f_phys, rel=5e-9)


@pytest.mark.parametrize("b", [0.0, -0.6])
def test_anchor_flux_level_at_half_light_radius(b):
    """The refraction-only flux at rho(r_h) must equal referenceFluxLevel,
    regardless of the haze/twoLimb/surface options."""
    m = EY92.ElliotYoung1992ModelTraditional(
        radiusHalfLight=1215.0,
        lambdaHalfLight=22.5,
        b=b,
        planetDistance=D,
        position=np.array([1000.0]),
        twoLimb=True,
        surfaceRadius=900.0,
        hazeOnsetRadius=1190.0,
        hazeKappa1=2.0e-3,
        hazeScaleHeight=30.0,
    )
    args = (D, m.referenceRadius, m.nu0, m.lambda_g0, m.a, m.b)
    flux_at_anchor = float(EY92.phi_ref(m.radiusHalfLight, *args))
    assert flux_at_anchor == pytest.approx(0.5, abs=1e-12)


def test_scale_height_attributes_follow_maintainer_convention():
    r_h, lam_iso, b = 1215.0, 22.5, -0.6
    m = EY92.ElliotYoung1992ModelTraditional(
        radiusHalfLight=r_h, lambdaHalfLight=lam_iso, b=b,
        planetDistance=D, position=np.array([1000.0]),
    )
    lam_true = lam_iso - 2.5 * b  # (3a+5b)/2 at a = 0
    assert m.lambdaTrueHalfLight == pytest.approx(lam_true, rel=1e-15)
    assert m.scaleHeight == pytest.approx(r_h / lam_true, rel=1e-15)
    assert m.isothermalScaleHeight == pytest.approx(r_h / lam_iso, rel=1e-15)
    # Isothermal, constant-mu: the conventions coincide.
    iso = EY92.ElliotYoung1992ModelTraditional(
        radiusHalfLight=r_h, lambdaHalfLight=lam_iso, b=0.0,
        planetDistance=D, position=np.array([1000.0]),
    )
    assert iso.lambdaTrueHalfLight == iso.lambdaHalfLight
    assert iso.scaleHeight == iso.isothermalScaleHeight


def test_parameterization_is_anchor_invariant_under_options():
    """(nu0, lambda_g0) must not depend on twoLimb/haze/surface."""
    common = dict(
        radiusHalfLight=1215.0, lambdaHalfLight=22.5, b=-0.6,
        planetDistance=D, position=np.array([800.0]),
    )
    bare = EY92.ElliotYoung1992ModelTraditional(**common)
    loaded = EY92.ElliotYoung1992ModelTraditional(
        **common, twoLimb=True, surfaceRadius=1000.0,
        hazeOnsetRadius=1219.0, hazeKappa1=2.27e-3, hazeScaleHeight=29.3,
    )
    assert bare.nu0 == loaded.nu0
    assert bare.lambda_g0 == loaded.lambda_g0


def test_no_physical_constants_needed():
    """The traditional parameterization must be constructible and
    runnable from geometry alone (no physicalData values enter)."""
    m = EY92.ElliotYoung1992ModelTraditional(
        radiusHalfLight=1215.0, lambdaHalfLight=21.0, b=0.0,
        planetDistance=4.323e9, position=np.array([900.0, 1200.0, 1500.0]),
    )
    flux = m.main()
    assert np.all((flux > 0) & (flux <= 1.0))


def test_unphysical_lambda_rejected():
    with pytest.raises(ValueError):
        EY92.ElliotYoung1992ModelTraditional(
            radiusHalfLight=1215.0, lambdaHalfLight=10.0, b=5.0,
            planetDistance=D, position=np.array([1000.0]),
        )


# ---------------------------------------------------------------------------
# ElliotYoung1992ModelTraditionalScaleHeight
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("a,b", [(0.0, 0.0), (0.0, -0.6), (0.3, -0.4)])
def test_scale_height_class_equivalent_to_lambda_class(a, b):
    """(rH, H) and (rH, lambdaHiso = rH/H + (3a+5b)/2) must be the same
    model, bit for bit."""
    t = np.arange(-150.0, 150.0, 2.0)
    position = np.sqrt(60.0**2 + (18.5 * t) ** 2)
    r_h, h = 1215.0, 55.0
    lam_iso = r_h / h + (3.0 * a + 5.0 * b) / 2.0
    options = dict(
        b=b, a=a, planetDistance=D, position=position,
        twoLimb=True, surfaceRadius=1000.0,
        hazeOnsetRadius=1219.0, hazeKappa1=2.27e-3, hazeScaleHeight=29.3,
    )
    via_h = EY92.ElliotYoung1992ModelTraditionalScaleHeight(
        radiusHalfLight=r_h, scaleHeightH=h, **options
    )
    via_lam = EY92.ElliotYoung1992ModelTraditional(
        radiusHalfLight=r_h, lambdaHalfLight=lam_iso, **options
    )
    assert via_h.nu0 == via_lam.nu0
    assert via_h.lambda_g0 == via_lam.lambda_g0
    assert np.array_equal(via_h.main(), via_lam.main())


def test_scale_height_class_attributes_round_trip():
    r_h, h, b = 1215.0, 55.0, -0.6
    m = EY92.ElliotYoung1992ModelTraditionalScaleHeight(
        radiusHalfLight=r_h, scaleHeightH=h, b=b,
        planetDistance=D, position=np.array([1000.0]),
    )
    # scaleHeight (inherited, = rH/lambda_true) equals the input exactly.
    assert m.scaleHeight == pytest.approx(h, rel=1e-15)
    assert m.scaleHeightH == h
    assert m.lambdaHalfLight == pytest.approx(r_h / h + 2.5 * b, rel=1e-15)
    assert m.lambdaTrueHalfLight == pytest.approx(r_h / h, rel=1e-15)


def test_scale_height_class_rejects_nonpositive_h():
    with pytest.raises(ValueError):
        EY92.ElliotYoung1992ModelTraditionalScaleHeight(
            radiusHalfLight=1215.0, scaleHeightH=-5.0, b=0.0,
            planetDistance=D, position=np.array([1000.0]),
        )
