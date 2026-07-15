"""CE97 Atmosphere builders: injected constants and the unified refractivity path.

Until 2026-07-15 the three Atmosphere classes hardcoded their own R/G/kB
(2014-2018 vintages) and converted nu_STP with a 1-bar reference state —
~1.3% above the package's 1-atm Loschmidt convention, the documented
"class-internal refractivity" gotcha. The migration injects a
physicalData.ConstantSet (default DEFAULT_CONSTANTS = CODATA2022) and
unifies the conversion, closing that gotcha: the class Refractivity column
now equals physicalData.refractivity() exactly, under either vintage.

Migration characterization (2026-07-15, vs the pre-migration hardcodes):
temperature columns unchanged; pressure/number-density shifted <= 2e-5
(kB -3.5e-7, G +2.2e-8 accumulated hydrostatically); refractivity column
scaled by x0.98692 (the 1-bar -> 1-atm fix). The pins below freeze the
post-migration default outputs.
"""
import numpy as np
import pytest

from jlegroup import CE97, physicalData
from jlegroup.physicalData import CODATA1986, CODATA2022, DEFAULT_CONSTANTS

COMMON = dict(
    referencePressure=17.18, referenceRadius=1200.0, planetRadius=1200.0,
    planetMass=2.1398e22, meanMolecularMass=physicalData.MOLAR_MASS["N2"],
    polarizability=None,
    refractivityAtSTP=physicalData.refractivitySTP("N2", 0.7),
    topOfAtmosphere=2500.0, resolution=1.0,
)


def iso_atmosphere(**kw):
    atm = CE97.Atmosphere(referenceTemperature=114.5, temperatureGradient=0.0,
                          **COMMON, **kw)
    atm.main()
    return atm


def test_default_constants_are_the_package_default():
    atm = iso_atmosphere()
    assert atm.constants is DEFAULT_CONSTANTS
    assert atm.r0 == float(DEFAULT_CONSTANTS.gas_constant)
    assert atm.G == float(DEFAULT_CONSTANTS.gravitational)
    assert atm.kB == float(DEFAULT_CONSTANTS.boltzmann)


@pytest.mark.parametrize("constants", [CODATA2022, CODATA1986],
                         ids=["codata2022", "codata1986"])
def test_refractivity_column_matches_physicaldata(constants):
    """The former ~1.3% class-internal gotcha is gone: identical conversion,
    whichever vintage is injected."""
    atm = iso_atmosphere(constants=constants)
    p = atm.atmosphericProfile
    nu = physicalData.refractivity(p["NumDensity"].to_numpy(), "N2", 0.7,
                                   constants=constants)
    assert p["Refractivity"].to_numpy() == pytest.approx(nu, rel=1e-14)


def test_default_output_pins():
    """Post-migration regression pins (CODATA-2022 defaults)."""
    p = iso_atmosphere().atmosphericProfile
    assert p["NumDensity"].iloc[0] == pytest.approx(1.0867618643280858e22, rel=1e-12)
    assert p["Refractivity"].iloc[0] == pytest.approx(1.2011729796843493e-07, rel=1e-12)
    assert p["NumDensity"].iloc[650] == pytest.approx(5.0155848934726104e16, rel=1e-12)
    assert p["Refractivity"].iloc[650] == pytest.approx(5.543611023816277e-13, rel=1e-12)


def test_vintage_switch_moves_hydrostatics_as_G_predicts():
    """CODATA-1986 G is 2.56e-4 smaller -> larger scale height -> higher
    pressure aloft; a per-mille-class effect over ~38 scale heights, and the
    temperature column must not move at all."""
    p22 = iso_atmosphere().atmosphericProfile
    p86 = iso_atmosphere(constants=CODATA1986).atmosphericProfile
    ratio = p86["Pressure"].iloc[1299] / p22["Pressure"].iloc[1299]
    assert 1.001 < ratio < 1.01
    assert (p86["Temperature"] == p22["Temperature"]).all()


def test_tprofile_class_accepts_constants():
    r = np.arange(1200.0, 1501.0, 1.0)
    t = np.full_like(r, 114.5)
    for constants in (None, CODATA1986):
        atm = CE97.AtmospherefromTprofile(
            referencePressure=17.18, referenceRadius=1200.0,
            temperatureProfile=t, radius=r, planetRadius=1200.0,
            planetMass=2.1398e22,
            meanMolecularMass=physicalData.MOLAR_MASS["N2"],
            polarizability=None,
            refractivityAtSTP=physicalData.refractivitySTP("N2", 0.7),
            constants=constants,
        )
        atm.main()
        used = constants or DEFAULT_CONSTANTS
        p = atm.atmosphericProfile
        nu = physicalData.refractivity(p["NumDensity"].to_numpy(), "N2", 0.7,
                                       constants=used)
        assert p["Refractivity"].to_numpy() == pytest.approx(nu, rel=1e-14)


def test_tpprofile_class_accepts_constants():
    pressures = np.geomspace(17.18, 1e-4, 40)
    temps = np.full_like(pressures, 114.5)
    atm = CE97.AtmospherefromTpProfile(
        referencePressure=17.18, referenceRadius=1200.0,
        temperatureProfile=temps, pressureProfile=pressures,
        planetRadius=1200.0, planetMass=2.1398e22,
        meanMolecularMass=physicalData.MOLAR_MASS["N2"],
        polarizability=None,
        refractivityAtSTP=physicalData.refractivitySTP("N2", 0.7),
        constants=CODATA1986,
    )
    atm.main()
    assert atm.constants is CODATA1986
    p = atm.atmosphericProfile
    nu = physicalData.refractivity(p["NumDensity"].to_numpy(), "N2", 0.7,
                                   constants=CODATA1986)
    assert p["Refractivity"].to_numpy() == pytest.approx(nu, rel=1e-14)
