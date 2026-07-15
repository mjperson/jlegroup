"""physicalData: two-vintage pins, provenance policy, gas registry, re-exports.

The pinned values ARE the point. Two immutable vintages coexist:
CODATA-1986 (the jleGroup validation vintage — every reference light curve
and paper table was generated with it; archived in VINTAGE_1986 and the
CODATA1986 set) and CODATA-2022/SI-2019 (the module-level defaults, for
current research). NEITHER may drift: a silent change in either vintage
shifts results at up to the 2.6e-4 level (G). These tests make any drift a
loud failure.
"""
import numpy as np
import pytest

from jlegroup import EPQ03, physicalData
from jlegroup.physicalData import (
    CODATA1986,
    CODATA2022,
    DEFAULT_CONSTANTS,
    VINTAGE_1986,
    Constant,
    GASES,
    MOLAR_MASS,
)

# --- vintage pins (deliberate; see module docstring before touching) --------

PINNED_1986 = {
    "BOLTZMANN": 1.380658e-23,
    "GRAVITATIONAL": 6.67259e-11,
    "AVOGADRO": 6.0221367e23,
    "LOSCHMIDT": 2.686763e25,
    "AU_KM": 1.49597870691e8,
    "SPEED_OF_LIGHT": 2.99792458e8,
    "PLANCK": 6.6260755e-34,
    "STEFAN_BOLTZMANN": 5.67051e-8,
}

PINNED_2022 = {
    "BOLTZMANN": 1.380649e-23,
    "GRAVITATIONAL": 6.67430e-11,
    "AVOGADRO": 6.02214076e23,
    "LOSCHMIDT": 2.686780111e25,
    "AU_KM": 1.495978707e8,
    "SPEED_OF_LIGHT": 2.99792458e8,
    "PLANCK": 6.62607015e-34,
    "STEFAN_BOLTZMANN": 5.670374419e-8,
}


@pytest.mark.parametrize("name,value", sorted(PINNED_1986.items()))
def test_validation_vintage_pinned(name, value):
    """The frozen 1986 archive: the values the references were made with."""
    assert float(VINTAGE_1986[name]) == value


@pytest.mark.parametrize("name,value", sorted(PINNED_2022.items()))
def test_module_defaults_are_codata2022(name, value):
    assert float(getattr(physicalData, name)) == value


def test_constant_sets_wire_to_their_vintages():
    assert DEFAULT_CONSTANTS is CODATA2022
    assert float(CODATA1986.gravitational) == PINNED_1986["GRAVITATIONAL"]
    assert float(CODATA1986.boltzmann) == PINNED_1986["BOLTZMANN"]
    assert float(CODATA1986.loschmidt) == PINNED_1986["LOSCHMIDT"]
    assert float(CODATA1986.amu) == 1e-3 / PINNED_1986["AVOGADRO"]
    assert float(CODATA2022.gravitational) == PINNED_2022["GRAVITATIONAL"]
    assert float(CODATA2022.boltzmann) == PINNED_2022["BOLTZMANN"]
    assert float(CODATA2022.loschmidt) == PINNED_2022["LOSCHMIDT"]
    assert float(CODATA2022.amu) == 1e-3 / PINNED_2022["AVOGADRO"]


def test_derived_set_properties():
    """avogadro/gas_constant derive from amu (molar-mass-constant convention)."""
    assert CODATA1986.avogadro == pytest.approx(PINNED_1986["AVOGADRO"], rel=1e-12)
    assert CODATA1986.gas_constant == pytest.approx(8.31451, abs=2e-5)  # CODATA-1986 R
    assert CODATA2022.avogadro == pytest.approx(PINNED_2022["AVOGADRO"], rel=1e-12)
    # SI-2019 exact R = k_B N_A
    assert CODATA2022.gas_constant == pytest.approx(8.31446261815324, rel=1e-12)


def test_amu_derived_from_avogadro():
    assert float(physicalData.AMU) == 1e-3 / float(physicalData.AVOGADRO)
    # equals the CODATA-2022 measured m_u at the molar-mass-constant level
    assert physicalData.AMU == pytest.approx(1.66053906892e-27, rel=2e-9)
    # and the frozen 1986 record equals the 1986 printed value
    assert VINTAGE_1986["AMU"] == pytest.approx(1.6605402e-27, rel=1e-7)


# --- provenance policy -------------------------------------------------------

ALL_CONSTANTS = ["BOLTZMANN", "GRAVITATIONAL", "AVOGADRO", "LOSCHMIDT",
                 "AU_KM", "AMU", "SPEED_OF_LIGHT", "PLANCK", "STEFAN_BOLTZMANN"]


@pytest.mark.parametrize("name", ALL_CONSTANTS)
def test_constants_carry_provenance(name):
    c = getattr(physicalData, name)
    assert isinstance(c, Constant) and isinstance(c, float)
    assert c.symbol and c.units and c.source
    if c.uncertainty is None:
        assert c.note, f"{name}: uncertainty None requires an explanatory note"
    else:
        assert 0.0 < c.uncertainty < abs(float(c))


def test_constants_behave_as_floats_in_arithmetic():
    r_gas = physicalData.BOLTZMANN * physicalData.AVOGADRO
    assert r_gas == pytest.approx(8.31446261815324, rel=1e-12)  # SI-2019 exact R
    assert f"{physicalData.BOLTZMANN:.6e}" == "1.380649e-23"


def test_loschmidt_consistent_with_ideal_gas_at_classic_stp():
    """L = p/(kT) at 273.15 K, 101325 Pa within the 1986 rounding."""
    implied = 101325.0 / (float(physicalData.BOLTZMANN) * 273.15)
    assert implied == pytest.approx(float(physicalData.LOSCHMIDT), rel=2e-5)


# --- gas registry (EY92 Table 9) ---------------------------------------------

TABLE9 = {
    "N2": (28.01, 2.980e-4),
    "CH4": (16.04, 4.401e-4),
    "CO": (28.01, 3.364e-4),
    "50%CH4-50%Ar": (28.00, 3.614e-4),
}


def test_gas_registry_pins_table9():
    assert set(GASES) == set(TABLE9)
    for name, (mu, nu) in TABLE9.items():
        assert GASES[name].mu == mu and GASES[name].nu_stp == nu


def test_molar_mass_mirrors_gas_registry():
    for name, gas in GASES.items():
        assert MOLAR_MASS[name] == pytest.approx(gas.mu * 1e-3, rel=1e-12)


def test_bandpass_and_monochromatic_n2_are_deliberately_distinct():
    """Table 9's KAO-passband value and Peck & Khanna at 0.7 um are both
    legitimate and must NOT be 'reconciled' — they differ by ~0.35%."""
    mono = physicalData.refractivitySTP("N2", 0.7)
    band = GASES["N2"].nu_stp
    assert mono == pytest.approx(2.9696e-4, rel=1e-3)
    rel = band / mono - 1.0
    assert 2e-3 < rel < 5e-3


# --- ConstantSet / re-exports --------------------------------------------------

def test_default_constant_set_matches_module_values():
    """Module-level names carry the DEFAULT (2022) vintage, not 1986."""
    assert float(CODATA2022.gravitational) == float(physicalData.GRAVITATIONAL)
    assert float(CODATA2022.boltzmann) == float(physicalData.BOLTZMANN)
    assert float(CODATA2022.loschmidt) == float(physicalData.LOSCHMIDT)
    assert CODATA2022.amu == 1e-3 / float(physicalData.AVOGADRO)
    assert float(CODATA1986.boltzmann) != float(physicalData.BOLTZMANN)


def test_epq03_reexports_are_the_same_objects():
    """The consolidation must not fork the constants: EPQ03's names are the
    physicalData objects themselves."""
    assert EPQ03.CODATA1986 is physicalData.CODATA1986
    assert EPQ03.CODATA2022 is physicalData.CODATA2022
    assert EPQ03.DEFAULT_CONSTANTS is physicalData.DEFAULT_CONSTANTS
    assert EPQ03.ConstantSet is physicalData.ConstantSet
    assert EPQ03.Gas is physicalData.Gas
    assert EPQ03.GASES is physicalData.GASES
    assert float(EPQ03.AU_KM) == float(physicalData.AU_KM)
    # method-specific set stays with its method
    assert isinstance(EPQ03.EPQ03_TABLE10, physicalData.ConstantSet)
    assert not hasattr(physicalData, "EPQ03_TABLE10")


def test_refractivity_helper_identity():
    """n = L gives nu = nu_STP exactly."""
    nu = physicalData.refractivity(float(physicalData.LOSCHMIDT), "N2", 0.7)
    assert nu == pytest.approx(physicalData.refractivitySTP("N2", 0.7), rel=1e-14)


# --- gas dispersion formulas (pdRefractivityAtSTP port) ------------------------

# Regression pins at lambda = 0.5 um — the MATHEMATICA package's default
# wavelength — computed from the ported formulas at merge time.
DISPERSION_AT_HALF_MICRON = {
    "N2": 0.00030016022315837914,
    "H2": 0.00013988486400000002,      # package default = reference [2], Allen form
    "Ar": 0.00028340241830010666,
    "CO2": 0.0004502384,
    "He": 3.875328e-05,
    "Uranus": 0.00012471512640000002,  # 0.85 H2[2] + 0.15 He[1]
}


@pytest.mark.parametrize("gas,value", sorted(DISPERSION_AT_HALF_MICRON.items()))
def test_dispersion_pins_at_pd_default_wavelength(gas, value):
    assert physicalData.refractivitySTP(gas, 0.5) == pytest.approx(value, rel=1e-12)


def test_h2_default_reference_is_two():
    """Fidelity to physicalData_2.3.m: its ["H2", 0] rule is overridden to
    reference [2] (Allen form) by a later definition — [1] (Peck & Huang
    1977) remains available explicitly and differs by ~0.1%."""
    lam = 0.5
    default = physicalData.refractivitySTP("H2", lam)
    assert default == physicalData.refractivitySTP("H2", lam, reference=2)
    ph = physicalData.refractivitySTP("H2", lam, reference=1)
    assert ph == pytest.approx(0.00014002141121495328, rel=1e-12)
    assert default != ph


def test_uranus_mixture_identities():
    """"Uranus" = 0.85 H2[2] + 0.15 He[1], in both refractivity and MW."""
    lam = 0.5
    mix = (0.85 * physicalData.refractivitySTP("H2", lam, reference=2)
           + 0.15 * physicalData.refractivitySTP("He", lam))
    assert physicalData.refractivitySTP("Uranus", lam) == pytest.approx(mix, rel=1e-15)
    mw = 0.85 * 2.016e-3 + 0.15 * 4.0026e-3
    assert MOLAR_MASS["Uranus"] == pytest.approx(mw, rel=1e-15)


def test_n2_default_unchanged_by_multigas_port():
    """Backward compatibility: the validated N2-at-0.7-um value is untouched."""
    assert physicalData.refractivitySTP("N2", 0.7) == 0.0002969636474759682
    assert physicalData.refractivitySTP("N2", 0.7) == physicalData.refractivitySTP(
        "N2", 0.7, reference=1
    )


def test_unknown_gas_raises():
    with pytest.raises(NotImplementedError, match="available gases"):
        physicalData.refractivitySTP("SF6", 0.5)
    with pytest.raises(NotImplementedError):
        physicalData.refractivitySTP("N2", 0.5, reference=9)


def test_refractivity_sources_cover_all_formulas():
    assert set(physicalData.REFRACTIVITY_SOURCES) == set(
        physicalData._REFRACTIVITY_FORMULAS
    )
    assert all(physicalData.REFRACTIVITY_SOURCES.values())


# --- body registry (pdMass/pdGM/pdEquatorialRadius port) -----------------------

def test_bodies_registry_pins():
    t = physicalData.BODIES["Triton"]
    assert t.mass_kg == 2.1398e22          # the benchmark-suite body mass
    assert t.mass_uncertainty_kg == 0.0053e22
    assert t.radius_km == 1352.6
    p = physicalData.BODIES["Pluto"]
    assert p.mass_kg == 1.305e22           # Buie et al. 2006
    assert p.gm_m3s2 == 8.70773e11         # Person 2006
    assert physicalData.BODIES["Sun"].gm_m3s2 == 1.32712440018e20  # DE-405


def test_bodies_mass_gm_internally_consistent():
    """Where both are given, GM and G*mass (independent literature values)
    agree well within a percent."""
    g = float(physicalData.GRAVITATIONAL)
    for body in physicalData.BODIES.values():
        assert body.source
        if body.mass_kg is not None and body.gm_m3s2 is not None:
            assert body.gm_m3s2 / (g * body.mass_kg) == pytest.approx(1.0, abs=0.01)
