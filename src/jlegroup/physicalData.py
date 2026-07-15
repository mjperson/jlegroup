"""Physical constants and shared gas data for jlegroup.

TWO VINTAGES, ONE SWITCH (design adopted 2026-07-15). Fundamental constants come in
two immutable ``ConstantSet`` bundles:

    CODATA2022 — current values (several exact by the 2019 SI redefinition). The
        module-level names (BOLTZMANN, GRAVITATIONAL, ...) carry THESE values, and
        DEFAULT_CONSTANTS points here: research use gets current physics by default.
    CODATA1986 — the vintage of the Mathematica ``jleGroup`physicalData`` package and
        therefore of EVERY reference light curve and paper table this code is
        validated against (records archived in VINTAGE_1986). This is the
        VERIFICATION mode: any comparison against a Mathematica reference or a
        published table must pass ``constants=CODATA1986``.

Every model constructor and conversion helper accepts ``constants=`` and defaults to
DEFAULT_CONSTANTS. The dominant difference between vintages is G (+2.56e-4 relative,
1986 -> 2022); everything else moves by <~1e-5. Both vintages are test-pinned
(tests/test_physicalData.py); neither set's values may ever drift. What does NOT
participate in the vintage switch: measured quantities with their own provenance —
the Table-9 passband refractivities (GASES), the dispersion formulas, and the BODIES
registry. (Note GM values are measured directly and are G-independent; masses derived
as GM/G inherit the G vintage.)

Design (adopted 2026-07-08 from the EY92 development effort's constants module): every
universal constant is a ``Constant`` — a float subclass carrying symbol, units, 1-sigma
uncertainty, source, and provenance notes — so values participate in arithmetic unchanged
while their provenance travels with them. Policy: no bare numeric literals for physical
constants in other modules; take them from here (method-specific paper-reproduction sets,
e.g. ``EPQ03.EPQ03_TABLE10``, are the documented exception and live with their method).

Mapping to the Mathematica package (the 1986 archive, NOT the module defaults):
    VINTAGE_1986["BOLTZMANN"]   = pdBoltzmannConstant[2]
    VINTAGE_1986["LOSCHMIDT"]   = pdLoschmidtNumber[3]
    refractivitySTP("N2", lambda) ~ pdRefractivityAtSTP["N2", 1, lambda]

Two refractivity conventions coexist deliberately — do not "reconcile" them:
    refractivitySTP(gas, wavelength_um, reference) : monochromatic dispersion
        formulas ported from pdRefractivityAtSTP — N2, H2, Ar, CO2, He, and the
        85/15 H2-He "Uranus" mixture (citations in REFRACTIVITY_SOURCES).
        NOTE: this module defaults to 0.7 um (the validation references'
        wavelength); the Mathematica package defaults to 0.5 um — pass
        wavelength_um=0.5 to reproduce pd default outputs.
    GASES[name].nu_stp : EY92 Table 9 values, integrated over the 1988 KAO CCD
        passband (Dunham et al. 1985 QE x 4900 K blackbody; EY92 Sec. 11) — the
        values the EY92/EPQ03 papers' numbers are built on.

Body data: BODIES carries occultation-target entries ported from pdMass/pdGM/
pdEquatorialRadius (Pluto, Triton, Sun) with the package's citations — including
Triton's 2.1398e22 kg (Anderson et al. 1992), the benchmark suite's body mass.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Constant",
    "BOLTZMANN",
    "GRAVITATIONAL",
    "AVOGADRO",
    "LOSCHMIDT",
    "AU_KM",
    "AMU",
    "SPEED_OF_LIGHT",
    "PLANCK",
    "STEFAN_BOLTZMANN",
    "MOLAR_MASS",
    "Gas",
    "GASES",
    "Body",
    "BODIES",
    "ConstantSet",
    "CODATA1986",
    "CODATA2022",
    "DEFAULT_CONSTANTS",
    "VINTAGE_1986",
    "REFRACTIVITY_SOURCES",
    "refractivitySTP",
    "refractivity",
]


class Constant(float):
    """A physical constant: a float carrying its provenance.

    Behaves exactly like a float in arithmetic; the metadata rides along.

    uncertainty : 1-sigma standard uncertainty in the same units, or None —
        permitted only when ``note`` explains why (exact/adopted by
        definition, derived, or the source states no uncertainty).
        Enforced by tests/test_physicalData.py. Treat instances as
        immutable.
    """

    def __new__(cls, value, symbol, units, source, uncertainty=None, note=""):
        obj = super().__new__(cls, value)
        obj.symbol = symbol
        obj.units = units
        obj.source = source
        obj.uncertainty = uncertainty
        obj.note = note
        return obj

    def __repr__(self):
        return f"Constant({float(self)!r}, symbol={self.symbol!r}, units={self.units!r})"


# --- CODATA-1986 / jleGroup validation vintage (FROZEN ARCHIVE) ----------------------
# These are the values the Mathematica jleGroup package carries and therefore the
# vintage every reference light curve and paper table was generated with. They are
# permanently frozen here (test-pinned) and exposed as the CODATA1986 ConstantSet —
# the package's VERIFICATION mode. Never edit these records.

VINTAGE_1986 = {
    "BOLTZMANN": Constant(
        1.380658e-23,
        symbol="k_B",
        units="J/K",
        uncertainty=0.000012e-23,
        source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
               "by jleGroup`physicalData as pdBoltzmannConstant[2]",
        note="1986 recommended value 1.380658(12)e-23. Exact SI (2019) value is "
             "1.380649e-23; frozen at 1986 vintage for benchmark verification.",
    ),
    "GRAVITATIONAL": Constant(
        6.67259e-11,
        symbol="G",
        units="m^3 kg^-1 s^-2",
        uncertainty=0.00085e-11,
        source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
               "by jleGroup`physicalData",
        note="1986 recommended value 6.67259(85)e-11.",
    ),
    "AVOGADRO": Constant(
        6.0221367e23,
        symbol="N_A",
        units="mol^-1",
        uncertainty=0.0000036e23,
        source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
               "by jleGroup`physicalData",
        note="1986 recommended value 6.0221367(36)e23.",
    ),
    "LOSCHMIDT": Constant(
        2.686763e25,
        symbol="L",
        units="m^-3",
        uncertainty=0.000023e25,
        source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
               "by jleGroup`physicalData as pdLoschmidtNumber[3]",
        note="Ideal-gas number density at classic STP (273.15 K, 101325 Pa = 1 atm) — "
             "the reference state gas refractivity tables use. Beware the NIST "
             "100-kPa 'standard state' variant (~2.6516e25), which is NOT this.",
    ),
    "AU_KM": Constant(
        1.49597870691e8,
        symbol="au",
        units="km",
        uncertainty=3e-3,
        source="JPL planetary ephemeris DE-405 (ssd.jpl.nasa.gov); adopted by "
               "jleGroup`physicalData as pdAstronomicalUnit[3]",
        note="Adopted ephemeris value; pdError gives 3 m (= 3e-3 km).",
    ),
    "SPEED_OF_LIGHT": Constant(
        2.99792458e8,
        symbol="c",
        units="m/s",
        uncertainty=None,
        source="Exact by SI definition (17th CGPM, 1983); adopted by "
               "jleGroup`physicalData as pdSpeedOfLight[2]",
        note="Exact: defines the metre.",
    ),
    "PLANCK": Constant(
        6.6260755e-34,
        symbol="h",
        units="J s",
        uncertainty=4.0e-40,
        source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
               "by jleGroup`physicalData as pdPlanckConstant[2]",
        note="1986 recommended value 6.6260755(40)e-34.",
    ),
    "STEFAN_BOLTZMANN": Constant(
        5.67051e-8,
        symbol="sigma_SB",
        units="W m^-2 K^-4",
        uncertainty=1.9e-12,
        source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
               "by jleGroup`physicalData as pdStefanBoltzmannConstant[2]",
        note="1986 recommended value 5.67051(19)e-8.",
    ),
}
VINTAGE_1986["AMU"] = Constant(
    1e-3 / float(VINTAGE_1986["AVOGADRO"]),
    symbol="m_amu",
    units="kg",
    uncertainty=None,
    source="Derived: (1e-3 kg/mol) / N_A with the CODATA-1986 Avogadro above",
    note="Derived, so no independent uncertainty (relative uncertainty follows "
         "N_A, ~6e-7). Equals the CODATA-1986 recommended 1.6605402(10)e-27 kg "
         "at printed precision.",
)


# --- CODATA-2022 / SI-2019 vintage (MODULE DEFAULT) -----------------------------------
# Current recommended values, for current research. Several are exact by the 2019
# SI redefinition. The validation references were NOT generated with these — pass
# constants=CODATA1986 when reproducing them (see module docstring).

BOLTZMANN = Constant(
    1.380649e-23,
    symbol="k_B",
    units="J/K",
    uncertainty=None,
    source="Exact by SI definition (26th CGPM, 2019) / CODATA 2022",
    note="Exact: defines the kelvin. (Validation vintage: 1.380658e-23.)",
)

GRAVITATIONAL = Constant(
    6.67430e-11,
    symbol="G",
    units="m^3 kg^-1 s^-2",
    uncertainty=0.00015e-11,
    source="CODATA 2022 (identical to CODATA 2018)",
    note="2022 recommended value 6.67430(15)e-11 — a +2.56e-4 relative shift from "
         "the 1986 validation vintage; the largest single-vintage effect in this "
         "package.",
)

AVOGADRO = Constant(
    6.02214076e23,
    symbol="N_A",
    units="mol^-1",
    uncertainty=None,
    source="Exact by SI definition (26th CGPM, 2019) / CODATA 2022",
    note="Exact: defines the mole.",
)

LOSCHMIDT = Constant(
    2.686780111e25,
    symbol="L",
    units="m^-3",
    uncertainty=None,
    source="CODATA 2022 n_0 (273.15 K, 101325 Pa); exactly derivable as "
           "p/(k_B T) with the exact SI k_B",
    note="Ideal-gas number density at classic STP (273.15 K, 101325 Pa = 1 atm). "
         "Beware the NIST 100-kPa 'standard state' variant (~2.6516e25), which "
         "is NOT this.",
)

AU_KM = Constant(
    1.495978707e8,
    symbol="au",
    units="km",
    uncertainty=None,
    source="IAU 2012 Resolution B2",
    note="Exact by definition (149 597 870.700 km) since 2012.",
)

AMU = Constant(
    1e-3 / float(AVOGADRO),
    symbol="m_amu",
    units="kg",
    uncertainty=None,
    source="Derived: (1e-3 kg/mol) / N_A with the exact SI Avogadro above",
    note="Molar-mass-constant convention (M_u = 1e-3 kg/mol, exact to ~1e-9 "
         "since SI-2019); CODATA-2022 measured m_u = 1.66053906892(52)e-27 kg "
         "differs only at that level. Derived here so the set is internally "
         "consistent, mirroring the 1986 set's convention.",
)

SPEED_OF_LIGHT = Constant(
    2.99792458e8,
    symbol="c",
    units="m/s",
    uncertainty=None,
    source="Exact by SI definition (17th CGPM, 1983)",
    note="Exact: defines the metre. Identical in both vintages.",
)

PLANCK = Constant(
    6.62607015e-34,
    symbol="h",
    units="J s",
    uncertainty=None,
    source="Exact by SI definition (26th CGPM, 2019) / CODATA 2022",
    note="Exact: defines the kilogram.",
)

STEFAN_BOLTZMANN = Constant(
    5.670374419e-8,
    symbol="sigma_SB",
    units="W m^-2 K^-4",
    uncertainty=None,
    source="CODATA 2022; exactly derivable as 2 pi^5 k_B^4 / (15 h^3 c^2) from "
           "the exact SI constants",
    note="Exact (derived).",
)


#: Molar masses, kg/mol. N2/CH4/CO/mixture mirror the GASES registry (EY92
#: Table 9); H2/He/CO2/"Uranus" are ported from pdMolecularWeight (the Uranus
#: entry is the package's 85/15 H2-He mixture, ~2.314e-3).
MOLAR_MASS = {
    "N2": 28.01e-3,
    "CH4": 16.04e-3,
    "CO": 28.01e-3,
    "50%CH4-50%Ar": 28.00e-3,
    "H2": 2.016e-3,
    "He": 4.0026e-3,
    "CO2": 44.0095e-3,
    "Uranus": 0.85 * 2.016e-3 + 0.15 * 4.0026e-3,
}


# --- Gas registry (EY92 Table 9 / EPQ03 Table 9) -------------------------------------


@dataclass(frozen=True)
class Gas:
    """Homogeneous-atmosphere gas data (EY92 Table 9 / EPQ03 Table 9).

    mu : mean molecular weight, amu.
    nu_stp : refractivity at STP, integrated over the EY92 KAO CCD passband
        (Dunham et al. 1985 QE x 4900 K blackbody; EY92 Sec. 11) — a
        passband-weighted value, deliberately distinct from the
        monochromatic ``refractivitySTP(gas, wavelength_um)``. EY92 states
        no uncertainties for these.
    """

    name: str
    mu: float
    nu_stp: float


#: EY92 Table 9 candidate-composition values (EPQ03 Table 9/Table 1 uses the
#: same N2 pair). Keys are the Table 9 composition labels.
GASES = {
    "N2": Gas("N2", 28.01, 2.980e-4),
    "CH4": Gas("CH4", 16.04, 4.401e-4),
    "CO": Gas("CO", 28.01, 3.364e-4),
    "50%CH4-50%Ar": Gas("50%CH4-50%Ar", 28.00, 3.614e-4),
}


# --- Occultation-target body registry (jleGroup`physicalData port) --------------------


@dataclass(frozen=True)
class Body:
    """Occultation-target body data ported from jleGroup`physicalData
    (pdMass / pdGM / pdEquatorialRadius), with that package's citations.

    mass_kg and gm_m3s2 are independent literature values (not derived from
    one another); fields are None where the package gives no value.
    Uncertainties are 1-sigma per pdError.
    """

    name: str
    mass_kg: float | None
    mass_uncertainty_kg: float | None
    gm_m3s2: float | None
    gm_uncertainty_m3s2: float | None
    radius_km: float | None
    radius_uncertainty_km: float | None
    source: str


#: Minimal registry for the group's occultation science; extend as needed.
BODIES = {
    "Earth": Body(
        "Earth", None, None, None, None, 6378.14, None,
        "radius: IAU (1976) System of Astronomical Constants equatorial "
        "radius, 6378.140 km — the value hardcoded in Mathematica jleGroup "
        "shadowMap 4.1.4 smDist; added for the shadowmap module. Other "
        "fields not ported (no package-sourced values in hand).",
    ),
    "Pluto": Body(
        "Pluto", 1.305e22, 0.0065e22, 8.70773e11, 0.043e11, None, None,
        "mass: pdMass['Pluto',1] = Buie et al. 2006, AJ 132, 290; "
        "GM: pdGM['Pluto',1] = Person 2006",
    ),
    "Triton": Body(
        "Triton", 2.1398e22, 0.0053e22, 1.4279e12, 0.035e12, 1352.6, 2.4,
        "mass: pdMass['Triton',1] = Anderson et al. 1992 (this is the jlegroup "
        "benchmark-suite body mass); GM and radius: pdGM/pdEquatorialRadius"
        "['Triton',1] = McKinnon, Lunine & Banfield 1995, in Neptune and "
        "Triton (U. Arizona Press), p. 809",
    ),
    "Sun": Body(
        "Sun", 1.98892e30, 0.00025e30, 1.32712440018e20, 8e9, 6.95508e5, 26.0,
        "mass: pdMass['Sun',2] = Rev. Mod. Phys. 68, 611 (1996); GM: "
        "pdGM['Sun',1] = JPL DE-405 (ssd.jpl.nasa.gov); radius: "
        "pdRadius['Sun',1] = Allen 4th ed.",
    ),
}


# --- Constant-set injection (for method modules) --------------------------------------


@dataclass(frozen=True)
class ConstantSet:
    """The physical constants the method equations use, as one injectable
    bundle (e.g. ``EPQ03.invert_light_curve(..., constants=...)``). This is
    the package's vintage switch: pass ``CODATA2022`` (the default) for
    current research, ``CODATA1986`` to reproduce the validation references.

    gravitational : G, m^3 kg^-1 s^-2
    boltzmann : k, J/K
    loschmidt : L, m^-3 (number density at STP)
    amu : atomic mass unit, kg

    Derived (molar-mass-constant convention, M_u = 1e-3 kg/mol):
    avogadro : N_A = 1e-3 / amu, mol^-1
    gas_constant : R = k N_A, J mol^-1 K^-1
    """

    name: str
    gravitational: float
    boltzmann: float
    loschmidt: float
    amu: float

    @property
    def avogadro(self):
        return 1e-3 / self.amu

    @property
    def gas_constant(self):
        return self.boltzmann * (1e-3 / self.amu)


#: VERIFICATION vintage: the values the Mathematica jleGroup package and every
#: validation reference were generated with. Frozen forever; test-pinned.
CODATA1986 = ConstantSet(
    name="CODATA-1986 (jleGroup validation vintage)",
    gravitational=VINTAGE_1986["GRAVITATIONAL"],
    boltzmann=VINTAGE_1986["BOLTZMANN"],
    loschmidt=VINTAGE_1986["LOSCHMIDT"],
    amu=VINTAGE_1986["AMU"],
)

#: CURRENT vintage (the package default): CODATA-2022 / SI-2019 exact values.
CODATA2022 = ConstantSet(
    name="CODATA-2022 / SI-2019 (package default)",
    gravitational=GRAVITATIONAL,
    boltzmann=BOLTZMANN,
    loschmidt=LOSCHMIDT,
    amu=AMU,
)

#: What every model in the package uses when no ``constants=`` is passed.
DEFAULT_CONSTANTS = CODATA2022


# --- Refractivity conversions ----------------------------------------------------------


#: Peck-style 15 C -> 273.16 K density scaling used by the N2/Ar formulas.
_T_SCALE = (273.16 + 15.0) / 273.16


def _allen_form(a, b):
    """Allen Astrophysical Quantities dispersion form nu = A (1 + B/lambda^2)."""
    def formula(lam):
        return a * (1.0 + b * (1.0 / lam) ** 2)
    return formula


def _n2_peck_khanna(lam):
    sigma = (1.0 / lam) ** 2
    return _T_SCALE * (6497.378 + 3.0738649e6 / (144.0 - sigma)) / 1e8


def _h2_peck_huang(lam):
    sigma = (1.0 / lam) ** 2
    return (21.113 + 12723.2 / (111.0 - sigma)) / 1e6


def _ar_peck_fisher(lam):
    sigma = (1.0 / lam) ** 2
    return _T_SCALE * (643.2135 + 286060.21 / (144.0 - sigma)) / 1e7


def _uranus_mixture(lam):
    return (0.85 * _REFRACTIVITY_FORMULAS[("H2", 2)](lam)
            + 0.15 * _REFRACTIVITY_FORMULAS[("He", 1)](lam))


#: (gas, reference) -> formula, mirroring pdRefractivityAtSTP's reference values.
_REFRACTIVITY_FORMULAS = {
    ("N2", 1): _n2_peck_khanna,
    ("H2", 1): _h2_peck_huang,
    ("H2", 2): _allen_form(13.58e-5, 7.52e-3),
    ("Ar", 1): _ar_peck_fisher,
    ("CO2", 1): _allen_form(43.9e-5, 6.4e-3),
    ("He", 1): _allen_form(3.84e-5, 2.3e-3),
    ("Uranus", 1): _uranus_mixture,
}

#: The pd reference-[0] rule: each gas's default reference. H2 defaults to [2]
#: because the Mathematica package's ["H2", 0] rule is overridden to [2] by a
#: later definition in physicalData_2.3.m (and its "Uranus" mixture uses H2[2]).
_REFRACTIVITY_DEFAULT = {"N2": 1, "H2": 2, "Ar": 1, "CO2": 1, "He": 1, "Uranus": 1}

#: Citations per (gas, reference), following the package's pdReference entries.
REFRACTIVITY_SOURCES = {
    ("N2", 1): "Peck & Khanna 1966, JOSA 56, 8 (15 C formula scaled to 273.16 K); "
               "valid ~0.47-2.1 um",
    ("H2", 1): "Peck & Huang 1977, JOSA 67, 11",
    ("H2", 2): "Allen dispersion form A(1 + B/lambda^2) (package default for H2)",
    ("Ar", 1): "Peck & Fisher 1964, JOSA 54, 11 (15 C formula scaled to 273.16 K)",
    ("CO2", 1): "Allen, Astrophysical Quantities, 2nd ed. (dispersion form)",
    ("He", 1): "Allen dispersion form (no reference stated in the Mathematica package)",
    ("Uranus", 1): "0.85 x H2[2] + 0.15 x He[1] (jleGroup`physicalData 85/15 mixture)",
}


def refractivitySTP(gas="N2", wavelength_um=0.7, reference=0):
    """Monochromatic refractivity (n - 1) of the gas at STP.

    Ported from jleGroup`physicalData pdRefractivityAtSTP[gas, reference,
    lambda]. ``reference=0`` selects the gas's package-default formula (see
    REFRACTIVITY_SOURCES for citations, _REFRACTIVITY_DEFAULT for the map).
    Available gases: N2, H2, Ar, CO2, He, and "Uranus" (85/15 H2-He mixture).

    NOTE: the default wavelength here is 0.7 um — the wavelength of this
    package's validation references — while the Mathematica package defaults
    to 0.5 um. Pass wavelength_um explicitly when reproducing pd numbers.

    N2 at 0.7 um returns ~2.970e-4 (Peck & Khanna 1966).

    For the EY92/EPQ03 papers' KAO-passband-integrated values (N2, CH4, CO,
    50%CH4-50%Ar), use the ``GASES`` registry instead.
    """
    ref = _REFRACTIVITY_DEFAULT.get(gas, 1) if reference == 0 else reference
    try:
        formula = _REFRACTIVITY_FORMULAS[(gas, ref)]
    except KeyError:
        available = sorted({g for g, _ in _REFRACTIVITY_FORMULAS})
        raise NotImplementedError(
            f"refractivitySTP: no formula for gas={gas!r}, reference={ref}; "
            f"available gases: {available}"
        ) from None
    return formula(wavelength_um)


def refractivity(number_density, gas="N2", wavelength_um=0.7, reference=0,
                 constants=None):
    """Refractivity profile nu(r) from a number-density profile [m^-3]:

        nu(r) = nu_STP(gas, lambda) * n(r) / L

    with L from ``constants`` (default DEFAULT_CONSTANTS; pass
    ``constants=CODATA1986`` when reproducing validation references).
    """
    if constants is None:
        constants = DEFAULT_CONSTANTS
    return (refractivitySTP(gas, wavelength_um, reference)
            / constants.loschmidt * number_density)
