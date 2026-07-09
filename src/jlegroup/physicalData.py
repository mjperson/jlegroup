"""Physical constants and shared gas data for jlegroup, mirroring the Mathematica
``jleGroup`physicalData`` package.

Vintage matters: the Mathematica package — and therefore every reference light curve used
to validate this code — uses CODATA-1986 values. Do NOT "upgrade" these to current CODATA
without revalidating the benchmark suite; a relative change of a few 1e-4 in a constant is
detectable at the accuracy this package is validated to. The pinned values are enforced
by tests/test_physicalData.py.

Design (adopted 2026-07-08 from the EY92 development effort's constants module): every
universal constant is a ``Constant`` — a float subclass carrying symbol, units, 1-sigma
uncertainty, source, and provenance notes — so values participate in arithmetic unchanged
while their provenance travels with them. Policy: no bare numeric literals for physical
constants in other modules; take them from here (method-specific paper-reproduction sets,
e.g. ``EPQ03.EPQ03_TABLE10``, are the documented exception and live with their method).

Mapping to the Mathematica package:
    BOLTZMANN   = pdBoltzmannConstant[2]
    LOSCHMIDT   = pdLoschmidtNumber[3]
    refractivitySTP("N2", lambda) ~ pdRefractivityAtSTP["N2", 1, lambda]

Two refractivity conventions coexist deliberately — do not "reconcile" them:
    refractivitySTP(gas, wavelength_um) : monochromatic (Peck & Khanna 1966 for N2).
    GASES[name].nu_stp : EY92 Table 9 values, integrated over the 1988 KAO CCD
        passband (Dunham et al. 1985 QE x 4900 K blackbody; EY92 Sec. 11) — the
        values the EY92/EPQ03 papers' numbers are built on.
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
    "MOLAR_MASS",
    "Gas",
    "GASES",
    "ConstantSet",
    "CODATA1986",
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


# --- CODATA 1986 vintage (jleGroup`physicalData) ------------------------------------

BOLTZMANN = Constant(
    1.380658e-23,
    symbol="k_B",
    units="J/K",
    uncertainty=0.000012e-23,
    source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
           "by jleGroup`physicalData as pdBoltzmannConstant[2]",
    note="1986 recommended value 1.380658(12)e-23. Exact SI (2019) value is "
         "1.380649e-23; kept at 1986 vintage for benchmark consistency.",
)

GRAVITATIONAL = Constant(
    6.67259e-11,
    symbol="G",
    units="m^3 kg^-1 s^-2",
    uncertainty=0.00085e-11,
    source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
           "by jleGroup`physicalData",
    note="1986 recommended value 6.67259(85)e-11.",
)

AVOGADRO = Constant(
    6.0221367e23,
    symbol="N_A",
    units="mol^-1",
    uncertainty=0.0000036e23,
    source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
           "by jleGroup`physicalData",
    note="1986 recommended value 6.0221367(36)e23.",
)

LOSCHMIDT = Constant(
    2.686763e25,
    symbol="L",
    units="m^-3",
    uncertainty=0.000023e25,
    source="CODATA 1986 (Cohen & Taylor 1987, Rev. Mod. Phys. 59, 1121); adopted "
           "by jleGroup`physicalData as pdLoschmidtNumber[3]",
    note="Ideal-gas number density at classic STP (273.15 K, 101325 Pa = 1 atm) — "
         "the reference state gas refractivity tables use. Beware the NIST "
         "100-kPa 'standard state' variant (~2.6516e25), which is NOT this.",
)

AU_KM = Constant(
    1.49597870691e8,
    symbol="au",
    units="km",
    uncertainty=None,
    source="DE405 (Standish 1998) vintage value; adopted by jleGroup`physicalData",
    note="Adopted ephemeris value (formal uncertainty at the metre level). The "
         "SI value has been exact by IAU 2012 Resolution B2 definition since "
         "2012 (149 597 870.700 km); kept at package vintage for consistency.",
)

AMU = Constant(
    1e-3 / float(AVOGADRO),
    symbol="m_amu",
    units="kg",
    uncertainty=None,
    source="Derived: (1e-3 kg/mol) / N_A with the CODATA-1986 Avogadro above",
    note="Derived, so no independent uncertainty (relative uncertainty follows "
         "N_A, ~6e-7). Equals the CODATA-1986 recommended 1.6605402(10)e-27 kg "
         "at printed precision.",
)


#: Molar masses, kg/mol (mirrors the GASES registry's mu values, EY92 Table 9).
MOLAR_MASS = {
    "N2": 28.01e-3,
    "CH4": 16.04e-3,
    "CO": 28.01e-3,
    "50%CH4-50%Ar": 28.00e-3,
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


# --- Constant-set injection (for method modules) --------------------------------------


@dataclass(frozen=True)
class ConstantSet:
    """The four physical constants the method equations use, as one injectable
    bundle (e.g. ``EPQ03.invert_light_curve(..., constants=...)``).

    gravitational : G, m^3 kg^-1 s^-2
    boltzmann : k, J/K
    loschmidt : L, m^-3 (number density at STP)
    amu : atomic mass unit, kg
    """

    name: str
    gravitational: float
    boltzmann: float
    loschmidt: float
    amu: float


#: The package-default set (this module's CODATA-1986 vintage values).
CODATA1986 = ConstantSet(
    name="CODATA-1986 (jlegroup.physicalData)",
    gravitational=GRAVITATIONAL,
    boltzmann=BOLTZMANN,
    loschmidt=LOSCHMIDT,
    amu=1e-3 / AVOGADRO,
)


# --- Refractivity conversions ----------------------------------------------------------


def refractivitySTP(gas="N2", wavelength_um=0.7):
    """Monochromatic refractivity (n - 1) of the gas at STP.

    N2: Peck & Khanna 1966 (JOSA 56, 8), 15 C dispersion formula scaled to 273.16 K.
    Valid roughly 0.47-2.1 um. At 0.7 um returns ~2.970e-4.

    For the EY92/EPQ03 papers' KAO-passband-integrated values (N2, CH4, CO,
    50%CH4-50%Ar), use the ``GASES`` registry instead.

    wavelength_um: wavelength in microns.
    """
    if gas != "N2":
        raise NotImplementedError("refractivitySTP: only N2 is implemented so far")
    sigma = (1.0 / wavelength_um) ** 2
    return (273.16 + 15.0) / 273.16 * (6497.378 + 3.0738649e6 / (144.0 - sigma)) / 1e8


def refractivity(number_density, gas="N2", wavelength_um=0.7):
    """Refractivity profile nu(r) from a number-density profile [m^-3]:

        nu(r) = nu_STP(gas, lambda) * n(r) / LOSCHMIDT
    """
    return refractivitySTP(gas, wavelength_um) / LOSCHMIDT * number_density
