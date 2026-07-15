"""shadowmap validation.

The module implements the functionality of Mathematica jleGroup shadowMap
4.1.4 with modern machinery (astropy time/coordinates, Natural Earth
coastlines); there is no reference publication, so validation is layered
(reference data provenance: tests/data/shadowmap-mathematica/README.md):

1. **Exact Mathematica oracles** for the parts ported verbatim: track-line
   construction (measured <= 1e-15 vs drawtracks output), smDist / smOffset
   methods (exact but for the documented 1.42e-5 au-rounding difference),
   compass directions (byte-equal), MJD date parsing and sexagesimal
   conventions (<= 6e-14).

2. **Superseded-theory cross-checks** for the parts astropy replaces: the
   astropy chain must agree with the original's output at the accuracy class
   of the theory it supersedes — GMST to <= 1 s of hour angle (the original
   used IAU-1982 with UTC-as-UT1; measured <= 0.82 s, dUT1-dominated), the
   sub-star point to <= 25 arcsec when fed the original's apparent
   coordinates as TETE (measured <= 18"; equation-of-equinoxes + dUT1 +
   polar motion, all neglected by the original), the sub-solar point to
   <= 1.5 arcmin (measured <= 0.61'; the smSunPosn theory's arcminute
   class).

3. **Physical invariants** for the geometry: the drawn terminator lies at
   sun altitude -horizon_angle (geocentric, spherical model) everywhere;
   night-cap boundary points sit at angular radius (90 - h) deg from the
   antisolar point; orthographic output stays on the unit disk; rotation
   round-trips.  (The original's night shading fails the first invariant by
   up to tens of degrees — the defect documented in the data README — so
   these invariants, not the original's polygons, are the oracle here.)

IERS configuration: auto_download is disabled for offline-deterministic CI
(astropy falls back to its bundled tables; the 2050 test date is beyond
them, hence degraded_accuracy="warn" — the affected quantities are dUT1 and
polar motion, both far below the tolerances used here).
"""
import csv
import math
import os

import numpy as np
import pytest

pytest.importorskip("astropy", reason="shadowmap extra not installed")

from astropy.utils import iers  # noqa: E402

iers.conf.auto_download = False
iers.conf.iers_degraded_accuracy = "warn"

from astropy import units as u                                  # noqa: E402
from astropy.coordinates import ITRS, SkyCoord, TETE, get_sun   # noqa: E402
from astropy.time import Time                                   # noqa: E402

from jlegroup import physicalData, shadowmap as sm              # noqa: E402

DATA = os.path.join(os.path.dirname(__file__), "data", "shadowmap-mathematica")


def _rows(fname):
    with open(os.path.join(DATA, fname)) as f:
        return list(csv.DictReader(f))


GLOBE_CASES = {r["case"]: r for r in _rows("globe_cases.csv")}


# ---------------------------------------------------------------------------
# Input coercion
# ---------------------------------------------------------------------------


def test_as_time_mjd_reference():
    # Exact pins of the legacy 6-field parsing (all separators, day 0).
    for r in _rows("tc_dates.csv"):
        assert sm.as_time(r["date"]).mjd == pytest.approx(float(r["mjd"]),
                                                          abs=1e-9)


def test_as_time_input_forms():
    t = Time("2015-06-29 16:55:00", scale="utc")
    assert sm.as_time(t) is t
    assert sm.as_time(57202.70486111111).mjd == pytest.approx(
        57202.70486111111, abs=1e-12)
    assert sm.as_time("2015-06-29 16:55:00").mjd == pytest.approx(
        t.mjd, abs=1e-12)
    assert sm.as_time("2015:06:29:16:55:00").mjd == pytest.approx(
        t.mjd, abs=1e-9)
    from datetime import datetime
    assert sm.as_time(datetime(2015, 6, 29, 16, 55)).mjd == pytest.approx(
        t.mjd, abs=1e-9)
    with pytest.raises(ValueError):
        sm.as_time("not a time at all")


def test_star_coord_reference():
    for r in _rows("tc_ra.csv"):
        c = sm.star_coord(r["ra_string"], "00 00 00")
        assert c.ra.to_value(u.rad) == pytest.approx(float(r["radians"]),
                                                     abs=1e-12)
    for r in _rows("tc_dec.csv"):
        c = sm.star_coord("00 00 00", r["dec_string"])
        assert c.dec.to_value(u.rad) == pytest.approx(float(r["radians"]),
                                                      abs=1e-12)


def test_star_coord_input_forms():
    ref = sm.star_coord("17 45 31.20", "-22 28 45.5")
    combined = sm.star_coord("17 45 31.20 -22 28 45.5")
    assert ref.separation(combined).to_value(u.arcsec) < 1e-9
    sky = SkyCoord(ra=10.5 * u.deg, dec=-3.25 * u.deg)
    assert sm.star_coord(sky) is sky
    floats = sm.star_coord(266.38, -22.479305555555555)
    assert floats.ra.to_value(u.deg) == pytest.approx(266.38, abs=1e-12)
    assert floats.dec.to_value(u.deg) == pytest.approx(-22.479305555555555,
                                                       abs=1e-12)
    with pytest.raises(TypeError):
        sm.star_coord(1.25)


# ---------------------------------------------------------------------------
# Astropy chain vs the superseded Mathematica chain
# ---------------------------------------------------------------------------


def test_gmst_cross_check():
    # astropy mean sidereal time vs tcGMST (IAU-1982, UTC treated as UT1):
    # measured <= 0.82 s of hour angle, dUT1-dominated.
    for r in _rows("tc_dates.csv"):
        t = sm.as_time(r["date"])
        g = t.sidereal_time("mean", "greenwich").hourangle
        d = abs(g - float(r["gmst_hours"])) * 3600.0
        d = min(d, 86400.0 - d)
        assert d < 1.0, r["date"]


def test_substar_point_vs_original():
    # Feed the original's apparent-of-date star coordinates as TETE; the
    # ITRS sub-star point then differs from the original's view point only
    # by what the original neglected (equation of equinoxes, dUT1, polar
    # motion): measured <= 0.5" lat, <= 18" lon.
    for c in GLOBE_CASES.values():
        t = sm.as_time(c["date"])
        star = SkyCoord(ra=float(c["ra_rad"]) * u.rad,
                        dec=float(c["dec_rad"]) * u.rad,
                        frame=TETE(obstime=t))
        lat, lon = sm.substar_point(star, t)
        dlat = abs(lat - float(c["view_lat_deg"])) * 3600.0
        dlon = abs((lon - float(c["view_long_deg"]) + 180.0) % 360.0
                   - 180.0) * 3600.0
        assert dlat < 1.0, c["case"]
        assert dlon < 25.0, c["case"]


def test_subsolar_point_vs_original():
    # get_sun vs the smSunPosn low-precision theory: measured <= 0.61'.
    for c in GLOBE_CASES.values():
        t = sm.as_time(c["date"])
        lat, lon = sm.subsolar_point(t)
        dlat = abs(lat - float(c["sun_dec_deg"])) * 60.0
        dlon = abs((lon - float(c["sun_long_deg"]) + 180.0) % 360.0
                   - 180.0) * 60.0
        assert dlat < 1.5, c["case"]
        assert dlon < 1.5, c["case"]


def test_sun_vs_smsunposn_reference():
    # Angular separation between get_sun and the original theory's RA/Dec,
    # at the theory's arcminute class (both opKepler branches).  Measured
    # <= 2.35', worst at 2050 — 60 yr from the theory's fixed 1990.0
    # elements — as expected for a fixed-epoch low-precision theory.
    for r in _rows("sun_position.csv"):
        t = sm.as_time(r["date"])
        old = SkyCoord(ra=float(r["ra_hours"]) * u.hourangle,
                       dec=float(r["dec_deg"]) * u.deg,
                       frame=TETE(obstime=t))
        sep = get_sun(t).transform_to(TETE(obstime=t)).separation(old)
        assert sep.to_value(u.arcmin) < 3.0, r["date"]


def test_antisolar_is_antipode():
    t = sm.as_time("2015:06:29:16:55:00")
    slat, slon = sm.subsolar_point(t)
    alat, alon = sm.antisolar_point(t)
    assert alat == pytest.approx(-slat, abs=1e-12)
    assert (alon - slon) % 360.0 == pytest.approx(180.0, abs=1e-12)


# ---------------------------------------------------------------------------
# View geometry
# ---------------------------------------------------------------------------


def test_view_rotation_round_trip():
    rng = np.random.default_rng(42)
    lat = np.arcsin(rng.uniform(-1, 1, 200))
    lon = rng.uniform(-math.pi, math.pi, 200)
    latp, lonp = sm.view_rotation(lat, lon, -22.5, 95.2)
    lat2, lon2 = sm.view_rotation_inverse(latp, lonp, -22.5, 95.2)
    assert np.max(np.abs(lat2 - lat)) < 1e-12
    dlon = (lon2 - lon + math.pi) % (2 * math.pi) - math.pi
    assert np.max(np.abs(dlon)) < 1e-12


def test_view_rotation_center_and_pole():
    # The view center maps to (0, 0); the north pole to lat' = 90 - view_lat.
    latp, lonp = sm.view_rotation(math.radians(-22.5), math.radians(95.2),
                                  -22.5, 95.2)
    assert abs(latp) < 1e-12 and abs(lonp) < 1e-12
    latp, _ = sm.view_rotation(math.pi / 2.0, 0.0, -22.5, 95.2)
    assert math.degrees(latp) == pytest.approx(90.0 - 22.5, abs=1e-9)


def test_unknown_projection_raises():
    with pytest.raises(ValueError):
        sm.night_polygon(0.0, 0.0, 10.0, 100.0, 0.0, "gnomonic")
    with pytest.raises(ValueError):
        sm.coastline_outlines(0.0, 0.0, "gnomonic")


# ---------------------------------------------------------------------------
# Coastlines (Natural Earth 1:110m)
# ---------------------------------------------------------------------------


def test_coastline_database_pins():
    db = sm.coastlines()
    assert len(db) == 134
    total = sum(len(pts) for pts in db)
    assert total == 5128
    alldeg = np.degrees(np.vstack(db))
    assert np.all(np.abs(alldeg[:, 0]) <= 90.0)
    assert np.all(np.abs(alldeg[:, 1]) <= 180.0)
    assert alldeg[:, 0].min() < -85.0      # Antarctica present
    assert alldeg[:, 0].max() > 83.0       # Greenland present
    # checksum of the bundled data (float32 payload, degrees)
    assert float(np.abs(alldeg).sum()) == pytest.approx(663527.355, abs=0.01)


def test_coastline_outlines_orthographic():
    view_lat, view_lon = -22.5, 95.2
    out = sm.coastline_outlines(view_lat, view_lon, "orthographic")
    assert len(out) > 30
    allpts = np.vstack(out)
    # everything on the unit disk
    assert np.max(np.hypot(allpts[:, 0], allpts[:, 1])) <= 1.0 + 1e-9
    for piece in out:
        assert len(piece) >= 2
    # splitting must not lose visible vertices: every visible vertex of
    # every polyline appears in some output piece
    n_visible = 0
    for pts in sm.coastlines():
        _, lonp = sm.view_rotation(pts[:, 0], pts[:, 1], view_lat, view_lon)
        vis = np.abs(lonp) < math.pi / 2.0
        for a, b in sm._split_runs(vis):
            if b - a >= 2:
                n_visible += b - a
    assert len(allpts) == n_visible


def test_coastline_outlines_split_at_limb():
    # A synthetic equatorial polyline sweeping -179..179 deg, seen from
    # (0, 0): exactly one visible run, the vertices with |lon| < 90.
    lon = np.radians(np.arange(-179.0, 180.0, 2.0))
    lat = np.zeros_like(lon)
    latp, lonp = sm.view_rotation(lat, lon, 0.0, 0.0)
    runs = sm._split_runs(np.abs(lonp) < math.pi / 2.0)
    assert runs == [(45, 135)]  # lon -89..+89 deg inclusive, 90 vertices
    # and seen from the antipode (0, 180): two visible runs, one per end
    latp, lonp = sm.view_rotation(lat, lon, 0.0, 180.0)
    runs = sm._split_runs(np.abs(lonp) < math.pi / 2.0)
    assert len(runs) == 2
    assert sum(b - a for a, b in runs) == 90


def test_coastline_outlines_cylindrical_no_jumps():
    for proj in ("mercator", "equirectangular"):
        out = sm.coastline_outlines(37.0, -155.0, proj)
        for piece in out:
            assert np.max(np.abs(np.diff(piece[:, 0]))) <= math.pi


# ---------------------------------------------------------------------------
# Night polygon
# ---------------------------------------------------------------------------


def test_night_cap_closure_and_radius():
    for h in (0.0, 6.0, 18.0):
        lat, lon = sm._night_cap(h)
        assert len(lat) == 203
        assert lat[0] == pytest.approx(lat[-1], abs=1e-12)
        assert lon[0] == pytest.approx(lon[-1], abs=1e-12)
        # every point at angular distance (90 - h) from (0, 0)
        cosd = np.cos(lat) * np.cos(lon)
        assert np.max(np.abs(np.degrees(np.arccos(cosd)) - (90.0 - h))) < 1e-5


@pytest.mark.parametrize("h,tol_deg", [(0.0, 1e-4), (6.0, 1e-8), (18.0, 1e-8)])
def test_night_polygon_boundary_at_horizon_angle(h, tol_deg):
    # Pure-geometry invariant: boundary points sit at angular radius
    # (90 - h) deg from the antisolar point.  (h=0 tolerance is sqrt(eps)
    # class: the cap parametrization touches its coordinate poles there.)
    view_lat, view_lon = 37.3, -14.2
    anti_lat, anti_lon = -21.4, 133.7
    poly = sm.night_polygon(view_lat, view_lon, anti_lat, anti_lon, h,
                            "equirectangular")
    lat, lon = sm.view_rotation_inverse(poly[:, 1], poly[:, 0],
                                        view_lat, view_lon)
    cosd = (math.sin(math.radians(anti_lat)) * np.sin(lat)
            + math.cos(math.radians(anti_lat)) * np.cos(lat)
            * np.cos(lon - math.radians(anti_lon)))
    d = np.degrees(np.arccos(np.clip(cosd, -1.0, 1.0)))
    assert np.max(np.abs(d - (90.0 - h))) < tol_deg


def test_night_polygon_terminator_via_astropy_sun():
    # End-to-end: with the antisolar point from astropy, boundary points
    # rotated back to earth sit at geocentric sun altitude -h.
    t = sm.as_time("2015:06:29:16:55:00")
    h = 12.0
    view_lat, view_lon = sm.substar_point(
        sm.star_coord("17 45 31.20", "-22 28 45.5"), t)
    anti_lat, anti_lon = sm.antisolar_point(t)
    poly = sm.night_polygon(view_lat, view_lon, anti_lat, anti_lon, h,
                            "equirectangular")
    lat, lon = sm.view_rotation_inverse(poly[:, 1], poly[:, 0],
                                        view_lat, view_lon)
    sun = get_sun(t).transform_to(ITRS(obstime=t)).spherical
    slat, slon = sun.lat.to_value(u.rad), sun.lon.to_value(u.rad)
    sinalt = (np.sin(lat) * math.sin(slat)
              + np.cos(lat) * math.cos(slat) * np.cos(lon - slon))
    alt = np.degrees(np.arcsin(np.clip(sinalt, -1.0, 1.0)))
    assert np.max(np.abs(alt + h)) < 1e-6


def test_night_polygon_orthographic_clipping():
    # Partially visible night side: polygon nonempty, on the unit disk.
    for c in GLOBE_CASES.values():
        t = sm.as_time(c["date"])
        anti_lat, anti_lon = sm.antisolar_point(t)
        poly = sm.night_polygon(float(c["view_lat_deg"]),
                                float(c["view_long_deg"]),
                                anti_lat, anti_lon,
                                float(c["horizon_angle_deg"]),
                                "orthographic")
        assert len(poly) > 100, c["case"]
        assert np.max(np.hypot(poly[:, 0], poly[:, 1])) <= 1.0 + 1e-9


def test_night_polygon_fully_visible_no_clip():
    # View centered on the antisolar point, deep cap: entirely visible,
    # so all 203 boundary points survive unclipped.
    poly = sm.night_polygon(-21.4, 133.7, -21.4, 133.7, 18.0, "orthographic")
    assert poly.shape == (203, 2)
    assert np.max(np.hypot(poly[:, 0], poly[:, 1])) < math.sin(
        math.radians(72.0)) + 1e-9


def test_night_polygon_invisible_returns_empty():
    # Sun behind the observer (antisolar on the far hemisphere, deep cap):
    # nothing to shade.  (The original's clipper errored in this case.)
    poly = sm.night_polygon(0.0, 0.0, 0.0, 180.0, 18.0, "orthographic")
    assert poly.shape == (0, 2)


# ---------------------------------------------------------------------------
# Tracks (ported drawtracks) and directions — exact Mathematica oracles
# ---------------------------------------------------------------------------


def test_directions_reference():
    for r in _rows("directions.csv"):
        assert sm.find_direction(float(r["angle_deg"])) == r["direction"]


def test_opposite_direction():
    assert sm.opposite_direction("North ") == "South "
    assert sm.opposite_direction("South-East ") == "North-West "


@pytest.mark.parametrize("case", ["G1", "G2", "G3", "G4"])
def test_shadow_tracks_reference(case):
    c = GLOBE_CASES[case]
    ts = sm.shadow_tracks(float(c["dist"]), float(c["pa_deg"]),
                          float(c["radius"]), float(c["pred_error"]))
    got = {"near": ts.near, "minus": ts.error_minus, "plus": ts.error_plus,
           ("centerfar", 1): ts.center, ("centerfar", 2): ts.far}
    for r in _rows(f"{case}_tracks.csv"):
        label = r["line_label"]
        key = (label, int(r["segment"])) if label == "centerfar" else label
        point = got[key][int(r["point"]) - 1]
        assert point[0] == pytest.approx(float(r["x"]), abs=1e-12), (case, key)
        assert point[1] == pytest.approx(float(r["y"]), abs=1e-12), (case, key)


def test_track_geometry_invariants():
    # Parallel lines perpendicular to the PA direction, at signed offsets
    # dist-radius / dist / dist+radius from the Earth center.
    dist, pa, radius = 0.42, 157.5, 0.18
    ts = sm.shadow_tracks(dist, pa, radius, 0.25)
    for seg, offset in [(ts.near, dist - radius), (ts.center, dist),
                        (ts.far, dist + radius)]:
        d = seg[1] - seg[0]
        n = np.array([-d[1], d[0]]) / np.hypot(*d)
        assert abs(n @ seg[0]) == pytest.approx(abs(offset), abs=1e-12)
    assert ts.note == ""


def test_track_ground_paths_lie_on_their_lines():
    # Forward-projecting a ground path orthographically must land back on
    # the fundamental-plane line: perpendicular distance = the line offset,
    # endpoints on the limb.
    dist, pa, radius, pred = 0.42, 157.5, 0.18, 0.25
    tp = track_paths = sm.track_ground_paths(dist, pa, radius, pred)
    theta = math.radians(90.0 - pa)
    nvec = np.array([math.cos(theta), math.sin(theta)])
    for path, offset in [(tp.near, dist - radius), (tp.center, dist),
                         (tp.far, dist + radius),
                         (tp.error_minus, dist - radius - pred),
                         (tp.error_plus, dist + radius + pred)]:
        assert path is not None
        xy = np.column_stack([np.cos(path[:, 0]) * np.sin(path[:, 1]),
                              np.sin(path[:, 0])])
        assert np.max(np.abs(xy @ nvec - offset)) < 1e-12
        r_end = np.hypot(*xy[0]), np.hypot(*xy[-1])
        assert r_end[0] == pytest.approx(1.0, abs=1e-9)
        assert r_end[1] == pytest.approx(1.0, abs=1e-9)
    assert track_paths.note == ""


def test_track_ground_paths_miss_cases():
    # G3 geometry: all five lines miss the Earth (subdist 1.4 > 1).
    tp = sm.track_ground_paths(1.6, 200.0, 0.2, 0.0)
    assert all(p is None for p in
               [tp.near, tp.center, tp.far, tp.error_minus, tp.error_plus])
    assert "No tracks cross the earth" in tp.note
    # G4 geometry: only the center line crosses (near = -1.2, far = 1.8).
    tp = sm.track_ground_paths(0.3, 310.0, 1.5, 0.5)
    assert tp.near is None and tp.far is None
    assert tp.center is not None
    assert tp.error_minus is None and tp.error_plus is None


def test_track_notes():
    assert sm.shadow_tracks(1.6, 200.0, 0.2).note == (
        " No tracks cross the earth\n "
        "All three tracks are South of the earth")
    assert sm.shadow_tracks(0.3, 310.0, 1.5).note == (
        " The entire earth is in the shadow \n "
        "One track is North-West of the earth, \n "
        "and one track is South-East of the earth")
    assert sm.shadow_tracks(0.95, 45.0, 0.2).note == (
        " One track is North-East of the earth")
    assert sm.shadow_tracks(1.2, 90.0, 0.5).note == (
        " Two tracks are East of the earth")


# ---------------------------------------------------------------------------
# smDist / smOffset — exact Mathematica oracles (au-rounding budgeted)
# ---------------------------------------------------------------------------


def test_dist_from_impact_parameter_reference():
    for r in _rows("smdist.csv"):
        got = sm.dist_from_impact_parameter(float(r["b_arcsec"]),
                                            float(r["d_au"]))
        assert got == pytest.approx(float(r["dist_earth_radii"]), rel=5e-5)


def test_dist_uses_physicaldata_earth():
    assert physicalData.BODIES["Earth"].radius_km == 6378.14
    b, d_au = 0.5, 31.9
    expected = b / (math.atan2(6378.14, d_au * physicalData.AU_KM)
                    * 3600.0 * 180.0 / math.pi)
    assert sm.dist_from_impact_parameter(b, d_au) == pytest.approx(
        expected, rel=1e-14)


def test_offset_prediction_reference():
    # b and PA are au-free (exact); the time shift carries the au
    # difference: delta_t ~ 3840 s here, x 1.42e-5 ~ 0.055 s ~ 6.3e-7 d.
    for r in _rows("smoffset.csv"):
        res = sm.offset_prediction(
            "17 45 31.20", "-22 28 45.5", r["ra_off"], r["dec_off"],
            0.31, 157.5, "2015 06 29 16 55 00", 24.2, 31.9,
            retrograde=bool(int(r["retrograde"])),
            add_offset_to_star=bool(int(r["addoff"])))
        assert res.coord.ra.to_value(u.rad) == pytest.approx(
            float(r["ra_new_rad"]), abs=1e-12)
        assert res.coord.dec.to_value(u.rad) == pytest.approx(
            float(r["dec_new_rad"]), abs=1e-12)
        assert res.b_arcsec == pytest.approx(float(r["b_new_arcsec"]),
                                             abs=1e-9)
        assert res.pa_deg == pytest.approx(float(r["pa_new_deg"]), abs=1e-9)
        assert res.time.mjd == pytest.approx(float(r["dt_new_mjd"]), abs=2e-6)
        # formatted strings round-trip to the same values
        rt = sm.star_coord(res.ra_string, res.dec_string)
        assert rt.separation(res.coord).to_value(u.arcsec) < 1e-3


def test_offset_prediction_input_validation():
    with pytest.raises(ValueError):
        sm.offset_prediction("17 45 31.20", "-22 28 45.5", "00 00 00.35",
                             "-00 00 01.20", 0.0, 157.5,
                             "2015 06 29 16 55 00", 24.2, 31.9)
    with pytest.raises(ValueError):
        sm.offset_prediction("17 45 31.20", "-22 28 45.5", 0.0, 0.0,
                             0.31, 157.5, "2015 06 29 16 55 00", 24.2, 31.9)


# ---------------------------------------------------------------------------
# The full map
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("projection", sm.PROJECTIONS)
def test_globe_smoke(projection, tmp_path):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    ax = sm.globe("17 45 31.20 -22 28 45.5", "2015:06:29:16:55:00",
                  projection=projection, horizon_angle=18.0, tracks=True,
                  dist=0.42, pa=157.5, radius=0.18, pred_error=0.25,
                  plot_label=f"test {projection}")
    assert len(ax.lines) > 30           # coastline pieces
    out = tmp_path / f"globe_{projection}.png"
    ax.figure.savefig(out, dpi=72)
    plt.close(ax.figure)
    assert out.stat().st_size > 5000


def test_globe_input_forms_and_diagnostic(tmp_path, capsys):
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    star = SkyCoord(ra=266.38 * u.deg, dec=-22.479 * u.deg)
    t = Time(57202.70486111111, format="mjd", scale="utc")
    ax = sm.globe(star, t, tracks=True, dist=1.6, pa=200.0, radius=0.2,
                  print_diagnostic=True)
    assert "No tracks cross the earth" in capsys.readouterr().out
    plt.close(ax.figure)

    ax = sm.globe(("21:30:00", "-45:00:00"), "2003:07:01:10:00:00",
                  horizon_angle=6.0)
    ax.figure.savefig(tmp_path / "globe_tuple.png", dpi=72)
    plt.close(ax.figure)
